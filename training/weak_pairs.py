from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.rendering import find_font, line_pitch

from .review_corpus import load_snapshot_records, sha256_bytes


WEAK_PAIR_SCHEMA_VERSION = 1
WEAK_PAIR_DATASET_NAME = "yaruyomi-weak-v1"
VARIANT_VERSION = 1
DEFAULT_VARIANTS = ("skeleton", "contour", "simplified", "dropout")
V2_VARIANTS = (
    "skeleton",
    "tone_density",
    "tone_density_coarse",
    "tone_quantized",
    "tone_contour",
    "warped_tone",
)


def render_aa_text(text: str, *, font_size: int = 16) -> np.ndarray:
    font = ImageFont.truetype(str(find_font()), font_size)
    lines = text.splitlines() or [""]
    pitch = line_pitch(font_size)
    width = max(2, int(math.ceil(max(font.getlength(line) for line in lines))) + 2)
    canvas = Image.new("L", (width, max(pitch, len(lines) * pitch)), 255)
    draw = ImageDraw.Draw(canvas)
    ascent, _ = font.getmetrics()
    for index, line in enumerate(lines):
        draw.text((0, index * pitch + ascent), line, font=font, fill=0, anchor="ls")
    return np.asarray(canvas)


def _skeletonize(ink: np.ndarray) -> np.ndarray:
    work = np.where(ink > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    return skeleton


def _remove_small_components(ink: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    kept = np.zeros_like(ink)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            kept[labels == label] = 255
    return kept


def _prune_endpoints(ink: np.ndarray, iterations: int) -> np.ndarray:
    work = np.where(ink > 0, 1, 0).astype(np.uint8)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    for _ in range(iterations):
        neighbors = cv2.filter2D(work, cv2.CV_16U, neighbor_kernel)
        endpoints = (work > 0) & (neighbors <= 1)
        if not np.any(endpoints):
            break
        work[endpoints] = 0
    return work * 255


def _dropout_regions(
    ink: np.ndarray,
    *,
    rng: random.Random,
    erase_count: int,
    maximum_width: int,
    maximum_height: int,
    target_ink: bool = False,
    target_ink_ratio: float | None = None,
) -> np.ndarray:
    result = ink.copy()
    height, width = result.shape
    ink_positions = np.argwhere(result > 0) if target_ink else np.empty((0, 2))
    target_removed = (
        round(len(ink_positions) * target_ink_ratio)
        if target_ink_ratio is not None
        else 0
    )
    removed = 0
    operations = 0
    operation_limit = max(erase_count, 512 if target_removed else erase_count)
    while operations < operation_limit and (
        operations < erase_count or removed < target_removed
    ):
        operations += 1
        erase_width = rng.randint(2, maximum_width)
        erase_height = rng.randint(1, maximum_height)
        if len(ink_positions):
            center_y, center_x = ink_positions[rng.randrange(len(ink_positions))]
            left = max(0, int(center_x) - erase_width // 2)
            top = max(0, int(center_y) - erase_height // 2)
        else:
            left = rng.randrange(max(1, width))
            top = rng.randrange(max(1, height))
        right = min(width, left + erase_width)
        bottom = min(height, top + erase_height)
        removed += int(cv2.countNonZero(result[top:bottom, left:right]))
        result[top:bottom, left:right] = 0
    return result


def _dropout_line(ink: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    result = _skeletonize(ink)
    rng = random.Random(seed)
    height, width = result.shape
    erase_count = max(2, min(12, round((height * width) / 180_000)))
    maximum_width = max(3, width // 30)
    maximum_height = max(2, height // 45)
    result = _dropout_regions(
        result,
        rng=rng,
        erase_count=erase_count,
        maximum_width=maximum_width,
        maximum_height=maximum_height,
    )
    result = _remove_small_components(result, 3)
    return result, {
        "erase_count": erase_count,
        "maximum_erase_width": maximum_width,
        "maximum_erase_height": maximum_height,
        "minimum_component_area": 3,
    }


def _smooth_warp(ink: np.ndarray, *, rng: random.Random) -> tuple[np.ndarray, dict[str, float]]:
    height, width = ink.shape
    amplitude_x = max(1.0, min(3.0, width / 500))
    amplitude_y = max(0.75, min(2.0, height / 400))
    period_x = max(90.0, width / 3.2)
    period_y = max(70.0, height / 3.5)
    phase_x = rng.random() * 2 * math.pi
    phase_y = rng.random() * 2 * math.pi
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    map_x = grid_x + amplitude_x * np.sin(2 * math.pi * grid_y / period_y + phase_x)
    map_y = grid_y + amplitude_y * np.sin(2 * math.pi * grid_x / period_x + phase_y)
    warped = cv2.remap(
        ink,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = cv2.threshold(warped, 80, 255, cv2.THRESH_BINARY)[1]
    return warped, {
        "amplitude_x": round(amplitude_x, 3),
        "amplitude_y": round(amplitude_y, 3),
        "period_x": round(period_x, 3),
        "period_y": round(period_y, 3),
        "phase_x": round(phase_x, 6),
        "phase_y": round(phase_y, 6),
    }


def _tone_density(
    ink: np.ndarray,
    *,
    sigma: float,
    gain: float,
    sigma_y: float | None = None,
) -> np.ndarray:
    density = cv2.GaussianBlur(
        ink,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma_y if sigma_y is not None else sigma * 0.75,
    )
    return np.clip(density.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _connected_skeleton(ink: np.ndarray) -> np.ndarray:
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
    )
    connected = cv2.morphologyEx(
        horizontal,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    )
    return _skeletonize(_remove_small_components(connected, 4))


def generate_weak_variant(
    rendered: np.ndarray,
    variant: str,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    ink = cv2.threshold(rendered, 200, 255, cv2.THRESH_BINARY_INV)[1]
    if variant == "skeleton":
        result_ink = _skeletonize(ink)
        parameters: dict[str, object] = {"threshold": 200}
    elif variant == "contour":
        blurred = cv2.GaussianBlur(rendered, (0, 0), sigmaX=0.8, sigmaY=0.8)
        result_ink = cv2.Canny(blurred, 70, 170, L2gradient=True)
        parameters = {"sigma": 0.8, "canny_low": 70, "canny_high": 170}
    elif variant == "simplified":
        connected = cv2.morphologyEx(
            ink,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        result_ink = _skeletonize(_remove_small_components(connected, 6))
        parameters = {"closing_kernel": 3, "minimum_component_area": 6}
    elif variant == "dropout":
        result_ink, parameters = _dropout_line(ink, seed=seed)
    elif variant == "connected":
        result_ink = _connected_skeleton(ink)
        parameters = {
            "horizontal_closing_kernel": [3, 1],
            "vertical_closing_kernel": [1, 3],
            "minimum_component_area": 4,
        }
    elif variant == "endpoint_pruned":
        result_ink = _prune_endpoints(_skeletonize(ink), 2)
        result_ink = _remove_small_components(result_ink, 3)
        parameters = {"endpoint_pruning_iterations": 2, "minimum_component_area": 3}
    elif variant == "smooth_warp":
        rng = random.Random(seed)
        blurred = cv2.GaussianBlur(ink, (0, 0), sigmaX=0.65, sigmaY=0.65)
        smoothed = cv2.threshold(blurred, 72, 255, cv2.THRESH_BINARY)[1]
        warped, warp_parameters = _smooth_warp(smoothed, rng=rng)
        result_ink = _skeletonize(_remove_small_components(warped, 4))
        parameters = {
            "gaussian_sigma": 0.65,
            "threshold": 72,
            "minimum_component_area": 4,
            **warp_parameters,
        }
    elif variant in {"dropout_mild", "dropout_medium"}:
        result_ink = _skeletonize(ink)
        rng = random.Random(seed)
        height, width = result_ink.shape
        scale = 0.55 if variant == "dropout_mild" else 1.25
        erase_count = max(1, min(18, round((height * width) / 180_000 * scale)))
        maximum_width = max(3, round(width / (48 if variant == "dropout_mild" else 26)))
        maximum_height = max(2, round(height / (70 if variant == "dropout_mild" else 40)))
        result_ink = _dropout_regions(
            result_ink,
            rng=rng,
            erase_count=erase_count,
            maximum_width=maximum_width,
            maximum_height=maximum_height,
            target_ink=True,
            target_ink_ratio=0.01 if variant == "dropout_mild" else 0.03,
        )
        parameters = {
            "strength": "mild" if variant == "dropout_mild" else "medium",
            "target": "ink_pixel",
            "target_ink_ratio": 0.01 if variant == "dropout_mild" else 0.03,
            "erase_count": erase_count,
            "maximum_erase_width": maximum_width,
            "maximum_erase_height": maximum_height,
            "minimum_component_area": None,
        }
    elif variant == "tone_density":
        result_ink = _tone_density(ink, sigma=4.0, gain=2.4)
        parameters = {
            "tone_sigma_x": 4.0,
            "tone_sigma_y": 3.0,
            "tone_gain": 2.4,
            "tone_levels": "continuous",
        }
    elif variant == "tone_density_coarse":
        result_ink = _tone_density(ink, sigma=8.0, sigma_y=10.0, gain=2.7)
        parameters = {
            "tone_sigma_x": 8.0,
            "tone_sigma_y": 10.0,
            "tone_gain": 2.7,
            "tone_levels": "continuous",
        }
    elif variant == "tone_quantized":
        density = _tone_density(ink, sigma=7.0, sigma_y=10.0, gain=2.7)
        result_ink = np.zeros_like(density)
        result_ink[density >= 24] = 64
        result_ink[density >= 64] = 128
        result_ink[density >= 120] = 192
        result_ink[density >= 184] = 255
        parameters = {
            "tone_sigma_x": 7.0,
            "tone_sigma_y": 10.0,
            "tone_gain": 2.7,
            "tone_thresholds": [24, 64, 120, 184],
            "tone_ink_levels": [0, 64, 128, 192, 255],
        }
    elif variant == "line_tone":
        tone = _tone_density(ink, sigma=4.0, gain=2.0)
        result_ink = cv2.max(_skeletonize(ink), tone)
        parameters = {
            "line": "skeleton",
            "tone_sigma_x": 4.0,
            "tone_sigma_y": 3.0,
            "tone_gain": 2.0,
        }
    elif variant == "connected_tone":
        tone = _tone_density(ink, sigma=4.5, gain=2.0)
        result_ink = cv2.max(_connected_skeleton(ink), tone)
        parameters = {
            "line": "connected_skeleton",
            "horizontal_closing_kernel": [3, 1],
            "vertical_closing_kernel": [1, 3],
            "tone_sigma_x": 4.5,
            "tone_sigma_y": 3.375,
            "tone_gain": 2.0,
        }
    elif variant == "tone_contour":
        tone = _tone_density(ink, sigma=7.0, sigma_y=10.0, gain=2.5)
        tone_image = 255 - tone
        contour = cv2.Canny(tone_image, 12, 36, L2gradient=True)
        result_ink = cv2.max(tone, contour)
        parameters = {
            "tone_sigma_x": 7.0,
            "tone_sigma_y": 10.0,
            "tone_gain": 2.5,
            "tone_contour_canny_low": 12,
            "tone_contour_canny_high": 36,
        }
    elif variant == "warped_tone":
        rng = random.Random(seed)
        warped, warp_parameters = _smooth_warp(ink, rng=rng)
        tone = _tone_density(warped, sigma=7.0, sigma_y=10.0, gain=2.5)
        contour = cv2.Canny(255 - tone, 12, 36, L2gradient=True)
        result_ink = cv2.max(contour, tone)
        parameters = {
            "line": "warped_tone_contour",
            "tone_sigma_x": 7.0,
            "tone_sigma_y": 10.0,
            "tone_gain": 2.5,
            "tone_contour_canny_low": 12,
            "tone_contour_canny_high": 36,
            **warp_parameters,
        }
    else:
        raise ValueError(f"Unknown weak-pair variant: {variant}")
    return 255 - result_ink, parameters


def select_pilot_records(
    records: list[dict[str, object]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, object]]:
    accepted = [record for record in records if record["decision"] == "accept"]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in accepted:
        grouped[str(record["corrected_category"])].append(record)
    rng = random.Random(seed)
    for category_records in grouped.values():
        category_records.sort(key=lambda record: int(record["entry_id"]))
        rng.shuffle(category_records)
    selected: list[dict[str, object]] = []
    categories = sorted(grouped)
    while len(selected) < min(limit, len(accepted)):
        added = False
        for category in categories:
            if grouped[category] and len(selected) < limit:
                selected.append(grouped[category].pop())
                added = True
        if not added:
            break
    return selected


def generate_weak_pair_dataset(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    limit: int = 20,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    font_size: int = 16,
    seed: int = 42,
    dataset_name: str = WEAK_PAIR_DATASET_NAME,
    purpose: str = "preference pilot; not a real source-image pairing",
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Weak-pair output is not empty: {output_dir}")
    snapshot_payload = (snapshot_dir / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_payload)
    records = load_snapshot_records(snapshot_dir)
    selected = select_pilot_records(records, limit=limit, seed=seed)
    if not selected:
        raise ValueError("Snapshot has no accepted records")
    output_dir.mkdir(parents=True, exist_ok=True)
    font_hash = hashlib.sha256(find_font().read_bytes()).hexdigest()
    pair_records: list[dict[str, object]] = []
    for record in selected:
        entry_id = int(record["entry_id"])
        text_path = snapshot_dir / str(record["text"])
        text = text_path.read_text(encoding="utf-8")
        rendered = render_aa_text(text, font_size=font_size)
        rendered_path = Path("rendered") / f"{entry_id:07d}.png"
        (output_dir / rendered_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rendered).save(output_dir / rendered_path, optimize=True)
        rendered_payload = (output_dir / rendered_path).read_bytes()
        for variant_index, variant in enumerate(variants):
            variant_seed = seed * 1_000_003 + entry_id * 101 + variant_index
            candidate, parameters = generate_weak_variant(
                rendered,
                variant,
                seed=variant_seed,
            )
            candidate_path = Path("candidates") / f"{entry_id:07d}" / f"{variant}.png"
            (output_dir / candidate_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(candidate).save(output_dir / candidate_path, optimize=True)
            candidate_payload = (output_dir / candidate_path).read_bytes()
            pair_records.append(
                {
                    "schema_version": WEAK_PAIR_SCHEMA_VERSION,
                    "dataset": dataset_name,
                    "pair_type": "weak_pair",
                    "entry_id": entry_id,
                    "category": record["corrected_category"],
                    "split": record["split"],
                    "sensitive": record["sensitive"],
                    "source_text": str(record["text"]),
                    "source_text_sha256": record["unicode_text_sha256"],
                    "rendered": rendered_path.as_posix(),
                    "rendered_sha256": sha256_bytes(rendered_payload),
                    "candidate": candidate_path.as_posix(),
                    "candidate_sha256": sha256_bytes(candidate_payload),
                    "method": variant,
                    "method_version": VARIANT_VERSION,
                    "parameters": parameters,
                    "seed": variant_seed,
                    "font": str(find_font()),
                    "font_sha256": font_hash,
                    "font_size": font_size,
                    "rights_status": "unknown; local curation only",
                }
            )
    manifest_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in pair_records
    ).encode("utf-8")
    (output_dir / "pairs.jsonl").write_bytes(manifest_payload)
    category_counts = defaultdict(int)
    for record in selected:
        category_counts[str(record["corrected_category"])] += 1
    dataset: dict[str, object] = {
        "schema_version": WEAK_PAIR_SCHEMA_VERSION,
        "dataset": dataset_name,
        "snapshot": snapshot["snapshot"],
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "seed": seed,
        "font_size": font_size,
        "font_sha256": font_hash,
        "selected_aa_count": len(selected),
        "variants": list(variants),
        "weak_pair_count": len(pair_records),
        "category_counts": dict(sorted(category_counts.items())),
        "pairs": "pairs.jsonl",
        "pairs_sha256": sha256_bytes(manifest_payload),
        "rights_status": "unknown; local curation only",
        "purpose": purpose,
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return dataset
