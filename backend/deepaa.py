from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .aa_semantics import FILL_REPEAT_CHARACTERS
from .contracts import ConversionOptions, ConversionResult
from .image_io import decode_image, image_to_data_url
from .line_ops import (
    remove_small_components as _remove_small_components,
    skeletonize as _skeletonize,
)
from .random_forest import predict_probabilities as predict_random_forest
from .rendering import render_glyph_mask


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "deepaa-light.onnx"
START_LOCATOR_PATH = ROOT / "models" / "deepaa-start-locator.onnx"
RANDOM_FOREST_PATH = ROOT / "models" / "deepaa-random-forest.joblib"
CHAR_LIST_PATH = ROOT / "models" / "deepaa-charset.csv"

GLYPH_HEIGHT = 16
LINE_PITCH = 18
HALF_WIDTH = 8
CONTEXT_SIZE = 64
CONTEXT_MARGIN = (CONTEXT_SIZE - LINE_PITCH) // 2
PRIOR_CORRECTION_STRENGTH = 0.06
MODEL_COST_WEIGHT = 0.82
LOCAL_SHAPE_WEIGHT = 2.2
REPEAT_TRANSITION_WEIGHT = 0.09
ROW_REPEAT_WEIGHT = 1.4


@dataclass(frozen=True)
class DeepAAAssets:
    session: ort.InferenceSession
    characters: tuple[str, ...]
    frequencies: tuple[int, ...]
    glyphs: dict[str, np.ndarray]
    half_space_index: int


@dataclass(frozen=True)
class BeamState:
    position: int
    characters: tuple[str, ...]
    model_cost: float
    previous_half_space: bool
    last_character: str | None
    repeat_count: int


def _abstract_linework(
    ink: np.ndarray,
    options: ConversionOptions,
) -> np.ndarray:
    """Merge sub-character detail into sparse representative strokes."""
    if options.abstraction <= 0 or not np.any(ink):
        return ink
    profile_scale = {
        "lineart": 1.0,
        "person": 0.85,
        "auto": 0.65,
        "background": 0.25,
        "background_lineart": 0.25,
    }[options.profile]
    strength = min(1.0, options.abstraction / 100.0 * profile_scale)
    if strength < 0.04:
        return ink

    sigma = 0.45 + 0.95 * strength
    blurred = cv2.GaussianBlur(ink, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # A lower threshold expands nearby strokes until they share one region;
    # skeletonization then replaces that region with a representative centerline.
    merge_threshold = int(round(125 - 55 * strength))
    merged = cv2.threshold(
        blurred,
        merge_threshold,
        255,
        cv2.THRESH_BINARY,
    )[1]
    if strength >= 0.35:
        merged = cv2.morphologyEx(
            merged,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    abstracted = _skeletonize(merged)
    minimum_length = max(1, int(round(1 + 7 * strength)))
    abstracted = _remove_small_components(abstracted, minimum_length)

    if options.profile in {"person", "lineart"}:
        # Dense eyes collapse to a dot under ordinary skeletonization. Treat
        # the expected portrait face area as a semantic priority region and
        # retain its source linework while simplifying hair and clothing.
        height, width = ink.shape
        face_mask = np.zeros_like(ink)
        cv2.ellipse(
            face_mask,
            (round(width * 0.50), round(height * 0.40)),
            (max(1, round(width * 0.24)), max(1, round(height * 0.18))),
            0,
            0,
            360,
            255,
            -1,
        )
        abstracted = cv2.bitwise_or(
            cv2.bitwise_and(abstracted, cv2.bitwise_not(face_mask)),
            cv2.bitwise_and(ink, face_mask),
        )
    return abstracted


def _simplify_background_lineart(
    ink: np.ndarray,
    options: ConversionOptions,
) -> np.ndarray:
    """Keep architectural envelopes and long segments from dense clean line art."""
    if options.abstraction <= 0 or not np.any(ink):
        return ink

    strength = min(1.0, options.abstraction / 100.0)
    sigma = 0.9 + 1.45 * strength
    blurred = cv2.GaussianBlur(ink, (0, 0), sigmaX=sigma, sigmaY=sigma)
    merge_threshold = int(round(65 - 28 * strength))
    merged = cv2.threshold(
        blurred,
        merge_threshold,
        255,
        cv2.THRESH_BINARY,
    )[1]
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    # Unlike portrait simplification, use the envelope of merged detail. A
    # skeleton through a dense roof or stone texture creates a false mosaic.
    structural = cv2.Canny(merged, 40, 120)

    # Reintroduce genuine long architectural strokes without restoring short
    # brick, foliage, and ornament texture.
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(255 - ink)[0]
    minimum_length = 12 + 18 * strength
    if detected is not None:
        for x1, y1, x2, y2 in detected.reshape(-1, 4):
            if np.hypot(x2 - x1, y2 - y1) < minimum_length:
                continue
            cv2.line(
                structural,
                (round(float(x1)), round(float(y1))),
                (round(float(x2)), round(float(y2))),
                255,
                1,
                cv2.LINE_AA,
            )
    structural = cv2.threshold(structural, 1, 255, cv2.THRESH_BINARY)[1]
    return _remove_small_components(structural, options.min_component)


def _crop_image(
    image: np.ndarray,
    options: ConversionOptions,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape
    left = int(round(options.crop_x * width))
    top = int(round(options.crop_y * height))
    right = int(round((options.crop_x + options.crop_width) * width))
    bottom = int(round((options.crop_y + options.crop_height) * height))
    right = min(max(right, left + 1), width)
    bottom = min(max(bottom, top + 1), height)
    return image[top:bottom, left:right], (left, top, right, bottom)


def preprocess_for_deepaa(
    image: np.ndarray,
    options: ConversionOptions,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create the thin black-on-white line image expected by DeepAA."""
    normalized = options.normalized()
    cropped, crop_box = _crop_image(image, normalized)
    target_width = normalized.columns * HALF_WIDTH
    natural_height = int(round(cropped.shape[0] * target_width / cropped.shape[1]))
    rows = min(
        normalized.max_rows,
        max(1, int(round(natural_height / LINE_PITCH))),
    )
    target_height = rows * LINE_PITCH

    resized = cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    sigma = 0.8 + (100 - normalized.detail) / 24.0
    specialized_abstraction = False
    if normalized.profile in {"lineart", "background_lineart"}:
        # The input is already a clean drawing. Avoid Canny, which would turn
        # every stroke into two parallel edges after resizing.
        line_threshold = min(250, max(180, int(205 + normalized.detail * 0.42)))
        ink = cv2.threshold(
            resized,
            line_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )[1]
        ink = _remove_small_components(ink, max(1, normalized.min_component // 2))
        if normalized.profile == "background_lineart":
            edges = _simplify_background_lineart(ink, normalized)
            specialized_abstraction = True
        else:
            edges = ink
    elif normalized.profile == "person":
        smoothed = cv2.GaussianBlur(
            resized,
            (0, 0),
            sigmaX=max(0.65, sigma * 0.68),
            sigmaY=max(0.65, sigma * 0.68),
        )
        equalized = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(
            smoothed
        )
        fine = cv2.Canny(
            equalized,
            max(0, int(normalized.threshold_low * 0.68)),
            max(1, int(normalized.threshold_high * 0.82)),
            L2gradient=True,
        )
        coarse = cv2.Canny(
            cv2.GaussianBlur(equalized, (0, 0), sigmaX=1.35, sigmaY=1.35),
            normalized.threshold_low,
            normalized.threshold_high,
            L2gradient=True,
        )
        edges = cv2.bitwise_or(fine, coarse)
        edges = _remove_small_components(edges, max(1, normalized.min_component // 2))
    elif normalized.profile == "background":
        smoothed = cv2.GaussianBlur(
            resized,
            (0, 0),
            sigmaX=sigma * 0.92,
            sigmaY=sigma * 0.92,
        )
        equalized = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(10, 8)).apply(
            smoothed
        )
        edges = cv2.Canny(
            equalized,
            normalized.threshold_low,
            normalized.threshold_high,
            L2gradient=True,
        )
        structural = cv2.Canny(
            cv2.GaussianBlur(equalized, (0, 0), sigmaX=1.55, sigmaY=1.55),
            max(0, int(normalized.threshold_low * 0.72)),
            max(1, int(normalized.threshold_high * 0.86)),
            L2gradient=True,
        )
        edges = cv2.bitwise_or(edges, structural)
        lines = cv2.HoughLinesP(
            structural,
            1,
            np.pi / 180,
            threshold=18,
            minLineLength=max(12, target_width // 55),
            maxLineGap=4,
        )
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(edges, (x1, y1), (x2, y2), 255, 1, cv2.LINE_8)
        edges = _remove_small_components(edges, max(2, normalized.min_component))
    else:
        smoothed = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
        equalized = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(
            smoothed
        )
        edges = cv2.Canny(
            equalized,
            normalized.threshold_low,
            normalized.threshold_high,
            L2gradient=True,
        )
        edges = _remove_small_components(edges, normalized.min_component)
    if normalized.profile not in {"lineart", "background_lineart"} and normalized.detail < 75:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if not specialized_abstraction:
        edges = _abstract_linework(edges, normalized)
    return 255 - edges, crop_box


@lru_cache(maxsize=1)
def load_assets() -> DeepAAAssets:
    missing = [
        str(path)
        for path in (MODEL_PATH, CHAR_LIST_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("DeepAA assets are missing: " + ", ".join(missing))

    with CHAR_LIST_PATH.open("r", encoding="cp932", newline="") as handle:
        rows = csv.DictReader(handle)
        selected_rows = [row for row in rows if int(row["frequency"]) >= 10]
        characters = tuple(row["char"] for row in selected_rows)
        frequencies = tuple(int(row["frequency"]) for row in selected_rows)
    if len(characters) != 411:
        raise ValueError(f"Expected 411 DeepAA characters, got {len(characters)}")

    glyphs = {
        char: render_glyph_mask(char, GLYPH_HEIGHT, GLYPH_HEIGHT)
        for char in characters
    }

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return DeepAAAssets(
        session=session,
        characters=characters,
        frequencies=frequencies,
        glyphs=glyphs,
        half_space_index=characters.index(" "),
    )


def _adjusted_log_scores(
    scores: np.ndarray,
    assets: DeepAAAssets,
    diversity_strength: float = 1.0,
) -> np.ndarray:
    """Apply a conservative correction for the training-set class prior."""
    log_scores = np.log(np.clip(scores.astype(np.float64), 1e-8, 1.0))
    frequencies = np.asarray(assets.frequencies, dtype=np.float64)
    rarity_boost = diversity_strength * PRIOR_CORRECTION_STRENGTH * np.log(
        frequencies.max() / frequencies
    )
    # Never turn a probability into a negative transition cost.
    return np.minimum(log_scores + rarity_boost, 0.0)


def _predict_character_probabilities(
    windows: np.ndarray,
    assets: DeepAAAssets,
    *,
    classifier: str,
    random_forest_model: Path | None,
) -> np.ndarray:
    if classifier == "cnn":
        inputs = (windows.astype(np.float32) / 255.0)[..., np.newaxis]
        return assets.session.run(["probabilities"], {"input": inputs})[0]
    if classifier == "random-forest":
        return predict_random_forest(
            windows,
            assets.characters,
            random_forest_model or RANDOM_FOREST_PATH,
        )
    raise ValueError("classifier must be 'cnn' or 'random-forest'")


def _candidate_indices(
    scores: np.ndarray,
    assets: DeepAAAssets,
    top_k: int,
    diversity_strength: float = 1.0,
) -> np.ndarray:
    """Keep both raw-model favorites and prior-corrected alternatives."""
    count = min(max(1, top_k), len(assets.characters))
    raw = np.argpartition(scores, -count)[-count:]
    if diversity_strength <= 0:
        return raw[np.argsort(scores[raw])[::-1]]
    adjusted = _adjusted_log_scores(scores, assets, diversity_strength)
    calibrated = np.argpartition(adjusted, -count)[-count:]
    combined = np.unique(np.concatenate((raw, calibrated)))
    return combined[np.argsort(adjusted[combined])[::-1]]


def _local_shape_cost(
    target_patch: np.ndarray,
    glyph_ink: np.ndarray,
    density_weight: float = 1.25,
    structure_strength: float = 0.0,
) -> float:
    missed = float(np.logical_and(target_patch, ~glyph_ink).mean())
    extra = float(np.logical_and(glyph_ink, ~target_patch).mean())
    target_density = float(target_patch.mean())
    glyph_density = float(glyph_ink.mean())
    # A small tolerance avoids punishing the antialiasing/binarization delta.
    density_excess = max(0.0, glyph_density - target_density - 0.015)
    pixel_cost = 1.35 * missed + extra + density_weight * density_excess
    return pixel_cost + 0.45 * structure_strength * _structure_mismatch_cost(
        target_patch,
        glyph_ink,
    )


def _direction_histogram(ink: np.ndarray) -> np.ndarray:
    if not np.any(ink):
        return np.zeros(4, dtype=np.float64)
    image = ink.astype(np.float32)
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    active = magnitude > 1e-4
    if not np.any(active):
        return np.zeros(4, dtype=np.float64)
    # Gradient direction is normal to the stroke. Modulo pi keeps opposite
    # edge normals in the same line-orientation bin.
    angles = np.mod(np.arctan2(gradient_y[active], gradient_x[active]), np.pi)
    bins = np.minimum((angles * (4.0 / np.pi)).astype(np.int32), 3)
    histogram = np.bincount(
        bins,
        weights=magnitude[active],
        minlength=4,
    ).astype(np.float64)
    total = histogram.sum()
    return histogram / total if total else histogram


def _profile_mismatch(target: np.ndarray, glyph: np.ndarray) -> float:
    target = target.astype(bool)
    glyph = glyph.astype(bool)
    if not target.any() and not glyph.any():
        return 0.0

    def expand(profile: np.ndarray) -> np.ndarray:
        padded = np.pad(profile.astype(bool), 1, mode="constant")
        return padded[:-2] | padded[1:-1] | padded[2:]

    target_near = expand(target)
    glyph_near = expand(glyph)
    missed = np.logical_and(target, ~glyph_near).mean()
    extra = np.logical_and(glyph, ~target_near).mean()
    return float(missed + extra)


def _structure_mismatch_cost(
    target_patch: np.ndarray,
    glyph_ink: np.ndarray,
) -> float:
    """Measure line direction and entry/exit disagreement for one glyph cell."""
    if target_patch.shape != glyph_ink.shape:
        raise ValueError("Target and glyph patches must have the same shape")
    direction_cost = 0.5 * float(
        np.abs(
            _direction_histogram(target_patch) - _direction_histogram(glyph_ink)
        ).sum()
    )
    edge_profiles = (
        (target_patch[:, :2].any(axis=1), glyph_ink[:, :2].any(axis=1)),
        (target_patch[:, -2:].any(axis=1), glyph_ink[:, -2:].any(axis=1)),
        (target_patch[:2, :].any(axis=0), glyph_ink[:2, :].any(axis=0)),
        (target_patch[-2:, :].any(axis=0), glyph_ink[-2:, :].any(axis=0)),
    )
    profile_costs = [
        _profile_mismatch(target, glyph) for target, glyph in edge_profiles
    ]
    # Horizontal character placement makes the left/right boundaries more
    # important, while top/bottom still preserve cross-row structure.
    boundary_cost = 0.35 * (profile_costs[0] + profile_costs[1]) + 0.15 * (
        profile_costs[2] + profile_costs[3]
    )
    return direction_cost + boundary_cost


def _repeat_transition_cost(
    last_character: str | None,
    repeat_count: int,
    char: str,
) -> tuple[float, int]:
    next_count = repeat_count + 1 if char == last_character else 1
    if char in FILL_REPEAT_CHARACTERS:
        return 0.0, next_count
    excess = max(0, next_count - 2)
    return REPEAT_TRANSITION_WEIGHT * (excess ** 1.35), next_count


def _sequence_repeat_cost(characters: tuple[str, ...] | list[str]) -> float:
    if not characters:
        return 0.0
    total = 0.0
    run_length = 1
    for previous, current in zip(characters, characters[1:]):
        if current == previous:
            run_length += 1
        else:
            if previous not in FILL_REPEAT_CHARACTERS:
                total += max(0, run_length - 3) ** 1.25
            run_length = 1
    if characters[-1] not in FILL_REPEAT_CHARACTERS:
        total += max(0, run_length - 3) ** 1.25
    return total / len(characters)


@lru_cache(maxsize=1)
def load_start_locator() -> ort.InferenceSession:
    if not START_LOCATOR_PATH.exists():
        raise FileNotFoundError(f"DeepAA start locator is missing: {START_LOCATOR_PATH}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(START_LOCATOR_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def generate_text(
    line_image: np.ndarray,
    *,
    vertical_offset: int = 0,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
) -> tuple[str, list[list[str]]]:
    """Run the original DeepAA-style variable-width greedy decoder."""
    if line_image.ndim != 2:
        raise ValueError("DeepAA expects a two-dimensional grayscale line image.")
    if line_image.shape[0] % LINE_PITCH:
        raise ValueError("DeepAA line image height must be a multiple of 18 pixels.")
    if not 0 <= vertical_offset < LINE_PITCH:
        raise ValueError("vertical_offset must be between 0 and 17.")

    assets = load_assets()
    rows = line_image.shape[0] // LINE_PITCH
    target_width = line_image.shape[1]
    padded = np.full(
        (
            line_image.shape[0] + 2 * CONTEXT_MARGIN + LINE_PITCH,
            target_width + 2 * CONTEXT_MARGIN + LINE_PITCH,
        ),
        255,
        dtype=np.uint8,
    )
    source_top = CONTEXT_MARGIN + vertical_offset
    padded[
        source_top : source_top + line_image.shape[0],
        CONTEXT_MARGIN : CONTEXT_MARGIN + target_width,
    ] = line_image

    positions = np.zeros(rows, dtype=np.int32)
    previous_half_space = np.zeros(rows, dtype=bool)
    output: list[list[str]] = [[] for _ in range(rows)]
    active = list(range(rows))

    # Batch one current window from every active row. This preserves the
    # original variable-width greedy path without one runtime call per glyph.
    while active:
        windows = np.stack(
            [
                padded[
                    row * LINE_PITCH : row * LINE_PITCH + CONTEXT_SIZE,
                    positions[row] : positions[row] + CONTEXT_SIZE,
                ]
                for row in active
            ]
        )
        probabilities = _predict_character_probabilities(
            windows,
            assets,
            classifier=classifier,
            random_forest_model=random_forest_model,
        )

        next_active: list[int] = []
        for batch_index, row in enumerate(active):
            scores = probabilities[batch_index]
            if previous_half_space[row]:
                scores = scores.copy()
                scores[assets.half_space_index] = -1.0
            class_index = int(np.argmax(scores))
            char = assets.characters[class_index]
            output[row].append(char)
            previous_half_space[row] = class_index == assets.half_space_index
            positions[row] += assets.glyphs[char].shape[1]
            if positions[row] < target_width:
                next_active.append(row)
        active = next_active

    lines = ["".join(line).rstrip() for line in output]
    return "\n".join(lines).rstrip() + "\n", output


def _padded_line_image(line_image: np.ndarray, vertical_offset: int) -> np.ndarray:
    target_width = line_image.shape[1]
    padded = np.full(
        (
            line_image.shape[0] + 2 * CONTEXT_MARGIN + LINE_PITCH,
            target_width + 2 * CONTEXT_MARGIN + LINE_PITCH,
        ),
        255,
        dtype=np.uint8,
    )
    source_top = CONTEXT_MARGIN + vertical_offset
    padded[
        source_top : source_top + line_image.shape[0],
        CONTEXT_MARGIN : CONTEXT_MARGIN + target_width,
    ] = line_image
    return padded


def _render_row(
    characters: tuple[str, ...] | list[str],
    target_width: int,
    glyphs: dict[str, np.ndarray],
) -> np.ndarray:
    canvas = np.full((LINE_PITCH, target_width), 255, dtype=np.uint8)
    x = 0
    for char in characters:
        glyph = glyphs[char]
        width = min(glyph.shape[1], target_width - x)
        if width <= 0:
            break
        patch = canvas[:GLYPH_HEIGHT, x : x + width]
        patch[glyph[:, :width].astype(bool)] = 0
        x += glyph.shape[1]
    return canvas


def generate_text_beam(
    line_image: np.ndarray,
    *,
    vertical_offset: int,
    beam_width: int = 4,
    top_k: int = 6,
    diversity_strength: float = 1.0,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
) -> tuple[str, list[list[str]]]:
    """Decode multiple variable-width paths and select by rendered line fit."""
    if beam_width < 1 or top_k < 1:
        raise ValueError("beam_width and top_k must be positive.")
    assets = load_assets()
    rows = line_image.shape[0] // LINE_PITCH
    target_width = line_image.shape[1]
    padded = _padded_line_image(line_image, vertical_offset)
    active: list[list[BeamState]] = [
        [BeamState(0, (), 0.0, False, None, 0)] for _ in range(rows)
    ]
    completed: list[list[BeamState]] = [[] for _ in range(rows)]

    for _ in range(target_width // 3 + 1):
        locations = {
            (row, state.position)
            for row, states in enumerate(active)
            for state in states
            if state.position < target_width
        }
        if not locations:
            break
        ordered_locations = sorted(locations)
        windows = np.stack(
            [
                padded[
                    row * LINE_PITCH : row * LINE_PITCH + CONTEXT_SIZE,
                    x : x + CONTEXT_SIZE,
                ]
                for row, x in ordered_locations
            ]
        )
        probabilities = _predict_character_probabilities(
            windows,
            assets,
            classifier=classifier,
            random_forest_model=random_forest_model,
        )
        probability_at = {
            location: probabilities[index]
            for index, location in enumerate(ordered_locations)
        }

        any_active = False
        for row, states in enumerate(active):
            candidates: list[BeamState] = []
            target_row = line_image[
                row * LINE_PITCH : (row + 1) * LINE_PITCH
            ]
            for state in states:
                if state.position >= target_width:
                    completed[row].append(state)
                    continue
                scores = probability_at[(row, state.position)]
                adjusted_log_scores = _adjusted_log_scores(
                    scores,
                    assets,
                    diversity_strength,
                )
                candidate_indices = _candidate_indices(
                    scores,
                    assets,
                    top_k,
                    diversity_strength,
                )
                for class_index_value in candidate_indices:
                    class_index = int(class_index_value)
                    if (
                        state.previous_half_space
                        and class_index == assets.half_space_index
                    ):
                        continue
                    char = assets.characters[class_index]
                    glyph = assets.glyphs[char]
                    width = glyph.shape[1]
                    visible_width = min(width, target_width - state.position)
                    if visible_width <= 0:
                        continue
                    target_patch = target_row[
                        :GLYPH_HEIGHT,
                        state.position : state.position + visible_width,
                    ] < 128
                    glyph_ink = glyph[:, :visible_width].astype(bool)
                    local_cost = _local_shape_cost(
                        target_patch,
                        glyph_ink,
                        # Backgrounds may legitimately repeat line glyphs, but
                        # that must not disable the penalty for glyphs that are
                        # much darker than the source patch.
                        density_weight=1.25,
                        structure_strength=structure_strength,
                    )
                    repeat_cost, repeat_count = _repeat_transition_cost(
                        state.last_character,
                        state.repeat_count,
                        char,
                    )
                    candidate = BeamState(
                        position=state.position + width,
                        characters=state.characters + (char,),
                        model_cost=(
                            state.model_cost
                            - (1.0 - 0.18 * diversity_strength)
                            * adjusted_log_scores[class_index]
                            + (1.75 + 0.45 * diversity_strength) * local_cost
                            + diversity_strength * repeat_cost
                        ),
                        previous_half_space=(
                            class_index == assets.half_space_index
                        ),
                        last_character=char,
                        repeat_count=repeat_count,
                    )
                    if candidate.position >= target_width:
                        completed[row].append(candidate)
                    else:
                        candidates.append(candidate)

            # Every state expanded by one character, so raw cumulative model
            # cost is comparable at this pruning point.
            candidates.sort(key=lambda item: item.model_cost)
            active[row] = candidates[:beam_width]
            any_active = any_active or bool(active[row])
        if not any_active:
            break

    predictions: list[list[str]] = []
    for row in range(rows):
        choices = completed[row] or active[row]
        if not choices:
            predictions.append([])
            continue
        target_row = line_image[row * LINE_PITCH : (row + 1) * LINE_PITCH]

        def final_cost(state: BeamState) -> float:
            rendered = _render_row(state.characters, target_width, assets.glyphs)
            confidence_tiebreaker = 0.015 * state.model_cost / max(
                1, len(state.characters)
            )
            return (
                line_match_cost(target_row, rendered)
                + diversity_strength
                * ROW_REPEAT_WEIGHT
                * _sequence_repeat_cost(state.characters)
                + confidence_tiebreaker
            )

        best = min(choices, key=final_cost)
        predictions.append(list(best.characters))

    lines = ["".join(line).rstrip() for line in predictions]
    return "\n".join(lines).rstrip() + "\n", predictions


def _infer_row_probabilities(
    padded: np.ndarray,
    row: int,
    target_width: int,
    assets: DeepAAAssets,
    *,
    batch_size: int = 512,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    for start in range(0, target_width, batch_size):
        stop = min(target_width, start + batch_size)
        windows = np.stack(
            [
                padded[
                    row * LINE_PITCH : row * LINE_PITCH + CONTEXT_SIZE,
                    x : x + CONTEXT_SIZE,
                ]
                for x in range(start, stop)
            ]
        )
        probabilities.append(
            _predict_character_probabilities(
                windows,
                assets,
                classifier=classifier,
                random_forest_model=random_forest_model,
            )
        )
    return np.concatenate(probabilities, axis=0)


def _decode_row_viterbi(
    target_row: np.ndarray,
    probabilities: np.ndarray | dict[int, np.ndarray],
    assets: DeepAAAssets,
    *,
    top_k: int,
    structure_strength: float = 0.0,
) -> list[str]:
    """Find the minimum-cost variable-width path across one complete row."""
    target_width = target_row.shape[1]
    maximum_width = max(assets.glyphs[char].shape[1] for char in assets.characters)
    limit = target_width + maximum_width
    costs = np.full((limit + 1, 2), np.inf, dtype=np.float64)
    previous_position = np.full((limit + 1, 2), -1, dtype=np.int32)
    previous_space_state = np.full((limit + 1, 2), -1, dtype=np.int8)
    previous_character = np.full((limit + 1, 2), -1, dtype=np.int32)
    costs[0, 0] = 0.0

    if isinstance(probabilities, np.ndarray):
        scores_at = {position: probabilities[position] for position in range(target_width)}
    else:
        scores_at = probabilities
    candidates_by_position = {
        position: _candidate_indices(scores, assets, top_k)
        for position, scores in scores_at.items()
    }

    for position in sorted(scores_at):
        if position >= target_width:
            continue
        for was_half_space in (0, 1):
            current_cost = costs[position, was_half_space]
            if not np.isfinite(current_cost):
                continue
            scores = scores_at[position]
            adjusted_log_scores = _adjusted_log_scores(scores, assets)
            for class_index_value in candidates_by_position[position]:
                class_index = int(class_index_value)
                is_half_space = int(class_index == assets.half_space_index)
                if was_half_space and is_half_space:
                    continue
                char = assets.characters[class_index]
                glyph = assets.glyphs[char]
                glyph_width = glyph.shape[1]
                visible_width = min(glyph_width, target_width - position)
                if visible_width <= 0:
                    continue
                target_patch = target_row[
                    :GLYPH_HEIGHT,
                    position : position + visible_width,
                ] < 128
                glyph_ink = glyph[:, :visible_width].astype(bool)
                local_cost = _local_shape_cost(
                    target_patch,
                    glyph_ink,
                    structure_strength=structure_strength,
                )
                coverage = visible_width / HALF_WIDTH
                transition_cost = coverage * (
                    -MODEL_COST_WEIGHT * adjusted_log_scores[class_index]
                    + LOCAL_SHAPE_WEIGHT * local_cost
                )
                next_position = position + glyph_width
                new_cost = current_cost + transition_cost
                if next_position > target_width:
                    new_cost += 0.08 * (next_position - target_width)
                if new_cost < costs[next_position, is_half_space]:
                    costs[next_position, is_half_space] = new_cost
                    previous_position[next_position, is_half_space] = position
                    previous_space_state[next_position, is_half_space] = was_half_space
                    previous_character[next_position, is_half_space] = class_index

    end_position, end_space_state = min(
        (
            (position, space_state)
            for position in range(target_width, limit + 1)
            for space_state in (0, 1)
            if np.isfinite(costs[position, space_state])
        ),
        key=lambda state: costs[state],
    )
    characters: list[str] = []
    position = end_position
    space_state = end_space_state
    while position > 0:
        class_index = int(previous_character[position, space_state])
        if class_index < 0:
            raise RuntimeError("Viterbi backtracking encountered an incomplete path.")
        characters.append(assets.characters[class_index])
        next_position = int(previous_position[position, space_state])
        space_state = int(previous_space_state[position, space_state])
        position = next_position
    characters.reverse()
    return characters


def generate_text_viterbi(
    line_image: np.ndarray,
    *,
    vertical_offset: int,
    top_k: int = 8,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
) -> tuple[str, list[list[str]]]:
    """Decode every row with a full-width dynamic-programming search."""
    if line_image.ndim != 2:
        raise ValueError("DeepAA expects a two-dimensional grayscale line image.")
    if line_image.shape[0] % LINE_PITCH:
        raise ValueError("DeepAA line image height must be a multiple of 18 pixels.")
    if not 0 <= vertical_offset < LINE_PITCH:
        raise ValueError("vertical_offset must be between 0 and 17.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    assets = load_assets()
    rows = line_image.shape[0] // LINE_PITCH
    target_width = line_image.shape[1]
    padded = _padded_line_image(line_image, vertical_offset)
    predictions: list[list[str]] = []
    for row in range(rows):
        probabilities = _infer_row_probabilities(
            padded,
            row,
            target_width,
            assets,
            classifier=classifier,
            random_forest_model=random_forest_model,
        )
        target_row = line_image[row * LINE_PITCH : (row + 1) * LINE_PITCH]
        predictions.append(
            _decode_row_viterbi(
                target_row,
                probabilities,
                assets,
                top_k=top_k,
                structure_strength=structure_strength,
            )
        )
    lines = ["".join(line).rstrip() for line in predictions]
    return "\n".join(lines).rstrip() + "\n", predictions


def generate_text_viterbi_pruned(
    line_image: np.ndarray,
    *,
    vertical_offset: int,
    baseline: list[list[str]],
    start_threshold: float = 0.2,
    top_k: int = 8,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
) -> tuple[str, list[list[str]]]:
    """Use a fast line locator, then run the heavy CNN only at candidate starts."""
    assets = load_assets()
    rows = line_image.shape[0] // LINE_PITCH
    target_width = line_image.shape[1]
    if len(baseline) != rows:
        raise ValueError("The baseline row count does not match the line image.")
    padded = _padded_line_image(line_image, vertical_offset)
    contexts = np.stack(
        [
            padded[
                row * LINE_PITCH : row * LINE_PITCH + CONTEXT_SIZE,
                CONTEXT_MARGIN : CONTEXT_MARGIN + target_width,
            ]
            for row in range(rows)
        ]
    ).astype(np.float32)
    start_logits = load_start_locator().run(
        ["start_logits"],
        {"line_context": (contexts / 255.0)[:, np.newaxis]},
    )[0]
    start_probabilities = 1.0 / (1.0 + np.exp(-np.clip(start_logits, -30, 30)))

    candidate_positions: list[set[int]] = []
    for row, baseline_line in enumerate(baseline):
        candidates = set(
            int(position)
            for position in np.flatnonzero(start_probabilities[row] >= start_threshold)
        )
        position = 0
        candidates.add(0)
        for char in baseline_line:
            candidates.add(position)
            position += assets.glyphs[char].shape[1]
        candidate_positions.append(
            {position for position in candidates if position < target_width}
        )

    locations = [
        (row, position)
        for row, positions in enumerate(candidate_positions)
        for position in sorted(positions)
    ]
    probabilities_at: dict[tuple[int, int], np.ndarray] = {}
    for start in range(0, len(locations), 512):
        batch_locations = locations[start : start + 512]
        windows = np.stack(
            [
                padded[
                    row * LINE_PITCH : row * LINE_PITCH + CONTEXT_SIZE,
                    position : position + CONTEXT_SIZE,
                ]
                for row, position in batch_locations
            ]
        )
        batch_probabilities = _predict_character_probabilities(
            windows,
            assets,
            classifier=classifier,
            random_forest_model=random_forest_model,
        )
        probabilities_at.update(
            {
                location: batch_probabilities[index]
                for index, location in enumerate(batch_locations)
            }
        )

    predictions: list[list[str]] = []
    for row in range(rows):
        row_probabilities = {
            position: probabilities_at[(row, position)]
            for position in candidate_positions[row]
        }
        target_row = line_image[row * LINE_PITCH : (row + 1) * LINE_PITCH]
        predictions.append(
            _decode_row_viterbi(
                target_row,
                row_probabilities,
                assets,
                top_k=top_k,
                structure_strength=structure_strength,
            )
        )
    lines = ["".join(line).rstrip() for line in predictions]
    return "\n".join(lines).rstrip() + "\n", predictions


def _select_rows_by_render_cost(
    line_image: np.ndarray,
    baseline: list[list[str]],
    candidate: list[list[str]],
    glyphs: dict[str, np.ndarray],
    *,
    repeat_weight: float = ROW_REPEAT_WEIGHT,
    structure_strength: float = 0.0,
) -> list[list[str]]:
    target_width = line_image.shape[1]
    selected: list[list[str]] = []
    for row, (baseline_line, candidate_line) in enumerate(zip(baseline, candidate)):
        target = line_image[row * LINE_PITCH : (row + 1) * LINE_PITCH]
        selected.append(
            candidate_line
            if _row_prediction_cost(
                target,
                candidate_line,
                glyphs,
                repeat_weight=repeat_weight,
                structure_strength=structure_strength,
            )
            <= _row_prediction_cost(
                target,
                baseline_line,
                glyphs,
                repeat_weight=repeat_weight,
                structure_strength=structure_strength,
            )
            else baseline_line
        )
    return selected


def _row_prediction_cost(
    target_row: np.ndarray,
    characters: tuple[str, ...] | list[str],
    glyphs: dict[str, np.ndarray],
    *,
    repeat_weight: float = ROW_REPEAT_WEIGHT,
    structure_strength: float = 0.0,
) -> float:
    rendered = _render_row(characters, target_row.shape[1], glyphs)
    return (
        line_match_cost(target_row, rendered)
        + repeat_weight * _sequence_repeat_cost(characters)
        + 0.55
        * structure_strength
        * _row_structure_cost(target_row, characters, glyphs)
    )


def _row_structure_cost(
    target_row: np.ndarray,
    characters: tuple[str, ...] | list[str],
    glyphs: dict[str, np.ndarray],
) -> float:
    x = 0
    weighted_cost = 0.0
    visible_pixels = 0
    for char in characters:
        glyph = glyphs[char]
        width = min(glyph.shape[1], target_row.shape[1] - x)
        if width <= 0:
            break
        target_patch = target_row[:GLYPH_HEIGHT, x : x + width] < 128
        glyph_ink = glyph[:, :width].astype(bool)
        weighted_cost += width * _structure_mismatch_cost(target_patch, glyph_ink)
        visible_pixels += width
        x += glyph.shape[1]
    return weighted_cost / visible_pixels if visible_pixels else 0.0


def _predictions_cost(
    line_image: np.ndarray,
    predictions: list[list[str]],
    glyphs: dict[str, np.ndarray],
    *,
    repeat_weight: float = ROW_REPEAT_WEIGHT,
    structure_strength: float = 0.0,
) -> float:
    if not predictions:
        return float("inf")
    costs = [
        _row_prediction_cost(
            line_image[row * LINE_PITCH : (row + 1) * LINE_PITCH],
            characters,
            glyphs,
            repeat_weight=repeat_weight,
            structure_strength=structure_strength,
        )
        for row, characters in enumerate(predictions)
    ]
    return float(np.mean(costs))


def render_predictions(
    predictions: list[list[str]],
    target_width: int,
    glyphs: dict[str, np.ndarray] | None = None,
) -> Image.Image:
    glyphs = glyphs or load_assets().glyphs
    canvas = np.full(
        (max(1, len(predictions) * LINE_PITCH), max(1, target_width)),
        255,
        dtype=np.uint8,
    )
    for row, characters in enumerate(predictions):
        x = 0
        for char in characters:
            glyph = glyphs[char]
            width = min(glyph.shape[1], target_width - x)
            if width <= 0:
                break
            ink = glyph[:, :width].astype(bool)
            patch = canvas[
                row * LINE_PITCH : row * LINE_PITCH + GLYPH_HEIGHT,
                x : x + width,
            ]
            patch[ink] = 0
            x += glyph.shape[1]
    return Image.fromarray(canvas)


def line_match_cost(target: np.ndarray, rendered: np.ndarray) -> float:
    """Symmetric distance-transform cost between target and rendered ink."""
    target_ink = target < 128
    rendered_ink = rendered < 128
    if not target_ink.any():
        return float(rendered_ink.mean())
    if not rendered_ink.any():
        return 10.0

    distance_to_target = cv2.distanceTransform(
        (~target_ink).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distance_to_rendered = cv2.distanceTransform(
        (~rendered_ink).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    missed = float(np.minimum(distance_to_rendered[target_ink], 8.0).mean())
    extra = float(np.minimum(distance_to_target[rendered_ink], 8.0).mean())
    return 1.35 * missed + extra


def choose_vertical_alignment(
    line_image: np.ndarray,
    *,
    repeat_weight: float = ROW_REPEAT_WEIGHT,
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
) -> tuple[int, str, list[list[str]], Image.Image, float]:
    """Evaluate all 18 legal line-grid offsets and keep the best rendering."""
    best: tuple[int, str, list[list[str]], Image.Image, float] | None = None
    for offset in range(LINE_PITCH):
        text, predictions = generate_text(
            line_image,
            vertical_offset=offset,
            classifier=classifier,
            random_forest_model=random_forest_model,
        )
        rendered = render_predictions(predictions, line_image.shape[1])
        cost = _predictions_cost(
            line_image,
            predictions,
            load_assets().glyphs,
            repeat_weight=repeat_weight,
            structure_strength=structure_strength,
        )
        candidate = (offset, text, predictions, rendered, cost)
        if best is None or cost < best[4]:
            best = candidate
    assert best is not None
    return best


def convert_deepaa(
    data: bytes,
    options: ConversionOptions,
    *,
    decoder: str = "beam",
    classifier: str = "cnn",
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
    line_image_override: np.ndarray | None = None,
    crop_box_override: tuple[int, int, int, int] | None = None,
) -> ConversionResult:
    if decoder not in {"beam", "viterbi", "viterbi-pruned"}:
        raise ValueError(f"Unknown DeepAA decoder: {decoder}")
    if classifier not in {"cnn", "random-forest"}:
        raise ValueError(f"Unknown DeepAA classifier: {classifier}")
    if not 0.0 <= structure_strength <= 2.0:
        raise ValueError("structure_strength must be between 0 and 2")
    normalized = options.normalized()
    if normalized.profile == "background":
        repeat_weight = 0.0
        beam_top_k = 4
        diversity_strength = 0.0
    elif normalized.profile == "background_lineart":
        repeat_weight = 0.45
        beam_top_k = 6
        diversity_strength = 0.6
    else:
        repeat_weight = ROW_REPEAT_WEIGHT
        beam_top_k = 6
        diversity_strength = 1.0
    if (line_image_override is None) != (crop_box_override is None):
        raise ValueError("line image and crop overrides must be provided together")
    if line_image_override is None:
        source = decode_image(data)
        line_image, crop_box = preprocess_for_deepaa(source, normalized)
    else:
        line_image = np.asarray(line_image_override, dtype=np.uint8).copy()
        if line_image.ndim != 2 or line_image.shape[0] % LINE_PITCH:
            raise ValueError("line image override must be a 2D 18px-row grid")
        if line_image.shape[1] != normalized.columns * HALF_WIDTH:
            raise ValueError("line image override width does not match columns")
        assert crop_box_override is not None
        crop_box = tuple(int(value) for value in crop_box_override)
    offset, ascii_text, greedy_predictions, rendered, greedy_cost = choose_vertical_alignment(
        line_image,
        repeat_weight=repeat_weight,
        classifier=classifier,
        random_forest_model=random_forest_model,
        structure_strength=structure_strength,
    )
    beam_predictions: list[list[str]] | None = None
    if decoder != "beam":
        _, raw_beam_predictions = generate_text_beam(
            line_image,
            vertical_offset=offset,
            top_k=beam_top_k,
            diversity_strength=diversity_strength,
            classifier=classifier,
            random_forest_model=random_forest_model,
            structure_strength=structure_strength,
        )
        beam_predictions = _select_rows_by_render_cost(
            line_image,
            greedy_predictions,
            raw_beam_predictions,
            load_assets().glyphs,
            repeat_weight=repeat_weight,
            structure_strength=structure_strength,
        )
        beam_rendered = render_predictions(beam_predictions, line_image.shape[1])
        beam_cost = _predictions_cost(
            line_image,
            beam_predictions,
            load_assets().glyphs,
            repeat_weight=repeat_weight,
            structure_strength=structure_strength,
        )
        if beam_cost <= greedy_cost:
            baseline_predictions = beam_predictions
            baseline_rendered = beam_rendered
            baseline_cost = beam_cost
        else:
            baseline_predictions = greedy_predictions
            baseline_rendered = rendered
            baseline_cost = greedy_cost
    else:
        baseline_predictions = greedy_predictions
        baseline_rendered = rendered
        baseline_cost = greedy_cost

    if decoder == "viterbi":
        _, candidate_predictions = generate_text_viterbi(
            line_image,
            vertical_offset=offset,
            classifier=classifier,
            random_forest_model=random_forest_model,
            structure_strength=structure_strength,
        )
    elif decoder == "viterbi-pruned":
        _, candidate_predictions = generate_text_viterbi_pruned(
            line_image,
            vertical_offset=offset,
            baseline=baseline_predictions,
            classifier=classifier,
            random_forest_model=random_forest_model,
            structure_strength=structure_strength,
        )
    else:
        _, candidate_predictions = generate_text_beam(
            line_image,
            vertical_offset=offset,
            top_k=beam_top_k,
            diversity_strength=diversity_strength,
            classifier=classifier,
            random_forest_model=random_forest_model,
            structure_strength=structure_strength,
        )
    selected_predictions = _select_rows_by_render_cost(
        line_image,
        baseline_predictions,
        candidate_predictions,
        load_assets().glyphs,
        repeat_weight=repeat_weight,
        structure_strength=structure_strength,
    )
    selected_rendered = render_predictions(selected_predictions, line_image.shape[1])
    selected_cost = _predictions_cost(
        line_image,
        selected_predictions,
        load_assets().glyphs,
        repeat_weight=repeat_weight,
        structure_strength=structure_strength,
    )
    if selected_cost <= baseline_cost:
        selected_lines = ["".join(line).rstrip() for line in selected_predictions]
        ascii_text = "\n".join(selected_lines).rstrip() + "\n"
        rendered = selected_rendered
    elif decoder != "beam":
        baseline_lines = ["".join(line).rstrip() for line in baseline_predictions]
        ascii_text = "\n".join(baseline_lines).rstrip() + "\n"
        rendered = baseline_rendered
    visible_lines = ascii_text.rstrip("\n").split("\n")
    return ConversionResult(
        ascii_text=ascii_text,
        rows=len(visible_lines),
        columns=normalized.columns,
        processed_png=image_to_data_url(line_image),
        rendered_png=image_to_data_url(rendered),
        crop=crop_box,
    )
