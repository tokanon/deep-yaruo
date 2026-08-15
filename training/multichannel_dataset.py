from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backend.input_channels import (
    ChannelExtractionConfig,
    channel_preview,
    extract_tone,
    normalize_geometry,
)
from backend.rendering import find_font
from training.aa_fill_layers import detect_fill_runs
from training.review_corpus import load_snapshot_records, sha256_bytes
from training.structure_augment import (
    apply_render_augment,
    choose_augment,
    get_augment_policy,
)
from training.structure_proxy import extract_proxy_structure
from training.structure_proxy_dataset import load_vocabulary, text_character_examples
from training.weak_pairs import render_aa_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_tone_nuisance(
    preview: np.ndarray,
    key: str,
    *,
    probability: float,
) -> np.ndarray:
    """Add label-independent broad tone so tone alone cannot imply a fill glyph."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("tone nuisance probability must be between zero and one")
    if probability == 0.0:
        return preview
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if draw >= probability:
        return preview
    density = 1.0 - preview.astype(np.float32) / 255.0
    height, width = preview.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(1, width - 1)
    yy /= max(1, height - 1)
    level = (1 + digest[9] % 2) / 3.0
    mode = digest[8] % 4
    if mode == 0:
        nuisance = np.full_like(density, level)
    elif mode == 1:
        coordinate = xx if digest[10] % 2 else yy
        if digest[11] % 2:
            coordinate = 1.0 - coordinate
        nuisance = coordinate * level
    elif mode == 2:
        center_x = 0.15 + 0.7 * digest[10] / 255.0
        center_y = 0.15 + 0.7 * digest[11] / 255.0
        radius = 0.22 + 0.22 * digest[12] / 255.0
        nuisance = level * np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * radius**2)
        )
    else:
        coordinate = xx + yy if digest[10] % 2 else xx - yy + 1.0
        center = 0.45 + 0.9 * digest[11] / 255.0
        radius = 0.18 + 0.18 * digest[12] / 255.0
        nuisance = level * np.exp(-((coordinate - center) ** 2) / (2.0 * radius**2))
    steps = 3
    augmented_density = np.rint(np.maximum(density, nuisance) * steps) / steps
    return np.clip(255.0 - augmented_density * 255.0, 0, 255).astype(np.uint8)


def prepare_c2_character_dataset(
    snapshot_dir: Path,
    charset_path: Path,
    output_dir: Path,
    *,
    tone_window_width: int,
    tone_window_height: int,
    tone_sigma: float,
    tone_gamma: float,
    training_augment_policy: str = "processing_50",
    seed: int = 42,
    target_width: int = 512,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"P6 dataset output is not empty: {output_dir}")
    snapshot_payload = (snapshot_dir / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_payload)
    records = [
        record
        for record in load_snapshot_records(snapshot_dir)
        if record["decision"] == "accept"
        and record["split"] in {"train", "validation", "test"}
    ]
    if not records:
        raise ValueError("accepted-v1 contains no accepted train/validation/test records")
    vocabulary = load_vocabulary(charset_path)
    policy = get_augment_policy(training_augment_policy)
    channel_config = ChannelExtractionConfig(
        target_width=target_width,
        tone_window_width=tone_window_width,
        tone_window_height=tone_window_height,
        tone_sigma=tone_sigma,
        tone_gamma=tone_gamma,
    )
    channel_config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    channels_dir = output_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    example_records: list[dict[str, object]] = []
    work_records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    fill_counts: Counter[str] = Counter()
    diagnostic_totals: Counter[str] = Counter()
    for record in sorted(records, key=lambda item: int(item["entry_id"])):
        entry_id = int(record["entry_id"])
        text_path = snapshot_dir / str(record["text"])
        text_payload = text_path.read_bytes()
        if sha256_bytes(text_payload) != record["unicode_text_sha256"]:
            raise ValueError(f"Accepted AA text hash mismatch: {text_path}")
        text = text_payload.decode("utf-8")
        rendered = render_aa_text(text, font_size=16)
        normalized = normalize_geometry(rendered, channel_config)
        tone = extract_tone(normalized, channel_config)
        rendered_variants = {"raw": rendered}
        if record["split"] == "train":
            for weighted in policy.variants:
                rendered_variants[weighted.augment] = apply_render_augment(
                    rendered,
                    weighted.augment,
                )
        channel_relatives: dict[str, Path] = {}
        for augment, rendered_variant in rendered_variants.items():
            structure = extract_proxy_structure(
                rendered_variant,
                channel_config=channel_config,
            )
            if structure.shape != tone.shape:
                raise AssertionError("P6 ch0 and ch1 must share one geometry")
            channels = np.stack(
                (channel_preview(structure), channel_preview(tone)),
            ).astype(np.uint8)
            suffix = "" if augment == "raw" else f"-{augment}"
            relative = Path("channels") / f"{entry_id:07d}{suffix}.npz"
            np.savez_compressed(output_dir / relative, channels=channels)
            channel_relatives[augment] = relative

        examples, diagnostics = text_character_examples(
            text,
            vocabulary,
            source_width=rendered.shape[1],
            target_width=target_width,
            font_size=16,
        )
        diagnostic_totals.update(diagnostics)
        fill_positions = {
            (run.line_index, index)
            for run in detect_fill_runs(text, font_size=16)
            for index in range(run.start_index, run.end_index)
        }
        work_fill_count = 0
        for example in examples:
            augment = "raw"
            if record["split"] == "train":
                augment = choose_augment(
                    policy,
                    (
                        f"{entry_id}:{example['x']}:{example['y']}:"
                        f"{example['label']}"
                    ),
                    seed=seed,
                )
            is_fill = (
                int(example["line_index"]),
                int(example["character_index"]),
            ) in fill_positions
            work_fill_count += int(is_fill)
            fill_counts[f"{record['split']}:{'fill' if is_fill else 'nonfill'}"] += 1
            example_records.append(
                {
                    "entry_id": entry_id,
                    "split": record["split"],
                    "channels": channel_relatives[augment].as_posix(),
                    "augment": augment,
                    "is_fill_instance": is_fill,
                    **example,
                }
            )
            split_counts[str(record["split"])] += 1
        work_records.append(
            {
                "entry_id": entry_id,
                "split": record["split"],
                "category": record["corrected_category"],
                "source_text": str(record["text"]),
                "source_text_sha256": record["unicode_text_sha256"],
                "raw_channels": channel_relatives["raw"].as_posix(),
                "raw_channels_sha256": _sha256(output_dir / channel_relatives["raw"]),
                "training_variants": [
                    {
                        "augment": augment,
                        "channels": relative.as_posix(),
                        "sha256": _sha256(output_dir / relative),
                    }
                    for augment, relative in channel_relatives.items()
                ],
                "source_shape": list(rendered.shape),
                "normalized_shape": list(tone.shape),
                "example_count": len(examples),
                "fill_instance_count": work_fill_count,
                "diagnostics": diagnostics,
            }
        )

    examples_payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in example_records
    ).encode("utf-8")
    (output_dir / "examples.jsonl").write_bytes(examples_payload)
    manifest = {
        "schema_version": 1,
        "phase": "P6",
        "dataset": f"accepted-v1-c2-w{tone_window_width}-g{tone_gamma:g}",
        "purpose": "synthetic C=2 structure+tone downstream character classification",
        "seed": seed,
        "snapshot": snapshot["snapshot"],
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "charset": str(charset_path.resolve()),
        "charset_sha256": _sha256(charset_path),
        "font": str(find_font().resolve()),
        "font_sha256": _sha256(find_font()),
        "channel_config": asdict(channel_config),
        "training_augment_policy": policy.manifest(),
        "vocabulary": [asdict(entry) for entry in vocabulary],
        "vocabulary_count": len(vocabulary),
        "work_count": len(work_records),
        "works": work_records,
        "example_count": len(example_records),
        "split_example_counts": dict(sorted(split_counts.items())),
        "fill_instance_counts": dict(sorted(fill_counts.items())),
        "diagnostic_totals": dict(sorted(diagnostic_totals.items())),
        "examples": "examples.jsonl",
        "examples_sha256": sha256_bytes(examples_payload),
        "rights_status": "unknown; local curation and derived training only; do not redistribute",
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
