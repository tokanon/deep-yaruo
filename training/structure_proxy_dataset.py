from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.input_channels import ChannelExtractionConfig, channel_preview
from backend.rendering import find_font, line_pitch, load_font
from training.review_corpus import load_snapshot_records, sha256_bytes
from training.structure_proxy import (
    StructureProxyConfig,
    StructureProxyMethod,
    extract_proxy_structure,
    generate_structure_proxy,
)
from training.structure_augment import (
    apply_render_augment,
    choose_augment,
    get_augment_policy,
)


@dataclass(frozen=True)
class VocabularyEntry:
    label: int
    char: str
    frequency: int
    frequency_band: str


def frequency_band(frequency: int) -> str:
    if frequency >= 1_000:
        return "1000+"
    if frequency >= 100:
        return "100-999"
    if frequency >= 20:
        return "20-99"
    return "10-19"


def load_vocabulary(path: Path, *, minimum_frequency: int = 10) -> tuple[VocabularyEntry, ...]:
    with path.open("r", encoding="cp932", newline="") as handle:
        rows = csv.DictReader(handle)
        selected = [row for row in rows if int(row["frequency"]) >= minimum_frequency]
    return tuple(
        VocabularyEntry(
            label=label,
            char=row["char"],
            frequency=int(row["frequency"]),
            frequency_band=frequency_band(int(row["frequency"])),
        )
        for label, row in enumerate(selected)
    )


def text_character_examples(
    text: str,
    vocabulary: tuple[VocabularyEntry, ...],
    *,
    source_width: int,
    target_width: int,
    font_size: int = 16,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    if source_width <= 0 or target_width <= 0:
        raise ValueError("source_width and target_width must be positive")
    by_character = {entry.char: entry for entry in vocabulary}
    font = load_font(font_size)
    pitch = line_pitch(font_size)
    scale = target_width / source_width
    examples: list[dict[str, object]] = []
    excluded = 0
    collisions = 0
    total = 0
    for line_index, line in enumerate(text.splitlines()):
        normalized_y = int(round(line_index * pitch * scale))
        starts: set[int] = set()
        source_x = 0.0
        for character_index, char in enumerate(line):
            total += 1
            normalized_x = int(round(source_x * scale))
            entry = by_character.get(char)
            if entry is None:
                excluded += 1
                source_x += float(font.getlength(char))
                continue
            if normalized_x in starts:
                collisions += 1
                source_x += float(font.getlength(char))
                continue
            starts.add(normalized_x)
            examples.append(
                {
                    "line_index": line_index,
                    "character_index": character_index,
                    "x": normalized_x,
                    "y": normalized_y,
                    "char": char,
                    "label": entry.label,
                    "frequency": entry.frequency,
                    "frequency_band": entry.frequency_band,
                }
            )
            source_x += float(font.getlength(char))
    return examples, {
        "total_characters": total,
        "included_characters": len(examples),
        "out_of_vocabulary_characters": excluded,
        "normalized_coordinate_collisions": collisions,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_structure_proxy_character_dataset(
    snapshot_dir: Path,
    charset_path: Path,
    output_dir: Path,
    *,
    method: StructureProxyMethod,
    seed: int = 42,
    target_width: int = 512,
    training_augment_policy: str | None = None,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"P3 dataset output is not empty: {output_dir}")
    snapshot_payload = (snapshot_dir / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_payload)
    records = [
        record
        for record in load_snapshot_records(snapshot_dir)
        if record["decision"] == "accept" and record["split"] in {"train", "validation", "test"}
    ]
    if not records:
        raise ValueError("accepted-v1 contains no accepted train/validation/test records")
    vocabulary = load_vocabulary(charset_path)
    channel_config = ChannelExtractionConfig(target_width=target_width)
    proxy_config = StructureProxyConfig()
    augment_policy = (
        get_augment_policy(training_augment_policy)
        if training_augment_policy is not None
        else None
    )
    if augment_policy is not None and method != "raw_raster":
        raise ValueError("P5 training augmentation currently requires raw_raster")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    example_records: list[dict[str, object]] = []
    work_records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    band_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    diagnostic_totals: Counter[str] = Counter()
    for record in sorted(records, key=lambda item: int(item["entry_id"])):
        entry_id = int(record["entry_id"])
        text_path = snapshot_dir / str(record["text"])
        text_payload = text_path.read_bytes()
        if sha256_bytes(text_payload) != record["unicode_text_sha256"]:
            raise ValueError(f"Accepted AA text hash mismatch: {text_path}")
        text = text_payload.decode("utf-8")
        proxy, parameters = generate_structure_proxy(
            text,
            method,
            config=proxy_config,
        )
        rendered_variants = {"raw": proxy}
        if augment_policy is not None and record["split"] == "train":
            for weighted in augment_policy.variants:
                rendered_variants[weighted.augment] = apply_render_augment(
                    proxy,
                    weighted.augment,
                )
        image_relatives: dict[str, Path] = {}
        normalized_shapes: dict[str, list[int]] = {}
        for augment, rendered_variant in rendered_variants.items():
            structure = extract_proxy_structure(
                rendered_variant,
                channel_config=channel_config,
            )
            training_image = channel_preview(structure)
            suffix = "" if augment == "raw" else f"-{augment}"
            image_relative = Path("images") / f"{entry_id:07d}{suffix}.png"
            Image.fromarray(training_image).save(output_dir / image_relative, optimize=True)
            image_relatives[augment] = image_relative
            normalized_shapes[augment] = list(training_image.shape)
        examples, diagnostics = text_character_examples(
            text,
            vocabulary,
            source_width=proxy.shape[1],
            target_width=target_width,
            font_size=proxy_config.font_size,
        )
        diagnostic_totals.update(diagnostics)
        for example in examples:
            augment = "raw"
            if augment_policy is not None and record["split"] == "train":
                augment = choose_augment(
                    augment_policy,
                    (
                        f"{entry_id}:{example['x']}:{example['y']}:"
                        f"{example['label']}"
                    ),
                    seed=seed,
                )
            item = {
                "entry_id": entry_id,
                "split": record["split"],
                "image": image_relatives[augment].as_posix(),
                "augment": augment,
                **example,
            }
            example_records.append(item)
            split_counts[str(record["split"])] += 1
            band_counts[str(example["frequency_band"])] += 1
            character_counts[str(example["char"])] += 1
        work_records.append(
            {
                "entry_id": entry_id,
                "split": record["split"],
                "category": record["corrected_category"],
                "source_text": str(record["text"]),
                "source_text_sha256": record["unicode_text_sha256"],
                "image": image_relatives["raw"].as_posix(),
                "image_sha256": _sha256(output_dir / image_relatives["raw"]),
                "training_variants": [
                    {
                        "augment": augment,
                        "image": relative.as_posix(),
                        "image_sha256": _sha256(output_dir / relative),
                    }
                    for augment, relative in image_relatives.items()
                ],
                "source_proxy_shape": list(proxy.shape),
                "normalized_shape": normalized_shapes["raw"],
                "example_count": len(examples),
                "diagnostics": diagnostics,
                "proxy_parameters": parameters,
            }
        )

    examples_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in example_records
    ).encode("utf-8")
    (output_dir / "examples.jsonl").write_bytes(examples_payload)
    vocabulary_payload = [asdict(entry) for entry in vocabulary]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "phase": "P5" if augment_policy is not None else "P3",
        "dataset": (
            f"accepted-v1-{method}-{augment_policy.policy_id}-c1"
            if augment_policy is not None
            else f"accepted-v1-{method}-c1"
        ),
        "method": method,
        "purpose": "synthetic C=1 downstream character classification",
        "seed": seed,
        "target_width": target_width,
        "snapshot": snapshot["snapshot"],
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "charset": str(charset_path.resolve()),
        "charset_sha256": _sha256(charset_path),
        "font": str(find_font().resolve()),
        "font_sha256": _sha256(find_font()),
        "channel_config": asdict(channel_config),
        "proxy_config": asdict(proxy_config),
        "training_augment_policy": (
            augment_policy.manifest() if augment_policy is not None else None
        ),
        "vocabulary": vocabulary_payload,
        "vocabulary_count": len(vocabulary),
        "work_count": len(work_records),
        "works": work_records,
        "example_count": len(example_records),
        "split_example_counts": dict(sorted(split_counts.items())),
        "frequency_band_counts": dict(sorted(band_counts.items())),
        "unique_in_vocabulary_characters": len(character_counts),
        "diagnostic_totals": dict(sorted(diagnostic_totals.items())),
        "in_vocabulary_coverage": round(
            diagnostic_totals["included_characters"]
            / max(1, diagnostic_totals["total_characters"]),
            8,
        ),
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
