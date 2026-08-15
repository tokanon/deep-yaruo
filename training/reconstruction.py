from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from backend.rendering import FONT_PATH, render_glyph_mask, render_text_mask
from training.splits import assign_series_splits, assign_work_splits


GLYPH_HEIGHT = 16
LINE_PITCH = 18


@dataclass(frozen=True, order=True)
class Placement:
    file_name: str
    y: int
    x: int
    char: str
    label: int


def load_placements(path: Path) -> tuple[dict[str, list[Placement]], int]:
    """Read DeepAA coordinates and remove its exact duplicate CSV records."""
    works: dict[str, list[Placement]] = defaultdict(list)
    seen: set[Placement] = set()
    duplicate_count = 0
    with path.open("r", encoding="cp932", newline="") as handle:
        for row in csv.DictReader(handle):
            placement = Placement(
                file_name=row["file_name"],
                x=int(row["x"]),
                y=int(row["y"]),
                char=row["char"],
                label=int(row["label"]),
            )
            if placement in seen:
                duplicate_count += 1
                continue
            seen.add(placement)
            works[placement.file_name].append(placement)

    for placements in works.values():
        placements.sort(key=lambda item: (item.y, item.x))
    return dict(sorted(works.items())), duplicate_count


def validate_placements(
    file_name: str,
    placements: Iterable[Placement],
    glyphs: dict[str, np.ndarray],
) -> None:
    rows: dict[int, list[Placement]] = defaultdict(list)
    for placement in placements:
        if placement.file_name != file_name:
            raise ValueError(f"Mixed file names in {file_name}")
        if placement.x < 0 or placement.y < 0:
            raise ValueError(f"Negative coordinate in {file_name}")
        if placement.y % LINE_PITCH:
            raise ValueError(f"Non-grid y coordinate in {file_name}: {placement.y}")
        if placement.char not in glyphs:
            raise ValueError(f"Missing glyph in {file_name}: {placement.char!r}")
        glyph = glyphs[placement.char]
        if glyph.ndim != 2 or glyph.shape[0] != GLYPH_HEIGHT:
            raise ValueError(f"Invalid glyph shape for {placement.char!r}: {glyph.shape}")
        rows[placement.y].append(placement)

    for y, row in rows.items():
        row.sort(key=lambda item: item.x)
        for current, following in zip(row, row[1:]):
            expected_x = current.x + glyphs[current.char].shape[1]
            if following.x != expected_x:
                raise ValueError(
                    f"Broken character advance in {file_name} at ({current.x}, {y}): "
                    f"expected {expected_x}, got {following.x}"
                )


def render_work(
    placements: Iterable[Placement],
    glyphs: dict[str, np.ndarray],
) -> tuple[np.ndarray, str]:
    ordered = sorted(placements, key=lambda item: (item.y, item.x))
    if not ordered:
        raise ValueError("Cannot render a work without placements.")
    height = max(item.y for item in ordered) + LINE_PITCH
    rows: dict[int, list[Placement]] = defaultdict(list)

    for item in ordered:
        rows[item.y].append(item)

    text_lines: list[str] = []
    for y in range(0, height, LINE_PITCH):
        text_lines.append("".join(item.char for item in rows.get(y, ())))
    text = "\n".join(text_lines) + "\n"
    mask = render_text_mask(
        text,
        GLYPH_HEIGHT,
        canvas_height_per_line=LINE_PITCH,
    )
    canvas = np.where(mask, 0, 255).astype(np.uint8)
    return canvas, text


def placement_digest(placements: Iterable[Placement]) -> str:
    payload = [
        [item.file_name, item.x, item.y, item.char, item.label]
        for item in sorted(placements, key=lambda value: (value.y, value.x))
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def assign_splits(
    file_names: Iterable[str],
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    return assign_work_splits(
        file_names,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def reconstruct_dataset(
    csv_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    selected_names: set[str] | None = None,
    split_strategy: str = "work-random",
) -> dict[str, object]:
    works, duplicate_count = load_placements(csv_path)
    characters = {
        placement.char
        for placements in works.values()
        for placement in placements
    }
    glyphs = {
        char: render_glyph_mask(char, GLYPH_HEIGHT, GLYPH_HEIGHT)
        for char in characters
    }
    if split_strategy == "work-random":
        splits = assign_splits(works, seed=seed)
    elif split_strategy == "series-holdout":
        splits = assign_series_splits(works)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    unknown_names = (selected_names or set()) - works.keys()
    if unknown_names:
        raise ValueError(f"Unknown work names: {', '.join(sorted(unknown_names))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for file_name, placements in works.items():
        if selected_names is not None and file_name not in selected_names:
            continue
        validate_placements(file_name, placements, glyphs)
        image, text = render_work(placements, glyphs)
        split = splits[file_name]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        image_path = split_dir / f"{file_name}.png"
        text_path = split_dir / f"{file_name}.txt"
        Image.fromarray(image).save(image_path)
        text_path.write_text(text, encoding="utf-8", newline="\n")
        records.append(
            {
                "file_name": file_name,
                "split": split,
                "placements": len(placements),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sha256": placement_digest(placements),
                "image": image_path.relative_to(output_dir).as_posix(),
                "text": text_path.relative_to(output_dir).as_posix(),
            }
        )

    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source": str(csv_path),
        "glyph_source": str(FONT_PATH),
        "seed": seed,
        "split_strategy": split_strategy,
        "source_work_count": len(works),
        "generated_work_count": len(records),
        "exact_duplicate_rows_removed": duplicate_count,
        "split_counts": counts,
        "works": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
