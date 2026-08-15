from __future__ import annotations

import csv
from pathlib import Path

from backend.rendering import render_glyph_mask
from training.reconstruction import (
    assign_splits,
    load_placements,
    reconstruct_dataset,
)
from training.splits import assign_series_splits, series_name


def write_fixture(csv_path: Path) -> tuple[int, int]:
    width_a = render_glyph_mask("A", 16, 16).shape[1]
    width_b = render_glyph_mask("B", 16, 16).shape[1]
    with csv_path.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_name", "x", "y", "char", "label"))
        writer.writeheader()
        rows = [
            {"file_name": "work", "x": 0, "y": 0, "char": "A", "label": 0},
            {"file_name": "work", "x": width_a, "y": 0, "char": "B", "label": 1},
            {"file_name": "work", "x": 0, "y": 0, "char": "A", "label": 0},
        ]
        writer.writerows(rows)
    return width_a, width_b


def test_reconstruction_removes_duplicates_and_preserves_advances() -> None:
    test_dir = Path(__file__).resolve().parents[1] / ".tmp" / "test-training-data"
    test_dir.mkdir(parents=True, exist_ok=True)
    csv_path = test_dir / "data.csv"
    output_dir = test_dir / "output"
    width_a, width_b = write_fixture(csv_path)

    works, duplicates = load_placements(csv_path)
    assert duplicates == 1
    assert [(item.x, item.char) for item in works["work"]] == [
        (0, "A"),
        (width_a, "B"),
    ]

    manifest = reconstruct_dataset(csv_path, output_dir)
    assert manifest["generated_work_count"] == 1
    assert manifest["exact_duplicate_rows_removed"] == 1
    record = manifest["works"][0]
    assert record["width"] == width_a + width_b
    assert record["height"] == 18
    assert (output_dir / record["image"]).exists()
    assert (output_dir / record["text"]).read_text(encoding="utf-8") == "AB\n"


def test_split_is_deterministic_and_work_level() -> None:
    names = [f"work-{index:03d}" for index in range(100)]
    first = assign_splits(names, seed=42)
    second = assign_splits(reversed(names), seed=42)
    assert first == second
    assert list(first.values()).count("train") == 80
    assert list(first.values()).count("validation") == 10
    assert list(first.values()).count("test") == 10


def test_series_split_keeps_source_series_isolated() -> None:
    names = [
        *(f"Alpha_{index:04d}" for index in range(8)),
        *(f"Beta{index:04d}" for index in range(2)),
        "Gamma-01",
    ]
    splits = assign_series_splits(names)
    seen: dict[str, set[str]] = {}
    for name, split in splits.items():
        seen.setdefault(series_name(name), set()).add(split)
    assert all(len(values) == 1 for values in seen.values())
    assert set(splits.values()) == {"train", "validation", "test"}
