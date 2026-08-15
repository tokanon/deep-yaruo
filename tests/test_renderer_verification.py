from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from backend.rendering import FONT_PATH, glyph_advance
from training.renderer_verification import WindowsGdiMeasurer, verify_renderer


TEST_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test-renderer-verification"


def _case_dir(name: str) -> Path:
    path = TEST_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_charset(path: Path) -> None:
    with path.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "char", "frequency"))
        writer.writeheader()
        writer.writerows(
            (
                {"index": 0, "char": "A", "frequency": 100},
                {"index": 1, "char": " ", "frequency": 90},
                {"index": 2, "char": "B", "frequency": 80},
            )
        )


def _write_dataset(path: Path, *, broken_x: bool = False, broken_label: bool = False) -> None:
    width_a = glyph_advance("A")
    width_space = glyph_advance(" ")
    with path.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("file_name", "x", "y", "char", "label"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"file_name": "fixture", "x": 0, "y": 0, "char": "A", "label": 0},
                {
                    "file_name": "fixture",
                    "x": width_a,
                    "y": 0,
                    "char": " ",
                    "label": 1,
                },
                {
                    "file_name": "fixture",
                    "x": width_a + width_space + int(broken_x),
                    "y": 0,
                    "char": "B",
                    "label": 1 if broken_label else 2,
                },
            )
        )


def test_renderer_verification_checks_independent_text_render_and_labels() -> None:
    case_dir = _case_dir("passing")
    dataset = case_dir / "dataset.csv"
    charset = case_dir / "charset.csv"
    _write_dataset(dataset)
    _write_charset(charset)

    report = verify_renderer(dataset, charset, include_gdi=False)

    assert report["gate_passed"] is True
    assert report["failure_counts"] == {}
    assert report["totals"]["works"] == 1
    assert report["totals"]["rows"] == 1
    assert report["totals"]["model_labeled_placements"] == 3
    assert report["totals"]["blank_glyph_characters"] == [" "]
    assert report["differences"]["dataset_right_edge_minus_sequential"] == {"0": 1}
    assert report["differences"]["full_run_advance_minus_sequential"] == {"0": 1}


def test_renderer_verification_reports_first_advance_mismatch() -> None:
    case_dir = _case_dir("broken-advance")
    dataset = case_dir / "dataset.csv"
    charset = case_dir / "charset.csv"
    _write_dataset(dataset, broken_x=True)
    _write_charset(charset)

    report = verify_renderer(dataset, charset, include_gdi=False)

    assert report["gate_passed"] is False
    assert report["failure_counts"]["character_advance"] == 1
    failure = next(
        item for item in report["first_failures"] if item["kind"] == "character_advance"
    )
    assert failure["char"] == "B"
    assert failure["x"] == failure["expected_x"] + 1


def test_renderer_verification_reports_label_character_mismatch() -> None:
    case_dir = _case_dir("broken-label")
    dataset = case_dir / "dataset.csv"
    charset = case_dir / "charset.csv"
    _write_dataset(dataset, broken_label=True)
    _write_charset(charset)

    report = verify_renderer(dataset, charset, include_gdi=False)

    assert report["gate_passed"] is False
    assert report["failure_counts"]["label_character"] == 1


@pytest.mark.skipif(os.name != "nt", reason="Classic GDI is Windows-only")
def test_gdi_and_pillow_use_the_same_saitamaar_advances() -> None:
    with WindowsGdiMeasurer(FONT_PATH) as gdi:
        for text in ("A", "/", " ", "AB"):
            assert gdi.measure(text).width == sum(glyph_advance(char) for char in text)
