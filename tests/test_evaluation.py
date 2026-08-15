from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.quality import geometric_metrics, text_usage_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_evaluation_sources_use_local_sample_directory() -> None:
    cases_path = ROOT / "evaluation" / "cases.json"
    specification = json.loads(cases_path.read_text(encoding="utf-8"))
    assert {case["id"] for case in specification["cases"]} == {
        "person-01-source",
        "background-01-source",
        "person-01-lineart",
        "background-01-lineart",
    }
    for case in specification["cases"]:
        source = Path(case["source"])
        assert source.parts[:2] == ("..", "samples")
        assert source.suffix.lower() in {".png", ".jpg", ".jpeg"}
    background_lineart = next(
        case
        for case in specification["cases"]
        if case["id"] == "background-01-lineart"
    )
    assert background_lineart["options"]["columns"] == 120
    assert background_lineart["options"]["max_rows"] >= 30


def test_geometric_metrics_prefer_identical_rendering() -> None:
    target = np.full((18, 64), 255, dtype=np.uint8)
    cv2.line(target, (5, 3), (58, 14), 0, 1)
    shifted = np.roll(target, 6, axis=1)
    identical = geometric_metrics(target, target)
    displaced = geometric_metrics(target, shifted)
    assert identical["distance_cost"] == 0
    assert identical["ink_f1_at_2px"] == 1
    assert displaced["distance_cost"] > identical["distance_cost"]
    assert displaced["ink_f1_at_2px"] < identical["ink_f1_at_2px"]


def test_text_usage_metrics_report_character_concentration() -> None:
    metrics = text_usage_metrics("||||,,ab\n")
    assert metrics["non_whitespace_characters"] == 8
    assert metrics["unique_characters"] == 4
    assert metrics["top_2_share"] == 0.75
