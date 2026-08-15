from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from training.weak_pairs import (
    DEFAULT_VARIANTS,
    V2_VARIANTS,
    generate_weak_pair_dataset,
    generate_weak_variant,
    render_aa_text,
)


def _snapshot(path: Path) -> None:
    text_path = Path("aa/train/0000001.txt")
    (path / text_path).parent.mkdir(parents=True, exist_ok=True)
    text = "／￣￣￣￣＼n｜　顔　｜\n＼＿＿＿＿／\n" * 4
    payload = text.encode("utf-8")
    (path / text_path).write_bytes(payload)
    record = {
        "entry_id": 1,
        "decision": "accept",
        "split": "train",
        "text": text_path.as_posix(),
        "unicode_text_sha256": hashlib.sha256(payload).hexdigest(),
        "corrected_category": "face",
        "sensitive": False,
    }
    records_payload = (json.dumps(record) + "\n").encode()
    (path / "records.jsonl").write_bytes(records_payload)
    (path / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot": "fixture",
                "records": "records.jsonl",
                "records_sha256": hashlib.sha256(records_payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_weak_variants_are_black_lines_on_white() -> None:
    rendered = render_aa_text("／￣￣＼\n＼＿＿／\n")

    for variant in DEFAULT_VARIANTS:
        candidate, parameters = generate_weak_variant(rendered, variant, seed=42)
        assert candidate.shape == rendered.shape
        assert candidate.dtype == np.uint8
        assert candidate.min() == 0
        assert candidate.max() == 255
        assert parameters


def test_weak_v2_variants_are_visible_on_white() -> None:
    rendered = render_aa_text("／￣￣＼\n＼＿＿／\n")

    for variant in V2_VARIANTS:
        candidate, parameters = generate_weak_variant(rendered, variant, seed=43)
        assert candidate.shape == rendered.shape
        assert candidate.dtype == np.uint8
        assert candidate.min() < 255
        assert candidate.max() > candidate.min()
        assert parameters


def test_weak_v2_tone_candidates_preserve_gray_density() -> None:
    rendered = render_aa_text(("／￣￣￣￣＼\n｜　顔　｜\n＼＿＿＿＿／\n") * 8)
    tone, _ = generate_weak_variant(rendered, "tone_density", seed=43)
    combined, _ = generate_weak_variant(rendered, "line_tone", seed=43)

    assert np.count_nonzero((tone > 0) & (tone < 255)) > 0
    assert np.count_nonzero((combined > 0) & (combined < 255)) > 0


def test_weak_pair_dataset_is_deterministic(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _snapshot(snapshot)

    first_dataset = generate_weak_pair_dataset(snapshot, first, limit=1, seed=7)
    second_dataset = generate_weak_pair_dataset(snapshot, second, limit=1, seed=7)

    assert first_dataset["weak_pair_count"] == 4
    assert first_dataset["pairs_sha256"] == second_dataset["pairs_sha256"]
    assert (first / "pairs.jsonl").read_bytes() == (second / "pairs.jsonl").read_bytes()
    image = cv2.imread(str(first / "candidates/0000001/skeleton.png"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert image.min() == 0
