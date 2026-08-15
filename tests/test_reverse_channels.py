from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.audit_reverse_channels_v1 import audit_reverse_channels
from training.prepare_reverse_channels_v1 import prepare_reverse_channels, tree_digest
from training.reverse_channels import decompose_text, surface_integrity_failures


def test_periodic_candidate_is_lossless_superset_without_glyph_width_cap() -> None:
    text = "┌:i:i:i:i:┐\n│漢漢漢漢漢漢│\n└::::::::┘"

    baseline, baseline_lattice = decompose_text(text, candidate="a-p6r1")
    periodic, periodic_lattice = decompose_text(text, candidate="b-periodic")

    assert np.all(baseline.fill_owned <= periodic.fill_owned)
    assert periodic.fill_owned.sum() > baseline.fill_owned.sum()
    assert any("漢" in run.text for run in periodic.runs)
    assert np.array_equal(baseline_lattice["target_mask"], periodic_lattice["target_mask"])
    assert np.array_equal(
        periodic_lattice["target_mask"],
        periodic.line_mask | periodic.fill_glyph_mask,
    )
    assert not surface_integrity_failures(periodic)


def _snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    text_dir = snapshot / "aa" / "train"
    text_dir.mkdir(parents=True)
    text = "┌:i:i:i:i:┐\n│::::::::│\n└────────┘\n"
    text_path = text_dir / "0000001.txt"
    text_path.write_text(text, encoding="utf-8")
    import hashlib

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "entry_id": 1,
        "decision": "accept",
        "split": "train",
        "text": "aa/train/0000001.txt",
        "source_file_id": 9,
        "corrected_category": "background",
        "unicode_text_sha256": text_hash,
    }
    records_payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    records_hash = hashlib.sha256(records_payload).hexdigest()
    (snapshot / "records.jsonl").write_bytes(records_payload)
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot": "test-accepted-v2",
                "records": "records.jsonl",
                "records_sha256": records_hash,
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_preparation_is_deterministic_and_records_integrity(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = prepare_reverse_channels(snapshot, first)
    second_manifest = prepare_reverse_channels(snapshot, second)

    assert first_manifest == second_manifest
    assert tree_digest(first) == tree_digest(second)
    assert first_manifest["integrity_passed"] is True
    assert first_manifest["training_candidate_selected"] is None
    assert len(first_manifest["preview_paths"]) == 1
    with np.load(first / "works" / "0000001" / "channels.npz") as arrays:
        assert np.array_equal(
            arrays["target_mask"].astype(bool),
            arrays["b_line_mask"].astype(bool)
            | arrays["b_fill_glyph_mask"].astype(bool),
        )

    audit = audit_reverse_channels(first, repeat_root=second)
    assert audit["passed"] is True
    assert audit["byte_deterministic"] is True
    assert audit["category_counts"] == {"background": 1}
