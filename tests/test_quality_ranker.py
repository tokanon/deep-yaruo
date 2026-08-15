from __future__ import annotations

import json
from pathlib import Path

from training.quality_ranker import build_quality_feature_matrix, text_structure_features


def test_text_structure_features_detect_repetition_and_symbols() -> None:
    features = text_structure_features("　┌──┐\n　│顔│\n　│顔│\n　└──┘\n")

    assert features["symbol_ratio"] > 0.5
    assert features["indented_line_ratio"] == 1.0
    assert features["repeated_line_ratio"] > 0


def test_quality_matrix_does_not_use_review_score_as_feature(tmp_path: Path) -> None:
    records = []
    for entry_id, decision, source_id in ((1, "accept", 10), (2, "reject", 20)):
        text_path = Path("aa") / "train" / f"{entry_id}.txt"
        (tmp_path / text_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / text_path).write_text("┌" + "─" * 60 + "┐\n" * 8, encoding="utf-8")
        records.append(
            {
                "entry_id": entry_id,
                "decision": decision,
                "text": text_path.as_posix(),
                "source_file_id": source_id,
                "source_path": f"group/source-{source_id}.mlt",
                "corrected_category": "background",
                "indexed_category": "background",
                "line_count": 8,
                "max_columns": 62,
                "character_count": 500,
                "art_score": 1.0,
                "sensitive": False,
                "text_transforms": [],
                "quality_score": 5 if decision == "accept" else 0,
            }
        )
    payload = "".join(json.dumps(record) + "\n" for record in records).encode()
    (tmp_path / "records.jsonl").write_bytes(payload)
    import hashlib

    (tmp_path / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot": "fixture",
                "records": "records.jsonl",
                "records_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    matrix, labels, groups, feature_names, loaded = build_quality_feature_matrix(tmp_path)

    assert matrix.shape[0] == 2
    assert labels.tolist() == [1, 0]
    assert groups.tolist() == [10, 20]
    assert "quality_score" not in feature_names
    assert "decision" not in feature_names
    assert len(loaded) == 2
