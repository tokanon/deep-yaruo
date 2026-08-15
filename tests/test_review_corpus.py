from __future__ import annotations

import json
import zipfile
from pathlib import Path

from backend.review import ReviewStore, ReviewSubmission
from training.review_corpus import (
    assign_source_group_splits,
    freeze_review_corpus,
    load_snapshot_records,
)
from training.review_corpus_audit import audit_review_snapshot
from training.yaruyomi import index_archive


def _art(symbol: str, label: str) -> str:
    return "".join(f"{symbol * 30}{label}{symbol * 30}\r\n" for _ in range(8))


def _fixture_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        symbol_pairs = (("─", "━"), ("＿", "＝"), ("／", "｜"), ("＼", "┼"))
        for index, (accepted_symbol, rejected_symbol) in enumerate(symbol_pairs, 1):
            payload = "[SPLIT]".join(
                (
                    "【背景】\r\n",
                    _art(accepted_symbol, f"採用{index}"),
                    _art(rejected_symbol, f"不採用{index}"),
                )
            ).encode("cp932")
            archive.writestr(f"fixture/汎用AA/背景/原本{index}.mlt", payload)


def test_group_split_keeps_source_files_isolated() -> None:
    records = [
        {"source_file_id": source, "corrected_category": category}
        for source, category in (
            (1, "character"),
            (1, "face"),
            (2, "background"),
            (3, "character"),
            (4, "effect"),
        )
    ]

    assignments = assign_source_group_splits(records, seed=42)

    assert set(assignments) == {1, 2, 3, 4}
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_freeze_review_corpus_writes_verified_unicode_snapshot(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews.sqlite3"
    output = tmp_path / "accepted-v1"
    _fixture_archive(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewStore(index, archive, reviews)

    for source_index in range(1, 5):
        candidate = store.next_candidate(category="background", start_id=0)
        assert candidate is not None
        store.save_review(
            candidate["id"],
            ReviewSubmission(
                decision="accept",
                category="background",
                quality_score=3,
            ),
        )
        rejected = store.next_candidate(category="background", start_id=0)
        assert rejected is not None
        store.save_review(
            rejected["id"],
            ReviewSubmission(
                decision="reject",
                category="background",
                quality_score=0,
            ),
        )

    snapshot = freeze_review_corpus(
        index,
        archive,
        reviews,
        output,
        expected_accepted=4,
    )
    records = load_snapshot_records(output)

    assert snapshot["accepted_count"] == 4
    assert snapshot["decision_counts"] == {"accept": 4, "reject": 4}
    assert set(snapshot["split_counts"]) == {"train", "validation", "test"}
    assert sum(snapshot["split_counts"].values()) == 4
    assert len(records) == 8
    assert all((output / record["text"]).exists() for record in records)
    assert all(record["rights_status"] == "unknown; local curation only" for record in records)
    persisted = json.loads((output / "snapshot.json").read_text(encoding="utf-8"))
    assert persisted["records_sha256"] == snapshot["records_sha256"]

    audit = audit_review_snapshot(output)
    assert audit["text_hash_failures"] == []
    assert audit["split_source_overlaps"] == []
    assert audit["accepted_count"] == 4
