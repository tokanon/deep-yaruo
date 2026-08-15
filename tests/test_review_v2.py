from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import backend.review_v2 as review_v2
from backend.review import ReviewStore, ReviewSubmission
from backend.review_v2 import ReviewV2Store, ReviewV2Submission
from training.review_corpus import (
    assign_source_group_splits,
    freeze_review_corpus,
    load_snapshot_records,
    related_source_groups,
)
from training.yaruyomi import index_archive


def _art(symbol: str, label: str) -> str:
    return "".join(f"{symbol * 30}{label}{symbol * 30}\r\n" for _ in range(8))


def _write_archive(path: Path) -> None:
    payload = "[SPLIT]".join(
        (
            "【背景】\r\n",
            _art("─", "候補1"),
            _art("＿", "候補2"),
            _art("／", "候補3"),
            _art("＼", "候補4"),
        )
    ).encode("cp932")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture/汎用AA/背景/原本.mlt", payload)


def _write_single_archive(path: Path) -> None:
    payload = "[SPLIT]".join(("【背景】\r\n", _art("─", "唯一候補"))).encode("cp932")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture/汎用AA/背景/唯一.mlt", payload)


def _write_multi_source_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, symbol in enumerate(("─", "＿", "／", "＼"), 1):
            payload = "[SPLIT]".join(
                ("【背景】\r\n", _art(symbol, f"採用{index}"))
            ).encode("cp932")
            archive.writestr(f"fixture/汎用AA/背景/原本{index}.mlt", payload)


def _write_cross_source_variants(path: Path) -> None:
    base = _art("─", "近似候補")
    changed = base.replace("近似候補", "近似候補差", 1)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "fixture/汎用AA/背景/原本A.mlt",
            "[SPLIT]".join(("【背景】\r\n", base)).encode("cp932"),
        )
        archive.writestr(
            "fixture/汎用AA/背景/原本B.mlt",
            "[SPLIT]".join(("【背景】\r\n", changed)).encode("cp932"),
        )


def _write_cross_source_whitespace_duplicate(path: Path) -> None:
    base = _art("─", "同一候補")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "fixture/汎用AA/背景/原本A.mlt",
            "[SPLIT]".join(("【背景】\r\n", base)).encode("cp932"),
        )
        archive.writestr(
            "fixture/汎用AA/背景/原本B.mlt",
            "[SPLIT]".join(("【背景】\r\n", "  " + base)).encode("cp932"),
        )


def test_v2_review_is_independent_of_old_review_database(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    old_reviews = tmp_path / "reviews.sqlite3"
    v2_reviews = tmp_path / "reviews-v2.sqlite3"
    _write_single_archive(archive)
    index_archive(archive, index, version="fixture")

    old_store = ReviewStore(index, archive, old_reviews)
    old_candidate = old_store.next_candidate(category="background", start_id=0)
    assert old_candidate is not None
    old_store.save_review(
        old_candidate["id"],
        ReviewSubmission(
            decision="reject",
            category="background",
            quality_score=0,
        ),
    )

    v2_store = ReviewV2Store(index, archive, v2_reviews)
    candidate = v2_store.next_candidate(category="background")
    assert candidate is not None
    assert candidate["normalizedSha256"] == old_candidate["normalizedSha256"]
    assert v2_store.status()["reviewedCount"] == 0


def test_v2_queue_is_persisted_and_optional_score_is_saved(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")

    first_store = ReviewV2Store(index, archive, reviews)
    first = first_store.next_candidate(category="background")
    assert first is not None
    restored = ReviewV2Store(index, archive, reviews).next_candidate(
        category="background"
    )
    assert restored is not None
    assert restored["id"] == first["id"]
    assert restored["batchId"] == first["batchId"]

    saved = first_store.save_review(
        int(first["id"]),
        ReviewV2Submission(decision="accept", category="background"),
    )
    assert saved["review"]["qualityScore"] is None
    status = first_store.status()
    assert status["corpus"] == "accepted-v2"
    assert status["acceptedCount"] == 1
    assert status["targetCount"] == 750
    assert status["reviewTargets"]["background"] == 60
    assert status["selectionPolicy"]["oldReviewInfluence"] == "none"


def test_v2_first_batch_is_deterministic_across_fresh_databases(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    first_reviews = tmp_path / "reviews-v2-a.sqlite3"
    second_reviews = tmp_path / "reviews-v2-b.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")

    first = ReviewV2Store(index, archive, first_reviews).next_candidate(
        category="background"
    )
    second = ReviewV2Store(index, archive, second_reviews).next_candidate(
        category="background"
    )

    assert first is not None and second is not None
    assert first["normalizedSha256"] == second["normalizedSha256"]


def test_v2_keeps_modified_works_reviewable_across_source_files(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_cross_source_variants(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewV2Store(index, archive, reviews)

    candidate = store.next_candidate(category="background")

    assert candidate is not None
    with sqlite3.connect(reviews) as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidate_queue").fetchone()[0] == 2


def test_v2_suppresses_whitespace_only_duplicates_across_source_files(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_cross_source_whitespace_duplicate(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewV2Store(index, archive, reviews)

    candidate = store.next_candidate(category="background")

    assert candidate is not None
    with sqlite3.connect(reviews) as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidate_queue").fetchone()[0] == 1


def test_related_variants_are_assigned_to_the_same_split() -> None:
    base = _art("─", "近似候補")
    records = [
        {"source_file_id": 1, "corrected_category": "character", "text": base},
        {
            "source_file_id": 2,
            "corrected_category": "character",
            "text": base.replace("近似候補", "近似候補差", 1),
        },
        {"source_file_id": 3, "corrected_category": "background", "text": _art("＿", "背景")},
        {"source_file_id": 4, "corrected_category": "effect", "text": _art("／", "効果")},
    ]

    source_groups, components = related_source_groups(records)
    assignments = assign_source_group_splits(records, source_groups=source_groups)

    assert components == [[1, 2]]
    assert assignments[1] == assignments[2]


def test_v2_skip_moves_to_next_persisted_candidate(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewV2Store(index, archive, reviews)

    first = store.next_candidate(category="background")
    assert first is not None
    second = store.next_candidate(
        category="background", skip_entry_id=int(first["id"])
    )
    assert second is not None
    assert second["id"] != first["id"]
    assert ReviewV2Store(index, archive, reviews).next_candidate(
        category="background"
    )["id"] == second["id"]


def test_v2_accept_requires_concrete_category() -> None:
    with pytest.raises(ValidationError):
        ReviewV2Submission(decision="accept", category="unknown")


def test_v2_checkpoint_freezes_nullable_scores_under_distinct_name(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    output = tmp_path / "accepted-v2" / "checkpoints" / "0004"
    _write_multi_source_archive(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewV2Store(index, archive, reviews)

    for _ in range(4):
        candidate = store.next_candidate(category="background")
        assert candidate is not None
        store.save_review(
            int(candidate["id"]),
            ReviewV2Submission(decision="accept", category="background"),
        )

    snapshot = freeze_review_corpus(
        index,
        archive,
        reviews,
        output,
        expected_accepted=4,
        snapshot_name="yaruyomi-accepted-v2-0004",
    )
    records = load_snapshot_records(output)

    assert snapshot["snapshot"] == "yaruyomi-accepted-v2-0004"
    assert snapshot["accepted_count"] == 4
    assert all(record["quality_score"] is None for record in records)


def test_v2_stops_presenting_candidates_at_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")
    monkeypatch.setattr(review_v2, "CHECKPOINT_ACCEPTED_COUNT", 1)
    monkeypatch.setattr(review_v2, "ACCEPTED_TARGETS", {"background": 1})
    store = ReviewV2Store(index, archive, reviews)

    candidate = store.next_candidate(category="background")
    assert candidate is not None
    store.save_review(
        int(candidate["id"]),
        ReviewV2Submission(decision="accept", category="background"),
    )

    assert store.status()["checkpointReady"] is True
    assert store.next_candidate(category="background") is None


def test_v2_checkpoint_waits_for_category_floors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")
    monkeypatch.setattr(review_v2, "CHECKPOINT_ACCEPTED_COUNT", 1)
    monkeypatch.setattr(review_v2, "ACCEPTED_TARGETS", {"background": 2})
    store = ReviewV2Store(index, archive, reviews)

    first = store.next_candidate(category="background")
    assert first is not None
    store.save_review(
        int(first["id"]),
        ReviewV2Submission(decision="accept", category="background"),
    )
    assert store.status()["acceptedCount"] == 1
    assert store.status()["checkpointReady"] is False
    assert store.next_candidate(category="background") is not None


def test_v2_expands_source_round_for_scarce_under_target_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews-v2.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")
    monkeypatch.setattr(review_v2, "CHECKPOINT_ACCEPTED_COUNT", 4)
    monkeypatch.setattr(review_v2, "ACCEPTED_TARGETS", {"background": 4})
    store = ReviewV2Store(index, archive, reviews)

    assert store.next_candidate(category="background") is not None
    with sqlite3.connect(reviews) as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidate_queue").fetchone()[0] == 4
