from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

from backend.review import ReviewStore, ReviewSubmission
from training.yaruyomi import index_archive


def _write_archive(path: Path) -> None:
    metadata = "最終更新日 2026/08/21\r\n"
    section = "【背景】\r\n"
    first = (
        "　┌" + "─" * 58 + "┐\r\n"
        "　│" + "　" * 29 + "│\r\n"
        "　│" + "　" * 10 + "部屋" + "　" * 17 + "│\r\n"
        "　│" + "　" * 29 + "│\r\n"
        "　│" + "＿" * 29 + "│\r\n"
        "　└" + "─" * 58 + "┘\r\n"
    )
    second = (
        "　／" + "￣" * 28 + "＼\r\n"
        "／" + "　" * 30 + "＼\r\n"
        "｜" + "　" * 12 + "山" + "　" * 17 + "｜\r\n"
        "｜" + "　" * 30 + "｜\r\n"
        "｜" + "　" * 30 + "｜\r\n"
        "｜" + "　" * 30 + "｜\r\n"
        "｜" + "＿" * 30 + "｜\r\n"
        "＼" + "＿" * 30 + "／\r\n"
    )
    payload = "[SPLIT]".join((metadata, section, first, first, second)).encode("cp932")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture/汎用AA/背景.mlt", payload)


def _write_source_cap_archive(path: Path) -> None:
    def art(symbol: str, label: str) -> str:
        return "".join(
            f"{symbol * 30}{label}{symbol * 30}\r\n"
            for _ in range(8)
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        first_payload = "[SPLIT]".join(
            ("【背景】\r\n",) + tuple(
                art(symbol, str(index))
                for index, symbol in enumerate(("─", "＿", "／", "＼", "┼"), 1)
            )
        ).encode("cp932")
        second_payload = "[SPLIT]".join(
            ("【背景】\r\n", art("━", "別原本"))
        ).encode("cp932")
        archive.writestr("fixture/汎用AA/背景/原本1.mlt", first_payload)
        archive.writestr("fixture/汎用AA/背景/原本2.mlt", second_payload)


def test_review_store_persists_hash_based_reviews_and_skips_duplicates(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews.sqlite3"
    _write_archive(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewStore(index, archive, reviews)

    status = store.status()
    assert status["available"] is True
    assert status["reviewedCount"] == 0
    assert status["targetCount"] == 500
    assert status["selectionPolicy"]["version"] == 3
    assert status["selectionPolicy"]["maximumReviewsPerSourceFile"] == 3

    first = store.next_candidate(start_id=0)
    assert first is not None
    assert first["category"] == "background"
    saved = store.save_review(
        first["id"],
        ReviewSubmission(
            decision="accept",
            category="architecture",
            quality_score=4,
            issue_tags=["category_mismatch"],
            comment="建物として採用",
        ),
    )
    assert saved["review"]["decision"] == "accept"
    assert saved["review"]["qualityScore"] == 4

    next_candidate = store.next_candidate(start_id=0)
    assert next_candidate is not None
    assert next_candidate["normalizedSha256"] != first["normalizedSha256"]
    assert store.status()["reviewedCount"] == 1

    restored = ReviewStore(index, archive, reviews).get_entry(first["id"])
    assert restored is not None
    assert restored["review"]["category"] == "architecture"


def test_review_selection_skips_non_work_candidates(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews.sqlite3"
    valid = (
        "┌" + "─" * 58 + "┐\r\n"
        "│" + "　" * 29 + "│\r\n"
        "│" + "　" * 12 + "建物" + "　" * 15 + "│\r\n"
        "│" + "　" * 29 + "│\r\n"
        "│" + "　" * 29 + "│\r\n"
        "│" + "＿" * 29 + "│\r\n"
        "└" + "─" * 58 + "┘\r\n"
    )
    entity = valid + ("　" * 90) + "&#65374;\r\n"
    payload = "[SPLIT]".join(
        ("最終更新日 2026/08/21\r\n", "【背景】\r\n", "《城》\r\n", entity, valid)
    ).encode("cp932")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("fixture/汎用AA/背景.mlt", payload)
        output.writestr("fixture/汎用AA/地図/案内図.mlt", payload[:-1])
    index_archive(archive, index, version="fixture")

    candidate = ReviewStore(index, archive, reviews).next_candidate(
        category="background", start_id=0
    )

    assert candidate is not None
    assert "地図" not in candidate["sourcePath"]
    assert "&#65374;" not in candidate["text"]
    assert "～" in candidate["text"]
    assert candidate["textTransforms"] == ["character_references"]


def test_review_selection_detects_multiple_vertical_blocks() -> None:
    text = ("／￣￣＼\n" * 6) + "\n\n" + ("＼＿＿／\n" * 6)

    assert ReviewStore._content_exclusion_reason(text) == "multiple_blocks"


def test_review_selection_detects_whitespace_and_small_difference_variants() -> None:
    base = "\n".join(("／￣￣￣￣＼", "|　顔　|", "＼＿＿＿＿／") * 12)
    whitespace_variant = "\n".join(f"　{line}" for line in base.splitlines())
    small_difference = base.replace("顔", "目", 1)
    unrelated = "┏" + "━" * 80 + "┓" + ("\n┃別の背景┃" * 20)

    assert ReviewStore._near_duplicate_text(base, whitespace_variant)
    assert ReviewStore._near_duplicate_text(base, small_difference)
    assert not ReviewStore._near_duplicate_text(base, unrelated)


def test_review_selection_caps_each_source_file_at_three_reviews(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    index = tmp_path / "index.sqlite3"
    reviews = tmp_path / "reviews.sqlite3"
    _write_source_cap_archive(archive)
    index_archive(archive, index, version="fixture")
    store = ReviewStore(index, archive, reviews)

    selected_sources: list[str] = []
    for _ in range(4):
        candidate = store.next_candidate(category="background", start_id=0)
        assert candidate is not None
        selected_sources.append(candidate["sourcePath"])
        store.save_review(
            candidate["id"],
            ReviewSubmission(
                decision="hold",
                category="background",
                quality_score=2,
            ),
        )

    assert Counter(selected_sources) == {
        "汎用AA/背景/原本1.mlt": 3,
        "汎用AA/背景/原本2.mlt": 1,
    }
    assert store.next_candidate(category="background", start_id=0) is None
