from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

from training.yaruyomi import (
    analyze_entry,
    decode_character_references,
    decode_numeric_character_references,
    decode_chunk,
    encode_unencodable_as_numeric_references,
    export_entries,
    index_archive,
    normalized_sha256,
    read_database_stats,
)

SENSITIVE_AA = "（成人向けの例）\r\n　／￣＼\r\n（ ﾟ ∀ ﾟ ）\r\n　＼＿／ &#65374; &#9608;\r\n"


def _write_archive(path: Path) -> None:
    heading = "最終更新日 2026/08/21\r\n"
    section = "【背景・室内】\r\n"
    aa = "　┌──┐\r\n　│部屋│\r\n　└──┘\r\n"
    payload = "[SPLIT]".join((heading, section, aa, aa, SENSITIVE_AA)).encode("cp932")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture/汎用AA/R18/背景.mlt", payload)


def test_analyze_entry_detects_aa_and_background() -> None:
    result = analyze_entry("　┌─┐\n　│家│\n　└─┘", "汎用AA/背景/建物.mlt", "【家】", 2)
    assert result.record_type == "aa"
    assert result.category == "architecture"
    assert result.max_columns >= 6


def test_negated_background_is_not_used_as_a_category() -> None:
    result = analyze_entry(
        "　（人物AA）\n　 /　ヽ\n　( ・∀・)",
        "あ行/作品/人物04（武器無し）.mlt",
        "背景無し",
        10,
    )
    assert result.category == "character"


def test_generic_mlt_path_takes_priority_over_section_heading() -> None:
    result = analyze_entry(
        "　　／|\n　／　|\n（ ロボ ）\n　＼＿|",
        "汎用AA/乗り物・メカ/ロボット01.mlt",
        "専門学校のCM",
        20,
    )
    assert result.category == "mecha"


def test_generic_character_libraries_are_not_misclassified_by_substrings() -> None:
    creature = analyze_entry(
        "　 /\\_∧\n　( ﾟДﾟ)\n　/　つつ\n （＿⌒ヽ",
        "汎用AA/伝承・伝説・空想の生物/ゴブリン.mlt",
        "【ゴブリン】",
        20,
    )
    animal = analyze_entry(
        "　 /\\_∧\n　(=・ω・)\n　/　つつ\n （＿⌒ヽ",
        "汎用AA/動植物/その他動物キャラ.mlt",
        "【キャラクター】",
        20,
    )
    clothing = analyze_entry(
        "　　○\n　／｜＼\n　　｜\n　／　＼",
        "汎用AA/衣装/その他水着.mlt",
        None,
        20,
    )

    assert creature.category == "character"
    assert animal.category == "character"
    assert clothing.category == "character"


def test_character_title_containing_fire_is_not_an_effect() -> None:
    result = analyze_entry(
        "　 /\\_∧\n　( ﾟДﾟ)\n　/　つつ\n （＿⌒ヽ",
        "は行/ふ/ファイアーエムブレム/蒼炎の軌跡/アイク.mlt",
        "蒼炎の軌跡（１７歳）",
        20,
    )

    assert result.category == "character"


def test_generic_small_props_and_military_vehicles_are_specific() -> None:
    tent = analyze_entry(
        "　　／＼\n　／　　＼\n／＿＿＿＿＼\n　｜　｜",
        "汎用AA/小道具/自然物・アウトドア系/テント.mlt",
        "【テント】",
        20,
    )
    vehicle = analyze_entry(
        "　＿＿＿＿\n／|＿＿＿|＼\n|　装甲車　|\n◎￣￣￣◎",
        "汎用AA/軍事/軍事兵器/軍事兵器（その他）.mlt",
        "【ストライカー装甲車】",
        20,
    )

    assert tent.category == "object"
    assert vehicle.category == "mecha"


def test_yukkuri_collections_are_faces() -> None:
    result = analyze_entry(
        "　＿＿＿\n／　　　＼\n|　・　・　|\n＼＿▽＿／",
        "2ch/元ネタ有り/ゆっくり/ゆっくりその他.mlt",
        "■基本",
        20,
    )

    assert result.category == "face"


def test_decode_chunk_recovers_utf8_fragments_inside_cp932() -> None:
    raw = "見出し".encode("cp932") + "▅—".encode("utf-8") + "終端".encode("cp932")
    decoded, encoding = decode_chunk(raw)
    assert decoded == "見出し▅—終端"
    assert encoding == "cp932+foreign-fragments"


def test_decode_chunk_recovers_cp1252_fragment_inside_cp932() -> None:
    raw = "引用".encode("cp932") + b"\x92'" + "終端".encode("cp932")
    decoded, encoding = decode_chunk(raw)
    assert decoded == "引用’'終端"
    assert encoding == "cp932+foreign-fragments"


def test_mlt_character_references_are_restored_without_parsing_html() -> None:
    text = (
        "&#8741; &#65374 &#x2225; &#8198x &nbsp; &thinsp; &x200A; "
        "&amp; & &Girls; invalid: &#0; &#xD800; &#12345678"
    )

    expected = (
        "∥ ～ ∥ \u2006x \u00a0 \u2009 \u200a "
        "& & &Girls; invalid: &#0; &#xD800; &#12345678"
    )
    assert decode_character_references(text) == expected
    assert decode_numeric_character_references(text) == expected


def test_cp932_transport_uses_numeric_references_only_when_needed() -> None:
    assert encode_unencodable_as_numeric_references("～∥█ A&<") == (
        "～∥&#9608;&#8201;A&<"
    )


def test_index_and_export_include_sensitive_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    database = tmp_path / "index.sqlite3"
    export_dir = tmp_path / "export"
    _write_archive(archive)

    result = index_archive(archive, database, version="fixture")
    assert result["source_file_count"] == 1
    assert result["aa_candidate_count"] == 3
    assert result["duplicate_occurrence_count"] == 1
    assert result["sensitive_aa_count"] == 3

    stats = read_database_stats(database)
    assert stats["metadata"]["schema_version"] == "2"
    assert stats["metadata"]["display_text_rule_version"] == "1"
    assert stats["metadata"]["sensitive_policy"] == "included and tagged"

    with sqlite3.connect(database) as connection:
        indexed_entity = connection.execute(
            "SELECT preview, normalized_sha256 FROM entries WHERE preview LIKE '%█%'"
        ).fetchone()
    assert indexed_entity is not None
    assert "&#9608;" not in indexed_entity[0]
    assert indexed_entity[1] == normalized_sha256(
        SENSITIVE_AA.rstrip("\r\n").replace("\r\n", "\n")
    )

    exported = export_entries(database, archive, export_dir, limit=0, generate_thumbnails=True)
    assert exported["exported_count"] == 2
    records = [json.loads(line) for line in (export_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(record["sensitive"] for record in records)
    assert all((export_dir / record["file"]).exists() for record in records)
    assert all((export_dir / record["thumbnail"]).exists() for record in records)
    texts = [
        (export_dir / record["file"]).read_text(encoding="utf-8")
        for record in records
    ]
    assert all("&#65374;" not in text for text in texts)
    assert any("～" in text for text in texts)
    assert any("█" in text for text in texts)
    transformed = [record for record in records if record["text_transforms"]]
    assert len(transformed) == 1
    assert transformed[0]["text_transforms"] == ["character_references"]
    assert transformed[0]["source_normalized_sha256"] != transformed[0]["exported_sha256"]

    cp932_dir = tmp_path / "export-cp932"
    export_entries(
        database,
        archive,
        cp932_dir,
        limit=0,
        output_encoding="cp932-ncr",
    )
    cp932_records = [
        json.loads(line)
        for line in (cp932_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cp932_texts = [
        (cp932_dir / record["file"]).read_text(encoding="cp932")
        for record in cp932_records
    ]
    assert any("～" in text and "&#9608;" in text for text in cp932_texts)
    assert all("█" not in text for text in cp932_texts)
    assert all(record["export_encoding"] == "cp932-ncr" for record in cp932_records)
