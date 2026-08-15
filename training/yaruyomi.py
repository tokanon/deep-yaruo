from __future__ import annotations

import hashlib
import html.entities
import json
import math
import os
import re
import sqlite3
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont

from backend.rendering import find_font, line_pitch


SCHEMA_VERSION = 2
SOURCE_URL = "https://aa.yaruyomi.com/"
DOWNLOAD_PAGE_URL = "https://s.yaruyomi.com/s/oJTZ"
SPLIT_MARKER = b"[SPLIT]"
_NUMERIC_CHARACTER_REFERENCE = re.compile(
    r"&#(?:"
    r"(?P<decimal>[0-9]{1,7})(?:;|(?![0-9]))"
    r"|[xX](?P<hexadecimal>[0-9A-Fa-f]{1,6})(?:;|(?![0-9A-Fa-f]))"
    r")"
)
_PSEUDO_HEX_CHARACTER_REFERENCE = re.compile(r"&[xX]([0-9A-Fa-f]{1,6});")
_NAMED_CHARACTER_REFERENCE = re.compile(r"&([A-Za-z][A-Za-z0-9]{1,31});")

_BRACKET_HEADING = re.compile(r"^[\s　]*[【〔［\[].+[】〕］\]][\s　]*$")
_METADATA = re.compile(
    r"(?:最終更新|更新日|作成日|作者|編集|注意|参照|出典|このMLT|このファイル)",
    re.IGNORECASE,
)
_ART_CHARACTERS = frozenset(
    r"/\|_-~=+*^<>[](){}.,:;!?'`@#$%&" "／＼｜＿－～＝＋＊＾＜＞［］（）｛｝・：；！？”’｀＠＃＄％＆" "┌┐└┘├┤┬┴┼─│━┃┏┓┗┛┣┫┳┻╋╭╮╰╯「」『』【】（）〈〉《》〔〕〆々" "○●◎◇◆□■△▲▽▼☆★※→←↑↓∧∨⌒彡ノヽ乂人八ハへくしつっッンソリルレロヾゝ仝个"
)

_SENSITIVE_WORDS = (
    "r18",
    "18禁",
    "エロ",
    "性的",
    "性行為",
    "性交",
    "裸",
    "陵辱",
    "触手",
    "成人向け",
)

_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multiple_people", ("複数", "集合", "群衆", "モブ", "二人", "二名")),
    ("architecture", ("建物", "建築", "住宅", "学校", "城", "寺", "神社", "店", "施設", "室内", "部屋")),
    ("nature", ("自然", "地形", "山岳", "森林", "海洋", "河川", "湖沼", "雲", "天候")),
    ("background", ("背景", "舞台", "道路", "街", "都市", "屋外", "景色")),
    ("mecha", ("メカ", "ロボ", "ガンダム", "機動兵器", "戦闘機", "軍用機", "宇宙船", "軍事兵器", "装甲車", "戦車")),
    ("effect", ("エフェクト", "効果", "漫符", "集中線", "爆発", "水しぶき")),
    ("text_or_joke", ("文字", "ロゴ", "台詞", "セリフ", "会話", "擬音", "吹き出し", "ネタ")),
    ("object", ("小物", "道具", "武器", "食べ物", "料理", "家具", "乗り物", "機械", "動植物")),
    ("face", ("顔文字", "表情", "顔アップ", "頭部", "首だけ")),
    ("upper_body", ("上半身", "バストアップ", "胸像", "半身")),
    ("full_body", ("全身", "立ち絵", "立像")),
)
_GENERIC_CHARACTER_KEYWORDS = (
    "キャラクター",
    "動物キャラ",
    "伝承・伝説・空想の生物",
    "衣装",
    "水着",
)
_NEGATED_CATEGORY_PATTERNS = tuple(
    re.compile(
        re.escape(keyword) + r"(?:は|が)?(?:無し|なし|無い|ない|無)",
        flags=re.IGNORECASE,
    )
    for keyword in sorted(
        {keyword for _, words in _CATEGORY_RULES for keyword in words},
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class EntryAnalysis:
    text: str
    record_type: str
    category: str
    category_reason: str
    sensitive: bool
    line_count: int
    max_columns: int
    character_count: int
    art_score: float
    preview: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_chunk_bytes(raw: bytes) -> bytes:
    if raw.startswith(b"\r\n"):
        raw = raw[2:]
    elif raw.startswith(b"\n") or raw.startswith(b"\r"):
        raw = raw[1:]
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith(b"\n") or raw.endswith(b"\r"):
        raw = raw[:-1]
    return raw


def _utf8_fragment(raw: bytes, offset: int) -> tuple[str, int] | None:
    first = raw[offset]
    if 0xC2 <= first <= 0xDF:
        size = 2
    elif 0xE0 <= first <= 0xEF:
        size = 3
    elif 0xF0 <= first <= 0xF4:
        size = 4
    else:
        return None
    candidate = raw[offset : offset + size]
    if len(candidate) != size:
        return None
    try:
        character = candidate.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        character.encode("cp932")
    except UnicodeEncodeError:
        return character, size
    return None


def _decode_mixed_cp932(raw: bytes) -> tuple[str, bool]:
    characters: list[str] = []
    offset = 0
    recovered_foreign_fragment = False
    while offset < len(raw):
        fragment = _utf8_fragment(raw, offset)
        if fragment is not None:
            character, size = fragment
            characters.append(character)
            offset += size
            recovered_foreign_fragment = True
            continue
        first = raw[offset]
        size = 1
        if 0x81 <= first <= 0x9F or 0xE0 <= first <= 0xFC:
            size = 2
        candidate = raw[offset : offset + size]
        try:
            characters.append(candidate.decode("cp932"))
            offset += size
        except UnicodeDecodeError:
            if 0x80 <= first <= 0x9F:
                try:
                    characters.append(bytes((first,)).decode("cp1252"))
                    offset += 1
                    recovered_foreign_fragment = True
                    continue
                except UnicodeDecodeError:
                    pass
            characters.append("\N{REPLACEMENT CHARACTER}")
            offset += 1
    return "".join(characters), recovered_foreign_fragment


def decode_chunk(raw: bytes) -> tuple[str, str]:
    try:
        return raw.decode("cp932"), "cp932"
    except UnicodeDecodeError:
        decoded, recovered_foreign_fragment = _decode_mixed_cp932(raw)
        if "\N{REPLACEMENT CHARACTER}" in decoded:
            return decoded, "cp932+replacement"
        if recovered_foreign_fragment:
            return decoded, "cp932+foreign-fragments"
        return decoded, "cp932+replacement"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _character_from_codepoint(value: int, source: str) -> str:
    if value == 0 or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
        return source
    return chr(value)


def decode_character_references(text: str) -> str:
    """Restore the HTML-like character-reference dialect used by MLT files."""

    def replace_numeric(match: re.Match[str]) -> str:
        decimal = match.group("decimal")
        value = int(decimal, 10) if decimal is not None else int(match.group("hexadecimal"), 16)
        return _character_from_codepoint(value, match.group(0))

    def replace_pseudo_hex(match: re.Match[str]) -> str:
        return _character_from_codepoint(int(match.group(1), 16), match.group(0))

    def replace_named(match: re.Match[str]) -> str:
        return html.entities.html5.get(match.group(1) + ";", match.group(0))

    restored = _NUMERIC_CHARACTER_REFERENCE.sub(replace_numeric, text)
    restored = _PSEUDO_HEX_CHARACTER_REFERENCE.sub(replace_pseudo_hex, restored)
    return _NAMED_CHARACTER_REFERENCE.sub(replace_named, restored)


def decode_numeric_character_references(text: str) -> str:
    """Backward-compatible alias for callers using the original function name."""

    return decode_character_references(text)


def encode_unencodable_as_numeric_references(text: str, encoding: str = "cp932") -> str:
    """Keep encodable text literal and serialize unsupported Unicode as NCRs."""

    output: list[str] = []
    for character in text:
        try:
            character.encode(encoding)
        except UnicodeEncodeError:
            output.append(f"&#{ord(character)};")
        else:
            output.append(character)
    return "".join(output)


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_newlines(text).encode("utf-8")).hexdigest()


def _display_columns(line: str) -> int:
    return len(line.encode("cp932", errors="replace"))


def _art_score(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    art = sum(char in _ART_CHARACTERS for char in visible)
    lines = text.splitlines() or [text]
    indented = sum(bool(line) and line[0] in " \t　" for line in lines)
    indent_bonus = min(0.25, indented / max(1, len(lines)) * 0.25)
    return min(1.0, art / len(visible) + indent_bonus)


def classify_record(text: str, chunk_index: int) -> tuple[str, float]:
    stripped_lines = [line for line in text.splitlines() if line.strip(" \t　")]
    if not stripped_lines:
        return "empty", 0.0
    score = _art_score(text)
    if chunk_index == 0 and _METADATA.search(text):
        return "metadata", score
    if len(stripped_lines) <= 3 and all(_BRACKET_HEADING.match(line) for line in stripped_lines):
        return "section", score
    max_columns = max(_display_columns(line) for line in stripped_lines)
    if len(stripped_lines) >= 4:
        return "aa", score
    if len(stripped_lines) >= 2 and (score >= 0.14 or max_columns >= 24):
        return "aa", score
    if len(stripped_lines) == 1 and score >= 0.30 and max_columns >= 3:
        return "aa", score
    if len(stripped_lines) <= 3 and max_columns <= 80:
        return "section", score
    if _METADATA.search(text):
        return "metadata", score
    return "text", score


def _without_negated_keywords(text: str) -> str:
    result = text
    for pattern in _NEGATED_CATEGORY_PATTERNS:
        result = pattern.sub("", result)
    return result


def _match_rules(haystack: str, categories: set[str] | None = None) -> tuple[str, str] | None:
    for category, keywords in _CATEGORY_RULES:
        if categories is not None and category not in categories:
            continue
        for keyword in keywords:
            if keyword.lower() in haystack:
                return category, keyword

    return None


def categorize(source_path: str, section: str | None) -> tuple[str, str]:
    top = source_path.partition("/")[0]
    cleaned_path = _without_negated_keywords(source_path).lower()
    cleaned_section = _without_negated_keywords(section or "").lower()
    haystack = f"{cleaned_path} {cleaned_section}"
    if top == "汎用AA":
        if "/小道具/" in f"/{cleaned_path}":
            return "object", "小道具"
        for keyword in _GENERIC_CHARACTER_KEYWORDS:
            if keyword.lower() in cleaned_path:
                return "character", keyword
        match = _match_rules(cleaned_path)
        if match is None:
            match = _match_rules(cleaned_section)
        return match if match is not None else ("unknown", "no-keyword")

    if "/ゆっくり/" in f"/{cleaned_path}":
        return "face", "ゆっくり"

    explicit_background = "背景" in cleaned_path or "風景" in cleaned_path
    if explicit_background:
        match = _match_rules(haystack, {"architecture", "nature"})
        return match if match is not None else ("background", "explicit-background")

    strong_categories = {
        "multiple_people",
        "mecha",
        "text_or_joke",
        "face",
        "upper_body",
        "full_body",
    }
    section_label = cleaned_section.strip(" \t　【】〔〕［］[]")
    if section_label in {"エフェクト", "効果", "漫符", "集中線", "爆発", "水しぶき"}:
        return "effect", section_label
    match = _match_rules(f"{cleaned_path.rsplit('/', 1)[-1]} {cleaned_section}", strong_categories)
    if match is not None:
        return match
    if top:
        return "character", f"top-level:{top}"
    return "unknown", "no-keyword"


def is_sensitive(source_path: str, section: str | None) -> bool:
    haystack = f"{source_path} {section or ''}".lower()
    return any(word.lower() in haystack for word in _SENSITIVE_WORDS)


def analyze_entry(text: str, source_path: str, section: str | None, chunk_index: int) -> EntryAnalysis:
    normalized = normalize_newlines(text)
    record_type, score = classify_record(normalized, chunk_index)
    category, reason = categorize(source_path, section)
    lines = normalized.split("\n") if normalized else []
    preview = " ".join(line.strip() for line in lines if line.strip())[:160]
    return EntryAnalysis(
        text=normalized,
        record_type=record_type,
        category=category,
        category_reason=reason,
        sensitive=is_sensitive(source_path, section),
        line_count=len(lines),
        max_columns=max((_display_columns(line) for line in lines), default=0),
        character_count=len(normalized),
        art_score=score,
        preview=preview,
    )


def _relative_member_path(name: str, root_prefix: str) -> str:
    if root_prefix and name.startswith(root_prefix):
        return name[len(root_prefix) :]
    return name


def _root_prefix(members: Iterable[zipfile.ZipInfo]) -> str:
    first_parts = {
        PurePosixPath(member.filename).parts[0]
        for member in members
        if PurePosixPath(member.filename).parts
    }
    if len(first_parts) == 1:
        return next(iter(first_parts)) + "/"
    return ""


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            zip_member TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL UNIQUE,
            byte_size INTEGER NOT NULL,
            crc32 TEXT NOT NULL,
            encoding TEXT NOT NULL,
            decode_ok INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL
        );

        CREATE TABLE entries (
            id INTEGER PRIMARY KEY,
            source_file_id INTEGER NOT NULL REFERENCES source_files(id),
            chunk_index INTEGER NOT NULL,
            section TEXT,
            record_type TEXT NOT NULL,
            category TEXT NOT NULL,
            category_reason TEXT NOT NULL,
            sensitive INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            normalized_sha256 TEXT NOT NULL,
            line_count INTEGER NOT NULL,
            max_columns INTEGER NOT NULL,
            character_count INTEGER NOT NULL,
            art_score REAL NOT NULL,
            preview TEXT NOT NULL,
            UNIQUE(source_file_id, chunk_index)
        );

        CREATE TABLE contents (
            normalized_sha256 TEXT PRIMARY KEY,
            canonical_entry_id INTEGER NOT NULL REFERENCES entries(id),
            occurrence_count INTEGER NOT NULL
        );
        """
    )


def _finalize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX entries_normalized_hash ON entries(normalized_sha256);
        INSERT INTO contents(normalized_sha256, canonical_entry_id, occurrence_count)
        SELECT normalized_sha256, MIN(id), COUNT(*)
        FROM entries
        GROUP BY normalized_sha256;
        CREATE INDEX entries_record_type ON entries(record_type);
        CREATE INDEX entries_category ON entries(category);
        CREATE INDEX entries_sensitive ON entries(sensitive);
        CREATE INDEX entries_source_file ON entries(source_file_id, chunk_index);
        ANALYZE;
        """
    )


def index_archive(
    archive_path: Path,
    database_path: Path,
    *,
    version: str,
    max_files: int | None = None,
    force: bool = False,
) -> dict[str, object]:
    archive_path = archive_path.resolve()
    database_path = database_path.resolve()
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    if database_path.exists() and not force:
        raise FileExistsError(f"Database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    archive_hash = sha256_file(archive_path)
    connection = connect_database(temporary_path)
    try:
        _create_schema(connection)
        acquired_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "archive_version": version,
            "archive_path": str(archive_path),
            "archive_filename": archive_path.name,
            "archive_sha256": archive_hash,
            "archive_byte_size": str(archive_path.stat().st_size),
            "source_url": SOURCE_URL,
            "download_page_url": DOWNLOAD_PAGE_URL,
            "acquired_at_utc": acquired_at,
            "rights_status": "unknown; local curation only",
            "source_encoding": "cp932",
            "display_text_rule_version": "1",
            "character_reference_policy": "MLT numeric, named HTML5, and pseudo-hex references decoded after source hashing",
            "sensitive_policy": "included and tagged",
            "category_rule_version": "7",
            "recategorization_status": "complete",
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())

        with zipfile.ZipFile(archive_path, metadata_encoding="cp932") as archive:
            all_members = archive.infolist()
            root_prefix = _root_prefix(all_members)
            members = [member for member in all_members if member.filename.lower().endswith(".mlt")]
            if max_files is not None:
                members = members[:max_files]

            entry_count = 0
            decode_error_files = 0
            mixed_encoding_files = 0
            for member in members:
                payload = archive.read(member)
                chunks = payload.split(SPLIT_MARKER)
                source_path = _relative_member_path(member.filename, root_prefix)
                decoded_chunks = [decode_chunk(clean_chunk_bytes(chunk)) for chunk in chunks]
                modes = {mode for _, mode in decoded_chunks}
                file_decode_ok = "cp932+replacement" not in modes
                file_encoding = (
                    "cp932+foreign-fragments"
                    if "cp932+foreign-fragments" in modes
                    else "cp932"
                )
                decode_error_files += not file_decode_ok
                mixed_encoding_files += file_encoding != "cp932"
                cursor = connection.execute(
                    """
                    INSERT INTO source_files(
                        zip_member, source_path, byte_size, crc32, encoding, decode_ok, chunk_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        member.filename,
                        source_path,
                        member.file_size,
                        f"{member.CRC:08x}",
                        file_encoding,
                        int(file_decode_ok),
                        len(chunks),
                    ),
                )
                source_file_id = int(cursor.lastrowid)
                current_section: str | None = None
                entry_rows: list[tuple[object, ...]] = []
                for chunk_index, (decoded, _) in enumerate(decoded_chunks):
                    cleaned = clean_chunk_bytes(chunks[chunk_index])
                    if not decoded and not cleaned:
                        continue
                    source_text = normalize_newlines(decoded)
                    display_text = decode_character_references(source_text)
                    analysis = analyze_entry(
                        display_text, source_path, current_section, chunk_index
                    )
                    if analysis.record_type == "section":
                        current_section = analysis.preview or current_section
                    content_hash = normalized_sha256(source_text)
                    entry_rows.append(
                        (
                            source_file_id,
                            chunk_index,
                            current_section,
                            analysis.record_type,
                            analysis.category,
                            analysis.category_reason,
                            int(analysis.sensitive),
                            hashlib.sha256(cleaned).hexdigest(),
                            content_hash,
                            analysis.line_count,
                            analysis.max_columns,
                            analysis.character_count,
                            analysis.art_score,
                            analysis.preview,
                        )
                    )
                    entry_count += 1
                connection.executemany(
                    """
                    INSERT INTO entries(
                        source_file_id, chunk_index, section, record_type, category,
                        category_reason, sensitive, raw_sha256, normalized_sha256,
                        line_count, max_columns, character_count, art_score, preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    entry_rows,
                )
                if source_file_id % 250 == 0:
                    connection.commit()

        _finalize_schema(connection)
        connection.commit()
        stats = database_stats(connection)
        stats["archive_sha256"] = archive_hash
        stats["decode_error_files"] = decode_error_files
        stats["mixed_encoding_files"] = mixed_encoding_files
    except Exception:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        connection.close()
    if database_path.exists():
        database_path.unlink()
    os.replace(temporary_path, database_path)
    return stats


def database_stats(connection: sqlite3.Connection) -> dict[str, object]:
    def scalar(query: str, parameters: tuple[object, ...] = ()) -> int:
        value = connection.execute(query, parameters).fetchone()[0]
        return int(value or 0)

    categories = {
        row["category"]: int(row["count"])
        for row in connection.execute(
            "SELECT category, COUNT(*) AS count FROM entries WHERE record_type = 'aa' GROUP BY category ORDER BY category"
        )
    }
    record_types = {
        row["record_type"]: int(row["count"])
        for row in connection.execute(
            "SELECT record_type, COUNT(*) AS count FROM entries GROUP BY record_type ORDER BY record_type"
        )
    }
    return {
        "source_file_count": scalar("SELECT COUNT(*) FROM source_files"),
        "entry_count": scalar("SELECT COUNT(*) FROM entries"),
        "aa_candidate_count": scalar("SELECT COUNT(*) FROM entries WHERE record_type = 'aa'"),
        "sensitive_aa_count": scalar(
            "SELECT COUNT(*) FROM entries WHERE record_type = 'aa' AND sensitive = 1"
        ),
        "unique_content_count": scalar("SELECT COUNT(*) FROM contents"),
        "duplicate_occurrence_count": scalar(
            "SELECT COALESCE(SUM(occurrence_count - 1), 0) FROM contents"
        ),
        "record_types": record_types,
        "categories": categories,
    }


def read_database_stats(database_path: Path) -> dict[str, object]:
    with connect_database(database_path) as connection:
        stats = database_stats(connection)
        stats["metadata"] = {
            row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")
        }
        return stats


def recategorize_database(database_path: Path) -> dict[str, object]:
    changed = 0
    scanned = 0
    with connect_database(database_path) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('recategorization_status', 'running')"
        )
        connection.commit()
        last_id = 0
        while True:
            rows = connection.execute(
                """
                SELECT e.id, e.section, e.category, e.category_reason, f.source_path
                FROM entries e
                JOIN source_files f ON f.id = e.source_file_id
                WHERE e.id > ?
                ORDER BY e.id
                LIMIT 25000
                """,
                (last_id,),
            ).fetchall()
            if not rows:
                break
            updates: list[tuple[str, str, int]] = []
            for row in rows:
                scanned += 1
                category, reason = categorize(row["source_path"], row["section"])
                if category != row["category"] or reason != row["category_reason"]:
                    updates.append((category, reason, row["id"]))
            if updates:
                connection.executemany(
                    "UPDATE entries SET category = ?, category_reason = ? WHERE id = ?",
                    updates,
                )
                changed += len(updates)
            connection.commit()
            last_id = int(rows[-1]["id"])
        connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("category_rule_version", "7"),
                ("recategorization_status", "complete"),
            )
        )
        connection.commit()
        stats = database_stats(connection)
    return {"scanned_count": scanned, "changed_count": changed, **stats}


def _export_rows(
    connection: sqlite3.Connection,
    *,
    category: str | None,
    record_type: str,
    include_duplicates: bool,
    exclude_sensitive: bool,
    sensitive_only: bool,
    limit: int,
) -> list[sqlite3.Row]:
    joins = ""
    conditions = ["e.record_type = ?"]
    parameters: list[object] = [record_type]
    if not include_duplicates:
        joins = "JOIN contents c ON c.normalized_sha256 = e.normalized_sha256"
        conditions.append("c.canonical_entry_id = e.id")
    if category:
        conditions.append("e.category = ?")
        parameters.append(category)
    if exclude_sensitive and sensitive_only:
        raise ValueError("exclude_sensitive and sensitive_only cannot be used together")
    if exclude_sensitive:
        conditions.append("e.sensitive = 0")
    elif sensitive_only:
        conditions.append("e.sensitive = 1")
    limit_sql = ""
    if limit > 0:
        limit_sql = " LIMIT ?"
        parameters.append(limit)
    query = f"""
        SELECT e.*, f.zip_member, f.source_path, f.encoding AS source_encoding
        FROM entries e
        JOIN source_files f ON f.id = e.source_file_id
        {joins}
        WHERE {' AND '.join(conditions)}
        ORDER BY f.source_path, e.chunk_index
        {limit_sql}
    """
    return list(connection.execute(query, parameters))


def render_text_preview(
    text: str,
    output_path: Path,
    *,
    font_size: int = 16,
    maximum_canvas: int = 4096,
    maximum_thumbnail: int = 512,
) -> None:
    font = ImageFont.truetype(str(find_font()), font_size)
    lines = normalize_newlines(text).split("\n") or [""]
    pitch = line_pitch(font_size)
    margin = 8
    measured_width = max((int(math.ceil(font.getlength(line))) for line in lines), default=1)
    width = max(1, min(maximum_canvas, measured_width + margin * 2))
    height = max(1, min(maximum_canvas, len(lines) * pitch + margin * 2))
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    for line_number, line in enumerate(lines):
        y = margin + line_number * pitch
        if y >= height:
            break
        draw.text((margin, y), line, font=font, fill=0)
    canvas.thumbnail((maximum_thumbnail, maximum_thumbnail), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def export_entries(
    database_path: Path,
    archive_path: Path,
    output_dir: Path,
    *,
    category: str | None = None,
    record_type: str = "aa",
    include_duplicates: bool = False,
    exclude_sensitive: bool = False,
    sensitive_only: bool = False,
    generate_thumbnails: bool = False,
    output_encoding: str = "utf-8",
    limit: int = 100,
) -> dict[str, object]:
    if output_encoding not in {"utf-8", "cp932-ncr"}:
        raise ValueError("output_encoding must be utf-8 or cp932-ncr")
    output_dir.mkdir(parents=True, exist_ok=True)
    aa_dir = output_dir / "aa"
    aa_dir.mkdir(parents=True, exist_ok=True)
    with connect_database(database_path) as connection:
        rows = _export_rows(
            connection,
            category=category,
            record_type=record_type,
            include_duplicates=include_duplicates,
            exclude_sensitive=exclude_sensitive,
            sensitive_only=sensitive_only,
            limit=limit,
        )

    by_member: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_member.setdefault(row["zip_member"], []).append(row)

    manifest_path = output_dir / "manifest.jsonl"
    exported = 0
    with zipfile.ZipFile(archive_path, metadata_encoding="cp932") as archive, manifest_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as manifest:
        for member, member_rows in by_member.items():
            chunks = archive.read(member).split(SPLIT_MARKER)
            for row in member_rows:
                raw = clean_chunk_bytes(chunks[row["chunk_index"]])
                text, _ = decode_chunk(raw)
                text = normalize_newlines(text)
                if normalized_sha256(text) != row["normalized_sha256"]:
                    raise ValueError(f"Archive content changed for entry {row['id']}")
                source_text = text
                text = decode_character_references(source_text)
                file_name = f"{row['id']:07d}.txt"
                exported_text = (
                    text
                    if output_encoding == "utf-8"
                    else encode_unencodable_as_numeric_references(text)
                )
                (aa_dir / file_name).write_text(
                    exported_text,
                    encoding="utf-8" if output_encoding == "utf-8" else "cp932",
                    newline="\n",
                )
                thumbnail_name: str | None = None
                if generate_thumbnails:
                    thumbnail_name = f"thumbnails/{row['id']:07d}.png"
                    render_text_preview(text, output_dir / thumbnail_name)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "id": row["id"],
                    "file": f"aa/{file_name}",
                    "source_path": row["source_path"],
                    "chunk_index": row["chunk_index"],
                    "section": row["section"],
                    "record_type": row["record_type"],
                    "category": row["category"],
                    "sensitive": bool(row["sensitive"]),
                    "source_encoding": row["source_encoding"],
                    "export_encoding": output_encoding,
                    "raw_sha256": row["raw_sha256"],
                    "normalized_sha256": row["normalized_sha256"],
                    "source_normalized_sha256": row["normalized_sha256"],
                    "exported_sha256": normalized_sha256(exported_text),
                    "text_transforms": (
                        ["character_references"] if text != source_text else []
                    ),
                    "output_transforms": (
                        ["cp932_numeric_character_references"]
                        if output_encoding == "cp932-ncr" and exported_text != text
                        else []
                    ),
                    "rights_status": "unknown; local curation only",
                }
                if thumbnail_name is not None:
                    record["thumbnail"] = thumbnail_name
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
    return {
        "exported_count": exported,
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sensitive_policy": "included" if not exclude_sensitive else "excluded",
        "sensitive_only": sensitive_only,
        "duplicates": "included" if include_duplicates else "canonical only",
        "thumbnails": generate_thumbnails,
        "output_encoding": output_encoding,
    }
