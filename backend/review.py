from __future__ import annotations

import json
import re
import secrets
import sqlite3
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from training.yaruyomi import (
    SPLIT_MARKER,
    clean_chunk_bytes,
    connect_database,
    decode_character_references,
    decode_chunk,
    normalize_newlines,
    normalized_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1"
DEFAULT_INDEX_DATABASE = REVIEW_ROOT / "index.sqlite3"
DEFAULT_ARCHIVE = REVIEW_ROOT / "source.zip"
DEFAULT_REVIEW_DATABASE = REVIEW_ROOT / "reviews.sqlite3"

REVIEW_CATEGORIES = (
    "character",
    "multiple_people",
    "face",
    "upper_body",
    "full_body",
    "background",
    "architecture",
    "nature",
    "object",
    "mecha",
    "effect",
    "text_or_joke",
    "unknown",
)
ISSUE_TAGS = (
    "fragment",
    "multiple_aa",
    "dialog_or_frame",
    "broken_alignment",
    "too_dense",
    "too_sparse",
    "text_or_logo",
    "category_mismatch",
)
INITIAL_REVIEW_TARGETS = {
    "character": 28,
    "multiple_people": 8,
    "face": 7,
    "upper_body": 7,
    "full_body": 7,
    "background": 8,
    "architecture": 8,
    "nature": 5,
    "object": 7,
    "mecha": 5,
    "effect": 5,
    "text_or_joke": 5,
}
REVIEW_TARGETS = {
    category: target * 5 for category, target in INITIAL_REVIEW_TARGETS.items()
}
REVIEW_TARGET_COUNT = sum(REVIEW_TARGETS.values())

# These thresholds preserve every accepted item in the first review pilot while
# removing headings, isolated marks, and almost-empty fragments.  They are a
# corpus-selection policy, not a statement that smaller AA can never be useful.
MIN_REVIEW_LINES = 6
MAX_REVIEW_LINES = 50
MIN_REVIEW_COLUMNS = 50
MIN_REVIEW_CHARACTERS = 250
MIN_SYMBOL_RATIO = 0.45
CANDIDATE_POOL_SIZE = 48
MAX_REVIEWS_PER_SOURCE_FILE = 3
NEAR_DUPLICATE_NGRAM_SIZE = 4
NEAR_DUPLICATE_MIN_LENGTH_RATIO = 0.88
NEAR_DUPLICATE_MIN_OVERLAP = 0.94
DEFERRED_PATH_PARTS = (
    "/汎用AA/地図/",
    "/汎用AA/体/",
    "/汎用AA/線・図形・図表・マーク/",
)
DEFERRED_KEYWORDS = ("差分", "省容量", "コピー能力")
MULTIPLE_BLOCK_PATTERN = re.compile(r"\n[ \t　]*\n[ \t　]*\n")


class ReviewSubmission(BaseModel):
    decision: Literal["accept", "hold", "reject"]
    category: str
    quality_score: int = Field(ge=0, le=5)
    issue_tags: list[str] = Field(default_factory=list, max_length=len(ISSUE_TAGS))
    comment: str = Field(default="", max_length=2000)

    @field_validator("category")
    @classmethod
    def valid_category(cls, value: str) -> str:
        if value not in REVIEW_CATEGORIES:
            raise ValueError(f"Unknown review category: {value}")
        return value

    @field_validator("issue_tags")
    @classmethod
    def valid_issue_tags(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(ISSUE_TAGS))
        if unknown:
            raise ValueError(f"Unknown issue tags: {', '.join(unknown)}")
        return list(dict.fromkeys(value))


class ReviewStore:
    def __init__(
        self,
        index_database: Path = DEFAULT_INDEX_DATABASE,
        archive_path: Path = DEFAULT_ARCHIVE,
        review_database: Path = DEFAULT_REVIEW_DATABASE,
    ) -> None:
        self.index_database = index_database.resolve()
        self.archive_path = archive_path.resolve()
        self.review_database = review_database.resolve()

    def _availability_error(self) -> str | None:
        missing = [
            str(path)
            for path in (self.index_database, self.archive_path)
            if not path.exists()
        ]
        if missing:
            return "Review data is missing: " + ", ".join(missing)
        return None

    def _archive_metadata(self) -> dict[str, str]:
        with connect_database(self.index_database) as connection:
            return {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def _ensure_review_schema(self) -> None:
        availability_error = self._availability_error()
        if availability_error:
            raise FileNotFoundError(availability_error)
        metadata = self._archive_metadata()
        archive_hash = metadata["archive_sha256"]
        self.review_database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.review_database) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    normalized_sha256 TEXT PRIMARY KEY,
                    entry_id INTEGER NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('accept', 'hold', 'reject')),
                    corrected_category TEXT NOT NULL,
                    quality_score INTEGER NOT NULL CHECK(quality_score BETWEEN 0 AND 5),
                    issue_tags_json TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS reviews_entry_id ON reviews(entry_id);
                CREATE INDEX IF NOT EXISTS reviews_decision ON reviews(decision);
                CREATE INDEX IF NOT EXISTS reviews_category ON reviews(corrected_category);
                """
            )
            stored_hash = connection.execute(
                "SELECT value FROM metadata WHERE key = 'archive_sha256'"
            ).fetchone()
            if stored_hash is not None and stored_hash[0] != archive_hash:
                raise ValueError(
                    "The review database belongs to a different source archive: "
                    f"{stored_hash[0]} != {archive_hash}"
                )
            connection.executemany(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("review_schema_version", "1"),
                    ("archive_sha256", archive_hash),
                    ("archive_version", metadata.get("archive_version", "unknown")),
                    ("reviewer_id", "local-user"),
                ),
            )
            connection.commit()

    def _attached_index_connection(self) -> sqlite3.Connection:
        self._ensure_review_schema()
        connection = connect_database(self.index_database)
        connection.execute("ATTACH DATABASE ? AS reviewdb", (str(self.review_database),))
        return connection

    def status(self) -> dict[str, object]:
        availability_error = self._availability_error()
        if availability_error:
            return {
                "available": False,
                "message": availability_error,
                "indexDatabase": str(self.index_database),
                "archive": str(self.archive_path),
            }
        self._ensure_review_schema()
        metadata = self._archive_metadata()
        with sqlite3.connect(self.review_database) as connection:
            decision_counts = {
                decision: count
                for decision, count in connection.execute(
                    "SELECT decision, COUNT(*) FROM reviews GROUP BY decision"
                )
            }
            reviewed_count = sum(decision_counts.values())
            last_reviewed = connection.execute(
                "SELECT entry_id FROM reviews ORDER BY updated_at_utc DESC LIMIT 1"
            ).fetchone()
        return {
            "available": True,
            "archiveVersion": metadata.get("archive_version", "unknown"),
            "archiveSha256": metadata.get("archive_sha256", ""),
            "reviewedCount": reviewed_count,
            "targetCount": REVIEW_TARGET_COUNT,
            "decisionCounts": {
                "accept": decision_counts.get("accept", 0),
                "hold": decision_counts.get("hold", 0),
                "reject": decision_counts.get("reject", 0),
            },
            "lastReviewedEntryId": last_reviewed[0] if last_reviewed else None,
            "categories": list(REVIEW_CATEGORIES),
            "issueTags": list(ISSUE_TAGS),
            "reviewTargets": REVIEW_TARGETS,
            "selectionPolicy": {
                "version": 3,
                "minimumLines": MIN_REVIEW_LINES,
                "maximumLines": MAX_REVIEW_LINES,
                "minimumColumns": MIN_REVIEW_COLUMNS,
                "minimumCharacters": MIN_REVIEW_CHARACTERS,
                "maximumReviewsPerSourceFile": MAX_REVIEWS_PER_SOURCE_FILE,
                "nearDuplicateRule": (
                    "same-source whitespace-insensitive four-gram overlap"
                ),
                "nearDuplicateMinimumLengthRatio": NEAR_DUPLICATE_MIN_LENGTH_RATIO,
                "nearDuplicateMinimumOverlap": NEAR_DUPLICATE_MIN_OVERLAP,
                "deferredPathParts": list(DEFERRED_PATH_PARTS),
                "deferredKeywords": list(DEFERRED_KEYWORDS),
                "characterReferences": "decoded after source hash verification",
                "sourceHistory": "used only for candidate ranking",
            },
            "sensitivePolicy": "included and tagged",
            "reviewDatabase": str(self.review_database),
        }

    def _candidate_query(
        self,
        *,
        category: str | None,
        sensitive: str,
        start_id: int,
        exclude_reviewed: bool,
        apply_selection_policy: bool = True,
        excluded_source_file_ids: tuple[int, ...] = (),
        limit: int = 1,
    ) -> tuple[str, list[object]]:
        conditions = ["e.record_type = 'aa'", "e.id > ?"]
        parameters: list[object] = [start_id]
        if category:
            if category not in REVIEW_CATEGORIES:
                raise ValueError(f"Unknown review category: {category}")
            conditions.append("e.category = ?")
            parameters.append(category)
        if sensitive == "sensitive":
            conditions.append("e.sensitive = 1")
        elif sensitive == "non_sensitive":
            conditions.append("e.sensitive = 0")
        elif sensitive != "all":
            raise ValueError("sensitive must be all, sensitive, or non_sensitive")
        if exclude_reviewed:
            conditions.append("r.normalized_sha256 IS NULL")
        if apply_selection_policy:
            conditions.extend(
                (
                    "e.line_count >= ?",
                    "e.line_count <= ?",
                    "e.max_columns >= ?",
                    "e.character_count >= ?",
                )
            )
            parameters.extend(
                (
                    MIN_REVIEW_LINES,
                    MAX_REVIEW_LINES,
                    MIN_REVIEW_COLUMNS,
                    MIN_REVIEW_CHARACTERS,
                )
            )
            for path_part in DEFERRED_PATH_PARTS:
                conditions.append(
                    "instr('/' || replace(f.source_path, '\\', '/'), ?) = 0"
                )
                parameters.append(path_part)
            for keyword in DEFERRED_KEYWORDS:
                conditions.append("instr(COALESCE(e.section, ''), ?) = 0")
                parameters.append(keyword)
                conditions.append("instr(f.source_path, ?) = 0")
                parameters.append(keyword)
            if excluded_source_file_ids:
                placeholders = ", ".join("?" for _ in excluded_source_file_ids)
                conditions.append(f"e.source_file_id NOT IN ({placeholders})")
                parameters.extend(excluded_source_file_ids)
        parameters.append(limit)
        query = f"""
            SELECT
                e.id, e.source_file_id, e.chunk_index, e.section, e.category, e.sensitive,
                e.normalized_sha256, e.line_count, e.max_columns,
                e.character_count, e.art_score, e.preview,
                f.zip_member, f.source_path, f.encoding AS source_encoding,
                r.decision, r.corrected_category, r.quality_score,
                r.issue_tags_json, r.comment, r.updated_at_utc
            FROM entries e
            JOIN source_files f ON f.id = e.source_file_id
            LEFT JOIN reviewdb.reviews r ON r.normalized_sha256 = e.normalized_sha256
            WHERE {' AND '.join(conditions)}
              AND EXISTS (
                SELECT 1 FROM contents c
                WHERE c.normalized_sha256 = e.normalized_sha256
                  AND c.canonical_entry_id = e.id
              )
            ORDER BY e.id
            LIMIT ?
        """
        return query, parameters

    def _read_text(
        self,
        zip_member: str,
        chunk_index: int,
        expected_hash: str,
        archive: zipfile.ZipFile | None = None,
    ) -> str:
        if archive is None:
            with zipfile.ZipFile(self.archive_path, metadata_encoding="cp932") as opened:
                return self._read_text(zip_member, chunk_index, expected_hash, opened)
        chunks = archive.read(zip_member).split(SPLIT_MARKER)
        if chunk_index >= len(chunks):
            raise ValueError(f"Chunk {chunk_index} is missing from {zip_member}")
        raw = clean_chunk_bytes(chunks[chunk_index])
        text, _ = decode_chunk(raw)
        text = normalize_newlines(text)
        if normalized_sha256(text) != expected_hash:
            raise ValueError(f"Archive content changed for {zip_member}#{chunk_index}")
        return decode_character_references(text)

    @staticmethod
    def _content_exclusion_reason(text: str) -> str | None:
        if MULTIPLE_BLOCK_PATTERN.search(normalize_newlines(text)):
            return "multiple_blocks"
        visible = [character for character in text if not character.isspace()]
        if not visible:
            return "empty"
        symbol_count = sum(
            unicodedata.category(character)[0] in {"P", "S"}
            for character in visible
        )
        if symbol_count / len(visible) < MIN_SYMBOL_RATIO:
            return "text_only"
        return None

    @staticmethod
    def _source_review_history(
        connection: sqlite3.Connection,
    ) -> dict[int, tuple[int, int]]:
        return {
            int(row["source_file_id"]): (int(row["accepted"]), int(row["rejected"]))
            for row in connection.execute(
                """
                SELECT
                    e.source_file_id,
                    SUM(r.decision = 'accept') AS accepted,
                    SUM(r.decision = 'reject') AS rejected
                FROM reviewdb.reviews r
                JOIN contents c ON c.normalized_sha256 = r.normalized_sha256
                JOIN entries e ON e.id = c.canonical_entry_id
                GROUP BY e.source_file_id
                """
            )
        }

    @staticmethod
    def _reviewed_entries_by_source(
        connection: sqlite3.Connection,
    ) -> dict[int, list[sqlite3.Row]]:
        reviewed: dict[int, list[sqlite3.Row]] = {}
        for row in connection.execute(
            """
            SELECT
                e.source_file_id, e.chunk_index, e.normalized_sha256,
                f.zip_member
            FROM reviewdb.reviews r
            JOIN contents c ON c.normalized_sha256 = r.normalized_sha256
            JOIN entries e ON e.id = c.canonical_entry_id
            JOIN source_files f ON f.id = e.source_file_id
            """
        ):
            reviewed.setdefault(int(row["source_file_id"]), []).append(row)
        return reviewed

    @staticmethod
    def _source_category_review_history(
        connection: sqlite3.Connection,
    ) -> dict[tuple[int, str], tuple[int, int]]:
        return {
            (int(row["source_file_id"]), str(row["corrected_category"])): (
                int(row["accepted"]),
                int(row["rejected"]),
            )
            for row in connection.execute(
                """
                SELECT
                    e.source_file_id, r.corrected_category,
                    SUM(r.decision = 'accept') AS accepted,
                    SUM(r.decision = 'reject') AS rejected
                FROM reviewdb.reviews r
                JOIN contents c ON c.normalized_sha256 = r.normalized_sha256
                JOIN entries e ON e.id = c.canonical_entry_id
                GROUP BY e.source_file_id, r.corrected_category
                """
            )
        }

    @staticmethod
    def _near_duplicate_text(first: str, second: str) -> bool:
        """Conservatively identify whitespace/difference variants in one MLT."""

        first_compact = "".join(character for character in first if not character.isspace())
        second_compact = "".join(character for character in second if not character.isspace())
        if first_compact == second_compact:
            return True
        longer = max(len(first_compact), len(second_compact))
        shorter = min(len(first_compact), len(second_compact))
        if shorter < NEAR_DUPLICATE_NGRAM_SIZE or shorter / max(1, longer) < NEAR_DUPLICATE_MIN_LENGTH_RATIO:
            return False

        size = NEAR_DUPLICATE_NGRAM_SIZE
        first_ngrams = Counter(
            first_compact[offset : offset + size]
            for offset in range(len(first_compact) - size + 1)
        )
        second_ngrams = Counter(
            second_compact[offset : offset + size]
            for offset in range(len(second_compact) - size + 1)
        )
        overlap = sum((first_ngrams & second_ngrams).values())
        total = sum(first_ngrams.values()) + sum(second_ngrams.values())
        return total > 0 and (2 * overlap / total) >= NEAR_DUPLICATE_MIN_OVERLAP

    @staticmethod
    def _candidate_rank(
        row: sqlite3.Row,
        source_history: dict[int, tuple[int, int]],
        source_category_history: dict[tuple[int, str], tuple[int, int]],
    ) -> float:
        source_file_id = int(row["source_file_id"])
        accepted, rejected = source_history.get(source_file_id, (0, 0))
        category_accepted, category_rejected = source_category_history.get(
            (source_file_id, str(row["category"])),
            (0, 0),
        )
        # A small Bayesian prior avoids overreacting to one decision while moving
        # repeatedly rejected source files to the back of each random candidate pool.
        source_quality = (accepted + 1.5) / (accepted + rejected + 3.0)
        category_quality = (category_accepted + 1.0) / (
            category_accepted + category_rejected + 2.0
        )
        line_score = min(int(row["line_count"]), 36) / 36
        width_score = min(int(row["max_columns"]), 140) / 140
        character_score = min(int(row["character_count"]), 1800) / 1800
        oversize_penalty = max(0, int(row["line_count"]) - 55) / 55
        return (
            source_quality * 2.0
            + category_quality
            + float(row["art_score"])
            + line_score
            + width_score
            + character_score
            - oversize_penalty
        )

    @staticmethod
    def _row_payload(row: sqlite3.Row, text: str) -> dict[str, object]:
        review = None
        if row["decision"] is not None:
            review = {
                "decision": row["decision"],
                "category": row["corrected_category"],
                "qualityScore": row["quality_score"],
                "issueTags": json.loads(row["issue_tags_json"]),
                "comment": row["comment"],
                "updatedAtUtc": row["updated_at_utc"],
            }
        return {
            "id": row["id"],
            "text": text,
            "sourcePath": row["source_path"],
            "sourceEncoding": row["source_encoding"],
            "chunkIndex": row["chunk_index"],
            "section": row["section"],
            "category": row["category"],
            "sensitive": bool(row["sensitive"]),
            "normalizedSha256": row["normalized_sha256"],
            "lineCount": row["line_count"],
            "maxColumns": row["max_columns"],
            "characterCount": row["character_count"],
            "artScore": row["art_score"],
            "preview": row["preview"],
            "textTransforms": (
                ["character_references"]
                if normalized_sha256(text) != row["normalized_sha256"]
                else []
            ),
            "review": review,
        }

    def next_candidate(
        self,
        *,
        category: str | None = None,
        sensitive: str = "all",
        start_id: int | None = None,
    ) -> dict[str, object] | None:
        with self._attached_index_connection() as connection:
            maximum_id = int(connection.execute("SELECT MAX(id) FROM entries").fetchone()[0] or 0)
            if maximum_id == 0:
                return None
            start = secrets.randbelow(maximum_id + 1) if start_id is None else max(0, start_id)
            if category is not None:
                category_order = [category]
            else:
                reviewed_by_category = {
                    row["category"]: int(row["count"])
                    for row in connection.execute(
                        """
                        SELECT e.category, COUNT(*) AS count
                        FROM reviewdb.reviews r
                        JOIN contents c ON c.normalized_sha256 = r.normalized_sha256
                        JOIN entries e ON e.id = c.canonical_entry_id
                        GROUP BY e.category
                        """
                    )
                }
                category_order = sorted(
                    REVIEW_TARGETS,
                    key=lambda item: (
                        reviewed_by_category.get(item, 0) / REVIEW_TARGETS[item],
                        reviewed_by_category.get(item, 0),
                        item,
                    ),
                )
            row = None
            text = None
            source_history = self._source_review_history(connection)
            source_category_history = self._source_category_review_history(connection)
            reviewed_by_source = self._reviewed_entries_by_source(connection)
            excluded_source_file_ids = tuple(
                sorted(
                    source_file_id
                    for source_file_id, reviewed_entries in reviewed_by_source.items()
                    if len(reviewed_entries) >= MAX_REVIEWS_PER_SOURCE_FILE
                )
            )
            reviewed_text_cache: dict[str, str] = {}
            for selected_category in category_order:
                query, parameters = self._candidate_query(
                    category=selected_category,
                    sensitive=sensitive,
                    start_id=start,
                    exclude_reviewed=True,
                    excluded_source_file_ids=excluded_source_file_ids,
                    limit=CANDIDATE_POOL_SIZE,
                )
                rows = connection.execute(query, parameters).fetchall()
                if not rows and start > 0:
                    query, parameters = self._candidate_query(
                        category=selected_category,
                        sensitive=sensitive,
                        start_id=0,
                        exclude_reviewed=True,
                        excluded_source_file_ids=excluded_source_file_ids,
                        limit=CANDIDATE_POOL_SIZE,
                    )
                    rows = connection.execute(query, parameters).fetchall()
                rows = sorted(
                    rows,
                    key=lambda item: self._candidate_rank(
                        item,
                        source_history,
                        source_category_history,
                    ),
                    reverse=True,
                )
                with zipfile.ZipFile(
                    self.archive_path, metadata_encoding="cp932"
                ) as archive:
                    for candidate in rows:
                        candidate_text = self._read_text(
                            candidate["zip_member"],
                            candidate["chunk_index"],
                            candidate["normalized_sha256"],
                            archive,
                        )
                        if self._content_exclusion_reason(candidate_text) is None:
                            near_duplicate = False
                            for reviewed_entry in reviewed_by_source.get(
                                int(candidate["source_file_id"]), []
                            ):
                                reviewed_hash = str(reviewed_entry["normalized_sha256"])
                                reviewed_text = reviewed_text_cache.get(reviewed_hash)
                                if reviewed_text is None:
                                    reviewed_text = self._read_text(
                                        reviewed_entry["zip_member"],
                                        reviewed_entry["chunk_index"],
                                        reviewed_hash,
                                        archive,
                                    )
                                    reviewed_text_cache[reviewed_hash] = reviewed_text
                                if self._near_duplicate_text(candidate_text, reviewed_text):
                                    near_duplicate = True
                                    break
                            if not near_duplicate:
                                row = candidate
                                text = candidate_text
                                break
                if row is not None:
                    break
        if row is None:
            return None
        assert text is not None
        return self._row_payload(row, text)

    def get_entry(self, entry_id: int) -> dict[str, object] | None:
        with self._attached_index_connection() as connection:
            query, parameters = self._candidate_query(
                category=None,
                sensitive="all",
                start_id=max(0, entry_id - 1),
                exclude_reviewed=False,
                apply_selection_policy=False,
            )
            row = connection.execute(query, parameters).fetchone()
        if row is None or row["id"] != entry_id:
            return None
        text = self._read_text(row["zip_member"], row["chunk_index"], row["normalized_sha256"])
        return self._row_payload(row, text)

    def save_review(self, entry_id: int, submission: ReviewSubmission) -> dict[str, object]:
        self._ensure_review_schema()
        with connect_database(self.index_database) as connection:
            entry = connection.execute(
                "SELECT normalized_sha256 FROM entries WHERE id = ? AND record_type = 'aa'",
                (entry_id,),
            ).fetchone()
        if entry is None:
            raise KeyError(f"AA entry {entry_id} was not found")
        timestamp = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.review_database) as connection:
            connection.execute(
                """
                INSERT INTO reviews(
                    normalized_sha256, entry_id, decision, corrected_category,
                    quality_score, issue_tags_json, comment, reviewer_id,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'local-user', ?, ?)
                ON CONFLICT(normalized_sha256) DO UPDATE SET
                    entry_id = excluded.entry_id,
                    decision = excluded.decision,
                    corrected_category = excluded.corrected_category,
                    quality_score = excluded.quality_score,
                    issue_tags_json = excluded.issue_tags_json,
                    comment = excluded.comment,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    entry["normalized_sha256"],
                    entry_id,
                    submission.decision,
                    submission.category,
                    submission.quality_score,
                    json.dumps(submission.issue_tags, ensure_ascii=False),
                    submission.comment.strip(),
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        candidate = self.get_entry(entry_id)
        if candidate is None:
            raise KeyError(f"AA entry {entry_id} was not found after saving")
        return candidate
