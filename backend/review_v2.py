from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from training.yaruyomi import connect_database, normalized_sha256

from .review import (
    DEFAULT_ARCHIVE,
    DEFAULT_INDEX_DATABASE,
    ISSUE_TAGS,
    MULTIPLE_BLOCK_PATTERN,
    REVIEW_CATEGORIES,
    ReviewStore,
)


DEFAULT_REVIEW_V2_DATABASE = DEFAULT_INDEX_DATABASE.parent / "reviews-v2.sqlite3"
REVIEW_V2_SCHEMA_VERSION = 2
SELECTION_POLICY_VERSION = 6
CHECKPOINT_ACCEPTED_COUNT = 750
BATCH_SIZE = 50
MAX_PRESENTATIONS_PER_SOURCE_FILE = 3
CANDIDATE_POOL_SIZE = 256

# Each collection checkpoint preserves the first 250-work category proportions.
ACCEPTED_TARGETS_250 = {
    "character": 70,
    "multiple_people": 20,
    "face": 18,
    "upper_body": 18,
    "full_body": 18,
    "background": 20,
    "architecture": 20,
    "nature": 12,
    "object": 18,
    "mecha": 12,
    "effect": 12,
    "text_or_joke": 12,
}
ACCEPTED_TARGETS = {
    category: target * 3 for category, target in ACCEPTED_TARGETS_250.items()
}


class ReviewV2Submission(BaseModel):
    decision: Literal["accept", "hold", "reject"]
    category: str
    quality_score: int | None = Field(default=None, ge=0, le=5)
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

    @model_validator(mode="after")
    def accepted_work_requires_category(self) -> "ReviewV2Submission":
        if self.decision == "accept" and self.category == "unknown":
            raise ValueError("An accepted work must have a concrete category")
        return self


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


class ReviewV2Store:
    def __init__(
        self,
        index_database: Path = DEFAULT_INDEX_DATABASE,
        archive_path: Path = DEFAULT_ARCHIVE,
        review_database: Path = DEFAULT_REVIEW_V2_DATABASE,
    ) -> None:
        self.index_database = index_database.resolve()
        self.archive_path = archive_path.resolve()
        self.review_database = review_database.resolve()
        self._reader = ReviewStore(
            self.index_database,
            self.archive_path,
            self.review_database,
        )

    def _availability_error(self) -> str | None:
        missing = [
            str(path)
            for path in (self.index_database, self.archive_path)
            if not path.exists()
        ]
        return "Review data is missing: " + ", ".join(missing) if missing else None

    def _archive_metadata(self) -> dict[str, str]:
        with connect_database(self.index_database) as connection:
            return {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }

    def _ensure_schema(self) -> None:
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
                    quality_score INTEGER CHECK(quality_score BETWEEN 0 AND 5),
                    issue_tags_json TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS reviews_entry_id ON reviews(entry_id);
                CREATE INDEX IF NOT EXISTS reviews_decision ON reviews(decision);
                CREATE INDEX IF NOT EXISTS reviews_category ON reviews(corrected_category);

                CREATE TABLE IF NOT EXISTS candidate_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_version INTEGER NOT NULL,
                    category_filter TEXT,
                    sensitive_filter TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidate_queue (
                    batch_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    entry_id INTEGER NOT NULL UNIQUE,
                    source_file_id INTEGER NOT NULL,
                    normalized_sha256 TEXT NOT NULL UNIQUE,
                    indexed_category TEXT NOT NULL,
                    compact_text TEXT NOT NULL,
                    compact_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending', 'skipped')),
                    queued_at_utc TEXT NOT NULL,
                    PRIMARY KEY(batch_id, position),
                    FOREIGN KEY(batch_id) REFERENCES candidate_batches(id)
                );

                CREATE INDEX IF NOT EXISTS candidate_queue_state
                    ON candidate_queue(state, batch_id, position);
                CREATE INDEX IF NOT EXISTS candidate_queue_source
                    ON candidate_queue(source_file_id);
                CREATE INDEX IF NOT EXISTS candidate_queue_compact
                    ON candidate_queue(compact_sha256);
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
            stored_schema = connection.execute(
                "SELECT value FROM metadata WHERE key = 'review_schema_version'"
            ).fetchone()
            if stored_schema is not None and stored_schema[0] != str(REVIEW_V2_SCHEMA_VERSION):
                raise ValueError(
                    "The active review database has an unsupported schema: "
                    f"{stored_schema[0]}"
                )
            connection.executemany(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("review_schema_version", str(REVIEW_V2_SCHEMA_VERSION)),
                    ("selection_policy_version", str(SELECTION_POLICY_VERSION)),
                    ("archive_sha256", archive_hash),
                    ("archive_version", metadata.get("archive_version", "unknown")),
                    ("reviewer_id", "local-user"),
                    ("corpus", "accepted-v2"),
                ),
            )
            connection.execute(
                """
                INSERT INTO metadata(key, value)
                VALUES ('selection_policy_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SELECTION_POLICY_VERSION),),
            )
            connection.commit()

    def _attached_connection(self) -> sqlite3.Connection:
        self._ensure_schema()
        connection = connect_database(self.index_database)
        connection.execute("ATTACH DATABASE ? AS reviewdb", (str(self.review_database),))
        return connection

    @staticmethod
    def _accepted_counts(connection: sqlite3.Connection) -> dict[str, int]:
        return {
            str(row["corrected_category"]): int(row["count"])
            for row in connection.execute(
                """
                SELECT corrected_category, COUNT(*) AS count
                FROM reviewdb.reviews
                WHERE decision = 'accept'
                GROUP BY corrected_category
                """
            )
        }

    @staticmethod
    def _checkpoint_ready(accepted_counts: dict[str, int]) -> bool:
        return bool(
            sum(accepted_counts.values()) >= CHECKPOINT_ACCEPTED_COUNT
            and all(
                accepted_counts.get(category, 0) >= target
                for category, target in ACCEPTED_TARGETS.items()
            )
        )

    @staticmethod
    def _direct_acceptance_rates(
        connection: sqlite3.Connection,
    ) -> dict[str, float]:
        rows = connection.execute(
            """
            SELECT
                q.indexed_category,
                COUNT(*) AS reviewed_count,
                SUM(
                    CASE
                        WHEN r.decision = 'accept'
                         AND r.corrected_category = q.indexed_category
                        THEN 1 ELSE 0
                    END
                ) AS direct_accept_count
            FROM reviewdb.candidate_queue q
            JOIN reviewdb.reviews r
              ON r.normalized_sha256 = q.normalized_sha256
            GROUP BY q.indexed_category
            """
        ).fetchall()
        rates: dict[str, float] = {}
        for row in rows:
            # Beta(2, 4) keeps unseen and very small strata from receiving an
            # unstable zero or one probability.
            rate = (int(row["direct_accept_count"]) + 2) / (
                int(row["reviewed_count"]) + 6
            )
            rates[str(row["indexed_category"])] = min(0.95, max(0.05, rate))
        return rates

    def status(self) -> dict[str, object]:
        availability_error = self._availability_error()
        if availability_error:
            return {
                "available": False,
                "message": availability_error,
                "indexDatabase": str(self.index_database),
                "archive": str(self.archive_path),
            }
        self._ensure_schema()
        metadata = self._archive_metadata()
        with sqlite3.connect(self.review_database) as connection:
            decision_counts = {
                str(decision): int(count)
                for decision, count in connection.execute(
                    "SELECT decision, COUNT(*) FROM reviews GROUP BY decision"
                )
            }
            accepted_by_category = {
                str(category): int(count)
                for category, count in connection.execute(
                    """
                    SELECT corrected_category, COUNT(*)
                    FROM reviews WHERE decision = 'accept'
                    GROUP BY corrected_category
                    """
                )
            }
            last_reviewed = connection.execute(
                "SELECT entry_id FROM reviews ORDER BY updated_at_utc DESC LIMIT 1"
            ).fetchone()
            current_batch = connection.execute(
                "SELECT MAX(id) FROM candidate_batches"
            ).fetchone()[0]
            pending_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM candidate_queue q
                    LEFT JOIN reviews r ON r.normalized_sha256 = q.normalized_sha256
                    WHERE q.state = 'pending' AND r.normalized_sha256 IS NULL
                    """
                ).fetchone()[0]
            )
        accepted_count = decision_counts.get("accept", 0)
        accepted_counts = {
            category: accepted_by_category.get(category, 0)
            for category in ACCEPTED_TARGETS
        }
        return {
            "available": True,
            "corpus": "accepted-v2",
            "archiveVersion": metadata.get("archive_version", "unknown"),
            "archiveSha256": metadata.get("archive_sha256", ""),
            "reviewedCount": sum(decision_counts.values()),
            "acceptedCount": accepted_count,
            "targetCount": CHECKPOINT_ACCEPTED_COUNT,
            "targetMetric": "accepted",
            "checkpointReady": self._checkpoint_ready(accepted_counts),
            "decisionCounts": {
                "accept": accepted_count,
                "hold": decision_counts.get("hold", 0),
                "reject": decision_counts.get("reject", 0),
            },
            "acceptedByCategory": accepted_counts,
            "lastReviewedEntryId": last_reviewed[0] if last_reviewed else None,
            "categories": list(REVIEW_CATEGORIES),
            "issueTags": list(ISSUE_TAGS),
            "reviewTargets": ACCEPTED_TARGETS,
            "currentBatchId": current_batch,
            "pendingCandidateCount": pending_count,
            "selectionPolicy": {
                "version": SELECTION_POLICY_VERSION,
                "batchSize": BATCH_SIZE,
                "maximumPresentationsPerSourceFile": MAX_PRESENTATIONS_PER_SOURCE_FILE,
                "allocation": "accepted-category-deficit",
                "allocationDetail": (
                    "each 50-item batch weights corrected-category deficits by "
                    "observed direct acceptance yield"
                ),
                "sizePolicy": "ranking-only; no hard line, column, or character minimum",
                "hardExclusions": [
                    "non-aa index records",
                    "empty content",
                    "multiple vertically separated works",
                    "v2 exact and whitespace-only duplicates",
                    "unreadable or hash-mismatched archive content",
                ],
                "nearDuplicateRule": (
                    "modified works remain reviewable; group related works into one "
                    "split and optionally prune only before training"
                ),
                "sourceDiversityRule": (
                    "three presentations per MLT by default; if every source in "
                    "an under-target indexed category is exhausted, expand the "
                    "per-source round evenly using observed acceptance yield"
                ),
                "oldReviewInfluence": "none",
            },
            "sensitivePolicy": "included and tagged",
            "reviewDatabase": str(self.review_database),
        }

    @staticmethod
    def _candidate_rank(
        row: sqlite3.Row,
        source_presentation_count: int = 0,
    ) -> tuple[int, float, int]:
        line_score = min(int(row["line_count"]), 60) / 60
        width_score = min(int(row["max_columns"]), 180) / 180
        character_score = min(int(row["character_count"]), 3000) / 3000
        oversize_penalty = max(0, int(row["line_count"]) - 100) / 100
        return (
            -source_presentation_count,
            float(row["art_score"])
            + line_score
            + width_score
            + character_score
            - oversize_penalty,
            -int(row["id"]),
        )

    @staticmethod
    def _category_source_limit(
        connection: sqlite3.Connection,
        *,
        category: str,
        accepted_count: int,
        acceptance_rate: float,
        source_counts: Counter[int],
    ) -> int:
        source_ids = {
            int(row["source_file_id"])
            for row in connection.execute(
                """
                SELECT DISTINCT source_file_id
                FROM entries
                WHERE record_type = 'aa' AND category = ?
                """,
                (category,),
            )
        }
        if not source_ids:
            return MAX_PRESENTATIONS_PER_SOURCE_FILE
        remaining = max(0, ACCEPTED_TARGETS.get(category, 0) - accepted_count)
        current_presentations = sum(source_counts[source_id] for source_id in source_ids)
        projected_presentations = current_presentations + math.ceil(
            remaining / max(0.05, acceptance_rate)
        )
        return max(
            MAX_PRESENTATIONS_PER_SOURCE_FILE,
            math.ceil(projected_presentations / len(source_ids)),
        )

    @staticmethod
    def _deterministic_start(
        archive_hash: str,
        batch_number: int,
        category: str,
        attempt: int,
        maximum_id: int,
    ) -> int:
        payload = (
            f"{archive_hash}:{SELECTION_POLICY_VERSION}:{batch_number}:"
            f"{category}:{attempt}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % max(1, maximum_id)

    @staticmethod
    def _content_exclusion_reason(text: str) -> str | None:
        if MULTIPLE_BLOCK_PATTERN.search(text):
            return "multiple_blocks"
        if not any(not character.isspace() for character in text):
            return "empty"
        return None

    @staticmethod
    def _candidate_query(
        *,
        category: str,
        sensitive: str,
        start_id: int,
        excluded_source_ids: tuple[int, ...],
        limit: int,
    ) -> tuple[str, list[object]]:
        conditions = [
            "e.record_type = 'aa'",
            "e.category = ?",
            "e.id > ?",
            "r.normalized_sha256 IS NULL",
            "q.normalized_sha256 IS NULL",
        ]
        parameters: list[object] = [category, start_id]
        if sensitive == "sensitive":
            conditions.append("e.sensitive = 1")
        elif sensitive == "non_sensitive":
            conditions.append("e.sensitive = 0")
        elif sensitive != "all":
            raise ValueError("sensitive must be all, sensitive, or non_sensitive")
        if excluded_source_ids:
            placeholders = ", ".join("?" for _ in excluded_source_ids)
            conditions.append(f"e.source_file_id NOT IN ({placeholders})")
            parameters.extend(excluded_source_ids)
        parameters.append(limit)
        return (
            f"""
            SELECT
                e.id, e.source_file_id, e.chunk_index, e.section, e.category, e.sensitive,
                e.normalized_sha256, e.line_count, e.max_columns,
                e.character_count, e.art_score, e.preview,
                f.zip_member, f.source_path, f.encoding AS source_encoding
            FROM entries e
            JOIN source_files f ON f.id = e.source_file_id
            LEFT JOIN reviewdb.reviews r ON r.normalized_sha256 = e.normalized_sha256
            LEFT JOIN reviewdb.candidate_queue q ON q.normalized_sha256 = e.normalized_sha256
            WHERE {' AND '.join(conditions)}
              AND EXISTS (
                SELECT 1 FROM contents c
                WHERE c.normalized_sha256 = e.normalized_sha256
                  AND c.canonical_entry_id = e.id
              )
            ORDER BY e.id
            LIMIT ?
            """,
            parameters,
        )

    @staticmethod
    def _prior_fingerprints(
        connection: sqlite3.Connection,
    ) -> set[str]:
        return {
            str(row["compact_text"])
            for row in connection.execute(
                "SELECT compact_text FROM reviewdb.candidate_queue"
            )
        }

    def _find_candidate(
        self,
        connection: sqlite3.Connection,
        archive: zipfile.ZipFile,
        *,
        category: str,
        sensitive: str,
        archive_hash: str,
        batch_number: int,
        attempt: int,
        maximum_id: int,
        source_counts: Counter[int],
        source_limit: int,
        prior_fingerprints: set[str],
    ) -> tuple[sqlite3.Row, str, str] | None:
        excluded_source_ids = tuple(
            sorted(
                source_id
                for source_id, count in source_counts.items()
                if count >= source_limit
            )
        )
        start = self._deterministic_start(
            archive_hash, batch_number, category, attempt, maximum_id
        )
        row_sets: list[list[sqlite3.Row]] = []
        for query_start in (start, 0):
            query, parameters = self._candidate_query(
                category=category,
                sensitive=sensitive,
                start_id=query_start,
                excluded_source_ids=excluded_source_ids,
                limit=CANDIDATE_POOL_SIZE,
            )
            rows = connection.execute(query, parameters).fetchall()
            row_sets.append(rows)
            if rows:
                break
        rows = sorted(
            row_sets[-1],
            key=lambda row: self._candidate_rank(
                row, source_counts[int(row["source_file_id"])]
            ),
            reverse=True,
        )
        for row in rows:
            text = self._reader._read_text(
                str(row["zip_member"]),
                int(row["chunk_index"]),
                str(row["normalized_sha256"]),
                archive,
            )
            if self._content_exclusion_reason(text) is not None:
                continue
            compact = _compact(text)
            if compact in prior_fingerprints:
                continue
            return row, text, compact
        return None

    def _create_batch(
        self,
        connection: sqlite3.Connection,
        *,
        category: str | None,
        sensitive: str,
    ) -> int | None:
        accepted_counts = self._accepted_counts(connection)
        if self._checkpoint_ready(accepted_counts):
            return None
        if category is not None and category not in REVIEW_CATEGORIES:
            raise ValueError(f"Unknown review category: {category}")
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO reviewdb.candidate_batches(
                policy_version, category_filter, sensitive_filter, created_at_utc
            ) VALUES (?, ?, ?, ?)
            """,
            (SELECTION_POLICY_VERSION, category, sensitive, timestamp),
        )
        batch_id = int(cursor.lastrowid)
        maximum_id = int(connection.execute("SELECT MAX(id) FROM entries").fetchone()[0] or 0)
        source_counts = Counter(
            {
                int(row["source_file_id"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT source_file_id, COUNT(*) AS count
                    FROM reviewdb.candidate_queue
                    GROUP BY source_file_id
                    """
                )
            }
        )
        acceptance_rates = self._direct_acceptance_rates(connection)
        source_limits = {
            item: self._category_source_limit(
                connection,
                category=item,
                accepted_count=accepted_counts.get(item, 0),
                acceptance_rate=acceptance_rates.get(item, 1 / 3),
                source_counts=source_counts,
            )
            for item in ACCEPTED_TARGETS
        }
        presentation_needs = {
            item: max(
                1.0,
                (ACCEPTED_TARGETS[item] - accepted_counts.get(item, 0))
                / acceptance_rates.get(item, 1 / 3),
            )
            for item in ACCEPTED_TARGETS
        }
        prior_fingerprints = self._prior_fingerprints(connection)
        scheduled_counts: Counter[str] = Counter()
        attempts: Counter[str] = Counter()
        exhausted: set[str] = set()
        inserted = 0
        archive_hash = self._archive_metadata()["archive_sha256"]
        with zipfile.ZipFile(self.archive_path, metadata_encoding="cp932") as archive:
            while inserted < BATCH_SIZE:
                if category is not None:
                    categories = [] if category in exhausted else [category]
                else:
                    categories = [
                        item
                        for item in ACCEPTED_TARGETS
                        if item not in exhausted
                        and accepted_counts.get(item, 0) < ACCEPTED_TARGETS[item]
                    ]
                if not categories:
                    break
                selected_category = min(
                    categories,
                    key=lambda item: (
                        scheduled_counts[item] / presentation_needs[item],
                        accepted_counts.get(item, 0) / ACCEPTED_TARGETS[item],
                        list(ACCEPTED_TARGETS).index(item)
                        if item in ACCEPTED_TARGETS
                        else len(ACCEPTED_TARGETS),
                    ),
                )
                attempt = attempts[selected_category]
                candidate = self._find_candidate(
                    connection,
                    archive,
                    category=selected_category,
                    sensitive=sensitive,
                    archive_hash=archive_hash,
                    batch_number=batch_id,
                    attempt=attempt,
                    maximum_id=maximum_id,
                    source_counts=source_counts,
                    source_limit=source_limits[selected_category],
                    prior_fingerprints=prior_fingerprints,
                )
                attempts[selected_category] += 1
                if candidate is None:
                    if attempts[selected_category] >= 8:
                        exhausted.add(selected_category)
                    continue
                row, _, compact = candidate
                connection.execute(
                    """
                    INSERT INTO reviewdb.candidate_queue(
                        batch_id, position, entry_id, source_file_id,
                        normalized_sha256, indexed_category, compact_text,
                        compact_sha256, queued_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        inserted,
                        int(row["id"]),
                        int(row["source_file_id"]),
                        str(row["normalized_sha256"]),
                        str(row["category"]),
                        compact,
                        hashlib.sha256(compact.encode("utf-8")).hexdigest(),
                        timestamp,
                    ),
                )
                source_counts[int(row["source_file_id"])] += 1
                scheduled_counts[selected_category] += 1
                prior_fingerprints.add(compact)
                inserted += 1
        connection.commit()
        return batch_id if inserted else None

    @staticmethod
    def _queued_candidate_row(
        connection: sqlite3.Connection,
        *,
        category: str | None,
        sensitive: str,
    ) -> sqlite3.Row | None:
        conditions = ["q.state = 'pending'", "r.normalized_sha256 IS NULL"]
        parameters: list[object] = []
        if category:
            conditions.append("e.category = ?")
            parameters.append(category)
        if sensitive == "sensitive":
            conditions.append("e.sensitive = 1")
        elif sensitive == "non_sensitive":
            conditions.append("e.sensitive = 0")
        elif sensitive != "all":
            raise ValueError("sensitive must be all, sensitive, or non_sensitive")
        return connection.execute(
            f"""
            SELECT
                e.id, e.source_file_id, e.chunk_index, e.section, e.category, e.sensitive,
                e.normalized_sha256, e.line_count, e.max_columns,
                e.character_count, e.art_score, e.preview,
                f.zip_member, f.source_path, f.encoding AS source_encoding,
                r.decision, r.corrected_category, r.quality_score,
                r.issue_tags_json, r.comment, r.updated_at_utc,
                q.batch_id, q.position
            FROM reviewdb.candidate_queue q
            JOIN entries e ON e.id = q.entry_id
            JOIN source_files f ON f.id = e.source_file_id
            LEFT JOIN reviewdb.reviews r ON r.normalized_sha256 = q.normalized_sha256
            WHERE {' AND '.join(conditions)}
            ORDER BY q.batch_id, q.position
            LIMIT 1
            """,
            parameters,
        ).fetchone()

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
            "id": int(row["id"]),
            "text": text,
            "sourcePath": row["source_path"],
            "sourceEncoding": row["source_encoding"],
            "chunkIndex": int(row["chunk_index"]),
            "section": row["section"],
            "category": row["category"],
            "sensitive": bool(row["sensitive"]),
            "normalizedSha256": row["normalized_sha256"],
            "lineCount": int(row["line_count"]),
            "maxColumns": int(row["max_columns"]),
            "characterCount": int(row["character_count"]),
            "artScore": float(row["art_score"]),
            "preview": row["preview"],
            "textTransforms": (
                ["character_references"]
                if normalized_sha256(text) != row["normalized_sha256"]
                else []
            ),
            "batchId": row["batch_id"] if "batch_id" in row.keys() else None,
            "batchPosition": row["position"] if "position" in row.keys() else None,
            "review": review,
        }

    def next_candidate(
        self,
        *,
        category: str | None = None,
        sensitive: str = "all",
        skip_entry_id: int | None = None,
    ) -> dict[str, object] | None:
        with self._attached_connection() as connection:
            if skip_entry_id is not None:
                connection.execute(
                    """
                    UPDATE reviewdb.candidate_queue
                    SET state = 'skipped'
                    WHERE entry_id = ?
                      AND normalized_sha256 NOT IN (
                          SELECT normalized_sha256 FROM reviewdb.reviews
                      )
                    """,
                    (skip_entry_id,),
                )
                connection.commit()
            if self._checkpoint_ready(self._accepted_counts(connection)):
                return None
            row = self._queued_candidate_row(
                connection, category=category, sensitive=sensitive
            )
            if row is None:
                if self._create_batch(
                    connection, category=category, sensitive=sensitive
                ) is None:
                    return None
                row = self._queued_candidate_row(
                    connection, category=category, sensitive=sensitive
                )
            if row is None:
                return None
        text = self._reader._read_text(
            str(row["zip_member"]),
            int(row["chunk_index"]),
            str(row["normalized_sha256"]),
        )
        return self._row_payload(row, text)

    def get_entry(self, entry_id: int) -> dict[str, object] | None:
        with self._attached_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    e.id, e.source_file_id, e.chunk_index, e.section, e.category, e.sensitive,
                    e.normalized_sha256, e.line_count, e.max_columns,
                    e.character_count, e.art_score, e.preview,
                    f.zip_member, f.source_path, f.encoding AS source_encoding,
                    r.decision, r.corrected_category, r.quality_score,
                    r.issue_tags_json, r.comment, r.updated_at_utc,
                    q.batch_id, q.position
                FROM entries e
                JOIN source_files f ON f.id = e.source_file_id
                LEFT JOIN reviewdb.reviews r ON r.normalized_sha256 = e.normalized_sha256
                LEFT JOIN reviewdb.candidate_queue q ON q.entry_id = e.id
                WHERE e.id = ? AND e.record_type = 'aa'
                LIMIT 1
                """,
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        text = self._reader._read_text(
            str(row["zip_member"]),
            int(row["chunk_index"]),
            str(row["normalized_sha256"]),
        )
        return self._row_payload(row, text)

    def save_review(
        self,
        entry_id: int,
        submission: ReviewV2Submission,
    ) -> dict[str, object]:
        self._ensure_schema()
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
                    str(entry["normalized_sha256"]),
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
