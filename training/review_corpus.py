from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .yaruyomi import (
    SPLIT_MARKER,
    clean_chunk_bytes,
    connect_database,
    decode_character_references,
    decode_chunk,
    normalize_newlines,
    normalized_sha256,
)


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_NAME = "yaruyomi-accepted-v1"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
RELATED_NGRAM_SIZE = 4
RELATED_MINIMUM_LENGTH_RATIO = 0.8
RELATED_OVERLAP_THRESHOLD = 0.88
MIRROR_PAIRS = (
    (47, 92),
    (40, 41),
    (91, 93),
    (123, 125),
    (60, 62),
    (65295, 65340),
    (65288, 65289),
    (65339, 65341),
    (65371, 65373),
    (65308, 65310),
    (9484, 9488),
    (9492, 9496),
    (9500, 9508),
    (9581, 9582),
    (9584, 9583),
    (8592, 8594),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _mirror_compact(text: str) -> str:
    translation: dict[int, int] = {}
    for left, right in MIRROR_PAIRS:
        translation[left] = right
        translation[right] = left
    return _compact(
        "\n".join(line[::-1].translate(translation) for line in text.splitlines())
    )


def _ngrams(text: str, size: int = RELATED_NGRAM_SIZE) -> Counter[str]:
    return Counter(
        text[offset : offset + size]
        for offset in range(max(0, len(text) - size + 1))
    )


def _overlap(first: Counter[str], second: Counter[str]) -> float:
    total = sum(first.values()) + sum(second.values())
    return 2 * sum((first & second).values()) / total if total else 0.0


def related_source_groups(
    records: Iterable[dict[str, object]],
    *,
    minimum_length_ratio: float = RELATED_MINIMUM_LENGTH_RATIO,
    overlap_threshold: float = RELATED_OVERLAP_THRESHOLD,
) -> tuple[dict[int, int], list[list[int]]]:
    """Group cross-MLT variants so accepted derivatives cannot leak across splits."""

    items = []
    source_ids: set[int] = set()
    for record in records:
        source_id = int(record["source_file_id"])
        text = str(record["text"])
        compact = _compact(text)
        source_ids.add(source_id)
        items.append(
            {
                "source_id": source_id,
                "compact": compact,
                "ngrams": _ngrams(compact),
                "mirror_ngrams": _ngrams(_mirror_compact(text)),
            }
        )

    parent = {source_id: source_id for source_id in source_ids}

    def find(source_id: int) -> int:
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        smaller, larger = sorted((first_root, second_root))
        parent[larger] = smaller

    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            if left["source_id"] == right["source_id"]:
                continue
            left_length = len(str(left["compact"]))
            right_length = len(str(right["compact"]))
            length_ratio = min(left_length, right_length) / max(
                1, left_length, right_length
            )
            if length_ratio < minimum_length_ratio:
                continue
            if (
                _overlap(left["ngrams"], right["ngrams"]) >= overlap_threshold
                or _overlap(left["mirror_ngrams"], right["ngrams"])
                >= overlap_threshold
            ):
                union(int(left["source_id"]), int(right["source_id"]))

    source_groups = {source_id: find(source_id) for source_id in source_ids}
    components: dict[int, set[int]] = defaultdict(set)
    for source_id, group_id in source_groups.items():
        components[group_id].add(source_id)
    related_components = sorted(
        (sorted(component) for component in components.values() if len(component) > 1),
        key=lambda component: (component[0], len(component)),
    )
    return source_groups, related_components


def _split_cost(
    total_counts: Counter[str],
    category_counts: dict[str, Counter[str]],
    total_target: dict[str, float],
    category_target: dict[str, dict[str, float]],
) -> float:
    total_cost = sum(
        ((total_counts[split] - total_target[split]) / max(1.0, total_target[split])) ** 2
        for split in SPLIT_NAMES
    )
    category_cost = 0.0
    for category, targets in category_target.items():
        for split in SPLIT_NAMES:
            category_cost += (
                (category_counts[category][split] - targets[split])
                / max(1.0, targets[split])
            ) ** 2
    return total_cost + category_cost * 0.35


def assign_source_group_splits(
    records: Iterable[dict[str, object]],
    *,
    seed: int = 42,
    source_groups: dict[int, int] | None = None,
) -> dict[int, str]:
    """Keep every MLT and supplied related-source group in one split."""

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_sources: dict[int, set[int]] = defaultdict(set)
    for record in records:
        source_id = int(record["source_file_id"])
        group_id = (source_groups or {}).get(source_id, source_id)
        grouped[group_id].append(record)
        grouped_sources[group_id].add(source_id)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("At least three source MLT files are required for a three-way split.")

    total = sum(len(items) for items in grouped.values())
    categories = Counter(
        str(record["corrected_category"])
        for items in grouped.values()
        for record in items
    )
    total_target = {
        split: total * SPLIT_FRACTIONS[split]
        for split in SPLIT_NAMES
    }
    category_target = {
        category: {
            split: count * SPLIT_FRACTIONS[split]
            for split in SPLIT_NAMES
        }
        for category, count in categories.items()
    }

    rng = random.Random(seed)
    tie_breakers = {source_id: rng.random() for source_id in grouped}
    ordered_groups = sorted(
        grouped,
        key=lambda source_id: (-len(grouped[source_id]), tie_breakers[source_id], source_id),
    )
    total_counts: Counter[str] = Counter()
    category_counts = {category: Counter() for category in categories}
    assignments: dict[int, str] = {}

    for group_index, source_id in enumerate(ordered_groups):
        group = grouped[source_id]
        group_categories = Counter(str(item["corrected_category"]) for item in group)
        candidates: list[tuple[float, int, str]] = []
        remaining_after = len(ordered_groups) - group_index - 1
        for split_index, split in enumerate(SPLIT_NAMES):
            next_totals = total_counts.copy()
            next_totals[split] += len(group)
            next_categories = {
                category: counts.copy()
                for category, counts in category_counts.items()
            }
            for category, count in group_categories.items():
                next_categories[category][split] += count
            empty_splits = sum(next_totals[item] == 0 for item in SPLIT_NAMES)
            impossible_empty_penalty = 1_000_000.0 if empty_splits > remaining_after else 0.0
            candidates.append(
                (
                    _split_cost(
                        next_totals,
                        next_categories,
                        total_target,
                        category_target,
                    )
                    + impossible_empty_penalty,
                    split_index,
                    split,
                )
            )
        selected = min(candidates)[2]
        for member_source_id in grouped_sources[source_id]:
            assignments[member_source_id] = selected
        total_counts[selected] += len(group)
        for category, count in group_categories.items():
            category_counts[category][selected] += count

    if set(assignments.values()) != set(SPLIT_NAMES):
        raise ValueError("Could not assign at least one source MLT to every split.")
    return assignments


def _review_rows(
    index_database: Path,
    review_database: Path,
) -> tuple[list[sqlite3.Row], dict[str, str]]:
    with connect_database(index_database) as connection:
        connection.execute("ATTACH DATABASE ? AS reviewdb", (str(review_database.resolve()),))
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM metadata")
        }
        rows = connection.execute(
            """
            SELECT
                e.id, e.source_file_id, e.chunk_index, e.section,
                e.category AS indexed_category, e.sensitive,
                e.raw_sha256, e.normalized_sha256, e.line_count,
                e.max_columns, e.character_count, e.art_score,
                f.zip_member, f.source_path, f.encoding AS source_encoding,
                r.decision, r.corrected_category, r.quality_score,
                r.issue_tags_json, r.comment, r.reviewer_id,
                r.created_at_utc, r.updated_at_utc
            FROM reviewdb.reviews r
            JOIN contents c ON c.normalized_sha256 = r.normalized_sha256
            JOIN entries e ON e.id = c.canonical_entry_id
            JOIN source_files f ON f.id = e.source_file_id
            ORDER BY e.id
            """
        ).fetchall()
    return rows, metadata


def freeze_review_corpus(
    index_database: Path,
    archive_path: Path,
    review_database: Path,
    output_dir: Path,
    *,
    expected_accepted: int = 100,
    seed: int = 42,
    snapshot_name: str = SNAPSHOT_NAME,
    group_near_duplicates: bool = False,
) -> dict[str, object]:
    """Freeze reviewed Unicode texts, provenance, and MLT-isolated splits."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Snapshot output is not empty: {output_dir}")
    rows, metadata = _review_rows(index_database, review_database)
    accepted_rows = [row for row in rows if row["decision"] == "accept"]
    if len(accepted_rows) != expected_accepted:
        raise ValueError(
            f"Expected {expected_accepted} accepted reviews, found {len(accepted_rows)}"
        )
    if metadata.get("archive_sha256") is None:
        raise ValueError("Index metadata has no archive_sha256")

    decoded_rows: list[tuple[sqlite3.Row, str, str, str]] = []
    with zipfile.ZipFile(archive_path, metadata_encoding="cp932") as archive:
        member_cache: dict[str, list[bytes]] = {}
        for row in rows:
            member = str(row["zip_member"])
            chunks = member_cache.get(member)
            if chunks is None:
                chunks = archive.read(member).split(SPLIT_MARKER)
                member_cache[member] = chunks
            raw = clean_chunk_bytes(chunks[int(row["chunk_index"])])
            source_text, decoded_encoding = decode_chunk(raw)
            source_text = normalize_newlines(source_text)
            if normalized_sha256(source_text) != row["normalized_sha256"]:
                raise ValueError(f"Archive content changed for entry {row['id']}")
            text = decode_character_references(source_text)
            decoded_rows.append((row, source_text, text, decoded_encoding))

    accepted_split_records = [
        {
            "source_file_id": int(row["source_file_id"]),
            "corrected_category": str(row["corrected_category"]),
            "text": text,
        }
        for row, _, text, _ in decoded_rows
        if row["decision"] == "accept"
    ]
    if group_near_duplicates:
        source_groups, related_components = related_source_groups(
            accepted_split_records
        )
    else:
        source_groups, related_components = {}, []
    source_splits = assign_source_group_splits(
        accepted_split_records,
        seed=seed,
        source_groups=source_groups,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for row, source_text, text, decoded_encoding in decoded_rows:
        decision = str(row["decision"])
        split = (
            source_splits[int(row["source_file_id"])]
            if decision == "accept"
            else "negative_pool"
        )
        relative_text_path = Path("aa") / split / f"{int(row['id']):07d}.txt"
        text_path = output_dir / relative_text_path
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_payload = text.encode("utf-8")
        text_path.write_bytes(text_payload)
        records.append(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "snapshot": snapshot_name,
                "entry_id": int(row["id"]),
                "decision": decision,
                "split": split,
                "text": relative_text_path.as_posix(),
                "source_file_id": int(row["source_file_id"]),
                "source_path": row["source_path"],
                "zip_member": row["zip_member"],
                "chunk_index": int(row["chunk_index"]),
                "section": row["section"],
                "source_encoding": row["source_encoding"],
                "decoded_encoding": decoded_encoding,
                "indexed_category": row["indexed_category"],
                "corrected_category": row["corrected_category"],
                "sensitive": bool(row["sensitive"]),
                "quality_score": (
                    int(row["quality_score"])
                    if row["quality_score"] is not None
                    else None
                ),
                "issue_tags": json.loads(row["issue_tags_json"]),
                "comment": row["comment"],
                "reviewer_id": row["reviewer_id"],
                "review_created_at_utc": row["created_at_utc"],
                "review_updated_at_utc": row["updated_at_utc"],
                "raw_sha256": row["raw_sha256"],
                "source_normalized_sha256": row["normalized_sha256"],
                "unicode_text_sha256": sha256_bytes(text_payload),
                "text_transforms": (
                    ["character_references"] if text != source_text else []
                ),
                "line_count": int(row["line_count"]),
                "max_columns": int(row["max_columns"]),
                "character_count": int(row["character_count"]),
                "art_score": float(row["art_score"]),
                "rights_status": "unknown; local curation only",
            }
        )

    records_path = output_dir / "records.jsonl"
    records_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    records_path.write_bytes(records_payload)
    accepted = [record for record in records if record["decision"] == "accept"]
    decisions = Counter(str(record["decision"]) for record in records)
    split_counts = Counter(str(record["split"]) for record in accepted)
    split_source_counts = {
        split: len(
            {
                int(record["source_file_id"])
                for record in accepted
                if record["split"] == split
            }
        )
        for split in SPLIT_NAMES
    }
    category_counts = Counter(str(record["corrected_category"]) for record in accepted)
    snapshot: dict[str, object] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot": snapshot_name,
        "archive_version": metadata.get("archive_version"),
        "archive_sha256": metadata["archive_sha256"],
        "source_url": metadata.get("source_url"),
        "rights_status": "unknown; local curation only",
        "seed": seed,
        "split_strategy": (
            "corrected-category-balanced source-MLT and near-duplicate-group holdout"
            if group_near_duplicates
            else "corrected-category-balanced source-MLT holdout"
        ),
        "near_duplicate_grouping": {
            "enabled": group_near_duplicates,
            "ngram_size": RELATED_NGRAM_SIZE,
            "minimum_length_ratio": RELATED_MINIMUM_LENGTH_RATIO,
            "overlap_threshold": RELATED_OVERLAP_THRESHOLD,
            "cross_mlt_group_count": len(related_components),
            "grouped_source_file_count": sum(
                len(component) for component in related_components
            ),
        },
        "review_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "accepted_count": len(accepted),
        "accepted_source_file_count": len(
            {int(record["source_file_id"]) for record in accepted}
        ),
        "accepted_sensitive_count": sum(bool(record["sensitive"]) for record in accepted),
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts": {split: split_counts[split] for split in SPLIT_NAMES},
        "split_source_counts": split_source_counts,
        "review_cutoff_utc": max(str(record["review_updated_at_utc"]) for record in records),
        "records": "records.jsonl",
        "records_sha256": sha256_bytes(records_payload),
    }
    snapshot_payload = (
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (output_dir / "snapshot.json").write_bytes(snapshot_payload)
    return snapshot


def load_snapshot_records(snapshot_dir: Path) -> list[dict[str, object]]:
    snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
    records_path = snapshot_dir / str(snapshot["records"])
    payload = records_path.read_bytes()
    if sha256_bytes(payload) != snapshot["records_sha256"]:
        raise ValueError("records.jsonl does not match snapshot.json")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]
