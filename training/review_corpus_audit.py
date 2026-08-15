from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .review_corpus import load_snapshot_records, sha256_bytes


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


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _mirror_compact(text: str) -> str:
    translation: dict[int, int] = {}
    for left, right in MIRROR_PAIRS:
        translation[left] = right
        translation[right] = left
    mirrored = "\n".join(line[::-1].translate(translation) for line in text.splitlines())
    return _compact(mirrored)


def _ngrams(text: str, size: int) -> Counter[str]:
    return Counter(text[offset : offset + size] for offset in range(max(0, len(text) - size + 1)))


def _overlap(first: Counter[str], second: Counter[str]) -> float:
    total = sum(first.values()) + sum(second.values())
    return 2 * sum((first & second).values()) / total if total else 0.0


def audit_review_snapshot(
    snapshot_dir: Path,
    *,
    ngram_size: int = 4,
    minimum_length_ratio: float = 0.8,
    report_threshold: float = 0.88,
) -> dict[str, object]:
    records = load_snapshot_records(snapshot_dir)
    accepted = [record for record in records if record["decision"] == "accept"]
    items: list[dict[str, object]] = []
    hash_failures: list[int] = []
    for record in accepted:
        payload = (snapshot_dir / str(record["text"])).read_bytes()
        if sha256_bytes(payload) != record["unicode_text_sha256"]:
            hash_failures.append(int(record["entry_id"]))
        text = payload.decode("utf-8")
        compact = _compact(text)
        items.append(
            {
                "entry_id": int(record["entry_id"]),
                "source_file_id": int(record["source_file_id"]),
                "category": str(record["corrected_category"]),
                "split": str(record["split"]),
                "compact": compact,
                "ngrams": _ngrams(compact, ngram_size),
                "mirror_ngrams": _ngrams(_mirror_compact(text), ngram_size),
            }
        )

    compact_groups: dict[str, list[int]] = defaultdict(list)
    for item in items:
        compact_groups[str(item["compact"])].append(int(item["entry_id"]))
    whitespace_duplicate_groups = [
        entry_ids for entry_ids in compact_groups.values() if len(entry_ids) > 1
    ]
    near_pairs: list[dict[str, object]] = []
    mirror_pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            left_length = len(str(left["compact"]))
            right_length = len(str(right["compact"]))
            length_ratio = min(left_length, right_length) / max(1, left_length, right_length)
            if length_ratio < minimum_length_ratio:
                continue
            score = _overlap(left["ngrams"], right["ngrams"])
            mirror_score = _overlap(left["mirror_ngrams"], right["ngrams"])
            common = {
                "left_entry_id": left["entry_id"],
                "right_entry_id": right["entry_id"],
                "left_category": left["category"],
                "right_category": right["category"],
                "left_split": left["split"],
                "right_split": right["split"],
                "same_source_file": left["source_file_id"] == right["source_file_id"],
                "length_ratio": round(length_ratio, 6),
            }
            if score >= report_threshold:
                near_pairs.append({**common, "overlap": round(score, 6)})
            if mirror_score >= report_threshold:
                mirror_pairs.append({**common, "overlap": round(mirror_score, 6)})
    near_pairs.sort(key=lambda pair: float(pair["overlap"]), reverse=True)
    mirror_pairs.sort(key=lambda pair: float(pair["overlap"]), reverse=True)
    cross_split_near_pairs = [
        pair for pair in near_pairs if pair["left_split"] != pair["right_split"]
    ]
    cross_split_mirror_pairs = [
        pair for pair in mirror_pairs if pair["left_split"] != pair["right_split"]
    ]

    split_sources: dict[str, set[int]] = defaultdict(set)
    split_categories: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        split = str(item["split"])
        split_sources[split].add(int(item["source_file_id"]))
        split_categories[split][str(item["category"])] += 1
    split_overlaps = []
    split_names = sorted(split_sources)
    for index, left_split in enumerate(split_names):
        for right_split in split_names[index + 1 :]:
            overlap = sorted(split_sources[left_split] & split_sources[right_split])
            if overlap:
                split_overlaps.append(
                    {
                        "left": left_split,
                        "right": right_split,
                        "source_file_ids": overlap,
                    }
                )
    return {
        "schema_version": 1,
        "snapshot": json.loads(
            (snapshot_dir / "snapshot.json").read_text(encoding="utf-8")
        )["snapshot"],
        "accepted_count": len(accepted),
        "text_hash_failures": hash_failures,
        "whitespace_duplicate_groups": whitespace_duplicate_groups,
        "near_duplicate_rule": {
            "ngram_size": ngram_size,
            "minimum_length_ratio": minimum_length_ratio,
            "report_threshold": report_threshold,
        },
        "near_duplicate_pairs": near_pairs,
        "mirror_near_duplicate_pairs": mirror_pairs,
        "cross_split_near_duplicate_pairs": cross_split_near_pairs,
        "cross_split_mirror_near_duplicate_pairs": cross_split_mirror_pairs,
        "split_source_overlaps": split_overlaps,
        "split_source_counts": {
            split: len(sources) for split, sources in sorted(split_sources.items())
        },
        "split_category_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_categories.items())
        },
        "passed": not (
            hash_failures
            or whitespace_duplicate_groups
            or cross_split_near_pairs
            or cross_split_mirror_pairs
            or split_overlaps
        ),
    }
