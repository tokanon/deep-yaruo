from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from backend.rendering import load_font
from backend.review_v2 import ACCEPTED_TARGETS_250

from .analyze_aa_fill_structure import analyze_snapshot
from .analyze_aa_fill_surfaces import analyze_surface_snapshot
from .audit_texture_motifs_p6r4t import MotifAuditConfig, audit_motif_catalog
from .review_corpus import load_snapshot_records
from .review_corpus_audit import audit_review_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT
    / "datasets"
    / "incoming"
    / "yaruyomi"
    / "v32.1"
    / "accepted-v2"
    / "checkpoints"
    / "0500"
)
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"


def _distribution(values: list[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"minimum": 0, "median": 0, "p90": 0, "p95": 0, "maximum": 0}

    def percentile(fraction: float) -> int | float:
        return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def _load_charset(path: Path) -> set[str]:
    with path.open("r", encoding="cp932", newline="") as handle:
        return {str(row["char"]) for row in csv.DictReader(handle)}


def _target_counts(target_count: int) -> dict[str, int]:
    multiplier = target_count / 250
    return {
        category: round(count * multiplier)
        for category, count in ACCEPTED_TARGETS_250.items()
    }


def audit_checkpoint(
    snapshot_root: Path,
    *,
    target_count: int,
    charset_path: Path = DEFAULT_CHARSET,
    font_size: int = 16,
) -> dict[str, object]:
    records = load_snapshot_records(snapshot_root)
    accepted = [record for record in records if record["decision"] == "accept"]
    texts = [
        (snapshot_root / str(record["text"])).read_text(encoding="utf-8")
        for record in accepted
    ]
    character_counts: Counter[str] = Counter(
        character
        for text in texts
        for character in text
        if not character.isspace()
    )
    occurrence_count = sum(character_counts.values())
    charset = _load_charset(charset_path)
    charset_occurrences = sum(
        count for character, count in character_counts.items() if character in charset
    )
    thresholds = (2, 5, 10, 20, 100)
    font = load_font(font_size)
    advance_occurrences: Counter[str] = Counter()
    for character, count in character_counts.items():
        advance_occurrences[f"{font.getlength(character):.3f}"] += count

    categories = Counter(str(record["corrected_category"]) for record in accepted)
    category_splits: dict[str, Counter[str]] = {}
    for record in accepted:
        category = str(record["corrected_category"])
        category_splits.setdefault(category, Counter())[str(record["split"])] += 1
    targets = _target_counts(target_count)
    fill = analyze_snapshot(snapshot_root, font_size=font_size)
    surfaces = analyze_surface_snapshot(snapshot_root, font_size=font_size)
    motifs = audit_motif_catalog(snapshot_root, config=MotifAuditConfig())
    core = audit_review_snapshot(snapshot_root)

    quality_scores = Counter(
        str(record["quality_score"])
        for record in accepted
        if record["quality_score"] is not None
    )
    issue_tags = Counter(
        str(tag) for record in accepted for tag in record["issue_tags"]
    )
    source_counts = Counter(int(record["source_file_id"]) for record in accepted)
    all_categories_in_all_splits = all(
        all(category_splits.get(category, Counter())[split] > 0 for split in ("train", "validation", "test"))
        for category in targets
    )
    category_deficits = {
        category: targets[category] - categories[category]
        for category in targets
        if categories[category] < targets[category]
    }

    return {
        "schema_version": 1,
        "snapshot": json.loads(
            (snapshot_root / "snapshot.json").read_text(encoding="utf-8")
        )["snapshot"],
        "target_accepted_count": target_count,
        "accepted_count": len(accepted),
        "review_count": len(records),
        "source_mlt_count": len(source_counts),
        "maximum_accepted_per_source_mlt": max(source_counts.values(), default=0),
        "sensitive_count": sum(bool(record["sensitive"]) for record in accepted),
        "core_integrity": core,
        "categories": {
            "counts": dict(sorted(categories.items())),
            "targets": targets,
            "deficits": category_deficits,
            "split_counts": {
                category: {
                    split: category_splits.get(category, Counter())[split]
                    for split in ("train", "validation", "test")
                }
                for category in targets
            },
            "all_categories_in_all_splits": all_categories_in_all_splits,
        },
        "dimensions": {
            "line_count": _distribution([int(record["line_count"]) for record in accepted]),
            "max_columns": _distribution([int(record["max_columns"]) for record in accepted]),
            "character_count": _distribution(
                [int(record["character_count"]) for record in accepted]
            ),
        },
        "quality": {
            "score_counts": dict(sorted(quality_scores.items())),
            "missing_score_count": sum(
                record["quality_score"] is None for record in accepted
            ),
            "issue_tag_counts": dict(sorted(issue_tags.items())),
        },
        "vocabulary": {
            "non_whitespace_occurrence_count": occurrence_count,
            "unique_character_count": len(character_counts),
            "singleton_character_count": sum(
                count == 1 for count in character_counts.values()
            ),
            "characters_at_or_above_frequency": {
                str(threshold): sum(
                    count >= threshold for count in character_counts.values()
                )
                for threshold in thresholds
            },
            "occurrence_coverage_at_or_above_frequency": {
                str(threshold): round(
                    sum(
                        count
                        for count in character_counts.values()
                        if count >= threshold
                    )
                    / max(1, occurrence_count),
                    8,
                )
                for threshold in thresholds
            },
            "deepaa_charset_size": len(charset),
            "deepaa_charset_occurrence_coverage": round(
                charset_occurrences / max(1, occurrence_count), 8
            ),
            "top_characters_outside_deepaa_charset": [
                {"codepoint": f"U+{ord(character):04X}", "count": count}
                for character, count in character_counts.most_common()
                if character not in charset
            ][:20],
        },
        "saitamaar_advances": {
            "font_size_px": font_size,
            "distinct_advance_count": len(advance_occurrences),
            "occurrence_counts_by_advance_px": dict(
                sorted(advance_occurrences.items(), key=lambda item: float(item[0]))
            ),
        },
        "fill": {
            "entries_with_fill_runs": int(fill["entries_with_fill"]),
            "entries_without_fill_runs": int(fill["entries_without_fill"]),
            "fill_entry_fraction": round(
                int(fill["entries_with_fill"]) / max(1, len(accepted)), 8
            ),
            "fill_run_count": int(surfaces["fill_run_count"]),
            "entries_with_multi_line_surfaces": int(
                surfaces["entries_with_multi_line_surfaces"]
            ),
            "multi_line_surface_entry_fraction": round(
                int(surfaces["entries_with_multi_line_surfaces"])
                / max(1, len(accepted)),
                8,
            ),
            "fill_surface_count": int(surfaces["fill_surface_count"]),
            "multi_line_surface_count": int(surfaces["multi_line_surface_count"]),
            "connected_run_fraction": float(surfaces["connected_run_fraction"]),
        },
        "motifs": {
            "candidate_count": int(motifs["candidate_motif_count"]),
            "supported_count": int(motifs["supported_motif_count"]),
            "catalog_sha256": str(motifs["catalog_sha256"]),
            "supported": [
                {
                    key: motif[key]
                    for key in (
                        "motif",
                        "natural_advance_px",
                        "ink_density",
                        "run_count",
                        "work_count",
                    )
                }
                for motif in motifs["motifs"]
                if motif["supported"]
            ],
        },
        "training_design_ready": bool(
            len(accepted) >= target_count
            and core["passed"]
            and all_categories_in_all_splits
        ),
        "corpus_saturation_claimed": False,
    }


def _comparison(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    return {
        "baseline_snapshot": baseline["snapshot"],
        "accepted_count_delta": int(current["accepted_count"])
        - int(baseline["accepted_count"]),
        "source_mlt_count_delta": int(current["source_mlt_count"])
        - int(baseline["source_mlt_count"]),
        "unique_character_count_delta": int(current["vocabulary"]["unique_character_count"])
        - int(baseline["vocabulary"]["unique_character_count"]),
        "fill_entry_fraction_delta": round(
            float(current["fill"]["fill_entry_fraction"])
            - float(baseline["fill"]["fill_entry_fraction"]),
            8,
        ),
        "multi_line_surface_entry_fraction_delta": round(
            float(current["fill"]["multi_line_surface_entry_fraction"])
            - float(baseline["fill"]["multi_line_surface_entry_fraction"]),
            8,
        ),
        "supported_motif_count_delta": int(current["motifs"]["supported_count"])
        - int(baseline["motifs"]["supported_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit accepted-AA checkpoint coverage and training readiness."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--target-accepted", type=int, default=500)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        default=[],
        help="Optional snapshot to compare; may be specified more than once.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_checkpoint(
        args.snapshot,
        target_count=args.target_accepted,
        charset_path=args.charset,
        font_size=args.font_size,
    )
    report["comparisons"] = [
        _comparison(
            report,
            audit_checkpoint(
                baseline,
                target_count=int(
                    json.loads((baseline / "snapshot.json").read_text(encoding="utf-8"))[
                        "accepted_count"
                    ]
                ),
                charset_path=args.charset,
                font_size=args.font_size,
            ),
        )
        for baseline in args.baseline
    ]
    output = args.output or args.snapshot / "checkpoint-audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "snapshot": report["snapshot"],
        "accepted_count": report["accepted_count"],
        "source_mlt_count": report["source_mlt_count"],
        "category_deficits": report["categories"]["deficits"],
        "all_categories_in_all_splits": report["categories"][
            "all_categories_in_all_splits"
        ],
        "dimensions": report["dimensions"],
        "vocabulary": report["vocabulary"],
        "fill": report["fill"],
        "motif_supported_count": report["motifs"]["supported_count"],
        "training_design_ready": report["training_design_ready"],
        "comparisons": report["comparisons"],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"audit={output.resolve()}")


if __name__ == "__main__":
    main()
