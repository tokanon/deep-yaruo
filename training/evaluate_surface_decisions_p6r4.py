from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend.correction_pairs import CorrectionPairStore, DEFAULT_CORRECTION_ROOT
from backend.surface_decisions import build_correction_pair_surface_analysis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "surface-decisions-p6r4" / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P6R-4 surface decisions on saved correction pairs."
    )
    parser.add_argument("--correction-root", type=Path, default=DEFAULT_CORRECTION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-extensions",
        action="store_true",
        help="Persist proposals, correspondence, and metrics into each record.",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "minimum": round(float(array.min()), 6),
        "maximum": round(float(array.max()), 6),
    }


def evaluate_correction_pairs(
    store: CorrectionPairStore,
    *,
    write_extensions: bool = False,
) -> dict[str, object]:
    records = store.list()
    entries: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    source_kind_counts: Counter[str] = Counter()
    layer_f1: dict[str, list[float]] = {}
    layer_pixel_iou: dict[str, list[float]] = {}
    for record_summary in records:
        record_id = str(record_summary["record_id"])
        prepared = build_correction_pair_surface_analysis(store, record_id)
        updates = prepared["updates"]
        if write_extensions:
            store.update_extensions(record_id, updates)
        category = str(prepared["subject_category"])
        source_kind = str(prepared["source_kind"])
        category_counts[category] += 1
        source_kind_counts[source_kind] += 1
        layer_entries: list[dict[str, object]] = []
        for layer in updates["surface_metrics"]["layers"]:
            layer_id = str(layer["layer_id"])
            evaluation = layer["evaluation"]
            f1 = float(evaluation["surface_metrics"]["f1"])
            pixel_iou = float(evaluation["pixel_metrics"]["iou"])
            layer_f1.setdefault(layer_id, []).append(f1)
            layer_pixel_iou.setdefault(layer_id, []).append(pixel_iou)
            layer_entries.append(
                {
                    "layer_id": layer_id,
                    "rule_status": layer["rule_status"],
                    "action_counts": layer["action_counts"],
                    "surface_metrics": evaluation["surface_metrics"],
                    "pixel_metrics": evaluation["pixel_metrics"],
                    "topology": evaluation["topology"],
                }
            )
        entries.append(
            {
                "record_id": record_id,
                "subject_category": category,
                "category_source": "conversion-profile",
                "profile": prepared["profile"],
                "source_kind": source_kind,
                "proposal_count": updates["surface_proposals"]["proposal_set"][
                    "proposal_count"
                ],
                "corrected_fill_run_count": updates["surface_correspondence"][
                    "corrected_fill_run_count"
                ],
                "corrected_fill_surface_count": updates["surface_correspondence"][
                    "corrected_fill_surface_count"
                ],
                "layers": layer_entries,
            }
        )

    person_count = category_counts["person"]
    background_count = category_counts["background"]
    missing_conditions: list[str] = []
    if person_count < 10:
        missing_conditions.append(f"person correction pairs: {person_count}/10")
    if background_count < 10:
        missing_conditions.append(f"background correction pairs: {background_count}/10")
    missing_conditions.extend(
        [
            "C1 comparison on the same correction pairs is not implemented",
            "line preservation after constrained AA placement is not implemented",
            "paired human evaluation is not collected",
        ]
    )
    if not entries:
        status = "blocked-no-correction-pairs"
    elif person_count < 10 or background_count < 10:
        status = "blocked-insufficient-person-background-strata"
    else:
        status = "foundation-ready-quality-gate-not-evaluated"
    return {
        "schema_version": 1,
        "phase": "P6R-4",
        "record_count": len(entries),
        "write_extensions": write_extensions,
        "status": status,
        "quality_gate_passed": None,
        "quality_gate_status": "not evaluated",
        "required_gate": {
            "minimum_person_pairs": 10,
            "minimum_background_pairs": 10,
            "comparison": "same-pair C1 baseline",
            "metrics": [
                "overfill",
                "missing",
                "surface line loss after AA placement",
                "paired human evaluation",
            ],
        },
        "missing_conditions": missing_conditions,
        "aggregate": {
            "subject_category_counts": dict(sorted(category_counts.items())),
            "source_kind_counts": dict(sorted(source_kind_counts.items())),
            "records_with_corrected_fill_surfaces": sum(
                int(entry["corrected_fill_surface_count"]) > 0 for entry in entries
            ),
            "layer_surface_f1": {
                layer_id: _summary(values)
                for layer_id, values in sorted(layer_f1.items())
            },
            "layer_pixel_iou": {
                layer_id: _summary(values)
                for layer_id, values in sorted(layer_pixel_iou.items())
            },
        },
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    report = evaluate_correction_pairs(
        CorrectionPairStore(args.correction_root),
        write_extensions=args.write_extensions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "record_count": report["record_count"],
                "status": report["status"],
                "quality_gate_passed": report["quality_gate_passed"],
                "missing_conditions": report["missing_conditions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
