from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import ImageFont

from backend.rendering import find_font, line_pitch
from evaluation.surface_metrics import (
    SurfaceSet,
    UnpairedSurfaceEvaluationError,
    evaluate_surface_sets,
)

from .aa_fill_layers import (
    connect_fill_runs,
    detect_fill_runs,
    render_fill_surface_labels,
)
from .review_corpus import load_snapshot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_OUTPUT = (
    ROOT / ".tmp" / "training-runs" / "surface-metrics-p6r2" / "report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P6R-2 surface-metric and pairing-contract checks."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-size", type=int, default=16)
    return parser.parse_args()


def _surface_set(
    labels: np.ndarray,
    *,
    case_id: str,
    correspondence_id: str | None = None,
    line_mask: np.ndarray | None = None,
) -> SurfaceSet:
    return SurfaceSet(
        labels=labels,
        coordinate_space_id=f"{case_id}:{labels.shape[1]}x{labels.shape[0]}",
        correspondence_id=correspondence_id,
        line_mask=line_mask,
    )


def _synthetic_cases() -> dict[str, dict[str, object]]:
    exact = np.zeros((12, 12), dtype=np.int32)
    exact[1:5, 1:5] = 1
    exact[7:10, 6:11] = 2

    overfill_reference = np.zeros((12, 12), dtype=np.int32)
    overfill_reference[1:4, 1:4] = 1
    overfill_predicted = overfill_reference.copy()
    overfill_predicted[7:10, 7:10] = 2

    missing_reference = overfill_predicted.copy()
    missing_predicted = overfill_reference.copy()

    split_reference = np.zeros((10, 12), dtype=np.int32)
    split_reference[2:8, 2:10] = 1
    split_predicted = np.zeros_like(split_reference)
    split_predicted[2:8, 2:6] = 1
    split_predicted[2:8, 6:10] = 2

    merge_reference = split_predicted.copy()
    merge_predicted = split_reference.copy()

    definitions = {
        "exact": (exact, exact.copy()),
        "overfill": (overfill_reference, overfill_predicted),
        "missing": (missing_reference, missing_predicted),
        "split": (split_reference, split_predicted),
        "merge": (merge_reference, merge_predicted),
    }
    return {
        case_id: evaluate_surface_sets(
            _surface_set(
                reference,
                case_id=case_id,
                correspondence_id=f"synthetic:{case_id}",
            ),
            _surface_set(
                predicted,
                case_id=case_id,
                correspondence_id=f"synthetic:{case_id}",
            ),
        )
        for case_id, (reference, predicted) in definitions.items()
    }


def evaluate_snapshot_contract(
    snapshot_root: Path,
    *,
    font_size: int = 16,
) -> dict[str, object]:
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record["decision"] == "accept"
    ]
    font = ImageFont.truetype(str(find_font()), font_size)
    pitch = line_pitch(font_size)
    exact_match_count = 0
    unpaired_rejection_count = 0
    surface_count = 0
    entries: list[dict[str, object]] = []

    for record in records:
        entry_id = int(record["entry_id"])
        text = (snapshot_root / str(record["text"])).read_text(encoding="utf-8")
        lines = text.splitlines() or [""]
        width = max(2, int(math.ceil(max(font.getlength(line) for line in lines))) + 2)
        height = max(pitch, len(lines) * pitch)
        surfaces = connect_fill_runs(
            detect_fill_runs(text, font_size=font_size),
            font_size=font_size,
        )
        labels = render_fill_surface_labels(surfaces, width=width, height=height)
        surface_count += len(surfaces)
        case_id = f"accepted-v1:{entry_id}"
        correspondence_id = f"synthetic-self:{entry_id}"
        paired = _surface_set(
            labels,
            case_id=case_id,
            correspondence_id=correspondence_id,
        )
        exact_report = evaluate_surface_sets(paired, paired)
        surface_metrics = exact_report["surface_metrics"]
        pixel_metrics = exact_report["pixel_metrics"]
        is_exact = (
            surface_metrics["precision"] == 1.0
            and surface_metrics["recall"] == 1.0
            and pixel_metrics["iou"] == 1.0
            and exact_report["topology"]["split_count"] == 0
            and exact_report["topology"]["merge_count"] == 0
        )
        exact_match_count += int(is_exact)

        unpaired = _surface_set(labels, case_id=case_id)
        try:
            evaluate_surface_sets(unpaired, unpaired)
        except UnpairedSurfaceEvaluationError:
            unpaired_rejected = True
            unpaired_rejection_count += 1
        else:
            unpaired_rejected = False
        entries.append(
            {
                "entry_id": entry_id,
                "surface_count": len(surfaces),
                "exact_self_match": is_exact,
                "unpaired_metrics_rejected": unpaired_rejected,
            }
        )

    synthetic = _synthetic_cases()
    return {
        "schema_version": 1,
        "phase": "P6R-2",
        "metric_scope": (
            "Correctness metrics are valid only for explicitly paired surface "
            "sets with identical correspondence and coordinate-space IDs."
        ),
        "matching_method": "greedy-descending-IoU one-to-one",
        "accepted_v1_self_check": {
            "entry_count": len(entries),
            "surface_count": surface_count,
            "exact_match_count": exact_match_count,
            "unpaired_rejection_count": unpaired_rejection_count,
            "entries": sorted(entries, key=lambda entry: int(entry["entry_id"])),
        },
        "synthetic_cases": {
            case_id: {
                "surface_metrics": report["surface_metrics"],
                "topology": report["topology"],
                "pixel_metrics": report["pixel_metrics"],
            }
            for case_id, report in synthetic.items()
        },
    }


def main() -> None:
    args = parse_args()
    report = evaluate_snapshot_contract(args.snapshot, font_size=args.font_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["accepted_v1_self_check"]
    print(
        json.dumps(
            {
                "entry_count": summary["entry_count"],
                "surface_count": summary["surface_count"],
                "exact_match_count": summary["exact_match_count"],
                "unpaired_rejection_count": summary["unpaired_rejection_count"],
                "synthetic_cases": {
                    case_id: {
                        "surface_metrics": values["surface_metrics"],
                        "split_count": values["topology"]["split_count"],
                        "merge_count": values["topology"]["merge_count"],
                    }
                    for case_id, values in report["synthetic_cases"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
