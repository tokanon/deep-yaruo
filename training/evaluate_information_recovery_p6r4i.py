from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from backend.contracts import ConversionOptions
from backend.information_recovery import generate_information_recovery_comparison
from backend.surface_fill_preferences import SurfaceFillPreferenceStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_ROOT = ROOT / "datasets" / "incoming" / "draft-preferences" / "v3"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "information-recovery-p6r4i"


def _write_data_url(path: Path, data_url: str) -> None:
    payload = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write comparison image: {path}")


def evaluate(
    v3_root: Path = DEFAULT_V3_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    store = SurfaceFillPreferenceStore(
        v3_root,
        comparison_kind="surface-recovery-v3",
        schema_version=3,
        require_common_processed=False,
    )
    cases = []
    for record in sorted(store.list(), key=lambda item: str(item["source"]["filename"])):
        record_dir = v3_root / str(record["record_id"])
        source_payload = (
            record_dir / record["source"]["artifact"]["path"]
        ).read_bytes()
        comparison = generate_information_recovery_comparison(
            source_payload,
            ConversionOptions(**record["options"]),
            store,
        )
        candidates = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for candidate in sorted(
            comparison["candidates"], key=lambda item: item["variantId"]
        ):
            surface = candidate["surface"]
            recovery = surface["recovery"]
            _write_data_url(
                output_dir
                / f"{Path(str(record['source']['filename'])).stem}-{candidate['variantId']}.png",
                candidate["result"]["renderedPng"],
            )
            candidates.append(
                {
                    "variant_id": candidate["variantId"],
                    "mechanical_safety_gate": surface["mechanical_safety_gate"],
                    "recovery_gate": surface["recovery_gate"],
                    "blank_information_reduction_fraction": surface[
                        "blank_information_reduction_fraction"
                    ],
                    "recovered_blank_information_cells": surface[
                        "recovered_blank_information_cells"
                    ],
                    "baseline_blank_information_cells": surface["audit"][
                        "blank_informative_cells"
                    ],
                    "replacement_count": recovery["replacement_count"],
                    "source_supported_changed_cells": recovery[
                        "source_supported_changed_cells"
                    ],
                    "missing_cell_budget": recovery["missing_cell_budget"],
                    "main_line_loss_at_1px": surface["line_diagnostics"][
                        "main_line_loss_at_1px"
                    ],
                    "exact_row_widths": surface["width_diagnostics"][
                        "exact_row_widths"
                    ],
                }
            )
        cases.append(
            {
                "filename": record["source"]["filename"],
                "source_v3_record_id": record["record_id"],
                "source_v3_variant_id": record["selected_variant"],
                "candidates": candidates,
            }
        )
    recovery_ids = {
        "information-surfaces-v4",
        "information-placement-v4",
        "information-combined-v4",
    }
    missing_total = sum(
        int(case["candidates"][0]["baseline_blank_information_cells"])
        for case in cases
    )
    recovered_by_variant = {
        variant_id: sum(
            int(candidate["recovered_blank_information_cells"])
            for case in cases
            for candidate in case["candidates"]
            if candidate["variant_id"] == variant_id
        )
        for variant_id in recovery_ids
    }
    aggregate_reduction = {
        variant_id: round(recovered / max(1, missing_total), 6)
        for variant_id, recovered in sorted(recovered_by_variant.items())
    }
    report = {
        "phase": "P6R-4I-recovery",
        "case_count": len(cases),
        "all_candidates_safe": bool(cases)
        and all(
            candidate["mechanical_safety_gate"]
            for case in cases
            for candidate in case["candidates"]
        ),
        "each_case_has_recovery_gate_candidate": bool(cases)
        and all(
            any(
                candidate["variant_id"] in recovery_ids
                and candidate["recovery_gate"]
                for candidate in case["candidates"]
            )
            for case in cases
        ),
        "aggregate_blank_information_reduction_fraction": aggregate_reduction,
        "aggregate_recovery_gate_passed": bool(cases)
        and any(value >= 0.25 for value in aggregate_reduction.values()),
        "human_gate_status": "not evaluated",
        "human_gate": "a recovery candidate is selected on at least 3 of 5 cases",
        "cases": cases,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.v3_root, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
