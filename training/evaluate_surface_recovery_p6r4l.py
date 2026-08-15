from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.contracts import ConversionOptions
from backend.surface_recovery import generate_surface_recovery_comparison


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_ROOT = ROOT / "datasets" / "incoming" / "draft-preferences" / "v2"


def evaluate_failed_v2_records(v2_root: Path = DEFAULT_V2_ROOT) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for record_path in sorted(v2_root.glob("*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("recipe_version") != "surface-fill-v2":
            continue
        if not record.get("none_usable"):
            continue
        record_dir = record_path.parent
        source = record["source"]
        source_artifact = source["artifact"]
        source_payload = (record_dir / source_artifact["path"]).read_bytes()
        comparison = generate_surface_recovery_comparison(
            source_payload,
            ConversionOptions(**record["options"]),
        )
        candidates = []
        for candidate in sorted(
            comparison["candidates"], key=lambda item: item["variantId"]
        ):
            surface = candidate["surface"]
            recovery = surface["line_recovery"]
            fill = surface["fill_summary"]
            candidates.append(
                {
                    "variant_id": candidate["variantId"],
                    "mechanical_gate": surface["mechanical_gate"],
                    "current_line_retention": recovery["current_line_retention"],
                    "source_supported_added_fraction": recovery[
                        "source_supported_added_fraction"
                    ],
                    "line_pixel_multiplier": recovery["line_pixel_multiplier"],
                    "added_line_pixels": recovery["added_line_pixels"],
                    "rescued_surface_count": surface["border_surface_recovery"][
                        "rescued_surface_count"
                    ],
                    "replaced_source_cell_count": fill[
                        "replaced_source_cell_count"
                    ],
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
                "record_id": record["record_id"],
                "filename": source["filename"],
                "profile": record["options"]["profile"],
                "candidates": candidates,
            }
        )

    return {
        "phase": "P6R-4L",
        "source_recipe": "surface-fill-v2",
        "failed_v2_record_count": len(cases),
        "mechanical_gate_passed": bool(cases)
        and all(
            candidate["mechanical_gate"]
            for case in cases
            for candidate in case["candidates"]
        ),
        "human_gate_status": "not evaluated",
        "human_acceptance": "a recovery candidate is selected on at least 3 of 5 cases",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_failed_v2_records(args.v2_root)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
