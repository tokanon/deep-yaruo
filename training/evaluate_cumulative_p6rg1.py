from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.contracts import ConversionOptions
from backend.cumulative_generator import P6RG1_RECIPE_VERSION, generate_p6rg1
from backend.correction_pairs import decode_png_data_url
from backend.surface_fill_preferences import (
    DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
    SurfaceFillPreferenceStore,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / ".tmp" / "training-runs" / "cumulative-p6rg1" / "report.json"
)


def _source_bytes(
    root: Path,
    record: dict[str, object],
) -> bytes:
    source = record.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("artifact"), dict):
        raise ValueError("Surface-fill record has no source artifact")
    artifact = source["artifact"]
    return (
        root / str(record["record_id"]) / str(artifact["path"])
    ).read_bytes()


def evaluate(
    preference_root: Path = DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
    *,
    case_names: set[str] | None = None,
    deterministic_repeat: bool = True,
) -> dict[str, object]:
    store = SurfaceFillPreferenceStore(preference_root)
    records = [
        record
        for record in store.list()
        if record.get("recipe_version") == "surface-fill-v2"
    ]
    if case_names is not None:
        records = [
            record
            for record in records
            if str(record["source"]["filename"]) in case_names
        ]
    records.sort(key=lambda record: str(record["source"]["filename"]))
    if case_names is None and len(records) != 15:
        raise ValueError(f"P6R-G1 requires 15 stress records; found {len(records)}")

    cases = []
    for index, record in enumerate(records, start=1):
        filename = str(record["source"]["filename"])
        print(f"[{index}/{len(records)}] {filename}", flush=True)
        data = _source_bytes(preference_root, record)
        options = ConversionOptions(**record["options"]).normalized()
        first = generate_p6rg1(data, options)
        first_text_hash = hashlib.sha256(
            first.result.ascii_text.encode("utf-8")
        ).hexdigest()
        first_rendered_hash = hashlib.sha256(
            decode_png_data_url(first.result.rendered_png)
        ).hexdigest()
        deterministic: bool | None = None
        if deterministic_repeat:
            second = generate_p6rg1(data, options)
            deterministic = bool(
                second.result.ascii_text == first.result.ascii_text
                and decode_png_data_url(second.result.rendered_png)
                == decode_png_data_url(first.result.rendered_png)
            )
        stages = first.pipeline["stages"]
        surface_stage, information_stage, texture_stage = stages
        cases.append(
            {
                "filename": filename,
                "source_record_id": record["record_id"],
                "profile": options.profile,
                "rows": first.result.rows,
                "columns": first.result.columns,
                "text_sha256": first_text_hash,
                "rendered_sha256": first_rendered_hash,
                "deterministic_checked": deterministic_repeat,
                "deterministic": deterministic,
                "surface_selected_variant": surface_stage["selected_variant"],
                "surface_fallback_reason": surface_stage["fallback_reason"],
                "information_applied": information_stage["applied"],
                "information_fallback_reason": information_stage["fallback_reason"],
                "information_blank_reduction_fraction": information_stage[
                    "blank_information_reduction_fraction"
                ],
                "texture_applied": texture_stage["applied"],
                "texture_fallback_reason": texture_stage["fallback_reason"],
                "texture_changed_character_fraction": texture_stage[
                    "changed_character_fraction"
                ],
                "maximum_texture_density_error": texture_stage["texture"][
                    "maximum_density_error"
                ],
                "maximum_texture_line_loss": texture_stage["line_diagnostics"][
                    "main_line_loss_at_1px"
                ],
                "exact_texture_row_widths": texture_stage["width_diagnostics"][
                    "exact_row_widths"
                ],
                "pipeline_safe": bool(
                    surface_stage["selected_variant"] != "deepaa-beam-v0.2"
                    and information_stage.get("output_safe", True)
                    and texture_stage.get("output_safe", True)
                    and texture_stage["width_diagnostics"]["exact_row_widths"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "recipe_version": P6RG1_RECIPE_VERSION,
        "case_count": len(cases),
        "deterministic_repeat": deterministic_repeat,
        "surface_variant_counts": {
            variant: sum(case["surface_selected_variant"] == variant for case in cases)
            for variant in sorted({case["surface_selected_variant"] for case in cases})
        },
        "information_applied_count": sum(case["information_applied"] for case in cases),
        "texture_applied_count": sum(case["texture_applied"] for case in cases),
        "minimum_texture_changed_character_fraction": min(
            (case["texture_changed_character_fraction"] for case in cases),
            default=0.0,
        ),
        "maximum_texture_density_error": max(
            (case["maximum_texture_density_error"] for case in cases),
            default=0.0,
        ),
        "maximum_texture_line_loss": max(
            (case["maximum_texture_line_loss"] for case in cases),
            default=0.0,
        ),
        "mechanical_gate_passed": bool(
            cases
            and all(
                case["pipeline_safe"]
                and (not deterministic_repeat or case["deterministic"])
                for case in cases
            )
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference-root", type=Path, default=DEFAULT_SURFACE_FILL_PREFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--no-repeat", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        args.preference_root,
        case_names=set(args.case) or None,
        deterministic_repeat=not args.no_repeat,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
