from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.contracts import ConversionOptions
from backend.correction_pairs import decode_png_data_url
from backend.cumulative_generator import (
    RECIPE_VERSION,
    _prepare_surface_model_inputs,
    _render_surface_model_text,
    generate_cumulative,
)
from backend.deepaa_surface import decode_surface_text
from backend.rendering import REFERENCE_FONT_SIZE, glyph_advance
from backend.surface_fill_preferences import (
    DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
    SurfaceFillPreferenceStore,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / ".tmp" / "training-runs" / "deepaa-surface-real-v0" / "report.json"
)


def _source_bytes(root: Path, record: dict[str, object]) -> bytes:
    source = record.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("artifact"), dict):
        raise ValueError("Surface-fill record has no source artifact")
    artifact = source["artifact"]
    return (root / str(record["record_id"]) / str(artifact["path"])).read_bytes()


def _line_widths(text: str) -> list[int]:
    return [
        sum(glyph_advance(character, REFERENCE_FONT_SIZE) for character in line)
        for line in text.rstrip("\n").split("\n")
    ]


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
        raise ValueError(f"DeepAA surface real gate requires 15 records; found {len(records)}")

    cases: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        filename = str(record["source"]["filename"])
        print(f"[{index}/{len(records)}] {filename}", flush=True)
        data = _source_bytes(preference_root, record)
        options = ConversionOptions(**record["options"]).normalized()
        first = generate_cumulative(data, options)
        first_rendered = decode_png_data_url(first.result.rendered_png)
        deterministic: bool | None = None
        if deterministic_repeat:
            second = generate_cumulative(data, options)
            deterministic = bool(
                second.result.ascii_text == first.result.ascii_text
                and decode_png_data_url(second.result.rendered_png) == first_rendered
                and second.pipeline == first.pipeline
            )
        widths = _line_widths(first.result.ascii_text)
        target_width = options.columns * 8
        fallback_used = bool(first.pipeline.get("fallback_used", False))
        if not fallback_used:
            decoder_stage = first.pipeline["stages"][-1]
            nonempty_whitespace_rows = list(decoder_stage["all_whitespace_rows"])
            exact_width = bool(
                decoder_stage["exact_row_widths"]
                and all(width == target_width for width in widths)
            )
            uses_target_start_count = bool(decoder_stage["uses_target_start_count"])
        else:
            nonempty_whitespace_rows = []
            fallback_pipeline = first.pipeline["fallback_pipeline"]
            stages = fallback_pipeline.get("stages", [])
            exact_width = bool(stages and stages[-1]["width_diagnostics"]["exact_row_widths"])
            uses_target_start_count = False
        cases.append(
            {
                "filename": filename,
                "source_record_id": record["record_id"],
                "profile": options.profile,
                "rows": first.result.rows,
                "columns": first.result.columns,
                "line_widths_px": widths,
                "target_width_px": target_width,
                "exact_width": exact_width,
                "nonempty_input_rows_rendered_whitespace": nonempty_whitespace_rows,
                "uses_target_start_count": uses_target_start_count,
                "fallback_used": fallback_used,
                "fallback_reason": first.pipeline.get("fallback_reason"),
                "text_sha256": hashlib.sha256(
                    first.result.ascii_text.encode("utf-8")
                ).hexdigest(),
                "rendered_sha256": hashlib.sha256(first_rendered).hexdigest(),
                "deterministic_checked": deterministic_repeat,
                "deterministic": deterministic,
            }
        )

    return {
        "schema_version": 1,
        "recipe_version": RECIPE_VERSION,
        "case_count": len(cases),
        "deterministic_repeat": deterministic_repeat,
        "fallback_count": sum(bool(case["fallback_used"]) for case in cases),
        "fallback_reasons": sorted(
            {
                str(case["fallback_reason"])
                for case in cases
                if case["fallback_reason"] is not None
            }
        ),
        "exact_width_count": sum(bool(case["exact_width"]) for case in cases),
        "deterministic_count": sum(case["deterministic"] is True for case in cases),
        "nonempty_whitespace_failure_count": sum(
            bool(case["nonempty_input_rows_rendered_whitespace"]) for case in cases
        ),
        "uses_target_start_count": any(
            bool(case["uses_target_start_count"]) for case in cases
        ),
        "mechanical_gate_passed": bool(
            cases
            and all(
                case["exact_width"]
                and not case["nonempty_input_rows_rendered_whitespace"]
                and not case["uses_target_start_count"]
                and (not deterministic_repeat or case["deterministic"])
                for case in cases
            )
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preference-root", type=Path, default=DEFAULT_SURFACE_FILL_PREFERENCE_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--no-repeat", action="store_true")
    parser.add_argument("--inspect-candidate", action="store_true")
    args = parser.parse_args()
    if args.inspect_candidate:
        records = [
            record
            for record in SurfaceFillPreferenceStore(args.preference_root).list()
            if record.get("recipe_version") == "surface-fill-v2"
            and (
                not args.case
                or str(record["source"]["filename"]) in set(args.case)
            )
        ]
        for record in records:
            data = _source_bytes(args.preference_root, record)
            options = ConversionOptions(**record["options"]).normalized()
            processed, line, surface, _crop, _metadata = _prepare_surface_model_inputs(
                data, options
            )
            decoded = decode_surface_text(line, surface)
            rendered = _render_surface_model_text(
                decoded.text,
                width=decoded.target_width,
                height=processed.shape[0],
            )
            print(
                json.dumps(
                    {
                        "filename": record["source"]["filename"],
                        "rows": decoded.rows,
                        "all_whitespace_rows": decoded.all_whitespace_rows,
                        "start_counts": decoded.start_counts,
                        "rendered_shape": rendered.shape,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return
    report = evaluate(
        args.preference_root,
        case_names=set(args.case) or None,
        deterministic_repeat=not args.no_repeat,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
