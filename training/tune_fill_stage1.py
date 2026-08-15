from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.contracts import ConversionOptions
from backend.deepaa import preprocess_for_deepaa
from backend.fill_postprocess import FillStage1Config, apply_fill_stage1
from backend.image_io import decode_image
from backend.input_channels import ChannelExtractionConfig, extract_input_channels
from training.evaluate_fill_stage1 import (
    DEFAULT_OUTPUT,
    _canonical_render,
    _replacement_diagnostic,
    _save_review_sheet,
    fill_stage_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep conservative P4 stage-1 boundary/structure thresholds."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _channel_from_preview(path: Path) -> np.ndarray:
    preview = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if preview is None:
        raise FileNotFoundError(path)
    return (255.0 - preview.astype(np.float32)) / 255.0


def _candidate_configs() -> list[FillStage1Config]:
    return [
        FillStage1Config(
            fill_fraction_minimum=fill_minimum,
            safe_fill_fraction_minimum=safe_minimum,
            structure_fraction_maximum=structure_maximum,
            boundary_margin_px=margin,
        )
        for fill_minimum, safe_minimum, structure_maximum, margin in itertools.product(
            (0.85, 0.95),
            (0.65, 0.85),
            (0.02, 0.04, 0.08, 0.16),
            (2, 3),
        )
    ]


def tune_fill_stage1(input_dir: Path) -> dict[str, object]:
    report_path = input_dir / "report.json"
    baseline_report = json.loads(report_path.read_text(encoding="utf-8"))
    empirical_counts = {
        str(character): int(count)
        for character, count in baseline_report["empirical_fill_run_counts"].items()
    }
    cases: list[dict[str, object]] = []
    for record in baseline_report["cases"]:
        case_id = str(record["id"])
        baseline_text = (input_dir / f"{case_id}-baseline.txt").read_text(
            encoding="utf-8"
        )
        baseline = cv2.imread(
            str(input_dir / f"{case_id}-baseline.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if baseline is None:
            raise FileNotFoundError(input_dir / f"{case_id}-baseline.png")
        cases.append(
            {
                "record": record,
                "id": case_id,
                "text": baseline_text,
                "baseline": baseline,
                "structure": _channel_from_preview(input_dir / f"{case_id}-ch0.png"),
                "tone": _channel_from_preview(input_dir / f"{case_id}-ch1.png"),
                "fill": _channel_from_preview(input_dir / f"{case_id}-ch2.png"),
            }
        )

    trials: list[dict[str, object]] = []
    for trial_index, config in enumerate(_candidate_configs()):
        trial_cases: list[dict[str, object]] = []
        source_line_losses: list[float] = []
        source_coverage_gains: list[float] = []
        outside_fractions: list[float] = []
        lineart_replacements = 0
        total_replacement_width = 0
        for case in cases:
            baseline = case["baseline"]
            result = apply_fill_stage1(
                case["text"],
                structure=case["structure"],
                tone=case["tone"],
                fill=case["fill"],
                empirical_run_counts=empirical_counts,
                canvas_width=baseline.shape[1],
                config=config,
            )
            rendered = _canonical_render(result.text, baseline.shape[1], baseline.shape[0])
            channels = type(
                "SweepChannels",
                (),
                {
                    "structure": case["structure"],
                    "tone": case["tone"],
                    "fill": case["fill"],
                },
            )()
            metrics = fill_stage_metrics(baseline, rendered, channels)
            kind = str(case["record"]["kind"])
            if "lineart" in kind:
                lineart_replacements += len(result.replacements)
            else:
                source_line_losses.append(float(metrics["main_line_loss_at_1px"]))
                source_coverage_gains.append(
                    float(metrics["ch2_ink_coverage_after"])
                    - float(metrics["ch2_ink_coverage_before"])
                )
                outside_fractions.append(
                    float(metrics["outside_ch2_added_ink_fraction"])
                )
            total_replacement_width += result.replaced_pixel_width
            trial_cases.append(
                {
                    "id": case["id"],
                    "replacement_count": len(result.replacements),
                    "replaced_source_cell_count": result.replaced_source_cell_count,
                    "replaced_pixel_width": result.replaced_pixel_width,
                    "metrics": metrics,
                }
            )
        maximum_line_loss = max(source_line_losses, default=0.0)
        maximum_outside = max(outside_fractions, default=0.0)
        mean_coverage_gain = float(np.mean(source_coverage_gains))
        passes_machine_gate = (
            maximum_line_loss <= 0.02
            and maximum_outside <= 0.01
            and lineart_replacements == 0
            and total_replacement_width > 0
        )
        trials.append(
            {
                "trial": trial_index,
                "config": asdict(config),
                "passes_machine_gate": passes_machine_gate,
                "aggregate": {
                    "maximum_source_main_line_loss_at_1px": maximum_line_loss,
                    "maximum_source_outside_ch2_added_ink_fraction": maximum_outside,
                    "mean_source_ch2_coverage_gain": round(mean_coverage_gain, 6),
                    "lineart_replacement_count": lineart_replacements,
                    "total_replaced_pixel_width": total_replacement_width,
                },
                "cases": trial_cases,
            }
        )

    passing = [trial for trial in trials if trial["passes_machine_gate"]]
    if passing:
        selected = max(
            passing,
            key=lambda trial: (
                trial["aggregate"]["mean_source_ch2_coverage_gain"],
                trial["aggregate"]["total_replaced_pixel_width"],
            ),
        )
        selection_reason = "maximum ch2 coverage gain among machine-gate passes"
    else:
        nonempty = [
            trial
            for trial in trials
            if trial["aggregate"]["total_replaced_pixel_width"] > 0
        ]
        selected = min(
            nonempty,
            key=lambda trial: (
                trial["aggregate"]["maximum_source_main_line_loss_at_1px"],
                -trial["aggregate"]["mean_source_ch2_coverage_gain"],
            ),
        )
        selection_reason = "no machine-gate pass; minimum source main-line loss"

    selected_config = FillStage1Config(**selected["config"])
    for case in cases:
        baseline = case["baseline"]
        fill_result = apply_fill_stage1(
            case["text"],
            structure=case["structure"],
            tone=case["tone"],
            fill=case["fill"],
            empirical_run_counts=empirical_counts,
            canvas_width=baseline.shape[1],
            config=selected_config,
        )
        filled = _canonical_render(
            fill_result.text,
            baseline.shape[1],
            baseline.shape[0],
        )
        record = case["record"]
        source_data = Path(str(record["source"])).read_bytes()
        source = decode_image(source_data)
        options = ConversionOptions(**record["options"])
        _, crop = preprocess_for_deepaa(source, options)
        left, top, right, bottom = crop
        source_kind = (
            "lineart"
            if options.profile in {"lineart", "background_lineart"}
            else "grayscale"
        )
        channels = extract_input_channels(
            source[top:bottom, left:right],
            source_kind=source_kind,
            config=ChannelExtractionConfig(target_width=baseline.shape[1]),
        )
        diagnostic = _replacement_diagnostic(
            channels,
            fill_result.replacements,
            canvas_width=baseline.shape[1],
            canvas_height=baseline.shape[0],
        )
        case_id = case["id"]
        (input_dir / f"{case_id}-selected.txt").write_text(
            fill_result.text,
            encoding="utf-8",
            newline="\n",
        )
        Image.fromarray(filled).save(
            input_dir / f"{case_id}-selected.png", optimize=True
        )
        Image.fromarray(diagnostic).save(
            input_dir / f"{case_id}-selected-diagnostic.png", optimize=True
        )
        _save_review_sheet(
            input_dir / f"{case_id}-selected-review.png",
            channels=channels,
            baseline=baseline,
            filled=filled,
            diagnostic=diagnostic,
        )

    output = {
        "schema_version": 1,
        "phase": "P4-stage1-threshold-sweep",
        "trial_count": len(trials),
        "machine_gate": {
            "maximum_source_main_line_loss_at_1px": 0.02,
            "maximum_source_outside_ch2_added_ink_fraction": 0.01,
            "lineart_replacement_count": 0,
            "requires_nonzero_fill": True,
            "semantic_review_still_required": True,
        },
        "passing_trial_count": len(passing),
        "selected_trial": selected["trial"],
        "selection_reason": selection_reason,
        "selected": selected,
        "trials": trials,
    }
    (input_dir / "sweep.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def main() -> None:
    args = parse_args()
    report = tune_fill_stage1(args.input)
    print(f"trials={report['trial_count']}")
    print(f"passing={report['passing_trial_count']}")
    print(f"selected={report['selected_trial']}")
    print(json.dumps(report["selected"]["aggregate"], indent=2))
    print(f"report={(args.input / 'sweep.json').resolve()}")


if __name__ == "__main__":
    main()
