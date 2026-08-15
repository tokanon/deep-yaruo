from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from backend.contracts import ConversionOptions
from backend.deepaa import convert_deepaa
from backend.fill_postprocess import FillReplacement, FillStage1Config, apply_fill_stage1
from backend.image_io import decode_image
from backend.input_channels import (
    ChannelExtractionConfig,
    ExtractedChannels,
    channel_preview,
    extract_input_channels,
)
from backend.rendering import line_pitch, render_text_mask
from evaluation.quality import decode_data_url, decode_png, geometric_metrics
from training.analyze_aa_fill_structure import analyze_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "cases.json"
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_OUTPUT = ROOT / "output" / "evaluation-fill-stage1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P4 stage-1 fill overwrite on the fixed real-image cases."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this fixed case ID; repeat to select multiple cases.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _canonical_render(text: str, width: int, height: int) -> np.ndarray:
    mask = render_text_mask(text, canvas_height_per_line=line_pitch(16))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    visible_height = min(height, mask.shape[0])
    visible_width = min(width, mask.shape[1])
    canvas[:visible_height, :visible_width][mask[:visible_height, :visible_width]] = 0
    return canvas


def _aligned_channel(channel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_NEAREST)


def fill_stage_metrics(
    baseline: np.ndarray,
    filled: np.ndarray,
    channels: ExtractedChannels,
) -> dict[str, object]:
    if baseline.shape != filled.shape:
        raise ValueError("P4 baseline and filled renders must share one shape")
    structure = _aligned_channel(channels.structure, baseline.shape) >= 0.5
    fill = _aligned_channel(channels.fill, baseline.shape) >= 0.5
    kernel = np.ones((3, 3), np.uint8)
    fill_u8 = fill.astype(np.uint8)
    boundary = cv2.morphologyEx(fill_u8, cv2.MORPH_GRADIENT, kernel).astype(bool)
    baseline_ink = baseline < 128
    filled_ink = filled < 128
    added = filled_ink & ~baseline_ink
    removed = baseline_ink & ~filled_ink
    changed = baseline_ink ^ filled_ink

    structure_support = cv2.dilate(structure.astype(np.uint8), kernel).astype(bool)
    baseline_structure_ink = baseline_ink & structure_support
    filled_nearby = cv2.dilate(filled_ink.astype(np.uint8), kernel).astype(bool)
    retained_structure = baseline_structure_ink & filled_nearby

    fill_count = int(fill.sum())
    added_count = int(added.sum())
    baseline_structure_count = int(baseline_structure_ink.sum())
    return {
        "added_ink_pixels": added_count,
        "removed_ink_pixels": int(removed.sum()),
        "outside_ch2_added_ink_fraction": round(
            float((added & ~fill).sum()) / max(1, added_count), 6
        ),
        "ch2_ink_coverage_before": round(
            float((baseline_ink & fill).sum()) / max(1, fill_count), 6
        ),
        "ch2_ink_coverage_after": round(
            float((filled_ink & fill).sum()) / max(1, fill_count), 6
        ),
        "ch2_boundary_changed_fraction": round(
            float((changed & boundary).sum()) / max(1, int(boundary.sum())), 6
        ),
        "baseline_structure_ink_pixels": baseline_structure_count,
        "main_line_retention_at_1px": round(
            float(retained_structure.sum()) / max(1, baseline_structure_count), 6
        ),
        "main_line_loss_at_1px": round(
            1.0 - float(retained_structure.sum()) / max(1, baseline_structure_count),
            6,
        ),
        "human_review_required": [
            "false semantic fills",
            "black hair coverage and flow",
            "black eye readability",
            "subject recognizability",
        ],
    }


def _replacement_diagnostic(
    channels: ExtractedChannels,
    replacements: tuple[FillReplacement, ...],
    *,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    gray = cv2.resize(
        channels.resized_gray,
        (canvas_width, canvas_height),
        interpolation=cv2.INTER_AREA,
    )
    structure = _aligned_channel(channels.structure, gray.shape) >= 0.5
    fill = _aligned_channel(channels.fill, gray.shape) >= 0.5
    diagnostic = np.repeat(gray[:, :, None], 3, axis=2)
    diagnostic[fill] = (
        diagnostic[fill].astype(np.float32) * 0.55
        + np.asarray([40, 105, 255], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    diagnostic[structure] = np.asarray([235, 48, 48], dtype=np.uint8)
    image = Image.fromarray(diagnostic)
    draw = ImageDraw.Draw(image)
    pitch = line_pitch(16)
    for replacement in replacements:
        draw.rectangle(
            (
                replacement.x_start,
                replacement.line_index * pitch,
                max(replacement.x_start, replacement.x_end - 1),
                min(canvas_height - 1, (replacement.line_index + 1) * pitch - 1),
            ),
            outline=(0, 225, 70),
            width=2,
        )
    return np.asarray(image)


def _labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    panel = np.full((image.shape[0] + 28, image.shape[1], 3), 255, dtype=np.uint8)
    panel[28:] = image
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return panel


def _save_review_sheet(
    path: Path,
    *,
    channels: ExtractedChannels,
    baseline: np.ndarray,
    filled: np.ndarray,
    diagnostic: np.ndarray,
) -> None:
    source = cv2.resize(
        channels.resized_gray,
        (baseline.shape[1], baseline.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    fill_preview = _aligned_channel(channel_preview(channels.fill), baseline.shape)
    panels = [
        _labeled_panel(source, "CROPPED SOURCE"),
        _labeled_panel(fill_preview, "CH2 FILL MASK"),
        _labeled_panel(baseline, "M BASELINE"),
        _labeled_panel(filled, "P4 STAGE 1"),
        _labeled_panel(diagnostic, "RED=CH0 BLUE=CH2 GREEN=REPLACED"),
    ]
    Image.fromarray(np.vstack(panels)).save(path, optimize=True)


def evaluate_fill_stage1(
    cases_path: Path,
    snapshot_path: Path,
    output_dir: Path,
    *,
    selected_ids: set[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"P4 output is not empty: {output_dir}; use --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    specification = json.loads(cases_path.read_text(encoding="utf-8"))
    fill_analysis = analyze_snapshot(snapshot_path)
    empirical_counts = {
        str(character): int(count)
        for character, count in fill_analysis["fill_character_counts"].items()
    }
    config = FillStage1Config()
    records: list[dict[str, object]] = []
    found: set[str] = set()

    for case in specification["cases"]:
        case_id = str(case["id"])
        if selected_ids is not None and case_id not in selected_ids:
            continue
        found.add(case_id)
        source_path = (cases_path.parent / str(case["source"])).resolve()
        source_data = source_path.read_bytes()
        options = ConversionOptions(**case["options"])
        started = time.perf_counter()
        baseline_result = convert_deepaa(source_data, options)
        target = decode_png(decode_data_url(baseline_result.processed_png))
        source = decode_image(source_data)
        left, top, right, bottom = baseline_result.crop
        cropped = source[top:bottom, left:right]
        source_kind = (
            "lineart"
            if options.profile in {"lineart", "background_lineart"}
            else "grayscale"
        )
        channels = extract_input_channels(
            cropped,
            source_kind=source_kind,
            config=ChannelExtractionConfig(target_width=target.shape[1]),
        )
        fill_result = apply_fill_stage1(
            baseline_result.ascii_text,
            structure=channels.structure,
            tone=channels.tone,
            fill=channels.fill,
            empirical_run_counts=empirical_counts,
            canvas_width=target.shape[1],
            config=config,
        )
        baseline_rendered = _canonical_render(
            baseline_result.ascii_text,
            target.shape[1],
            target.shape[0],
        )
        filled_rendered = _canonical_render(
            fill_result.text,
            target.shape[1],
            target.shape[0],
        )
        diagnostic = _replacement_diagnostic(
            channels,
            fill_result.replacements,
            canvas_width=target.shape[1],
            canvas_height=target.shape[0],
        )
        Image.fromarray(channel_preview(channels.structure)).save(
            output_dir / f"{case_id}-ch0.png", optimize=True
        )
        Image.fromarray(channel_preview(channels.tone)).save(
            output_dir / f"{case_id}-ch1.png", optimize=True
        )
        Image.fromarray(channel_preview(channels.fill)).save(
            output_dir / f"{case_id}-ch2.png", optimize=True
        )
        Image.fromarray(baseline_rendered).save(
            output_dir / f"{case_id}-baseline.png", optimize=True
        )
        Image.fromarray(filled_rendered).save(
            output_dir / f"{case_id}-filled.png", optimize=True
        )
        Image.fromarray(diagnostic).save(
            output_dir / f"{case_id}-diagnostic.png", optimize=True
        )
        _save_review_sheet(
            output_dir / f"{case_id}-review.png",
            channels=channels,
            baseline=baseline_rendered,
            filled=filled_rendered,
            diagnostic=diagnostic,
        )
        (output_dir / f"{case_id}-baseline.txt").write_text(
            baseline_result.ascii_text,
            encoding="utf-8",
            newline="\n",
        )
        (output_dir / f"{case_id}-filled.txt").write_text(
            fill_result.text,
            encoding="utf-8",
            newline="\n",
        )
        elapsed = time.perf_counter() - started
        records.append(
            {
                "id": case_id,
                "kind": case["kind"],
                "source": str(source_path),
                "source_sha256": hashlib.sha256(source_data).hexdigest(),
                "options": case["options"],
                "elapsed_seconds": round(elapsed, 3),
                "channel_statistics": channels.metadata["channel_statistics"],
                "stage1": fill_result.summary(),
                "fill_metrics": fill_stage_metrics(
                    baseline_rendered,
                    filled_rendered,
                    channels,
                ),
                "baseline_geometry": geometric_metrics(target, baseline_rendered),
                "filled_geometry": geometric_metrics(target, filled_rendered),
                "review_sheet": f"{case_id}-review.png",
            }
        )
        print(
            f"{case_id}: replacements={len(fill_result.replacements)}, "
            f"source_cells={fill_result.replaced_source_cell_count}, "
            f"elapsed={elapsed:.1f}s"
        )

    if selected_ids is not None and found != selected_ids:
        raise ValueError(
            "Unknown evaluation case IDs: " + ", ".join(sorted(selected_ids - found))
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "phase": "P4",
        "method": "stage-1 post-decoder overwrite inside ch2 only",
        "baseline": (
            "current runtime DeepAA/beam draft M; P3 selected raw_raster as the "
            "C=1 structure-proxy condition, but its checkpoint is not a runtime release model"
        ),
        "config": config.__dict__,
        "empirical_fill_run_counts": empirical_counts,
        "accepted_entries": int(fill_analysis["entry_count"]),
        "accepted_entries_with_fill": int(fill_analysis["entries_with_fill"]),
        "cases_file": str(cases_path.resolve()),
        "snapshot": str(snapshot_path.resolve()),
        "rights_status": "local evaluation only; source images and accepted AA are not tracked",
        "interpretation": (
            "Automatic metrics enforce ch2 locality, proportional-width preservation, and "
            "line retention. Semantic false fills and black hair/eye readability require review."
        ),
        "cases": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_fill_stage1(
        args.cases,
        args.snapshot,
        args.output,
        selected_ids=set(args.case_ids) if args.case_ids else None,
        force=args.force,
    )
    print(f"cases={len(report['cases'])}")
    print(f"report={(args.output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
