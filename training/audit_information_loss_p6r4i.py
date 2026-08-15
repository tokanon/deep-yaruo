from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backend.deepaa import HALF_WIDTH, LINE_PITCH
from backend.surface_fill_preferences import SurfaceFillPreferenceStore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_ROOT = ROOT / "datasets" / "incoming" / "draft-preferences" / "v3"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "information-loss-p6r4i"

OUTPUT_BLANK_MAXIMUM = 0.01
PROCESSED_INK_MINIMUM = 0.01
SELECTED_SURFACE_MINIMUM = 0.25
RELATIVE_DARKNESS_MINIMUM = 0.18
EDGE_PERCENTILE = 70.0
TEXTURE_PERCENTILE = 70.0
EDGE_ABSOLUTE_MINIMUM = 4.0
TEXTURE_ABSOLUTE_MINIMUM = 2.0


def _decode(payload: bytes, mode: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), mode)
    if image is None:
        raise ValueError("Could not decode an information-loss audit artifact")
    return image


def _cell_means(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    if height % LINE_PITCH or width % HALF_WIDTH:
        raise ValueError("Audit images must use the DeepAA 8x18 half-width grid")
    rows = height // LINE_PITCH
    columns = width // HALF_WIDTH
    return values.reshape(rows, LINE_PITCH, columns, HALF_WIDTH).mean(axis=(1, 3))


def _robust_threshold(
    values: np.ndarray,
    *,
    percentile: float,
    absolute_minimum: float,
) -> float:
    positive = values[values > 0]
    if not positive.size:
        return float("inf")
    return max(absolute_minimum, float(np.percentile(positive, percentile)))


def analyze_information_loss(
    source_gray: np.ndarray,
    *,
    crop: tuple[int, int, int, int],
    processed: np.ndarray,
    rendered: np.ndarray,
    fill_mask: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Audit source-supported evidence missing from a selected AA raster."""

    if processed.ndim != 2 or processed.shape != rendered.shape:
        raise ValueError("Processed and rendered images must be aligned grayscale images")
    if fill_mask.shape != processed.shape:
        raise ValueError("Fill mask must align with processed and rendered images")
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= source_gray.shape[1] and 0 <= y0 < y1 <= source_gray.shape[0]):
        raise ValueError("Audit crop is outside the source image")

    aligned = cv2.resize(
        source_gray[y0:y1, x0:x1],
        (processed.shape[1], processed.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    smoothed = cv2.GaussianBlur(aligned, (0, 0), sigmaX=1.0, sigmaY=1.0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    local_mean = cv2.GaussianBlur(aligned, (0, 0), sigmaX=3.0, sigmaY=3.0)
    texture = cv2.absdiff(aligned, local_mean)

    source_pixels = aligned.astype(np.float32)
    p10 = float(np.percentile(source_pixels, 10.0))
    p90 = float(np.percentile(source_pixels, 90.0))
    tone_span = max(1.0, p90 - p10)
    relative_darkness = np.clip((p90 - source_pixels) / tone_span, 0.0, 1.0)

    edge_cells = _cell_means(gradient)
    texture_cells = _cell_means(texture.astype(np.float32))
    tone_cells = _cell_means(relative_darkness)
    processed_ink = _cell_means((processed < 128).astype(np.float32))
    rendered_ink = _cell_means((rendered < 240).astype(np.float32))
    selected_surface = _cell_means((fill_mask >= 128).astype(np.float32))

    edge_threshold = _robust_threshold(
        edge_cells,
        percentile=EDGE_PERCENTILE,
        absolute_minimum=EDGE_ABSOLUTE_MINIMUM,
    )
    texture_threshold = _robust_threshold(
        texture_cells,
        percentile=TEXTURE_PERCENTILE,
        absolute_minimum=TEXTURE_ABSOLUTE_MINIMUM,
    )
    edge_evidence = edge_cells >= edge_threshold
    texture_evidence = texture_cells >= texture_threshold
    tone_evidence = tone_cells >= RELATIVE_DARKNESS_MINIMUM
    informative = edge_evidence | texture_evidence | tone_evidence
    output_blank = rendered_ink <= OUTPUT_BLANK_MAXIMUM
    missing = informative & output_blank

    processed_present = processed_ink > PROCESSED_INK_MINIMUM
    fill_selected = selected_surface >= SELECTED_SURFACE_MINIMUM
    categories = np.zeros(missing.shape, dtype=np.uint8)
    # Pipeline order makes these mutually exclusive: representation first,
    # followed by surface decision/placement, then glyph conversion.
    categories[missing & ~processed_present & (edge_evidence | texture_evidence)] = 1
    categories[
        missing
        & ~processed_present
        & ~(edge_evidence | texture_evidence)
        & tone_evidence
        & ~fill_selected
    ] = 2
    categories[
        missing
        & ~processed_present
        & ~(edge_evidence | texture_evidence)
        & tone_evidence
        & fill_selected
    ] = 3
    categories[missing & processed_present] = 4
    labels = {
        1: "preprocess-structure-loss",
        2: "surface-proposal-or-decision-loss",
        3: "selected-surface-placement-loss",
        4: "glyph-conversion-loss",
    }
    counts = Counter(labels[int(value)] for value in categories[categories > 0])
    informative_count = int(informative.sum())
    missing_count = int(missing.sum())
    total_cells = int(informative.size)
    report: dict[str, object] = {
        "grid": {
            "rows": int(informative.shape[0]),
            "columns": int(informative.shape[1]),
            "half_width_px": HALF_WIDTH,
            "line_pitch_px": LINE_PITCH,
        },
        "thresholds": {
            "output_blank_maximum": OUTPUT_BLANK_MAXIMUM,
            "processed_ink_minimum": PROCESSED_INK_MINIMUM,
            "selected_surface_minimum": SELECTED_SURFACE_MINIMUM,
            "relative_darkness_minimum": RELATIVE_DARKNESS_MINIMUM,
            "edge_percentile": EDGE_PERCENTILE,
            "edge_threshold": round(edge_threshold, 6),
            "texture_percentile": TEXTURE_PERCENTILE,
            "texture_threshold": round(texture_threshold, 6),
            "source_gray_p10": round(p10, 6),
            "source_gray_p90": round(p90, 6),
        },
        "total_cells": total_cells,
        "informative_source_cells": informative_count,
        "blank_informative_cells": missing_count,
        "blank_informative_fraction": round(
            missing_count / max(1, informative_count), 6
        ),
        "evidence_counts": {
            "edge": int(edge_evidence.sum()),
            "texture": int(texture_evidence.sum()),
            "tone": int(tone_evidence.sum()),
        },
        "cause_counts": {
            label: counts.get(label, 0) for label in labels.values()
        },
        "cause_fractions": {
            label: round(counts.get(label, 0) / max(1, missing_count), 6)
            for label in labels.values()
        },
    }
    return report, missing, categories


def _expanded_cell_mask(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(mask, LINE_PITCH, axis=0), HALF_WIDTH, axis=1)


def _write_diagnostics(
    output_dir: Path,
    stem: str,
    aligned_source: np.ndarray,
    missing: np.ndarray,
    categories: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_pixels = _expanded_cell_mask(missing)
    category_pixels = _expanded_cell_mask(categories)
    mask_path = output_dir / f"{stem}-missing-cells.png"
    overlay_path = output_dir / f"{stem}-cause-overlay.png"
    cv2.imwrite(str(mask_path), np.where(missing_pixels, 255, 0).astype(np.uint8))
    overlay = cv2.cvtColor(aligned_source, cv2.COLOR_GRAY2BGR)
    colors = {
        1: (40, 40, 230),
        2: (220, 80, 40),
        3: (40, 190, 230),
        4: (180, 40, 180),
    }
    for category, color in colors.items():
        selected = category_pixels == category
        overlay[selected] = (
            0.45 * overlay[selected] + 0.55 * np.asarray(color)
        ).astype(np.uint8)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "missing_mask": mask_path.name,
        "cause_overlay": overlay_path.name,
    }


def audit_v3_records(
    v3_root: Path = DEFAULT_V3_ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    store = SurfaceFillPreferenceStore(
        v3_root,
        comparison_kind="surface-recovery-v3",
        schema_version=3,
        require_common_processed=False,
    )
    for record in sorted(store.list(), key=lambda item: str(item["record_id"])):
        selected_variant = record.get("selected_variant")
        if not selected_variant:
            continue
        candidate = next(
            item
            for item in record["candidates"]
            if item["variant_id"] == selected_variant
        )
        record_dir = v3_root / str(record["record_id"])
        source = _decode(
            (record_dir / record["source"]["artifact"]["path"]).read_bytes()
        )
        artifacts = candidate["artifacts"]
        processed = _decode((record_dir / artifacts["processed"]["path"]).read_bytes())
        rendered = _decode((record_dir / artifacts["rendered"]["path"]).read_bytes())
        fill_mask = _decode((record_dir / artifacts["fill_mask"]["path"]).read_bytes())
        crop = tuple(int(value) for value in record["crop"])
        case_report, missing, categories = analyze_information_loss(
            source,
            crop=crop,
            processed=processed,
            rendered=rendered,
            fill_mask=fill_mask,
        )
        x0, y0, x1, y1 = crop
        aligned_source = cv2.resize(
            source[y0:y1, x0:x1],
            (processed.shape[1], processed.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        filename = str(record["source"]["filename"])
        stem = Path(filename).stem
        diagnostics = _write_diagnostics(
            output_dir,
            stem,
            aligned_source,
            missing,
            categories,
        )
        cases.append(
            {
                "record_id": record["record_id"],
                "filename": filename,
                "selected_variant": selected_variant,
                **case_report,
                "diagnostics": diagnostics,
            }
        )

    aggregate_counts = Counter()
    informative_total = 0
    missing_total = 0
    for case in cases:
        informative_total += int(case["informative_source_cells"])
        missing_total += int(case["blank_informative_cells"])
        aggregate_counts.update(case["cause_counts"])
    report = {
        "phase": "P6R-4I-information-loss-audit",
        "status": "diagnostic-only",
        "artifact_validation": "passed",
        "source_record_count": len(cases),
        "informative_source_cells": informative_total,
        "blank_informative_cells": missing_total,
        "blank_informative_fraction": round(
            missing_total / max(1, informative_total), 6
        ),
        "cause_counts": dict(sorted(aggregate_counts.items())),
        "cause_fractions": {
            label: round(count / max(1, missing_total), 6)
            for label, count in sorted(aggregate_counts.items())
        },
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
    report = audit_v3_records(args.v3_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
