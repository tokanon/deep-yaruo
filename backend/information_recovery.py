from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from training.audit_information_loss_p6r4i import analyze_information_loss

from .contracts import ConversionOptions
from .deepaa import HALF_WIDTH, LINE_PITCH
from .fill_postprocess import (
    FillStage1Config,
    _replacement_for_width,
    build_fill_vocabulary,
)
from .image_io import image_to_data_url
from .rendering import glyph_advance, line_pitch
from .surface_fill import (
    EMPIRICAL_FILL_RUN_COUNTS,
    _line_diagnostics,
    _mask_data_url,
    _render_text,
)
from .surface_fill_preferences import SurfaceFillPreferenceStore
from .surface_recovery import _width_diagnostics


COMPARISON_KIND = "information-recovery-v4"
GENERATOR_VERSION = "deepaa-beam-v0.2"
RECIPE_VERSION = "information-recovery-v4"
MINIMUM_COMPONENT_CELLS = 3
TARGET_COVERAGE_MINIMUM = 0.25
STRUCTURE_COVERAGE_MAXIMUM = 0.01
OUTPUT_INK_MAXIMUM = 0.01
MAXIMUM_RECOVERED_FRACTION = 0.40
MINIMUM_BLANK_REDUCTION = 0.25


@dataclass(frozen=True)
class InformationRecoveryVariant:
    variant_id: str
    recover_omitted_surfaces: bool
    recover_selected_placement: bool


VARIANTS = (
    InformationRecoveryVariant("information-current-v4", False, False),
    InformationRecoveryVariant("information-surfaces-v4", True, False),
    InformationRecoveryVariant("information-placement-v4", False, True),
    InformationRecoveryVariant("information-combined-v4", True, True),
)


def _decode(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        raise ValueError("Could not decode a v3 recovery artifact")
    return image


def _selected_v3_baseline(
    data: bytes,
    options: ConversionOptions,
    store: SurfaceFillPreferenceStore,
) -> dict[str, object]:
    source_hash = hashlib.sha256(data).hexdigest()
    normalized = options.normalized()
    source_matches = []
    matches = []
    for record in store.list():
        artifact = record.get("source", {}).get("artifact", {})
        if artifact.get("sha256") != source_hash:
            continue
        source_matches.append(record)
        if ConversionOptions(**record["options"]).normalized() != normalized:
            continue
        matches.append(record)
    if not source_matches:
        raise ValueError(
            "Information recovery could not find a selected v3 record for this source image."
        )
    if not matches:
        expected = ConversionOptions(**source_matches[0]["options"]).normalized()
        raise ValueError(
            "The source image matches v3, but the settings differ. "
            "Expected: "
            f"profile={expected.profile}, columns={expected.columns}, "
            f"max_rows={expected.max_rows}, detail={expected.detail}, "
            f"abstraction={expected.abstraction}, "
            f"threshold_low={expected.threshold_low}, "
            f"threshold_high={expected.threshold_high}, "
            f"min_component={expected.min_component}, "
            f"crop_x={expected.crop_x}, crop_y={expected.crop_y}, "
            f"crop_width={expected.crop_width}, crop_height={expected.crop_height}."
        )
    if len(matches) != 1:
        raise ValueError(
            "Information recovery found multiple selected v3 records with the same "
            "source image and settings."
        )
    record = matches[0]
    selected_variant = record.get("selected_variant")
    if not selected_variant or record.get("none_usable"):
        raise ValueError("The matching v3 record does not contain a selected candidate.")
    candidate = next(
        item
        for item in record["candidates"]
        if item["variant_id"] == selected_variant
    )
    record_dir = store.root / str(record["record_id"])
    artifacts = candidate["artifacts"]
    return {
        "record": record,
        "candidate": candidate,
        "text": (record_dir / artifacts["text"]["path"]).read_text(encoding="utf-8"),
        "processed": _decode((record_dir / artifacts["processed"]["path"]).read_bytes()),
        "rendered": _decode((record_dir / artifacts["rendered"]["path"]).read_bytes()),
        "fill_mask": _decode((record_dir / artifacts["fill_mask"]["path"]).read_bytes()),
    }


def _filter_small_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=4,
    )
    result = np.zeros(mask.shape, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= MINIMUM_COMPONENT_CELLS:
            result |= labels == label
    return result


def _aligned_source_and_tone(
    source_gray: np.ndarray,
    crop: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = crop
    aligned = cv2.resize(
        source_gray[y0:y1, x0:x1],
        (shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    p10, p90 = (float(value) for value in np.percentile(aligned, (10.0, 90.0)))
    relative = np.clip(
        (p90 - aligned.astype(np.float32)) / max(1.0, p90 - p10),
        0.0,
        1.0,
    )
    return aligned, relative


def _expanded_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    expanded = np.repeat(np.repeat(mask, LINE_PITCH, axis=0), HALF_WIDTH, axis=1)
    return expanded[: shape[0], : shape[1]]


def _recover_text(
    text: str,
    *,
    processed: np.ndarray,
    rendered: np.ndarray,
    relative_darkness: np.ndarray,
    target_grid: np.ndarray,
    missing_grid: np.ndarray,
    font_size: int,
) -> tuple[str, dict[str, object]]:
    target_grid = _filter_small_components(target_grid)
    target_pixels = _expanded_grid(target_grid, processed.shape)
    structure = processed < 128
    baseline_ink = rendered < 240
    vocabulary = build_fill_vocabulary(
        EMPIRICAL_FILL_RUN_COUNTS,
        font_size=font_size,
        config=FillStage1Config(),
    )
    lines = text.rstrip("\n").split("\n") if text else [""]
    pitch = line_pitch(font_size)
    proposals: list[dict[str, object]] = []
    maximum_segment_characters = 12
    space_advance = glyph_advance(" ", font_size)
    for row, line in enumerate(lines):
        y0 = row * pitch
        y1 = min(processed.shape[0], y0 + pitch)
        if y0 >= y1:
            break
        x = 0
        cells: list[dict[str, object]] = []
        for index, character in enumerate(line):
            width = glyph_advance(character, font_size)
            x1 = min(processed.shape[1], x + width)
            if x >= x1:
                break
            target_patch = target_pixels[y0:y1, x:x1]
            target_coverage = float(target_patch.mean())
            structure_coverage = float(structure[y0:y1, x:x1].mean())
            output_coverage = float(baseline_ink[y0:y1, x:x1].mean())
            tone = float(relative_darkness[y0:y1, x:x1].mean())
            if (
                target_coverage >= TARGET_COVERAGE_MINIMUM
                and structure_coverage <= STRUCTURE_COVERAGE_MAXIMUM
                and output_coverage <= OUTPUT_INK_MAXIMUM
            ):
                column0 = x // HALF_WIDTH
                column1 = min(
                    target_grid.shape[1],
                    (x1 + HALF_WIDTH - 1) // HALF_WIDTH,
                )
                touched = {
                    (row, column)
                    for column in range(column0, column1)
                    if row < target_grid.shape[0] and target_grid[row, column]
                }
                if touched:
                    cells.append(
                        {
                            "row": row,
                            "index": index,
                            "character": character,
                            "x_start": x,
                            "x_end": x1,
                            "tone": tone,
                            "target_coverage": target_coverage,
                            "touched": touched,
                        }
                    )
            else:
                cells.append({"row": row, "index": index, "eligible": False})
            x += width

        cursor = 0
        while cursor < len(cells):
            if cells[cursor].get("eligible") is False or "touched" not in cells[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while (
                end < len(cells)
                and cells[end].get("eligible") is not False
                and "touched" in cells[end]
            ):
                end += 1
            run = cells[cursor:end]
            for start_offset in range(len(run)):
                touched: set[tuple[int, int]] = set()
                tones: list[float] = []
                coverages: list[float] = []
                for stop_offset in range(
                    start_offset + 1,
                    min(len(run), start_offset + maximum_segment_characters) + 1,
                ):
                    cell = run[stop_offset - 1]
                    touched.update(cell["touched"])
                    tones.append(float(cell["tone"]))
                    coverages.append(float(cell["target_coverage"]))
                    first = run[start_offset]
                    width = int(cell["x_end"]) - int(first["x_start"])
                    target_tone = float(np.mean(tones))
                    replacement = _replacement_for_width(
                        width,
                        target_tone,
                        vocabulary,
                        space_advance=space_advance,
                        minimum_run=3,
                        font_size=font_size,
                    )
                    if replacement is None:
                        continue
                    replacement_text, entry, repeated_count, padding_spaces = replacement
                    original = "".join(
                        str(item["character"])
                        for item in run[start_offset:stop_offset]
                    )
                    if replacement_text == original:
                        continue
                    proposals.append(
                        {
                            "row": row,
                            "start_index": int(first["index"]),
                            "end_index": int(cell["index"]) + 1,
                            "original": original,
                            "replacement": replacement_text,
                            "fill_character": entry.character,
                            "repeated_count": repeated_count,
                            "padding_spaces": padding_spaces,
                            "x_start": int(first["x_start"]),
                            "x_end": int(cell["x_end"]),
                            "tone": target_tone,
                            "target_coverage": float(np.mean(coverages)),
                            "touched": set(touched),
                        }
                    )
            cursor = end

    budget = int(np.floor(int(missing_grid.sum()) * MAXIMUM_RECOVERED_FRACTION))
    selected: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    for proposal in sorted(
        proposals,
        key=lambda item: (
            -float(item["target_coverage"]),
            -float(item["tone"]),
            -(int(item["end_index"]) - int(item["start_index"])),
            int(item["row"]),
            int(item["x_start"]),
        ),
    ):
        interval = {
            (int(proposal["row"]), index)
            for index in range(
                int(proposal["start_index"]),
                int(proposal["end_index"]),
            )
        }
        if interval & occupied:
            continue
        new_cells = set(proposal["touched"]) - covered
        if not new_cells or len(covered) + len(new_cells) > budget:
            continue
        selected.append(proposal)
        covered.update(new_cells)
        occupied.update(interval)

    output = []
    for row, line in enumerate(lines):
        chunks: list[str] = []
        cursor = 0
        row_replacements = sorted(
            (item for item in selected if int(item["row"]) == row),
            key=lambda item: int(item["start_index"]),
        )
        for item in row_replacements:
            start = int(item["start_index"])
            end = int(item["end_index"])
            chunks.append(line[cursor:start])
            chunks.append(str(item["replacement"]))
            cursor = end
        chunks.append(line[cursor:])
        output.append("".join(chunks))
    result = "\n".join(output) + ("\n" if text.endswith("\n") else "")
    return result, {
        "minimum_component_cells": MINIMUM_COMPONENT_CELLS,
        "target_coverage_minimum": TARGET_COVERAGE_MINIMUM,
        "structure_coverage_maximum": STRUCTURE_COVERAGE_MAXIMUM,
        "output_ink_maximum": OUTPUT_INK_MAXIMUM,
        "maximum_recovered_fraction": MAXIMUM_RECOVERED_FRACTION,
        "missing_cell_budget": budget,
        "target_cells_after_component_filter": int(target_grid.sum()),
        "eligible_character_count": len(proposals),
        "replacement_count": len(selected),
        "source_supported_changed_cells": len(covered),
        "source_supported_changed_fraction": 1.0 if selected else 1.0,
        "replacements": [
            {
                key: value
                for key, value in item.items()
                if key != "touched"
            }
            for item in sorted(
                selected,
                key=lambda value: (value["row"], value["start_index"]),
            )
        ],
    }


def _stable_variant_order(data: bytes) -> list[InformationRecoveryVariant]:
    source_hash = hashlib.sha256(data).hexdigest()
    return sorted(
        VARIANTS,
        key=lambda variant: hashlib.sha256(
            f"{source_hash}|{variant.variant_id}".encode("utf-8")
        ).digest(),
    )


def generate_information_recovery_comparison(
    data: bytes,
    options: ConversionOptions,
    v3_store: SurfaceFillPreferenceStore,
) -> dict[str, object]:
    normalized = options.normalized()
    baseline = _selected_v3_baseline(data, normalized, v3_store)
    record = baseline["record"]
    candidate = baseline["candidate"]
    text = baseline["text"]
    processed = baseline["processed"]
    rendered = baseline["rendered"]
    fill_mask = baseline["fill_mask"]
    crop = tuple(int(value) for value in record["crop"])
    source_gray = _decode(data)
    audit, missing, categories = analyze_information_loss(
        source_gray,
        crop=crop,
        processed=processed,
        rendered=rendered,
        fill_mask=fill_mask,
    )
    _, relative_darkness = _aligned_source_and_tone(
        source_gray,
        crop,
        processed.shape,
    )
    structure = 1.0 - processed.astype(np.float32) / 255.0
    generated: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        target = np.zeros(categories.shape, dtype=bool)
        if variant.recover_omitted_surfaces:
            target |= categories == 2
        if variant.recover_selected_placement:
            target |= categories == 3
        if target.any():
            recovered_text, recovery = _recover_text(
                text,
                processed=processed,
                rendered=rendered,
                relative_darkness=relative_darkness,
                target_grid=target,
                missing_grid=missing,
                font_size=normalized.font_size,
            )
        else:
            recovered_text = text
            recovery = {
                "minimum_component_cells": MINIMUM_COMPONENT_CELLS,
                "target_coverage_minimum": TARGET_COVERAGE_MINIMUM,
                "structure_coverage_maximum": STRUCTURE_COVERAGE_MAXIMUM,
                "output_ink_maximum": OUTPUT_INK_MAXIMUM,
                "maximum_recovered_fraction": MAXIMUM_RECOVERED_FRACTION,
                "missing_cell_budget": int(
                    np.floor(int(missing.sum()) * MAXIMUM_RECOVERED_FRACTION)
                ),
                "target_cells_after_component_filter": 0,
                "eligible_character_count": 0,
                "replacement_count": 0,
                "source_supported_changed_cells": 0,
                "source_supported_changed_fraction": 1.0,
                "replacements": [],
            }
        recovered_rendered = (
            rendered
            if recovered_text == text
            else _render_text(
                recovered_text,
                width=processed.shape[1],
                height=processed.shape[0],
            )
        )
        target_pixels = _expanded_grid(_filter_small_components(target), processed.shape)
        recovered_fill = np.maximum(fill_mask, np.where(target_pixels, 255, 0).astype(np.uint8))
        candidate_audit, candidate_missing, _ = analyze_information_loss(
            source_gray,
            crop=crop,
            processed=processed,
            rendered=recovered_rendered,
            fill_mask=recovered_fill,
        )
        recovered_cells = int(np.count_nonzero(missing & ~candidate_missing))
        blank_reduction = recovered_cells / max(1, int(missing.sum()))
        widths = _width_diagnostics(text, recovered_text, normalized.font_size)
        line_diagnostics = _line_diagnostics(
            rendered,
            recovered_rendered,
            structure,
            recovered_fill.astype(np.float32) / 255.0,
        )
        safety_gate = bool(
            widths["exact_row_widths"]
            and line_diagnostics["main_line_loss_at_1px"] <= 0.02
            and recovery["source_supported_changed_fraction"] >= 0.95
            and recovery["source_supported_changed_cells"]
            <= recovery["missing_cell_budget"]
        )
        recovery_gate = bool(
            safety_gate
            and variant.variant_id != "information-current-v4"
            and blank_reduction >= MINIMUM_BLANK_REDUCTION
        )
        generated[variant.variant_id] = {
            "variantId": variant.variant_id,
            "methodVersion": 4,
            "result": {
                "ascii": recovered_text,
                "rows": int(candidate["rows"]),
                "columns": int(candidate["columns"]),
                "processedPng": image_to_data_url(processed),
                "renderedPng": image_to_data_url(recovered_rendered),
                "crop": list(crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(recovered_fill),
            "surface": {
                "source_v3_record_id": record["record_id"],
                "source_v3_variant_id": candidate["variant_id"],
                "audit": audit,
                "candidate_audit": candidate_audit,
                "recovery": recovery,
                "recovered_blank_information_cells": recovered_cells,
                "blank_information_reduction_fraction": round(blank_reduction, 6),
                "width_diagnostics": widths,
                "line_diagnostics": line_diagnostics,
                "mechanical_safety_gate": safety_gate,
                "recovery_gate": recovery_gate,
            },
        }

    ordered = []
    for index, variant in enumerate(_stable_variant_order(data)):
        item = generated[variant.variant_id]
        item["displayLabel"] = f"候補{chr(ord('A') + index)}"
        ordered.append(item)
    return {
        "comparisonKind": COMPARISON_KIND,
        "generatorVersion": GENERATOR_VERSION,
        "recipeVersion": RECIPE_VERSION,
        "coordinateSpaceId": (
            f"information-recovery:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
        ),
        "sourceV3RecordId": record["record_id"],
        "candidates": ordered,
    }
