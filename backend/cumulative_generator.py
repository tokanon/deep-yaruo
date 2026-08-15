from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from training.audit_information_loss_p6r4i import analyze_information_loss

from .contracts import ConversionOptions, ConversionResult
from .deepaa_surface import DECODER_VERSION, decode_surface_text
from .deepaa import convert_deepaa, preprocess_for_deepaa
from .image_io import decode_color_image, decode_image, image_to_data_url
from .rendering import REFERENCE_FONT_SIZE, render_text_mask
from .surface_channels import real_image_input_channels
from .information_recovery import (
    MAXIMUM_RECOVERED_FRACTION,
    _aligned_source_and_tone,
    _decode,
    _expanded_grid,
    _filter_small_components,
    _recover_text,
)
from .surface_fill import (
    MINIMUM_FILL_SCORE,
    SURFACE_LAYER_ID,
    SURFACE_PROPOSAL_CONFIG,
    _apply_tone_adaptive_fill,
    _decode_png_data_url,
    _line_diagnostics,
    _render_text,
)
from .surface_proposals import extract_surface_proposals
from .surface_recovery import (
    CONTACT_RATIO_SURFACE_DECISION,
    CURRENT_SURFACE_DECISION,
    SAFE_BORDER_RESCUE_RECIPE,
    STANDARD_RECIPE,
    _merge_fill_summaries,
    _surface_state,
    _width_diagnostics,
    recover_local_contrast_lines,
)
from .surface_texture import (
    ACCEPTED_SNAPSHOT_SHA256,
    DENSITY_TOLERANCE,
    PROXIMITY_METHOD_VERSION,
    apply_surface_texture_proximity,
    load_texture_catalog,
    _selected_owner_labels,
)


P6RG1_GENERATOR_VERSION = "deepaa-cumulative-g1"
P6RG1_RECIPE_VERSION = "p6r-cumulative-g1-v1"
GENERATOR_VERSION = "deepaa-surface-v0-ls"
RECIPE_VERSION = "deepaa-surface-v0-provisional-v1"
LINE_ART_PROFILES = frozenset({"lineart", "background_lineart"})


@dataclass(frozen=True)
class CumulativeGeneration:
    result: ConversionResult
    pipeline: dict[str, object]


def _select_surface_base(
    data: bytes,
    options: ConversionOptions,
) -> tuple[ConversionResult, np.ndarray, dict[str, object]]:
    normalized = options.normalized()
    source_gray = decode_image(data)
    current_line, crop = preprocess_for_deepaa(source_gray, normalized)
    recovered_line, line_recovery = recover_local_contrast_lines(
        source_gray,
        crop=crop,
        current_line_image=current_line,
        options=normalized,
    )
    line_applied = bool(line_recovery["mechanical_gate"])
    selected_line = recovered_line if line_applied else current_line
    base = convert_deepaa(
        data,
        normalized,
        line_image_override=selected_line,
        crop_box_override=crop,
    )
    processed = selected_line
    base_rendered = _decode_png_data_url(base.rendered_png)

    source = decode_color_image(data)
    coordinate_space_id = (
        f"cumulative-surface:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
    )
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=coordinate_space_id,
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=crop,
        target_shape_hw=processed.shape,
    )
    layer = next(
        item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID
    )
    current_state = _surface_state(
        layer,
        source=source,
        crop=crop,
        target_shape_hw=processed.shape,
        decision_config=CURRENT_SURFACE_DECISION,
    )
    recovered_state = _surface_state(
        layer,
        source=source,
        crop=crop,
        target_shape_hw=processed.shape,
        decision_config=CONTACT_RATIO_SURFACE_DECISION,
    )
    current_decisions, current_selected, current_tone, current_relative, gray = (
        current_state
    )
    recovered_decisions, _, recovered_tone, recovered_relative, _ = recovered_state
    current_actions = {
        int(decision["region_label"]): decision["action"]
        for decision in current_decisions
    }
    rescued_labels = sorted(
        int(decision["region_label"])
        for decision in recovered_decisions
        if decision["action"] == "fill"
        and current_actions.get(int(decision["region_label"])) != "fill"
    )
    structure = 1.0 - processed.astype(np.float32) / 255.0

    filled_text, initial_summary = _apply_tone_adaptive_fill(
        base.ascii_text,
        structure=structure,
        tone=current_tone,
        relative_darkness=current_relative,
        selected_surface=current_selected,
        recipe=STANDARD_RECIPE,
        canvas_width=processed.shape[1],
        font_size=normalized.font_size,
    )
    filled_rendered = (
        base_rendered
        if filled_text == base.ascii_text
        else _render_text(
            filled_text,
            width=processed.shape[1],
            height=processed.shape[0],
        )
    )
    fill_widths = _width_diagnostics(base.ascii_text, filled_text, normalized.font_size)
    fill_lines = _line_diagnostics(
        base_rendered,
        filled_rendered,
        structure,
        current_selected,
    )
    fill_safe = bool(
        fill_widths["exact_row_widths"]
        and fill_lines["main_line_loss_at_1px"] <= 0.02
    )
    if not fill_safe:
        filled_text = base.ascii_text
        filled_rendered = base_rendered
        selected = np.zeros(current_selected.shape, dtype=np.float32)
        fill_summary = {
            **initial_summary,
            "eligible_cell_count": 0,
            "replaced_source_cell_count": 0,
            "replacement_count": 0,
            "replaced_pixel_width": 0,
            "replacements": [],
            "band_summaries": [],
            "fallback_reason": "mechanical-safety-gate-failed"
        }
    else:
        selected = current_selected
        fill_summary = initial_summary

    border_applied = False
    border_fallback: str | None = None
    final_text = filled_text
    final_rendered = filled_rendered
    if rescued_labels:
        rescued_selected = np.isin(
            layer.labels,
            np.asarray(rescued_labels, dtype=np.int32),
        ).astype(np.float32)
        rescued_text, rescue_summary = _apply_tone_adaptive_fill(
            filled_text,
            structure=structure,
            tone=recovered_tone,
            relative_darkness=recovered_relative,
            selected_surface=rescued_selected,
            recipe=SAFE_BORDER_RESCUE_RECIPE,
            canvas_width=processed.shape[1],
            font_size=normalized.font_size,
        )
        rescued_rendered = (
            filled_rendered
            if rescued_text == filled_text
            else _render_text(
                rescued_text,
                width=processed.shape[1],
                height=processed.shape[0],
            )
        )
        rescue_widths = _width_diagnostics(
            filled_text,
            rescued_text,
            normalized.font_size,
        )
        rescue_lines = _line_diagnostics(
            filled_rendered,
            rescued_rendered,
            structure,
            rescued_selected,
        )
        if (
            rescue_widths["exact_row_widths"]
            and rescue_lines["main_line_loss_at_1px"] <= 0.02
        ):
            final_text = rescued_text
            final_rendered = rescued_rendered
            selected = np.maximum(selected, rescued_selected)
            fill_summary = _merge_fill_summaries(fill_summary, rescue_summary)
            border_applied = True
        else:
            border_fallback = "mechanical-safety-gate-failed"

    widths = _width_diagnostics(base.ascii_text, final_text, normalized.font_size)
    line_diagnostics = _line_diagnostics(
        base_rendered,
        final_rendered,
        structure,
        selected,
    )
    selected_id = (
        "recovery-combined-v3"
        if line_applied and border_applied
        else "recovery-lines-v3"
        if line_applied
        else "recovery-surfaces-v3"
        if border_applied
        else "recovery-current-v3"
    )
    result = ConversionResult(
        ascii_text=final_text,
        rows=base.rows,
        columns=base.columns,
        processed_png=image_to_data_url(processed),
        rendered_png=image_to_data_url(final_rendered),
        crop=base.crop,
    )
    return result, np.where(selected > 0, 255, 0).astype(np.uint8), {
        "stage": "surface-recovery-and-fill",
        "selected_variant": selected_id,
        "applied": bool(line_applied or fill_safe or border_applied),
        "fallback_reason": None,
        "line_recovery": line_recovery | {
            "applied": line_applied,
            "fallback_reason": None if line_applied else "mechanical-safety-gate-failed",
        },
        "surface_fill": {
            "variant": STANDARD_RECIPE.variant_id,
            "minimum_fill_score": MINIMUM_FILL_SCORE,
            "applied": fill_safe,
            "fallback_reason": None if fill_safe else "mechanical-safety-gate-failed",
        },
        "border_surface_recovery": {
            "attempted_region_labels": rescued_labels,
            "attempted_surface_count": len(rescued_labels),
            "applied": border_applied,
            "rescued_surface_count": len(rescued_labels) if border_applied else 0,
            "fallback_reason": border_fallback,
        },
        "fill_summary": fill_summary,
        "grayscale_reference": {
            "method": "crop-valid-p10-p90-v1",
            "p10": round(gray["p10"], 6),
            "p90": round(gray["p90"], 6),
        },
        "width_diagnostics": widths,
        "line_diagnostics": line_diagnostics,
        "mechanical_gate": bool(
            widths["exact_row_widths"]
            and line_diagnostics["main_line_loss_at_1px"] <= 0.02
        ),
    }


def _apply_information_stage(
    data: bytes,
    options: ConversionOptions,
    baseline: ConversionResult,
    fill_mask: np.ndarray,
) -> tuple[ConversionResult, np.ndarray, dict[str, object]]:
    processed = _decode_png_data_url(baseline.processed_png)
    rendered = _decode_png_data_url(baseline.rendered_png)
    source_gray = _decode(data)
    audit, missing, categories = analyze_information_loss(
        source_gray,
        crop=baseline.crop,
        processed=processed,
        rendered=rendered,
        fill_mask=fill_mask,
    )
    target = (categories == 2) | (categories == 3)
    _, relative_darkness = _aligned_source_and_tone(
        source_gray,
        baseline.crop,
        processed.shape,
    )
    if target.any():
        recovered_text, recovery = _recover_text(
            baseline.ascii_text,
            processed=processed,
            rendered=rendered,
            relative_darkness=relative_darkness,
            target_grid=target,
            missing_grid=missing,
            font_size=options.font_size,
        )
    else:
        recovered_text = baseline.ascii_text
        recovery = {
            "maximum_recovered_fraction": MAXIMUM_RECOVERED_FRACTION,
            "missing_cell_budget": int(
                np.floor(int(missing.sum()) * MAXIMUM_RECOVERED_FRACTION)
            ),
            "source_supported_changed_cells": 0,
            "source_supported_changed_fraction": 1.0,
            "replacement_count": 0,
            "replacements": [],
        }
    recovered_rendered = (
        rendered
        if recovered_text == baseline.ascii_text
        else _render_text(
            recovered_text,
            width=processed.shape[1],
            height=processed.shape[0],
        )
    )
    target_pixels = _expanded_grid(_filter_small_components(target), processed.shape)
    recovered_fill = np.maximum(
        fill_mask,
        np.where(target_pixels, 255, 0).astype(np.uint8),
    )
    candidate_audit, candidate_missing, _ = analyze_information_loss(
        source_gray,
        crop=baseline.crop,
        processed=processed,
        rendered=recovered_rendered,
        fill_mask=recovered_fill,
    )
    recovered_cells = int(np.count_nonzero(missing & ~candidate_missing))
    blank_reduction = recovered_cells / max(1, int(missing.sum()))
    widths = _width_diagnostics(
        baseline.ascii_text,
        recovered_text,
        options.font_size,
    )
    line_diagnostics = _line_diagnostics(
        rendered,
        recovered_rendered,
        1.0 - processed.astype(np.float32) / 255.0,
        recovered_fill.astype(np.float32) / 255.0,
    )
    safety_gate = bool(
        widths["exact_row_widths"]
        and line_diagnostics["main_line_loss_at_1px"] <= 0.02
        and recovery["source_supported_changed_fraction"] >= 0.95
        and recovery["source_supported_changed_cells"]
        <= recovery["missing_cell_budget"]
    )
    metadata = {
        "stage": "information-recovery",
        "variant": "information-combined-v4",
        "applied": safety_gate,
        "fallback_reason": None if safety_gate else "mechanical-safety-gate-failed",
        "audit": audit,
        "candidate_audit": candidate_audit,
        "recovery": recovery,
        "recovered_blank_information_cells": recovered_cells,
        "blank_information_reduction_fraction": round(blank_reduction, 6),
        "width_diagnostics": widths,
        "line_diagnostics": line_diagnostics,
        "mechanical_safety_gate": safety_gate,
        "output_safe": True,
    }
    if not safety_gate:
        return baseline, fill_mask, metadata
    return (
        ConversionResult(
            ascii_text=recovered_text,
            rows=baseline.rows,
            columns=baseline.columns,
            processed_png=baseline.processed_png,
            rendered_png=image_to_data_url(recovered_rendered),
            crop=baseline.crop,
        ),
        recovered_fill,
        metadata,
    )


def _apply_texture_stage(
    data: bytes,
    options: ConversionOptions,
    baseline: ConversionResult,
    fill_mask: np.ndarray,
) -> tuple[ConversionResult, dict[str, object]]:
    processed = _decode_png_data_url(baseline.processed_png)
    rendered = _decode_png_data_url(baseline.rendered_png)
    source = decode_color_image(data)
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=(
            f"cumulative-texture:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
        ),
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=baseline.crop,
        target_shape_hw=fill_mask.shape,
    )
    layer = next(
        item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID
    )
    owner = _selected_owner_labels(fill_mask > 127, layer.labels)
    palette, adjustments, catalog_hash, audit_hash = load_texture_catalog()
    textured_text, texture = apply_surface_texture_proximity(
        baseline.ascii_text,
        owner=owner,
        processed=processed,
        font_size=options.font_size,
        palette=palette,
        edge_adjustments=adjustments,
    )
    textured_rendered = (
        rendered
        if textured_text == baseline.ascii_text
        else _render_text(
            textured_text,
            width=rendered.shape[1],
            height=rendered.shape[0],
        )
    )
    widths = _width_diagnostics(
        baseline.ascii_text,
        textured_text,
        options.font_size,
    )
    line_diagnostics = _line_diagnostics(
        rendered,
        textured_rendered,
        1.0 - processed.astype(np.float32) / 255.0,
        (owner > 0).astype(np.float32),
    )
    safety_gate = bool(
        widths["exact_row_widths"]
        and line_diagnostics["main_line_loss_at_1px"] <= 0.02
        and float(texture["maximum_density_error"]) <= DENSITY_TOLERANCE + 1e-12
    )
    replacements = texture.get("replacements", [])
    changed_characters = sum(
        len(str(item.get("original", "")))
        for item in replacements
        if isinstance(item, dict)
    )
    fill_characters = int(texture.get("eligibility", {}).get("fill_cell_count", 0))
    metadata = {
        "stage": "surface-texture",
        "variant": "surface-texture-proximity-v6",
        "method_version": PROXIMITY_METHOD_VERSION,
        "applied": safety_gate,
        "fallback_reason": None if safety_gate else "mechanical-safety-gate-failed",
        "catalog_sha256": catalog_hash,
        "audit_config_sha256": audit_hash,
        "accepted_snapshot_sha256": ACCEPTED_SNAPSHOT_SHA256,
        "texture": texture,
        "changed_character_count": changed_characters,
        "fill_character_count": fill_characters,
        "changed_character_fraction": round(
            changed_characters / fill_characters if fill_characters else 0.0,
            8,
        ),
        "width_diagnostics": widths,
        "line_diagnostics": line_diagnostics,
        "mechanical_safety_gate": safety_gate,
        "output_safe": True,
    }
    if not safety_gate:
        return baseline, metadata
    return (
        ConversionResult(
            ascii_text=textured_text,
            rows=baseline.rows,
            columns=baseline.columns,
            processed_png=baseline.processed_png,
            rendered_png=image_to_data_url(textured_rendered),
            crop=baseline.crop,
        ),
        metadata,
    )


def generate_p6rg1(
    data: bytes,
    options: ConversionOptions,
) -> CumulativeGeneration:
    normalized = options.normalized()
    if normalized.profile in LINE_ART_PROFILES:
        result = convert_deepaa(data, normalized)
        return CumulativeGeneration(
            result=result,
            pipeline={
                "generator_version": P6RG1_GENERATOR_VERSION,
                "recipe_version": P6RG1_RECIPE_VERSION,
                "enabled": True,
                "applied": False,
                "reason": "line-art-profile-has-no-tone-surface",
                "stages": [],
            },
        )

    surface_result, fill_mask, surface_stage = _select_surface_base(data, normalized)
    information_result, recovered_fill, information_stage = _apply_information_stage(
        data,
        normalized,
        surface_result,
        fill_mask,
    )
    final_result, texture_stage = _apply_texture_stage(
        data,
        normalized,
        information_result,
        recovered_fill,
    )
    return CumulativeGeneration(
        result=final_result,
        pipeline={
            "generator_version": P6RG1_GENERATOR_VERSION,
            "recipe_version": P6RG1_RECIPE_VERSION,
            "enabled": True,
            "applied": True,
            "stages": [surface_stage, information_stage, texture_stage],
        },
    )


def _prepare_surface_model_inputs(
    data: bytes,
    options: ConversionOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int], dict[str, object]]:
    source_gray = decode_image(data)
    current_line, crop = preprocess_for_deepaa(source_gray, options)
    recovered_line, line_recovery = recover_local_contrast_lines(
        source_gray,
        crop=crop,
        current_line_image=current_line,
        options=options,
    )
    line_applied = bool(line_recovery["mechanical_gate"])
    selected_line = recovered_line if line_applied else current_line

    source = decode_color_image(data)
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=(
            f"surface-model:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
        ),
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=crop,
        target_shape_hw=selected_line.shape,
    )
    layer = next(
        item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID
    )
    decisions, selected, tone, _relative, gray = _surface_state(
        layer,
        source=source,
        crop=crop,
        target_shape_hw=selected_line.shape,
        decision_config=CURRENT_SURFACE_DECISION,
    )
    surface_ids = np.where(selected > 0, layer.labels, 0).astype(np.int32)
    surface_tone = np.rint(np.clip(tone, 0.0, 1.0) * 255.0).astype(np.uint8)
    line_channels, surface_channels = real_image_input_channels(
        selected_line,
        surface_ids,
        surface_tone,
    )
    metadata = {
        "stage": "p6r-real-image-inputs",
        "line_recovery": line_recovery
        | {
            "applied": line_applied,
            "fallback_reason": None
            if line_applied
            else "mechanical-safety-gate-failed",
        },
        "surface_layer_id": SURFACE_LAYER_ID,
        "surface_decision": "current-surface-decision-v1",
        "surface_proposal_count": len(layer.proposals),
        "selected_surface_count": sum(
            decision["action"] == "fill" for decision in decisions
        ),
        "selected_surface_pixel_count": int(np.count_nonzero(surface_ids)),
        "grayscale_reference": {
            "method": "crop-valid-p10-p90-v1",
            "p10": round(gray["p10"], 6),
            "p90": round(gray["p90"], 6),
        },
        "channels": {
            "line": ["p6r-4l-recovered-line" if line_applied else "deepaa-line"],
            "surface": ["membership", "id-boundary", "source-darkness-tone"],
            "raw_surface_ids_as_model_input": False,
        },
        "applied": True,
    }
    return selected_line, line_channels, surface_channels, crop, metadata


def _fallback_generation(
    data: bytes,
    options: ConversionOptions,
    *,
    reason: str,
) -> CumulativeGeneration:
    fallback = generate_p6rg1(data, options)
    return CumulativeGeneration(
        result=fallback.result,
        pipeline={
            "generator_version": GENERATOR_VERSION,
            "recipe_version": RECIPE_VERSION,
            "enabled": True,
            "applied": False,
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_generator_version": P6RG1_GENERATOR_VERSION,
            "fallback_recipe_version": P6RG1_RECIPE_VERSION,
            "fallback_pipeline": fallback.pipeline,
            "stages": [],
        },
    )


def _render_surface_model_text(text: str, *, width: int, height: int) -> np.ndarray:
    mask = render_text_mask(
        text,
        REFERENCE_FONT_SIZE,
        canvas_height_per_line=18,
    )
    canvas = np.full((height, width), 255, dtype=np.uint8)
    copy_height = min(height, mask.shape[0])
    copy_width = min(width, mask.shape[1])
    canvas[:copy_height, :copy_width][mask[:copy_height, :copy_width]] = 0
    return canvas


def generate_cumulative(
    data: bytes,
    options: ConversionOptions,
) -> CumulativeGeneration:
    normalized = options.normalized()
    if normalized.profile in LINE_ART_PROFILES:
        return generate_p6rg1(data, normalized)
    if normalized.font_size != REFERENCE_FONT_SIZE:
        return _fallback_generation(
            data,
            normalized,
            reason="surface-model-requires-reference-font-size",
        )
    try:
        processed, line_channels, surface_channels, crop, input_stage = (
            _prepare_surface_model_inputs(data, normalized)
        )
        decoded = decode_surface_text(line_channels, surface_channels)
        if decoded.all_whitespace_rows:
            return _fallback_generation(
                data,
                normalized,
                reason=(
                    "surface-model-produced-all-whitespace-on-nonempty-rows:"
                    + ",".join(str(row) for row in decoded.all_whitespace_rows)
                ),
            )
        rendered = _render_surface_model_text(
            decoded.text,
            width=decoded.target_width,
            height=processed.shape[0],
        )
        result = ConversionResult(
            ascii_text=decoded.text,
            rows=decoded.rows,
            columns=normalized.columns,
            processed_png=image_to_data_url(processed),
            rendered_png=image_to_data_url(rendered),
            crop=crop,
        )
        decoder_stage = {
            "stage": "deepaa-surface-ls-dense-decoder",
            "variant": "LS",
            "decoder_version": DECODER_VERSION,
            "inference_layout": "16-phase-shift-and-stitch-scanner-v1",
            "uses_target_character_identity": False,
            "uses_target_start_count": False,
            "target_width_px": decoded.target_width,
            "row_count": decoded.rows,
            "row_widths_px": list(decoded.row_widths),
            "start_counts": list(decoded.start_counts),
            "exact_row_widths": all(
                width == decoded.target_width for width in decoded.row_widths
            ),
            "all_whitespace_rows": list(decoded.all_whitespace_rows),
            "model_sha256": decoded.model_sha256,
            "vocabulary_sha256": decoded.vocabulary_sha256,
            "checkpoint_sha256": decoded.checkpoint_sha256,
            "applied": True,
            "mechanical_gate": True,
        }
        return CumulativeGeneration(
            result=result,
            pipeline={
                "generator_version": GENERATOR_VERSION,
                "recipe_version": RECIPE_VERSION,
                "enabled": True,
                "applied": True,
                "fallback_used": False,
                "fallback_reason": None,
                "fallback_generator_version": P6RG1_GENERATOR_VERSION,
                "fallback_recipe_version": P6RG1_RECIPE_VERSION,
                "stages": [input_stage, decoder_stage],
            },
        )
    except Exception as error:
        return _fallback_generation(
            data,
            normalized,
            reason=f"surface-model-inference-failed:{type(error).__name__}:{error}",
        )


def regenerate_p6rg1(data: bytes, options: ConversionOptions) -> ConversionResult:
    return generate_p6rg1(data, options).result


def regenerate_cumulative(data: bytes, options: ConversionOptions) -> ConversionResult:
    return generate_cumulative(data, options).result
