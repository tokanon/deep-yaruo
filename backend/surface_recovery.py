from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .contracts import ConversionOptions, ConversionResult
from .deepaa import convert_deepaa
from .image_io import decode_color_image, decode_image, image_to_data_url
from .rendering import glyph_advance
from .surface_decisions import SurfaceDecisionConfig, decide_surface_layer
from .surface_fill import (
    MINIMUM_FILL_SCORE,
    SURFACE_FILL_RECIPES,
    SURFACE_LAYER_ID,
    SURFACE_PROPOSAL_CONFIG,
    _apply_tone_adaptive_fill,
    _decode_png_data_url,
    _line_diagnostics,
    _mask_data_url,
    _relative_darkness_reference,
    _render_text,
)
from .surface_proposals import extract_surface_proposals


COMPARISON_KIND = "surface-recovery-v3"
GENERATOR_VERSION = "deepaa-beam-v0.2"
RECIPE_VERSION = "surface-recovery-v3"
STANDARD_RECIPE = next(
    recipe
    for recipe in SURFACE_FILL_RECIPES
    if recipe.variant_id == "surface-fill-standard-v2"
)
SAFE_BORDER_RESCUE_RECIPE = replace(
    STANDARD_RECIPE,
    variant_id="surface-border-rescue-safe-v3",
    placement_gates=tuple(
        replace(gate, structure_fraction_maximum=0.0)
        for gate in STANDARD_RECIPE.placement_gates
    ),
)


@dataclass(frozen=True)
class LineRecoveryConfig:
    clahe_clip_limit: float = 2.2
    canny_low_scale: float = 0.55
    canny_high_scale: float = 0.70
    gradient_percentile: float = 75.0
    maximum_line_pixel_multiplier: float = 1.5
    minimum_supported_fraction: float = 0.90


LINE_RECOVERY_CONFIG = LineRecoveryConfig()
CURRENT_SURFACE_DECISION = replace(
    SurfaceDecisionConfig(),
    minimum_fill_score=MINIMUM_FILL_SCORE,
)
CONTACT_RATIO_SURFACE_DECISION = replace(
    CURRENT_SURFACE_DECISION,
    border_penalty_mode="contact-ratio",
    border_minimum_penalty_fraction=2.0 / 3.0,
)


@dataclass(frozen=True)
class RecoveryVariant:
    variant_id: str
    recover_lines: bool
    recover_border_surfaces: bool


RECOVERY_VARIANTS = (
    RecoveryVariant("recovery-current-v3", False, False),
    RecoveryVariant("recovery-lines-v3", True, False),
    RecoveryVariant("recovery-surfaces-v3", False, True),
    RecoveryVariant("recovery-combined-v3", True, True),
)


def _minimum_component(options: ConversionOptions) -> int:
    if options.profile == "person":
        return max(1, options.min_component // 2)
    if options.profile == "background":
        return max(2, options.min_component)
    return max(1, options.min_component)


def recover_local_contrast_lines(
    source_gray: np.ndarray,
    *,
    crop: tuple[int, int, int, int],
    current_line_image: np.ndarray,
    options: ConversionOptions,
    config: LineRecoveryConfig = LINE_RECOVERY_CONFIG,
) -> tuple[np.ndarray, dict[str, object]]:
    """Add bounded, source-gradient-supported lines without removing current lines."""

    normalized = options.normalized()
    x0, y0, x1, y1 = crop
    cropped = source_gray[y0:y1, x0:x1]
    resized = cv2.resize(
        cropped,
        (current_line_image.shape[1], current_line_image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    sigma = 0.8 + (100 - normalized.detail) / 24.0
    profile_sigma = sigma * (0.68 if normalized.profile == "person" else 0.92)
    smoothed = cv2.GaussianBlur(
        resized,
        (0, 0),
        sigmaX=max(0.65, profile_sigma),
        sigmaY=max(0.65, profile_sigma),
    )
    tile_grid = (8, 8) if normalized.profile == "person" else (10, 8)
    equalized = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=tile_grid,
    ).apply(smoothed)
    low = max(0, int(round(normalized.threshold_low * config.canny_low_scale)))
    high = max(low + 1, int(round(normalized.threshold_high * config.canny_high_scale)))
    candidate = cv2.Canny(equalized, low, high, L2gradient=True)

    gx = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    positive = gradient[gradient > 0]
    gradient_threshold = (
        float(np.percentile(positive, config.gradient_percentile))
        if positive.size
        else float("inf")
    )
    support = cv2.dilate(
        (gradient >= gradient_threshold).astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    candidate_ink = (candidate > 0) & support
    current_ink = current_line_image < 128
    extra_ink = candidate_ink & ~current_ink

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_ink.astype(np.uint8),
        connectivity=8,
    )
    components: list[tuple[float, int, np.ndarray]] = []
    minimum_area = _minimum_component(normalized)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component_extra = (labels == label) & extra_ink
        added_area = int(component_extra.sum())
        if added_area == 0:
            continue
        components.append(
            (float(gradient[labels == label].mean()), added_area, component_extra)
        )
    components.sort(key=lambda item: (-item[0], -item[1]))

    current_count = int(current_ink.sum())
    addition_limit = max(
        1,
        int(current_count * (config.maximum_line_pixel_multiplier - 1.0)),
    )
    accepted_extra = np.zeros(current_ink.shape, dtype=bool)
    accepted_count = 0
    for _, added_area, component_extra in components:
        if accepted_count + added_area > addition_limit:
            continue
        accepted_extra |= component_extra
        accepted_count += added_area

    recovered_ink = current_ink | accepted_extra
    recovered = np.where(recovered_ink, 0, 255).astype(np.uint8)
    supported_added = int(np.count_nonzero(accepted_extra & support))
    support_fraction = supported_added / max(1, accepted_count)
    multiplier = int(recovered_ink.sum()) / max(1, current_count)
    diagnostics = {
        "method": "bounded-local-contrast-lines-v1",
        "config": asdict(config),
        "canny_thresholds": [low, high],
        "gradient_threshold": round(gradient_threshold, 6),
        "minimum_component_px": minimum_area,
        "current_line_pixels": current_count,
        "candidate_extra_line_pixels": int(extra_ink.sum()),
        "added_line_pixels": accepted_count,
        "recovered_line_pixels": int(recovered_ink.sum()),
        "current_line_retention": 1.0,
        "source_supported_added_fraction": round(support_fraction, 6),
        "line_pixel_multiplier": round(multiplier, 6),
        "mechanical_gate": bool(
            support_fraction >= config.minimum_supported_fraction
            and multiplier <= config.maximum_line_pixel_multiplier + 1e-9
        ),
    }
    return recovered, diagnostics


def _surface_state(
    layer,
    *,
    source: np.ndarray,
    crop: tuple[int, int, int, int],
    target_shape_hw: tuple[int, int],
    decision_config: SurfaceDecisionConfig,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    decisions, predicted = decide_surface_layer(
        layer,
        source_kind="color",
        config=decision_config,
    )
    selected = (predicted > 0).astype(np.float32)
    p10, p90 = _relative_darkness_reference(
        source,
        crop=crop,
        target_shape_hw=target_shape_hw,
    )
    span = max(1.0, p90 - p10)
    tone = np.zeros(target_shape_hw, dtype=np.float32)
    relative = np.zeros(target_shape_hw, dtype=np.float32)
    for decision in decisions:
        features = decision["features"]
        assert isinstance(features, dict)
        darkness = float(features["darkness"])
        gray_mean = 255.0 * (1.0 - darkness)
        value = float(np.clip((p90 - gray_mean) / span, 0.0, 1.0))
        features["relative_darkness"] = round(value, 6)
        if decision["action"] != "fill":
            continue
        mask = layer.labels == int(decision["region_label"])
        tone[mask] = darkness
        relative[mask] = value
    return decisions, selected, tone, relative, {"p10": p10, "p90": p90}


def _width_diagnostics(before: str, after: str, font_size: int) -> dict[str, object]:
    before_lines = before.rstrip("\n").split("\n")
    after_lines = after.rstrip("\n").split("\n")
    if len(before_lines) != len(after_lines):
        return {"exact_row_widths": False, "maximum_width_delta_px": None}
    deltas = []
    for first, second in zip(before_lines, after_lines, strict=True):
        first_width = sum(glyph_advance(character, font_size) for character in first)
        second_width = sum(glyph_advance(character, font_size) for character in second)
        deltas.append(abs(first_width - second_width))
    return {
        "exact_row_widths": all(delta == 0 for delta in deltas),
        "maximum_width_delta_px": max(deltas, default=0),
    }


def _merge_fill_summaries(
    initial: dict[str, object],
    rescue: dict[str, object],
) -> dict[str, object]:
    initial_bands = [
        {**band, "placement_phase": "current-surfaces"}
        for band in initial.get("band_summaries", [])
    ]
    rescue_bands = [
        {**band, "placement_phase": "border-rescue-surfaces"}
        for band in rescue.get("band_summaries", [])
    ]
    return {
        "cell_count": initial["cell_count"],
        "eligible_cell_count": int(initial["eligible_cell_count"])
        + int(rescue["eligible_cell_count"]),
        "replaced_source_cell_count": int(initial["replaced_source_cell_count"])
        + int(rescue["replaced_source_cell_count"]),
        "replacement_count": int(initial["replacement_count"])
        + int(rescue["replacement_count"]),
        "replaced_pixel_width": int(initial["replaced_pixel_width"])
        + int(rescue["replaced_pixel_width"]),
        "vocabulary": initial["vocabulary"] or rescue["vocabulary"],
        "replacements": [
            *initial.get("replacements", []),
            *rescue.get("replacements", []),
        ],
        "band_summaries": [*initial_bands, *rescue_bands],
    }


def _stable_variant_order(source_payload: bytes) -> list[RecoveryVariant]:
    source_hash = hashlib.sha256(source_payload).hexdigest()
    return sorted(
        RECOVERY_VARIANTS,
        key=lambda variant: hashlib.sha256(
            f"{source_hash}|{variant.variant_id}".encode("utf-8")
        ).digest(),
    )


def generate_surface_recovery_comparison(
    data: bytes,
    options: ConversionOptions,
) -> dict[str, object]:
    normalized = options.normalized()
    if normalized.profile in {"lineart", "background_lineart"}:
        raise ValueError("Surface recovery requires a color source profile.")

    current_result = convert_deepaa(data, normalized)
    current_line = _decode_png_data_url(current_result.processed_png)
    source_gray = decode_image(data)
    recovered_line, recovered_diagnostics = recover_local_contrast_lines(
        source_gray,
        crop=current_result.crop,
        current_line_image=current_line,
        options=normalized,
    )
    recovered_result = convert_deepaa(
        data,
        normalized,
        line_image_override=recovered_line,
        crop_box_override=current_result.crop,
    )
    if recovered_result.crop != current_result.crop:
        raise AssertionError("Recovered and current line branches must share one crop")

    source = decode_color_image(data)
    coordinate_space_id = (
        f"surface-recovery:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
    )
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=coordinate_space_id,
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=current_result.crop,
        target_shape_hw=current_line.shape,
    )
    layer = next(
        (item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID),
        None,
    )
    if layer is None:
        raise ValueError(f"Surface proposal layer was not generated: {SURFACE_LAYER_ID}")

    current_state = _surface_state(
        layer,
        source=source,
        crop=current_result.crop,
        target_shape_hw=current_line.shape,
        decision_config=CURRENT_SURFACE_DECISION,
    )
    recovered_state = _surface_state(
        layer,
        source=source,
        crop=current_result.crop,
        target_shape_hw=current_line.shape,
        decision_config=CONTACT_RATIO_SURFACE_DECISION,
    )
    current_actions = {
        int(decision["region_label"]): decision["action"]
        for decision in current_state[0]
    }
    rescued_labels = sorted(
        int(decision["region_label"])
        for decision in recovered_state[0]
        if decision["action"] == "fill"
        and current_actions.get(int(decision["region_label"])) != "fill"
    )

    generated: dict[str, dict[str, object]] = {}
    for variant in RECOVERY_VARIANTS:
        base: ConversionResult = recovered_result if variant.recover_lines else current_result
        processed = recovered_line if variant.recover_lines else current_line
        (
            current_decisions,
            current_selected,
            current_tone,
            current_relative,
            gray_reference,
        ) = current_state
        recovered_decisions, _, recovered_tone, recovered_relative, _ = recovered_state
        decisions = (
            recovered_decisions if variant.recover_border_surfaces else current_decisions
        )
        structure = 1.0 - processed.astype(np.float32) / 255.0
        text, initial_summary = _apply_tone_adaptive_fill(
            base.ascii_text,
            structure=structure,
            tone=current_tone,
            relative_darkness=current_relative,
            selected_surface=current_selected,
            recipe=STANDARD_RECIPE,
            canvas_width=processed.shape[1],
            font_size=normalized.font_size,
        )
        initial_text = text
        selected = current_selected
        fill_summary = initial_summary
        applied_rescued_labels: list[int] = []
        rescue_fallback_reason: str | None = None
        if variant.recover_border_surfaces and rescued_labels:
            rescued_selected = np.isin(
                layer.labels,
                np.asarray(rescued_labels, dtype=np.int32),
            ).astype(np.float32)
            text, rescue_summary = _apply_tone_adaptive_fill(
                text,
                structure=structure,
                tone=recovered_tone,
                relative_darkness=recovered_relative,
                selected_surface=rescued_selected,
                recipe=SAFE_BORDER_RESCUE_RECIPE,
                canvas_width=processed.shape[1],
                font_size=normalized.font_size,
            )
            selected = np.maximum(current_selected, rescued_selected)
            fill_summary = _merge_fill_summaries(initial_summary, rescue_summary)
            applied_rescued_labels = rescued_labels
        rendered = (
            _decode_png_data_url(base.rendered_png)
            if text == base.ascii_text
            else _render_text(text, width=processed.shape[1], height=processed.shape[0])
        )
        line_recovery = (
            recovered_diagnostics
            if variant.recover_lines
            else {
                "method": "current-lines-v2",
                "current_line_pixels": int(np.count_nonzero(current_line < 128)),
                "added_line_pixels": 0,
                "current_line_retention": 1.0,
                "source_supported_added_fraction": 1.0,
                "line_pixel_multiplier": 1.0,
                "mechanical_gate": True,
            }
        )
        widths = _width_diagnostics(base.ascii_text, text, normalized.font_size)
        fill_line = _line_diagnostics(
            _decode_png_data_url(base.rendered_png),
            rendered,
            structure,
            selected,
        )
        if (
            variant.recover_border_surfaces
            and fill_line["main_line_loss_at_1px"] > 0.02
        ):
            text = initial_text
            selected = current_selected
            fill_summary = initial_summary
            applied_rescued_labels = []
            rescue_fallback_reason = "main-line-loss-above-0.02"
            rendered = (
                _decode_png_data_url(base.rendered_png)
                if text == base.ascii_text
                else _render_text(
                    text,
                    width=processed.shape[1],
                    height=processed.shape[0],
                )
            )
            widths = _width_diagnostics(base.ascii_text, text, normalized.font_size)
            fill_line = _line_diagnostics(
                _decode_png_data_url(base.rendered_png),
                rendered,
                structure,
                selected,
            )
        mechanical_gate = bool(
            line_recovery["current_line_retention"] >= 0.98
            and line_recovery["source_supported_added_fraction"] >= 0.90
            and line_recovery["line_pixel_multiplier"] <= 1.5
            and widths["exact_row_widths"]
            and fill_line["main_line_loss_at_1px"] <= 0.02
        )
        action_counts = {
            action: sum(decision["action"] == action for decision in decisions)
            for action in ("fill", "reject", "abstain")
        }
        generated[variant.variant_id] = {
            "variantId": variant.variant_id,
            "methodVersion": 3,
            "result": {
                "ascii": text,
                "rows": base.rows,
                "columns": base.columns,
                "processedPng": image_to_data_url(processed),
                "renderedPng": image_to_data_url(rendered),
                "crop": list(base.crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(selected),
            "surface": {
                "layer_id": layer.layer_id,
                "minimum_fill_score": MINIMUM_FILL_SCORE,
                "decision_config": asdict(
                    CONTACT_RATIO_SURFACE_DECISION
                    if variant.recover_border_surfaces
                    else CURRENT_SURFACE_DECISION
                ),
                "proposal_config": asdict(SURFACE_PROPOSAL_CONFIG),
                "placement_config": {
                    "source_recipe": STANDARD_RECIPE.variant_id,
                    "tone_gates": [
                        asdict(gate) for gate in STANDARD_RECIPE.placement_gates
                    ],
                    "border_rescue_recipe": (
                        SAFE_BORDER_RESCUE_RECIPE.variant_id
                        if variant.recover_border_surfaces
                        else None
                    ),
                    "border_rescue_tone_gates": (
                        [
                            asdict(gate)
                            for gate in SAFE_BORDER_RESCUE_RECIPE.placement_gates
                        ]
                        if variant.recover_border_surfaces
                        else []
                    ),
                },
                "grayscale_reference": {
                    "method": "crop-valid-p10-p90-v1",
                    "p10": round(gray_reference["p10"], 6),
                    "p90": round(gray_reference["p90"], 6),
                },
                "action_counts": action_counts,
                "decisions": decisions,
                "fill_summary": fill_summary,
                "line_diagnostics": fill_line,
                "line_recovery": line_recovery,
                "border_surface_recovery": {
                    "enabled": variant.recover_border_surfaces,
                    "attempted_region_labels": rescued_labels
                    if variant.recover_border_surfaces
                    else [],
                    "attempted_surface_count": len(rescued_labels)
                    if variant.recover_border_surfaces
                    else 0,
                    "rescued_region_labels": applied_rescued_labels,
                    "rescued_surface_count": len(applied_rescued_labels),
                    "fallback_reason": rescue_fallback_reason,
                },
                "width_diagnostics": widths,
                "mechanical_gate": mechanical_gate,
            },
        }

    ordered: list[dict[str, object]] = []
    for index, variant in enumerate(_stable_variant_order(data)):
        candidate = generated[variant.variant_id]
        candidate["displayLabel"] = f"候補{chr(ord('A') + index)}"
        ordered.append(candidate)
    return {
        "comparisonKind": COMPARISON_KIND,
        "generatorVersion": GENERATOR_VERSION,
        "recipeVersion": RECIPE_VERSION,
        "coordinateSpaceId": coordinate_space_id,
        "candidates": ordered,
    }
