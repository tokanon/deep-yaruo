from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from .contracts import ConversionOptions, ConversionResult
from .deepaa import convert_deepaa, render_predictions
from .fill_postprocess import FillStage1Config, apply_fill_stage1
from .image_io import decode_color_image, image_to_data_url
from .surface_decisions import SurfaceDecisionConfig, decide_surface_layer
from .surface_proposals import (
    SurfaceProposalConfig,
    extract_surface_proposals,
    normalize_surface_geometry,
)


COMPARISON_KIND = "surface-fill-v2"
GENERATOR_VERSION = "deepaa-beam-v0.2"
RECIPE_VERSION = "surface-fill-v2"
SURFACE_LAYER_ID = "lab-l4-a4-b4"
MINIMUM_FILL_SCORE = 0.30
RELATIVE_DARKNESS_EDGES = (0.33, 0.67)

# Aggregate FillRun counts from accepted-v1. This contains no AA text.
EMPIRICAL_FILL_RUN_COUNTS = {
    ":": 1835,
    ".": 339,
    ";": 177,
    "'": 93,
    ",": 49,
    '"': 35,
}

SURFACE_PROPOSAL_CONFIG = SurfaceProposalConfig(
    color_quantizations=((4, 4, 4),),
)

@dataclass(frozen=True)
class TonePlacementGate:
    band_id: str
    relative_darkness_minimum: float
    relative_darkness_maximum: float
    fill_fraction_minimum: float
    safe_fill_fraction_minimum: float
    structure_fraction_maximum: float

    def placement_config(self) -> FillStage1Config:
        return FillStage1Config(
            minimum_run=3,
            minimum_empirical_runs=20,
            fill_fraction_minimum=self.fill_fraction_minimum,
            safe_fill_fraction_minimum=self.safe_fill_fraction_minimum,
            structure_fraction_maximum=self.structure_fraction_maximum,
            boundary_margin_px=1,
            coverage_scale=4.5,
        )


@dataclass(frozen=True)
class SurfaceFillRecipe:
    variant_id: str
    placement_gates: tuple[TonePlacementGate, ...]


def _placement_gates(
    *,
    dark: tuple[float, float, float],
    middle: tuple[float, float, float],
    light: tuple[float, float, float],
) -> tuple[TonePlacementGate, ...]:
    low, high = RELATIVE_DARKNESS_EDGES
    return (
        TonePlacementGate("dark", high, 1.000001, *dark),
        TonePlacementGate("middle", low, high, *middle),
        TonePlacementGate("light", 0.0, low, *light),
    )


SURFACE_FILL_RECIPES = (
    SurfaceFillRecipe("surface-fill-none-v2", ()),
    SurfaceFillRecipe(
        "surface-fill-conservative-v2",
        _placement_gates(
            dark=(0.70, 0.50, 0.06),
            middle=(0.80, 0.60, 0.04),
            light=(0.90, 0.70, 0.02),
        ),
    ),
    SurfaceFillRecipe(
        "surface-fill-standard-v2",
        _placement_gates(
            dark=(0.60, 0.40, 0.08),
            middle=(0.70, 0.50, 0.06),
            light=(0.80, 0.60, 0.04),
        ),
    ),
    SurfaceFillRecipe(
        "surface-fill-expanded-v2",
        _placement_gates(
            dark=(0.50, 0.30, 0.08),
            middle=(0.60, 0.40, 0.06),
            light=(0.70, 0.50, 0.04),
        ),
    ),
)


def _relative_darkness_reference(
    source: np.ndarray,
    *,
    crop: tuple[int, int, int, int],
    target_shape_hw: tuple[int, int],
) -> tuple[float, float]:
    normalized, valid, _ = normalize_surface_geometry(
        source,
        SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=crop,
        target_shape_hw=target_shape_hw,
    )
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    values = gray[valid]
    if values.size == 0:
        raise ValueError("Surface-fill crop contains no valid pixels")
    low, high = (float(value) for value in np.percentile(values, (10, 90)))
    if high - low < 1.0:
        return 0.0, 255.0
    return low, high


def _apply_tone_adaptive_fill(
    text: str,
    *,
    structure: np.ndarray,
    tone: np.ndarray,
    relative_darkness: np.ndarray,
    selected_surface: np.ndarray,
    recipe: SurfaceFillRecipe,
    canvas_width: int,
    font_size: int,
) -> tuple[str, dict[str, object]]:
    current_text = text
    band_summaries: list[dict[str, object]] = []
    replacements: list[dict[str, object]] = []
    eligible_cell_count = 0
    replaced_source_cell_count = 0
    replaced_pixel_width = 0
    vocabulary: list[dict[str, object]] = []
    for gate in recipe.placement_gates:
        band_mask = (
            (selected_surface >= 0.5)
            & (relative_darkness >= gate.relative_darkness_minimum)
            & (relative_darkness < gate.relative_darkness_maximum)
        ).astype(np.float32)
        result = apply_fill_stage1(
            current_text,
            structure=structure,
            tone=tone,
            fill=band_mask,
            empirical_run_counts=EMPIRICAL_FILL_RUN_COUNTS,
            canvas_width=canvas_width,
            font_size=font_size,
            config=gate.placement_config(),
        )
        summary = result.summary()
        tagged_replacements = [
            {**item, "relative_darkness_band": gate.band_id}
            for item in summary["replacements"]
        ]
        band_summaries.append(
            {
                "band_id": gate.band_id,
                "relative_darkness_minimum": gate.relative_darkness_minimum,
                "relative_darkness_maximum": gate.relative_darkness_maximum,
                "selected_pixel_count": int(np.count_nonzero(band_mask)),
                "eligible_cell_count": summary["eligible_cell_count"],
                "replaced_source_cell_count": summary["replaced_source_cell_count"],
                "replacement_count": summary["replacement_count"],
                "replaced_pixel_width": summary["replaced_pixel_width"],
            }
        )
        replacements.extend(tagged_replacements)
        eligible_cell_count += int(summary["eligible_cell_count"])
        replaced_source_cell_count += int(summary["replaced_source_cell_count"])
        replaced_pixel_width += int(summary["replaced_pixel_width"])
        if not vocabulary:
            vocabulary = list(summary["vocabulary"])
        current_text = result.text
    return current_text, {
        "cell_count": sum(len(line) for line in text.splitlines()),
        "eligible_cell_count": eligible_cell_count,
        "replaced_source_cell_count": replaced_source_cell_count,
        "replacement_count": len(replacements),
        "replaced_pixel_width": replaced_pixel_width,
        "vocabulary": vocabulary,
        "replacements": replacements,
        "band_summaries": band_summaries,
    }


def _decode_png_data_url(data_url: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("Expected a PNG data URL")
    payload = base64.b64decode(data_url[len(prefix) :], validate=True)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("PNG data URL could not be decoded")
    return image


def _render_text(text: str, *, width: int, height: int) -> np.ndarray:
    lines = text.rstrip("\n").split("\n") if text else [""]
    rendered = np.asarray(render_predictions([list(line) for line in lines], width))
    canvas = np.full((height, width), 255, dtype=np.uint8)
    visible_height = min(height, rendered.shape[0])
    visible_width = min(width, rendered.shape[1])
    canvas[:visible_height, :visible_width] = rendered[:visible_height, :visible_width]
    return canvas


def _line_diagnostics(
    baseline: np.ndarray,
    filled: np.ndarray,
    structure: np.ndarray,
    selected_surface: np.ndarray,
) -> dict[str, object]:
    if not (
        baseline.shape == filled.shape == structure.shape == selected_surface.shape
    ):
        raise ValueError("Surface-fill diagnostic images must share one shape")
    kernel = np.ones((3, 3), np.uint8)
    baseline_ink = baseline < 128
    filled_ink = filled < 128
    selected = selected_surface >= 0.5
    structure_support = cv2.dilate(
        (structure >= 0.5).astype(np.uint8), kernel
    ).astype(bool)
    baseline_structure = baseline_ink & structure_support
    filled_nearby = cv2.dilate(filled_ink.astype(np.uint8), kernel).astype(bool)
    retained = baseline_structure & filled_nearby
    added = filled_ink & ~baseline_ink
    removed = baseline_ink & ~filled_ink
    structure_count = int(baseline_structure.sum())
    return {
        "added_ink_pixels": int(added.sum()),
        "removed_ink_pixels": int(removed.sum()),
        "outside_selected_surface_added_fraction": round(
            float((added & ~selected).sum()) / max(1, int(added.sum())), 6
        ),
        "main_line_retention_at_1px": round(
            float((retained).sum()) / max(1, structure_count), 6
        ),
        "main_line_loss_at_1px": round(
            1.0 - float((retained).sum()) / max(1, structure_count), 6
        ),
    }


def _mask_data_url(mask: np.ndarray) -> str:
    return image_to_data_url(np.where(mask >= 0.5, 255, 0).astype(np.uint8))


def _stable_recipe_order(source_payload: bytes) -> list[SurfaceFillRecipe]:
    source_hash = hashlib.sha256(source_payload).hexdigest()
    return sorted(
        SURFACE_FILL_RECIPES,
        key=lambda recipe: hashlib.sha256(
            f"{source_hash}|{recipe.variant_id}".encode("utf-8")
        ).digest(),
    )


def generate_surface_fill_comparison(
    data: bytes,
    options: ConversionOptions,
) -> dict[str, object]:
    normalized = options.normalized()
    if normalized.profile in {"lineart", "background_lineart"}:
        raise ValueError(
            "Surface-fill comparison requires a color source profile; "
            "line-art tone is unavailable."
        )

    baseline = convert_deepaa(data, normalized)
    processed = _decode_png_data_url(baseline.processed_png)
    baseline_rendered = _decode_png_data_url(baseline.rendered_png)
    if baseline_rendered.shape != processed.shape:
        raise ValueError("Baseline line and rendered images must share one shape")
    structure = 1.0 - processed.astype(np.float32) / 255.0
    source = decode_color_image(data)
    coordinate_space_id = (
        f"surface-fill:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
    )
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=coordinate_space_id,
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=baseline.crop,
        target_shape_hw=processed.shape,
    )
    layer = next(
        (item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID),
        None,
    )
    if layer is None:
        raise ValueError(f"Surface proposal layer was not generated: {SURFACE_LAYER_ID}")

    gray_p10, gray_p90 = _relative_darkness_reference(
        source,
        crop=baseline.crop,
        target_shape_hw=processed.shape,
    )
    active_decision = replace(
        SurfaceDecisionConfig(),
        minimum_fill_score=MINIMUM_FILL_SCORE,
    )
    common_decisions, common_predicted = decide_surface_layer(
        layer,
        source_kind="color",
        config=active_decision,
    )
    selected_surface_common = (common_predicted > 0).astype(np.float32)
    tone = np.zeros(processed.shape, dtype=np.float32)
    relative_darkness = np.zeros(processed.shape, dtype=np.float32)
    gray_span = max(1.0, gray_p90 - gray_p10)
    for decision in common_decisions:
        features = decision["features"]
        assert isinstance(features, dict)
        absolute_darkness = float(features["darkness"])
        gray_mean = 255.0 * (1.0 - absolute_darkness)
        relative_value = float(np.clip((gray_p90 - gray_mean) / gray_span, 0.0, 1.0))
        features["relative_darkness"] = round(relative_value, 6)
        if decision["action"] != "fill":
            continue
        label = int(decision["region_label"])
        mask = layer.labels == label
        tone[mask] = absolute_darkness
        relative_darkness[mask] = relative_value

    generated_by_id: dict[str, dict[str, object]] = {}
    for recipe in SURFACE_FILL_RECIPES:
        if not recipe.placement_gates:
            selected_surface = np.zeros(processed.shape, dtype=np.float32)
            decisions: list[dict[str, object]] = []
            text = baseline.ascii_text
            rendered = baseline_rendered
            fill_summary = {
                "cell_count": sum(len(line) for line in text.splitlines()),
                "eligible_cell_count": 0,
                "replaced_source_cell_count": 0,
                "replacement_count": 0,
                "replaced_pixel_width": 0,
                "vocabulary": [],
                "replacements": [],
            }
            action_counts = {"fill": 0, "reject": 0, "abstain": 0}
            decision_config: dict[str, object] | None = None
            placement_config: dict[str, object] | None = None
        else:
            decisions = common_decisions
            selected_surface = selected_surface_common
            text, fill_summary = _apply_tone_adaptive_fill(
                baseline.ascii_text,
                structure=structure,
                tone=tone,
                relative_darkness=relative_darkness,
                selected_surface=selected_surface,
                recipe=recipe,
                canvas_width=processed.shape[1],
                font_size=normalized.font_size,
            )
            rendered = (
                baseline_rendered
                if text == baseline.ascii_text
                else _render_text(text, width=processed.shape[1], height=processed.shape[0])
            )
            action_counts = {
                action: sum(decision["action"] == action for decision in decisions)
                for action in ("fill", "reject", "abstain")
            }
            decision_config = asdict(active_decision)
            placement_config = {
                "relative_darkness_edges": list(RELATIVE_DARKNESS_EDGES),
                "boundary_margin_px": 1,
                "minimum_run": 3,
                "minimum_empirical_runs": 20,
                "coverage_scale": 4.5,
                "tone_gates": [asdict(gate) for gate in recipe.placement_gates],
            }

        generated_by_id[recipe.variant_id] = {
            "variantId": recipe.variant_id,
            "methodVersion": 2,
            "result": {
                "ascii": text,
                "rows": baseline.rows,
                "columns": baseline.columns,
                "processedPng": baseline.processed_png,
                "renderedPng": image_to_data_url(rendered),
                "crop": list(baseline.crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(selected_surface),
            "surface": {
                "layer_id": layer.layer_id,
                "proposal_labels_sha256": hashlib.sha256(
                    np.ascontiguousarray(layer.labels, dtype="<i4").tobytes()
                ).hexdigest(),
                "minimum_fill_score": (
                    None if not recipe.placement_gates else MINIMUM_FILL_SCORE
                ),
                "decision_config": decision_config,
                "proposal_config": asdict(SURFACE_PROPOSAL_CONFIG),
                "placement_config": placement_config,
                "grayscale_reference": {
                    "method": "crop-valid-p10-p90-v1",
                    "p10": round(gray_p10, 6),
                    "p90": round(gray_p90, 6),
                },
                "action_counts": action_counts,
                "decisions": decisions,
                "fill_summary": fill_summary,
                "line_diagnostics": _line_diagnostics(
                    baseline_rendered,
                    rendered,
                    structure,
                    selected_surface,
                ),
            },
        }

    ordered: list[dict[str, object]] = []
    for index, recipe in enumerate(_stable_recipe_order(data)):
        candidate = generated_by_id[recipe.variant_id]
        candidate["displayLabel"] = f"候補{chr(ord('A') + index)}"
        ordered.append(candidate)
    return {
        "comparisonKind": COMPARISON_KIND,
        "generatorVersion": GENERATOR_VERSION,
        "recipeVersion": RECIPE_VERSION,
        "coordinateSpaceId": coordinate_space_id,
        "candidates": ordered,
    }


def baseline_result_from_comparison(comparison: dict[str, object]) -> ConversionResult:
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Surface-fill comparison has no candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("variantId") != "surface-fill-none-v2":
            continue
        result = candidate.get("result")
        if not isinstance(result, dict):
            break
        crop = result.get("crop")
        if not isinstance(crop, list) or len(crop) != 4:
            break
        return ConversionResult(
            ascii_text=str(result["ascii"]),
            rows=int(result["rows"]),
            columns=int(result["columns"]),
            processed_png=str(result["processedPng"]),
            rendered_png=str(result["renderedPng"]),
            crop=tuple(int(value) for value in crop),
        )
    raise ValueError("Surface-fill comparison has no baseline candidate")
