from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from evaluation.surface_metrics import SurfaceSet, evaluate_surface_sets
from training.aa_fill_layers import (
    connect_fill_runs,
    detect_fill_runs,
    render_fill_surface_labels,
)

from .correction_pairs import CorrectionPairStore
from .image_io import decode_color_image
from .surface_proposals import (
    SurfaceProposal,
    SurfaceProposalConfig,
    SurfaceProposalLayer,
    SurfaceSourceKind,
    encode_label_spans,
    extract_surface_proposals,
)


@dataclass(frozen=True)
class SurfaceDecisionConfig:
    minimum_fill_score: float = 0.45
    darkness_weight: float = 0.55
    signed_contrast_weight: float = 0.25
    color_separation_weight: float = 0.10
    compactness_weight: float = 0.10
    border_penalty: float = 0.15
    border_penalty_mode: Literal["binary", "contact-ratio"] = "binary"
    border_full_contact_fraction: float = 0.25
    border_minimum_penalty_fraction: float = 0.0
    large_area_threshold: float = 0.50
    large_area_penalty: float = 0.15
    correspondence_overlap_threshold: float = 0.10

    def validate(self) -> None:
        if not 0.0 <= self.minimum_fill_score <= 1.0:
            raise ValueError("minimum_fill_score must be in [0, 1]")
        weights = (
            self.darkness_weight,
            self.signed_contrast_weight,
            self.color_separation_weight,
            self.compactness_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Surface decision weights cannot be negative")
        if self.border_penalty < 0.0 or self.large_area_penalty < 0.0:
            raise ValueError("Surface decision penalties cannot be negative")
        if self.border_penalty_mode not in {"binary", "contact-ratio"}:
            raise ValueError("Unknown border penalty mode")
        if not 0.0 < self.border_full_contact_fraction <= 1.0:
            raise ValueError("border_full_contact_fraction must be in (0, 1]")
        if not 0.0 <= self.border_minimum_penalty_fraction <= 1.0:
            raise ValueError("border_minimum_penalty_fraction must be in [0, 1]")
        if not 0.0 < self.large_area_threshold <= 1.0:
            raise ValueError("large_area_threshold must be in (0, 1]")
        if not 0.0 < self.correspondence_overlap_threshold <= 1.0:
            raise ValueError("correspondence_overlap_threshold must be in (0, 1]")


def _label_sha256(labels: np.ndarray) -> str:
    payload = np.ascontiguousarray(labels, dtype="<i4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _tone_level(darkness: float) -> int:
    if darkness < 0.20:
        return 0
    if darkness < 0.40:
        return 1
    if darkness < 0.65:
        return 2
    return 3


def _color_decision(
    proposal: SurfaceProposal,
    config: SurfaceDecisionConfig,
) -> dict[str, object]:
    darkness = _clip(1.0 - proposal.gray_mean / 255.0, 0.0, 1.0)
    signed_contrast = _clip(
        (proposal.surrounding_gray_delta or 0.0) / 64.0,
        -1.0,
        1.0,
    )
    color_separation = _clip(
        (proposal.surrounding_lab_delta or 0.0) / 64.0,
        0.0,
        1.0,
    )
    compactness = _clip(proposal.compactness, 0.0, 1.0)
    border_contact_fraction = _clip(
        proposal.domain_border_contact_px / max(1, proposal.perimeter_px),
        0.0,
        1.0,
    )
    if not proposal.touches_domain_border:
        border_penalty = 0.0
    elif config.border_penalty_mode == "contact-ratio":
        border_penalty = config.border_penalty * max(
            config.border_minimum_penalty_fraction,
            min(
                1.0,
                border_contact_fraction / config.border_full_contact_fraction,
            ),
        )
    else:
        border_penalty = config.border_penalty
    large_area_penalty = (
        config.large_area_penalty
        if proposal.area_fraction >= config.large_area_threshold
        else 0.0
    )
    score = (
        config.darkness_weight * darkness
        + config.signed_contrast_weight * signed_contrast
        + config.color_separation_weight * color_separation
        + config.compactness_weight * compactness
        - border_penalty
        - large_area_penalty
    )
    action = "fill" if score >= config.minimum_fill_score else "reject"
    reasons: list[str] = []
    if darkness >= 0.65:
        reasons.append("dark-region")
    if signed_contrast > 0.25:
        reasons.append("darker-than-surroundings")
    if proposal.touches_domain_border:
        reasons.append("domain-border-penalty")
    if proposal.area_fraction >= config.large_area_threshold:
        reasons.append("large-area-penalty")
    if not reasons:
        reasons.append("weak-fill-evidence")
    return {
        "proposal_id": proposal.proposal_id,
        "region_label": proposal.region_label,
        "action": action,
        "score": round(score, 6),
        "tone_level": _tone_level(darkness) if action == "fill" else None,
        "features": {
            "darkness": round(darkness, 6),
            "signed_surrounding_contrast": round(signed_contrast, 6),
            "color_separation": round(color_separation, 6),
            "compactness": round(compactness, 6),
            "border_penalty": border_penalty,
            "border_contact_fraction": round(border_contact_fraction, 6),
            "large_area_penalty": large_area_penalty,
        },
        "reasons": reasons,
    }


def decide_surface_layer(
    layer: SurfaceProposalLayer,
    *,
    source_kind: SurfaceSourceKind,
    config: SurfaceDecisionConfig | None = None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    active = config or SurfaceDecisionConfig()
    active.validate()
    decisions: list[dict[str, object]] = []
    selected_labels: list[int] = []
    for proposal in layer.proposals:
        if source_kind == "lineart":
            decision = {
                "proposal_id": proposal.proposal_id,
                "region_label": proposal.region_label,
                "action": "abstain",
                "score": None,
                "tone_level": None,
                "features": None,
                "reasons": ["lineart-tone-unavailable"],
            }
        else:
            decision = _color_decision(proposal, active)
        decisions.append(decision)
        if decision["action"] == "fill":
            selected_labels.append(proposal.region_label)
    predicted = np.where(
        np.isin(layer.labels, np.asarray(selected_labels, dtype=np.int32)),
        layer.labels,
        0,
    ).astype(np.int32)
    return decisions, predicted


def _overlap_records(
    reference_labels: np.ndarray,
    layer: SurfaceProposalLayer,
    *,
    minimum_overlap_fraction: float,
) -> list[dict[str, object]]:
    reference_areas = {
        int(label): int(np.count_nonzero(reference_labels == label))
        for label in np.unique(reference_labels)
        if label > 0
    }
    proposal_areas = {
        proposal.region_label: proposal.area_px for proposal in layer.proposals
    }
    shared = (reference_labels > 0) & (layer.labels > 0)
    if not np.any(shared):
        return []
    pairs, counts = np.unique(
        np.stack((reference_labels[shared], layer.labels[shared]), axis=1),
        axis=0,
        return_counts=True,
    )
    overlaps: list[dict[str, object]] = []
    for pair, count in zip(pairs, counts, strict=True):
        reference_label, proposal_label = (int(value) for value in pair)
        intersection = int(count)
        reference_area = reference_areas[reference_label]
        proposal_area = proposal_areas[proposal_label]
        overlap_fraction = intersection / min(reference_area, proposal_area)
        if overlap_fraction < minimum_overlap_fraction:
            continue
        union = reference_area + proposal_area - intersection
        overlaps.append(
            {
                "corrected_surface_label": reference_label,
                "corrected_surface_id": reference_label - 1,
                "proposal_id": f"{layer.layer_id}:{proposal_label:04d}",
                "intersection_px": intersection,
                "corrected_coverage": round(intersection / reference_area, 8),
                "proposal_coverage": round(intersection / proposal_area, 8),
                "iou": round(intersection / union, 8),
            }
        )
    return sorted(
        overlaps,
        key=lambda overlap: (
            int(overlap["corrected_surface_label"]),
            -float(overlap["iou"]),
            str(overlap["proposal_id"]),
        ),
    )


def analyze_surface_pair(
    source_image: np.ndarray,
    corrected_text: str,
    *,
    record_id: str,
    source_kind: SurfaceSourceKind,
    crop_box_xyxy: tuple[int, int, int, int],
    canvas_size_wh: tuple[int, int],
    font_size: int,
    proposal_config: SurfaceProposalConfig | None = None,
    decision_config: SurfaceDecisionConfig | None = None,
) -> dict[str, dict[str, object]]:
    active_decision = decision_config or SurfaceDecisionConfig()
    active_decision.validate()
    canvas_width, canvas_height = canvas_size_wh
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("AA canvas size must be positive")
    coordinate_space_id = f"correction-pair:{record_id}:aa-canvas-v1"
    correspondence_id = f"correction-pair:{record_id}:corrected-fill-v1"
    proposal_set = extract_surface_proposals(
        source_image,
        source_kind=source_kind,
        coordinate_space_id=coordinate_space_id,
        config=proposal_config,
        crop_box_xyxy=crop_box_xyxy,
        target_shape_hw=(canvas_height, canvas_width),
    )
    fill_runs = detect_fill_runs(corrected_text, font_size=font_size)
    fill_surfaces = connect_fill_runs(fill_runs, font_size=font_size)
    corrected_labels = render_fill_surface_labels(
        fill_surfaces,
        width=canvas_width,
        height=canvas_height,
    )

    proposal_payload = {
        "schema_version": 1,
        "record_id": record_id,
        "role": "input-surface-candidates-not-fill-decisions",
        "proposal_set": proposal_set.to_dict(include_label_spans=True),
    }
    correspondence_layers: list[dict[str, object]] = []
    metric_layers: list[dict[str, object]] = []
    for layer in proposal_set.layers:
        overlaps = _overlap_records(
            corrected_labels,
            layer,
            minimum_overlap_fraction=(
                active_decision.correspondence_overlap_threshold
            ),
        )
        correspondence_layers.append(
            {
                "layer_id": layer.layer_id,
                "minimum_overlap_fraction": (
                    active_decision.correspondence_overlap_threshold
                ),
                "overlap_count": len(overlaps),
                "overlaps": overlaps,
            }
        )
        decisions, predicted_labels = decide_surface_layer(
            layer,
            source_kind=source_kind,
            config=active_decision,
        )
        evaluation = evaluate_surface_sets(
            SurfaceSet(
                labels=corrected_labels,
                coordinate_space_id=coordinate_space_id,
                correspondence_id=correspondence_id,
            ),
            SurfaceSet(
                labels=predicted_labels,
                coordinate_space_id=coordinate_space_id,
                correspondence_id=correspondence_id,
            ),
        )
        action_counts = {
            action: sum(decision["action"] == action for decision in decisions)
            for action in ("fill", "reject", "abstain")
        }
        metric_layers.append(
            {
                "layer_id": layer.layer_id,
                "rule_status": (
                    "abstain-lineart-tone-unavailable"
                    if source_kind == "lineart"
                    else "evaluated-color-rule-v1"
                ),
                "action_counts": action_counts,
                "decisions": decisions,
                "predicted_labels_sha256": _label_sha256(predicted_labels),
                "evaluation": evaluation,
            }
        )

    correspondence_payload = {
        "schema_version": 1,
        "record_id": record_id,
        "correspondence_id": correspondence_id,
        "coordinate_space_id": coordinate_space_id,
        "reference_role": "corrected-aa-repeated-fill-surfaces",
        "corrected_fill_run_count": len(fill_runs),
        "corrected_fill_surface_count": len(fill_surfaces),
        "corrected_fill_surfaces": [
            surface.to_dict(include_runs=True) for surface in fill_surfaces
        ],
        "corrected_labels_sha256": _label_sha256(corrected_labels),
        "corrected_label_spans_yx0x1": encode_label_spans(corrected_labels),
        "layers": correspondence_layers,
    }
    metrics_payload = {
        "schema_version": 1,
        "record_id": record_id,
        "baseline": "transparent-rule-based-surface-decision-v1",
        "decision_config": asdict(active_decision),
        "paired_gold_status": "evaluated-against-corrected-aa-fill-surfaces",
        "limitations": [
            "repeated-character fill labels do not cover mixed-character hatching",
            "line-art tone is absent, so rule-v1 abstains instead of inventing tone",
            "line preservation after AA placement is not evaluated by this baseline",
        ],
        "layers": metric_layers,
    }
    return {
        "surface_proposals": proposal_payload,
        "surface_correspondence": correspondence_payload,
        "surface_metrics": metrics_payload,
    }


def analyze_correction_pair_surfaces(
    store: CorrectionPairStore,
    record_id: str,
    *,
    proposal_config: SurfaceProposalConfig | None = None,
    decision_config: SurfaceDecisionConfig | None = None,
) -> dict[str, object]:
    prepared = build_correction_pair_surface_analysis(
        store,
        record_id,
        proposal_config=proposal_config,
        decision_config=decision_config,
    )
    updates = prepared["updates"]
    extensions = store.update_extensions(record_id, updates)
    return {
        "record_id": record_id,
        "extension_revision": extensions["revision"],
        "source_kind": prepared["source_kind"],
        "subject_category": prepared["subject_category"],
        "surface_proposal_count": updates["surface_proposals"]["proposal_set"][
            "proposal_count"
        ],
        "corrected_fill_surface_count": updates["surface_correspondence"][
            "corrected_fill_surface_count"
        ],
        "extensions": extensions,
    }


def build_correction_pair_surface_analysis(
    store: CorrectionPairStore,
    record_id: str,
    *,
    proposal_config: SurfaceProposalConfig | None = None,
    decision_config: SurfaceDecisionConfig | None = None,
) -> dict[str, object]:
    record = store.get(record_id)
    manifest = record["manifest"]
    transform = manifest["coordinate_transform"]
    canvas_width, canvas_height = (
        int(value) for value in transform["aa_canvas_size_px"]
    )
    crop = tuple(int(value) for value in transform["source_crop_box_px"])
    profile = str(manifest["conversion"]["options"]["profile"])
    source_kind: SurfaceSourceKind = (
        "lineart" if profile in {"lineart", "background_lineart"} else "color"
    )
    if profile in {"person", "lineart"}:
        subject_category = "person"
    elif profile in {"background", "background_lineart"}:
        subject_category = "background"
    else:
        subject_category = "unknown"
    source_path, _ = store.asset(record_id, "source")
    source_image = decode_color_image(source_path.read_bytes())
    updates = analyze_surface_pair(
        source_image,
        str(record["corrected_text"]),
        record_id=record_id,
        source_kind=source_kind,
        crop_box_xyxy=crop,
        canvas_size_wh=(canvas_width, canvas_height),
        font_size=int(transform["font_size_px"]),
        proposal_config=proposal_config,
        decision_config=decision_config,
    )
    updates["surface_correspondence"]["stratification"] = {
        "subject_category": subject_category,
        "category_source": "conversion-profile",
        "profile": profile,
        "source_kind": source_kind,
    }
    return {
        "record_id": record_id,
        "source_kind": source_kind,
        "subject_category": subject_category,
        "profile": profile,
        "updates": updates,
    }
