from __future__ import annotations

from dataclasses import replace

import numpy as np

from backend.contracts import ConversionOptions
from backend.surface_decisions import SurfaceDecisionConfig, decide_surface_layer
from backend.surface_proposals import SurfaceProposal, SurfaceProposalLayer
from backend.surface_recovery import recover_local_contrast_lines


def test_local_contrast_recovery_preserves_current_lines_and_stays_bounded() -> None:
    source = np.full((72, 120), 220, dtype=np.uint8)
    source[:, :60] = 40
    current = np.full((72, 120), 255, dtype=np.uint8)
    current[10:62, 10:14] = 0

    recovered, diagnostics = recover_local_contrast_lines(
        source,
        crop=(0, 0, 120, 72),
        current_line_image=current,
        options=ConversionOptions(
            columns=24,
            detail=72,
            threshold_low=55,
            threshold_high=150,
            min_component=10,
            profile="person",
        ),
    )

    assert np.all(recovered[current < 128] < 128)
    assert diagnostics["added_line_pixels"] > 0
    assert diagnostics["current_line_retention"] == 1.0
    assert diagnostics["source_supported_added_fraction"] >= 0.90
    assert diagnostics["line_pixel_multiplier"] <= 1.5
    assert diagnostics["mechanical_gate"] is True


def test_contact_ratio_border_penalty_can_rescue_small_border_contact() -> None:
    labels = np.ones((18, 32), dtype=np.int32)
    proposal = SurfaceProposal(
        proposal_id="lab-l4-a4-b4:0001",
        layer_id="lab-l4-a4-b4",
        region_label=1,
        method="lab_connected_region",
        area_px=576,
        area_fraction=0.20,
        bbox_xyxy=(0, 0, 32, 18),
        centroid_xy=(15.5, 8.5),
        perimeter_px=100,
        compactness=0.50,
        touches_domain_border=True,
        domain_border_contact_px=2,
        gray_mean=89.25,
        gray_std=0.0,
        gray_quantiles=(89.25, 89.25, 89.25),
        surrounding_gray_contrast=0.0,
        surrounding_gray_delta=0.0,
        lab_mean=(89.25, 128.0, 128.0),
        surrounding_lab_delta=0.0,
        adjacent_proposal_ids=(),
        parameters={},
    )
    layer = SurfaceProposalLayer(
        layer_id="lab-l4-a4-b4",
        method="lab_connected_region",
        labels=labels,
        proposals=(proposal,),
        diagnostics={},
    )
    base = replace(SurfaceDecisionConfig(), minimum_fill_score=0.30)
    current, _ = decide_surface_layer(layer, source_kind="color", config=base)
    recovered, _ = decide_surface_layer(
        layer,
        source_kind="color",
        config=replace(base, border_penalty_mode="contact-ratio"),
    )

    assert current[0]["action"] == "reject"
    assert recovered[0]["action"] == "fill"
    assert recovered[0]["features"]["border_penalty"] < base.border_penalty
