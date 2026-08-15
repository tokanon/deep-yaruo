from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.contracts import ConversionOptions
from backend.correction_pairs import CorrectionPairStore, CorrectionPairSubmission
from backend.surface_decisions import (
    analyze_correction_pair_surfaces,
    analyze_surface_pair,
)
from backend.surface_proposals import SurfaceProposalConfig
from training.evaluate_surface_decisions_p6r4 import evaluate_correction_pairs


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def proposal_config() -> SurfaceProposalConfig:
    return SurfaceProposalConfig(
        target_width=32,
        line_pitch=18,
        minimum_area=1,
        minimum_area_fraction=0.0,
        maximum_area_fraction=0.99,
        max_proposals_per_layer=16,
        color_blur_sigma=0.0,
        color_quantizations=((4, 4, 4),),
        lineart_closing_kernels=(3,),
    )


def color_pair_image() -> np.ndarray:
    image = np.full((18, 32, 3), 245, dtype=np.uint8)
    image[:, :18] = (20, 20, 20)
    return image


def test_color_rule_matches_a_dark_corrected_fill_surface() -> None:
    result = analyze_surface_pair(
        color_pair_image(),
        ";;;;;;\n",
        record_id="synthetic-color",
        source_kind="color",
        crop_box_xyxy=(0, 0, 32, 18),
        canvas_size_wh=(32, 18),
        font_size=16,
        proposal_config=proposal_config(),
    )

    correspondence = result["surface_correspondence"]
    metrics = result["surface_metrics"]["layers"][0]
    assert correspondence["corrected_fill_surface_count"] == 1
    assert correspondence["layers"][0]["overlap_count"] >= 1
    assert metrics["action_counts"] == {"fill": 1, "reject": 1, "abstain": 0}
    assert metrics["evaluation"]["surface_metrics"]["precision"] == 1.0
    assert metrics["evaluation"]["surface_metrics"]["recall"] == 1.0
    assert metrics["evaluation"]["pixel_metrics"]["iou"] == 1.0


def test_lineart_rule_abstains_when_tone_is_unavailable() -> None:
    lineart = np.full((18, 32), 255, dtype=np.uint8)
    cv2.rectangle(lineart, (1, 1), (18, 16), 0, 1)

    result = analyze_surface_pair(
        lineart,
        ";;;;;;\n",
        record_id="synthetic-lineart",
        source_kind="lineart",
        crop_box_xyxy=(0, 0, 32, 18),
        canvas_size_wh=(32, 18),
        font_size=16,
        proposal_config=proposal_config(),
    )

    metrics = result["surface_metrics"]["layers"][0]
    assert metrics["rule_status"] == "abstain-lineart-tone-unavailable"
    assert metrics["action_counts"]["fill"] == 0
    assert metrics["action_counts"]["abstain"] == 1
    assert metrics["evaluation"]["surface_metrics"]["missing_count"] == 1


def test_correction_pair_surface_analysis_updates_all_slots_atomically(
    tmp_path: Path,
) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    image = color_pair_image()
    blank = np.full((18, 32), 255, dtype=np.uint8)
    saved = store.save(
        CorrectionPairSubmission(
            source_bytes=encode_png(image),
            source_filename="color-pair.png",
            source_media_type="image/png",
            source_origin="test-generated",
            rights_status="test-only",
            options=ConversionOptions(columns=32, profile="auto"),
            crop=(0, 0, 32, 18),
            draft_text="......\n",
            corrected_text=";;;;;;\n",
            processed_png=encode_png(blank),
            draft_rendered_png=encode_png(blank),
        )
    )
    record_id = saved["manifest"]["record_id"]
    record_path = store.root / record_id / "record.json"
    manifest_before = record_path.read_bytes()

    analyzed = analyze_correction_pair_surfaces(
        store,
        record_id,
        proposal_config=proposal_config(),
    )

    assert analyzed["extension_revision"] == 1
    assert analyzed["source_kind"] == "color"
    assert analyzed["corrected_fill_surface_count"] == 1
    extensions = store.get(record_id)["extensions"]
    assert all(extensions["data"][slot] is not None for slot in extensions["data"])
    assert extensions["data"]["surface_proposals"]["record_id"] == record_id
    assert extensions["data"]["surface_metrics"]["paired_gold_status"].startswith(
        "evaluated"
    )
    assert record_path.read_bytes() == manifest_before


def test_p6r4_report_refuses_to_claim_a_gate_without_pairs(tmp_path: Path) -> None:
    report = evaluate_correction_pairs(CorrectionPairStore(tmp_path / "empty"))

    assert report["record_count"] == 0
    assert report["status"] == "blocked-no-correction-pairs"
    assert report["quality_gate_passed"] is None
    assert report["quality_gate_status"] == "not evaluated"
    assert "person correction pairs: 0/10" in report["missing_conditions"]
    assert "background correction pairs: 0/10" in report["missing_conditions"]
