from __future__ import annotations

import cv2
import numpy as np
import pytest

import backend.surface_fill as surface_fill
from backend.contracts import ConversionOptions, ConversionResult
from backend.image_io import image_to_data_url
from backend.rendering import glyph_advance
from backend.surface_proposals import (
    SurfaceProposal,
    SurfaceProposalLayer,
    SurfaceProposalSet,
)


def _source_png() -> bytes:
    ramp = np.linspace(150, 230, 120, dtype=np.uint8)
    image = np.repeat(ramp[None, :, None], 18, axis=0)
    image = np.repeat(image, 3, axis=2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _proposal(
    label: int,
    x0: int,
    x1: int,
    *,
    gray_mean: float,
    surrounding_gray_delta: float,
    surrounding_lab_delta: float,
    compactness: float,
) -> SurfaceProposal:
    return SurfaceProposal(
        proposal_id=f"lab-l4-a4-b4:{label:04d}",
        layer_id="lab-l4-a4-b4",
        region_label=label,
        method="lab_connected_region",
        area_px=(x1 - x0) * 18,
        area_fraction=(x1 - x0) / 120,
        bbox_xyxy=(x0, 0, x1, 18),
        centroid_xy=((x0 + x1 - 1) / 2, 8.5),
        perimeter_px=2 * ((x1 - x0) + 18),
        compactness=compactness,
        touches_domain_border=False,
        domain_border_contact_px=0,
        gray_mean=gray_mean,
        gray_std=0.0,
        gray_quantiles=(gray_mean, gray_mean, gray_mean),
        surrounding_gray_contrast=abs(surrounding_gray_delta),
        surrounding_gray_delta=surrounding_gray_delta,
        lab_mean=(gray_mean, 128.0, 128.0),
        surrounding_lab_delta=surrounding_lab_delta,
        adjacent_proposal_ids=(),
        parameters={"lab_levels": [4, 4, 4]},
    )


def _proposal_set() -> SurfaceProposalSet:
    labels = np.zeros((18, 120), dtype=np.int32)
    labels[:, 0:35] = 1
    labels[:, 40:75] = 2
    labels[:, 80:115] = 3
    layer = SurfaceProposalLayer(
        layer_id="lab-l4-a4-b4",
        method="lab_connected_region",
        labels=labels,
        proposals=(
            _proposal(
                1,
                0,
                35,
                gray_mean=165.0,
                surrounding_gray_delta=64.0,
                surrounding_lab_delta=64.0,
                compactness=1.0,
            ),
            _proposal(
                2,
                40,
                75,
                gray_mean=195.0,
                surrounding_gray_delta=38.4,
                surrounding_lab_delta=51.2,
                compactness=0.8,
            ),
            _proposal(
                3,
                80,
                115,
                gray_mean=220.0,
                surrounding_gray_delta=38.4,
                surrounding_lab_delta=32.0,
                compactness=0.8,
            ),
        ),
        diagnostics={},
    )
    return SurfaceProposalSet(
        schema_version=1,
        coordinate_space_id="test-space",
        source_kind="color",
        source_image_shape_hw=(18, 120),
        source_crop_box_xyxy=(0, 0, 120, 18),
        image_shape_hw=(18, 120),
        content_shape_hw=(18, 120),
        layers=(layer,),
        cross_layer_relations=(),
        config=surface_fill.SURFACE_PROPOSAL_CONFIG,
    )


def _install_fixed_baseline(monkeypatch: pytest.MonkeyPatch) -> str:
    baseline_text = " " * 24
    blank = image_to_data_url(np.full((18, 120), 255, dtype=np.uint8))

    def fake_convert(data: bytes, options: ConversionOptions) -> ConversionResult:
        assert data
        assert options == options.normalized()
        return ConversionResult(
            ascii_text=baseline_text,
            rows=1,
            columns=24,
            processed_png=blank,
            rendered_png=blank,
            crop=(0, 0, 120, 18),
        )

    monkeypatch.setattr(surface_fill, "convert_deepaa", fake_convert)
    monkeypatch.setattr(
        surface_fill,
        "extract_surface_proposals",
        lambda *args, **kwargs: _proposal_set(),
    )
    return baseline_text


def test_surface_fill_comparison_shares_one_baseline_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_text = _install_fixed_baseline(monkeypatch)
    source = _source_png()
    options = ConversionOptions(columns=24, max_rows=12, profile="person").normalized()

    first = surface_fill.generate_surface_fill_comparison(source, options)
    second = surface_fill.generate_surface_fill_comparison(source, options)

    assert first == second
    assert first["comparisonKind"] == "surface-fill-v2"
    assert first["recipeVersion"] == "surface-fill-v2"
    assert len(first["candidates"]) == 4
    assert {item["displayLabel"] for item in first["candidates"]} == {
        "候補A",
        "候補B",
        "候補C",
        "候補D",
    }
    by_id = {item["variantId"]: item for item in first["candidates"]}
    assert by_id["surface-fill-none-v2"]["result"]["ascii"] == baseline_text
    assert {
        item["surface"]["minimum_fill_score"] for item in first["candidates"]
    } == {None, 0.30}
    fill_candidates = [
        item
        for item in first["candidates"]
        if item["surface"]["minimum_fill_score"] is not None
    ]
    assert all(
        tuple(item["surface"]["action_counts"].values()) == (3, 0, 0)
        for item in fill_candidates
    )
    relative_values = [
        decision["features"]["relative_darkness"]
        for decision in fill_candidates[0]["surface"]["decisions"]
    ]
    assert relative_values[0] >= 0.67
    assert 0.33 <= relative_values[1] < 0.67
    assert relative_values[2] < 0.33
    assert len({item["fillMaskPng"] for item in fill_candidates}) == 1

    baseline_width = sum(glyph_advance(character) for character in baseline_text)
    shared_processed = {
        item["result"]["processedPng"] for item in first["candidates"]
    }
    shared_options = {
        tuple(sorted(item["result"]["options"].items()))
        for item in first["candidates"]
    }
    shared_crop = {tuple(item["result"]["crop"]) for item in first["candidates"]}
    assert len(shared_processed) == len(shared_options) == len(shared_crop) == 1
    for candidate in first["candidates"]:
        assert sum(
            glyph_advance(character) for character in candidate["result"]["ascii"]
        ) == baseline_width
        assert candidate["result"]["columns"] == options.columns
        if candidate["surface"]["minimum_fill_score"] is not None:
            placement = candidate["surface"]["placement_config"]
            assert placement["relative_darkness_edges"] == [0.33, 0.67]
            assert placement["boundary_margin_px"] == 1
            assert placement["minimum_run"] == 3
        for replacement in candidate["surface"]["fill_summary"]["replacements"]:
            assert replacement["relative_darkness_band"] in {"dark", "middle", "light"}

    by_strictness = [
        by_id[variant]["surface"]["fill_summary"]["replaced_source_cell_count"]
        for variant in (
            "surface-fill-conservative-v2",
            "surface-fill-standard-v2",
            "surface-fill-expanded-v2",
        )
    ]
    assert by_strictness == sorted(by_strictness)
    standard_gates = by_id["surface-fill-standard-v2"]["surface"][
        "placement_config"
    ]["tone_gates"]
    expanded_gates = by_id["surface-fill-expanded-v2"]["surface"][
        "placement_config"
    ]["tone_gates"]
    assert [gate["structure_fraction_maximum"] for gate in expanded_gates] == [
        gate["structure_fraction_maximum"] for gate in standard_gates
    ] == [0.08, 0.06, 0.04]
    assert all(
        expanded["fill_fraction_minimum"] < standard["fill_fraction_minimum"]
        and expanded["safe_fill_fraction_minimum"]
        < standard["safe_fill_fraction_minimum"]
        for expanded, standard in zip(expanded_gates, standard_gates, strict=True)
    )


def test_surface_fill_comparison_rejects_lineart_before_generation(monkeypatch) -> None:
    called = False

    def fake_convert(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("line-art must be rejected before generation")

    monkeypatch.setattr(surface_fill, "convert_deepaa", fake_convert)
    with pytest.raises(ValueError, match="color source profile"):
        surface_fill.generate_surface_fill_comparison(
            b"not-an-image",
            ConversionOptions(profile="lineart"),
        )
    assert called is False
