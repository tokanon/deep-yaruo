from __future__ import annotations

import cv2
import numpy as np

from backend.image_io import decode_color_image, decode_image
from backend.surface_proposals import (
    SurfaceProposalConfig,
    decode_label_spans,
    extract_surface_proposals,
    render_proposal_layer,
)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def config(**updates) -> SurfaceProposalConfig:
    values = {
        "target_width": 64,
        "line_pitch": 8,
        "minimum_area": 8,
        "minimum_area_fraction": 0.0,
        "maximum_area_fraction": 0.99,
        "max_proposals_per_layer": 32,
        "color_blur_sigma": 0.0,
        "color_quantizations": ((4, 4, 4),),
        "lineart_closing_kernels": (3,),
    }
    values.update(updates)
    return SurfaceProposalConfig(**values)


def test_color_decode_preserves_bgr_and_composites_alpha_on_white() -> None:
    bgra = np.zeros((2, 2, 4), dtype=np.uint8)
    bgra[0, 0] = (20, 40, 200, 255)
    bgra[0, 1] = (0, 0, 0, 0)
    payload = encode_png(bgra)

    color = decode_color_image(payload)

    assert color.shape == (2, 2, 3)
    assert color[0, 0].tolist() == [20, 40, 200]
    assert color[0, 1].tolist() == [255, 255, 255]
    assert np.array_equal(decode_image(payload), cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE))


def test_lab_regions_are_deterministic_adjacent_and_round_trip_as_spans() -> None:
    image = np.zeros((40, 64, 3), dtype=np.uint8)
    image[:, :32] = (220, 60, 40)
    image[:, 32:] = (30, 60, 220)

    first = extract_surface_proposals(
        image,
        source_kind="color",
        coordinate_space_id="synthetic-color",
        config=config(color_quantizations=((4, 4, 4), (6, 6, 6))),
    )
    second = extract_surface_proposals(
        image,
        source_kind="color",
        coordinate_space_id="synthetic-color",
        config=config(color_quantizations=((4, 4, 4), (6, 6, 6))),
    )
    layer = first.layers[0]

    assert len(layer.proposals) == 2
    assert layer.labels[20, 12] > 0
    assert layer.labels[20, 52] > 0
    assert layer.labels[20, 12] != layer.labels[20, 52]
    assert all(proposal.adjacent_proposal_ids for proposal in layer.proposals)
    assert np.array_equal(layer.labels, second.layers[0].labels)
    encoded = layer.to_dict(include_label_spans=True)["label_spans_yx0x1"]
    assert np.array_equal(decode_label_spans(layer.labels.shape, encoded), layer.labels)
    assert first.to_dict()["coordinate_space_id"] == "synthetic-color"
    assert first.cross_layer_relations[0]["overlap_count"] == 2
    assert all(
        proposal.surrounding_lab_delta is not None for proposal in layer.proposals
    )
    assert all(
        proposal.surrounding_gray_delta is not None for proposal in layer.proposals
    )

    preview = render_proposal_layer(image, first, layer)
    assert preview.shape == (40, 64, 3)
    assert not np.array_equal(preview, image)


def test_lineart_closed_regions_exclude_the_external_background() -> None:
    lineart = np.full((48, 64), 255, dtype=np.uint8)
    cv2.rectangle(lineart, (10, 8), (52, 38), 0, 1)
    original = lineart.copy()

    proposals = extract_surface_proposals(
        lineart,
        source_kind="lineart",
        coordinate_space_id="synthetic-lineart",
        config=config(),
    )
    layer = proposals.layers[0]

    assert np.array_equal(lineart, original)
    assert len(layer.proposals) == 1
    assert layer.labels[20, 20] == 1
    assert layer.labels[2, 2] == 0
    assert layer.proposals[0].touches_domain_border is False
    assert layer.diagnostics["external_background_components"] == 1
    assert layer.diagnostics["external_background_fraction"] > 0.4


def test_lineart_closing_is_used_only_for_candidate_detection() -> None:
    lineart = np.full((48, 64), 255, dtype=np.uint8)
    cv2.rectangle(lineart, (10, 8), (52, 38), 0, 1)
    lineart[8, 30] = 255

    proposals = extract_surface_proposals(
        lineart,
        source_kind="lineart",
        coordinate_space_id="closing-test",
        config=config(lineart_closing_kernels=(1, 3)),
    )

    assert len(proposals.layers[0].proposals) == 0
    assert len(proposals.layers[1].proposals) == 1
    assert proposals.layers[1].labels[20, 20] == 1


def test_crop_and_explicit_target_match_the_correction_pair_canvas() -> None:
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    image[5:45, 10:40] = (30, 60, 220)
    image[5:45, 40:70] = (220, 60, 40)

    proposals = extract_surface_proposals(
        image,
        source_kind="color",
        coordinate_space_id="correction-pair:synthetic:aa-canvas",
        config=config(),
        crop_box_xyxy=(10, 5, 70, 45),
        target_shape_hw=(18, 32),
    )

    assert proposals.source_image_shape_hw == (60, 80)
    assert proposals.source_crop_box_xyxy == (10, 5, 70, 45)
    assert proposals.image_shape_hw == (18, 32)
    assert proposals.content_shape_hw == (18, 32)
    assert proposals.layers[0].labels.shape == (18, 32)
    preview = render_proposal_layer(image, proposals, proposals.layers[0])
    assert preview.shape == (18, 32, 3)
