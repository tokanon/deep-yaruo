from __future__ import annotations

import numpy as np

from backend.input_channels import ChannelExtractionConfig
from training.structure_proxy import (
    canonical_aa_skeleton,
    extract_proxy_structure,
    generate_structure_proxy,
    structure_descriptor,
    symmetric_chamfer,
)


def _channel_config() -> ChannelExtractionConfig:
    return ChannelExtractionConfig(
        target_width=128,
        line_pitch=18,
        tone_window_width=12,
        tone_window_height=12,
        tone_sigma=2.0,
    )


def test_p2_proxy_methods_are_reproducible_and_share_the_common_x_grid() -> None:
    text = " /\\\n( ﾟｰﾟ)\n"
    config = _channel_config()

    raw, raw_parameters = generate_structure_proxy(text, "raw_raster")
    smooth_a, smooth_parameters = generate_structure_proxy(
        text,
        "supersampled_blur_downsample",
    )
    smooth_b, _ = generate_structure_proxy(text, "supersampled_blur_downsample")
    raw_structure = extract_proxy_structure(raw, channel_config=config)
    smooth_structure = extract_proxy_structure(smooth_a, channel_config=config)

    assert raw_parameters["method"] == "raw_raster"
    assert smooth_parameters["render_scale"] == 4
    assert np.array_equal(smooth_a, smooth_b)
    assert raw.shape == smooth_a.shape
    assert raw_structure.shape == smooth_structure.shape
    assert raw_structure.shape[1] == 128
    assert raw_structure.shape[0] % 18 == 0
    assert np.count_nonzero(raw_structure) > 0
    assert np.count_nonzero(smooth_structure) > 0


def test_symmetric_chamfer_is_zero_for_identity_and_positive_for_shift() -> None:
    reference = np.zeros((24, 32), dtype=np.float32)
    reference[5:19, 10] = 1.0
    shifted = np.zeros_like(reference)
    shifted[5:19, 13] = 1.0

    identity = symmetric_chamfer(reference, reference)
    displaced = symmetric_chamfer(reference, shifted)

    assert identity["symmetric_px"] == 0.0
    assert displaced["symmetric_px"] > 2.5
    assert displaced["reference_to_candidate_px"] > 2.5
    assert displaced["candidate_to_reference_px"] > 2.5


def test_structure_descriptor_has_fixed_shape_and_no_dimension_feature() -> None:
    small = np.zeros((36, 64), dtype=np.float32)
    large = np.zeros((72, 128), dtype=np.float32)
    small[8:28, 20] = 1.0
    large[16:56, 40] = 1.0

    small_descriptor = structure_descriptor(small)
    large_descriptor = structure_descriptor(large)

    assert small_descriptor.shape == large_descriptor.shape
    assert small_descriptor.shape == (305,)
    assert np.all(np.isfinite(small_descriptor))
    assert np.all(np.isfinite(large_descriptor))


def test_canonical_skeleton_and_proxy_structure_use_one_geometry() -> None:
    text = "----\n / /\n"
    rendered, _ = generate_structure_proxy(text, "raw_raster")
    config = _channel_config()

    reference = canonical_aa_skeleton(rendered, channel_config=config)
    candidate = extract_proxy_structure(rendered, channel_config=config)
    result = symmetric_chamfer(reference, candidate)

    assert reference.shape == candidate.shape
    assert np.isfinite(result["symmetric_px"])
