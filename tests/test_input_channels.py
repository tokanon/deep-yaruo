from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.input_channels import (
    ChannelExtractionConfig,
    extract_input_channels,
)
from training.prepare_reference_channels import ROOT, _load_cases, _stratification


def _config(**overrides: object) -> ChannelExtractionConfig:
    values: dict[str, object] = {
        "target_width": 72,
        "line_pitch": 18,
        "tone_window_width": 9,
        "tone_window_height": 9,
        "tone_sigma": 2.0,
        "tone_gamma": 1.0,
        "tone_levels": 4,
        "fill_erosion_radius": 2,
        "fill_min_area": 9,
        "fill_max_hole": 4,
    }
    values.update(overrides)
    return ChannelExtractionConfig(**values)


def test_common_geometry_is_shared_and_uses_the_aa_line_grid() -> None:
    source = np.full((31, 50), 255, dtype=np.uint8)
    cv2.line(source, (4, 4), (45, 26), 0, 2)

    channels = extract_input_channels(
        source,
        source_kind="lineart",
        config=_config(),
    )

    assert channels.resized_gray.shape[1] == 72
    assert channels.resized_gray.shape[0] % 18 == 0
    assert channels.stacked.shape == (3, *channels.resized_gray.shape)
    assert channels.stacked.dtype == np.float32
    statistics = channels.metadata["channel_statistics"]
    assert isinstance(statistics, dict)
    assert sum(statistics["tone_level_fractions"]) == pytest.approx(1.0, abs=1e-5)


def test_tone_channel_has_only_the_configured_quantization_levels() -> None:
    gradient = np.tile(np.arange(72, dtype=np.uint8) * 3, (36, 1))

    channels = extract_input_channels(
        gradient,
        source_kind="grayscale",
        config=_config(),
    )

    quantized_indices = np.rint(channels.tone * 3).astype(np.uint8)
    assert set(int(value) for value in np.unique(quantized_indices)) <= {0, 1, 2, 3}
    assert np.allclose(channels.tone, quantized_indices.astype(np.float32) / 3.0)


def test_tone_gamma_makes_sparse_aa_density_visible_before_quantization() -> None:
    source = np.full((36, 72), 255, dtype=np.uint8)
    cv2.line(source, (4, 18), (68, 18), 0, 1)

    linear = extract_input_channels(
        source,
        source_kind="aa_proxy",
        config=_config(tone_window_width=24, tone_window_height=27, tone_gamma=1.0),
    )
    compressed = extract_input_channels(
        source,
        source_kind="aa_proxy",
        config=_config(tone_window_width=24, tone_window_height=27, tone_gamma=0.5),
    )

    assert np.count_nonzero(compressed.tone) > np.count_nonzero(linear.tone)


def test_tone_gamma_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tone_gamma"):
        _config(tone_gamma=0).validate()


def test_fill_reconstruction_keeps_a_dark_area_but_removes_a_thin_line() -> None:
    source = np.full((36, 72), 255, dtype=np.uint8)
    cv2.rectangle(source, (8, 8), (28, 27), 0, -1)
    cv2.line(source, (28, 18), (60, 18), 0, 1)
    cv2.line(source, (40, 4), (40, 14), 0, 1)

    channels = extract_input_channels(
        source,
        source_kind="lineart",
        config=_config(dark_threshold=128),
    )

    assert channels.fill[15, 15] == 1.0
    assert channels.fill[18, 50] == 0.0
    assert channels.fill[8, 40] == 0.0
    assert channels.structure[18, 50] == 1.0
    assert channels.structure[18, 40] == 1.0


def test_lineart_and_aa_proxy_share_the_same_structure_normalizer() -> None:
    source = np.full((36, 72), 255, dtype=np.uint8)
    cv2.circle(source, (36, 18), 11, 0, 3)

    lineart = extract_input_channels(
        source,
        source_kind="lineart",
        config=_config(),
    )
    proxy = extract_input_channels(
        source,
        source_kind="aa_proxy",
        config=_config(),
    )

    assert np.array_equal(lineart.structure, proxy.structure)
    assert np.array_equal(lineart.tone, proxy.tone)
    assert np.array_equal(lineart.fill, proxy.fill)


def test_reference_manifest_requires_explicit_local_rights_metadata() -> None:
    dataset, training_allowed, source_root, cases = _load_cases(
        ROOT / "samples",
        ROOT / "samples" / "p1-reference.example.json",
    )

    assert dataset == "p1-reference-local-v1"
    assert training_allowed is False
    assert source_root == ROOT / "samples"
    assert len(cases) == 4
    assert all(case["rights_status"] == "local-user-provided" for case in cases)


def test_reference_gate_requires_count_and_all_declared_strata() -> None:
    records = []
    for index in range(30):
        records.append(
            {
                "category": ("person", "background")[index % 2],
                "modality": ("source", "lineart")[index % 2],
                "style": ("photo", "illustration", "lineart")[index % 3],
                "tone_range": ("low", "medium", "high")[index % 3],
            }
        )

    report = _stratification(records)

    assert report["gate_passed"] is True
    assert report["missing_strata"] == {}
