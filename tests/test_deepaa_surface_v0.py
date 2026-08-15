from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from training.deepaa_surface_v0 import (
    DeepAASurfaceDenseScanner,
    DeepAASurfaceV0,
    extract_channel_window,
    load_deepaa_line_encoder,
    real_image_input_channels,
    surface_input_channels,
)
from training.evaluate_deepaa_surface_v0 import _decode_exact_width


ROOT = Path(__file__).resolve().parents[1]


def test_surface_channels_do_not_depend_on_ordinal_id_values() -> None:
    ids = np.asarray([[0, 4, 4], [9, 9, 4]], dtype=np.int32)
    remapped = np.asarray([[0, 27, 27], [3, 3, 27]], dtype=np.int32)
    tone = np.asarray([[0, 30, 30], [90, 90, 30]], dtype=np.uint8)
    first = surface_input_channels(ids, tone)
    second = surface_input_channels(remapped, tone)
    assert np.array_equal(first, second)
    assert first.shape == (3, 2, 3)
    assert np.all(first[2, ids == 0] == 0)


def test_channel_window_uses_channel_specific_padding() -> None:
    source = np.ones((3, 18, 8), dtype=np.float32)
    window = extract_channel_window(source, x=0, y=0, fill_value=0.0)
    assert window.shape == (3, 64, 64)
    assert np.all(window[:, :23, :] == 0)
    assert np.all(window[:, 23:41, 23:31] == 1)


def test_real_image_adapter_matches_reverse_channel_value_contract() -> None:
    line = np.asarray([[255, 0], [128, 255]], dtype=np.uint8)
    ids = np.asarray([[0, 2], [2, 2]], dtype=np.int32)
    tone = np.asarray([[0, 64], [64, 64]], dtype=np.uint8)
    line_channels, surface_channels = real_image_input_channels(line, ids, tone)
    assert line_channels.shape == (1, 2, 2)
    assert surface_channels.shape == (3, 2, 2)
    assert line_channels[0, 0, 0] == 1.0
    assert line_channels[0, 0, 1] == 0.0
    assert np.allclose(surface_channels[2, ids > 0], 64 / 255)


def test_l_and_ls_have_the_approved_inputs_and_three_heads() -> None:
    line = torch.rand(2, 1, 64, 64)
    surface = torch.rand(2, 3, 64, 64)
    for variant in ("L", "LS"):
        model = DeepAASurfaceV0(37, variant=variant)
        character, start, role = model(line, surface)
        assert character.shape == (2, 37)
        assert start.shape == (2,)
        assert role.shape == (2, 2)


def test_dense_scanner_matches_the_ls_model_for_one_exact_window() -> None:
    model = DeepAASurfaceV0(37, variant="LS")
    scanner = DeepAASurfaceDenseScanner(model)
    model.eval()
    scanner.eval()
    line = torch.rand(2, 1, 64, 64)
    surface = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = model(line, surface)
        actual = scanner(line, surface)
    for expected_head, actual_head in zip(expected, actual, strict=True):
        assert torch.allclose(expected_head, actual_head[:, 0], atol=1e-5)


def test_line_encoder_import_is_identical_for_l_and_ls() -> None:
    path = ROOT / "models" / "deepaa-light.onnx"
    line = torch.rand(2, 1, 64, 64)
    l_model = DeepAASurfaceV0(10, variant="L")
    ls_model = DeepAASurfaceV0(10, variant="LS")
    load_deepaa_line_encoder(l_model.line_encoder, path)
    load_deepaa_line_encoder(ls_model.line_encoder, path)
    l_model.eval()
    ls_model.eval()
    with torch.no_grad():
        assert torch.equal(l_model.line_encoder(line), ls_model.line_encoder(line))


def test_sequence_decoder_reaches_exact_width_without_target_identity() -> None:
    characters = ["a", "b", "c"]
    advances = np.asarray([3, 4, 5], dtype=np.int16)
    groups = {
        3: np.asarray([0]),
        4: np.asarray([1]),
        5: np.asarray([2]),
    }
    scores = np.log(
        np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.8, 0.1, 0.1],
            ],
            dtype=np.float32,
        )
    )
    decoded = _decode_exact_width(
        scores,
        characters=characters,
        advances=advances,
        advance_groups=groups,
        target_width=8,
        top_k=1,
    )
    assert decoded is not None
    assert sum(int(advances[characters.index(character)]) for character in decoded) == 8
