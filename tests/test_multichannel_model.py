from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.model import DeepAAMultitask, load_pretrained_backbone


ROOT = Path(__file__).resolve().parents[1]


def test_c2_initialization_exactly_preserves_the_c1_backbone() -> None:
    onnx_path = ROOT / "models" / "deepaa-light.onnx"
    c1 = DeepAAMultitask(input_channels=1)
    c2 = DeepAAMultitask(input_channels=2)
    load_pretrained_backbone(c1, onnx_path)
    load_pretrained_backbone(c2, onnx_path)
    c1.eval()
    c2.eval()
    structure = torch.rand(4, 1, 64, 64)
    arbitrary_tone = torch.rand(4, 1, 64, 64)
    with torch.no_grad():
        c1_logits, _ = c1(structure)
        c2_logits, _ = c2(torch.cat((structure, arbitrary_tone), dim=1))
    assert torch.equal(c1_logits, c2_logits)
    assert torch.count_nonzero(c2.convolutions[0].weight[:, 1:]) == 0


def test_multichannel_model_rejects_zero_inputs() -> None:
    try:
        DeepAAMultitask(input_channels=0)
    except ValueError as exc:
        assert "input_channels" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("zero input channels should fail")
