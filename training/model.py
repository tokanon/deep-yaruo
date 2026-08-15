from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class DeepAAMultitask(nn.Module):
    """DeepAA light backbone with character and character-start heads."""

    def __init__(self, class_count: int = 411, *, input_channels: int = 1) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        channels = (input_channels, 16, 32, 64, 128)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(channels[index], channels[index + 1], 3, padding=1)
                for index in range(4)
            ]
        )
        self.normalizations = nn.ModuleList(
            [nn.BatchNorm2d(channel, eps=0.001) for channel in channels[1:]]
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.character_head = nn.Linear(4 * 4 * 128, class_count)
        self.start_head = nn.Linear(4 * 4 * 128, 1)
        nn.init.zeros_(self.start_head.weight)
        nn.init.zeros_(self.start_head.bias)

    def features(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for convolution, normalization in zip(
            self.convolutions, self.normalizations
        ):
            value = self.pool(torch.relu(normalization(convolution(value))))
        # DeepAA was trained in Keras with channels-last Flatten ordering.
        return value.permute(0, 2, 3, 1).contiguous().view(value.shape[0], -1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs)
        return self.character_head(features), self.start_head(features).squeeze(1)


def load_pretrained_backbone(model: DeepAAMultitask, onnx_path: Path) -> None:
    graph = onnx.load(str(onnx_path)).graph
    tensors = {
        tensor.name: numpy_helper.to_array(tensor).copy()
        for tensor in graph.initializer
    }
    with torch.no_grad():
        for index, (convolution, normalization) in enumerate(
            zip(model.convolutions, model.normalizations), start=1
        ):
            prefix = f"conv2d_{index}"
            source_weight = torch.from_numpy(tensors[f"{prefix}.kernel"])
            if index == 1 and convolution.in_channels > 1:
                convolution.weight.zero_()
                convolution.weight[:, :1].copy_(source_weight)
            else:
                convolution.weight.copy_(source_weight)
            convolution.bias.copy_(torch.from_numpy(tensors[f"{prefix}.bias"]))
            prefix = f"batch_normalization_{index}"
            normalization.weight.copy_(torch.from_numpy(tensors[f"{prefix}.gamma"]))
            normalization.bias.copy_(torch.from_numpy(tensors[f"{prefix}.beta"]))
            normalization.running_mean.copy_(torch.from_numpy(tensors[f"{prefix}.mean"]))
            normalization.running_var.copy_(
                torch.from_numpy(tensors[f"{prefix}.variance"])
            )
        model.character_head.weight.copy_(
            torch.from_numpy(tensors["predictions.kernel"].T.copy())
        )
        model.character_head.bias.copy_(torch.from_numpy(tensors["predictions.bias"]))


def verify_against_onnx(
    model: DeepAAMultitask,
    onnx_path: Path,
    *,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    nhwc = rng.random((4, 64, 64, 1), dtype=np.float32)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    expected = session.run(["probabilities"], {"input": nhwc})[0]
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(nhwc).permute(0, 3, 1, 2)
        logits, _ = model(inputs)
        actual = torch.softmax(logits, dim=1).cpu().numpy()
    return float(np.max(np.abs(expected - actual)))


def export_multitask_onnx(model: DeepAAMultitask, output_path: Path) -> None:
    model.eval().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 64, 64),),
        str(output_path),
        input_names=["input_nchw"],
        output_names=["character_logits", "start_logits"],
        dynamic_axes={
            "input_nchw": {0: "batch"},
            "character_logits": {0: "batch"},
            "start_logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the PyTorch DeepAA import.")
    parser.add_argument(
        "--onnx", type=Path, default=ROOT / "models" / "deepaa-light.onnx"
    )
    args = parser.parse_args()
    model = DeepAAMultitask()
    load_pretrained_backbone(model, args.onnx)
    difference = verify_against_onnx(model, args.onnx)
    print(f"Maximum probability difference: {difference:.9g}")
    if difference > 1e-5:
        raise SystemExit("Imported weights do not reproduce the ONNX model.")


if __name__ == "__main__":
    main()
