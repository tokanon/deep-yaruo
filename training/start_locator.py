from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


class StartLocator(nn.Module):
    """Lightweight fully convolutional character-start locator."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        channels = (16, 24, 32, 48, 64)
        dilations = (1, 2, 4, 8)
        blocks: list[nn.Module] = []
        for index, dilation in enumerate(dilations):
            blocks.extend(
                [
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        3,
                        stride=(2, 1),
                        padding=(1, dilation),
                        dilation=(1, dilation),
                    ),
                    nn.BatchNorm2d(channels[index + 1]),
                    nn.ReLU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(64, 1, kernel_size=(4, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(inputs))
        return self.head(features).squeeze(1).squeeze(1)


def export_start_locator(model: StartLocator, output_path: Path) -> None:
    model.eval().cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.ones(1, 1, 64, 256),),
        str(output_path),
        input_names=["line_context"],
        output_names=["start_logits"],
        dynamic_axes={
            "line_context": {0: "batch", 3: "width"},
            "start_logits": {0: "batch", 1: "width"},
        },
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the start locator shape.")
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    model = StartLocator()
    output = model(torch.ones(2, 1, 64, args.width))
    print(f"input=(2, 1, 64, {args.width}) output={tuple(output.shape)}")


if __name__ == "__main__":
    main()
