from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .reconstruction import load_placements
from .torch_dataset import _read_image


VERTICAL_MARGIN = 23


def extract_line_context(
    image: np.ndarray,
    *,
    y: int,
    x: int,
    width: int,
) -> np.ndarray:
    context = np.full((64, width), 255, dtype=np.uint8)
    top = y - VERTICAL_MARGIN
    source_top = max(0, top)
    source_bottom = min(image.shape[0], top + 64)
    source_left = max(0, x)
    source_right = min(image.shape[1], x + width)
    if source_top < source_bottom and source_left < source_right:
        destination_top = source_top - top
        context[
            destination_top : destination_top + source_bottom - source_top,
            source_left - x : source_right - x,
        ] = image[source_top:source_bottom, source_left:source_right]
    return context


class StartLocatorDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        csv_path: Path,
        manifest_path: Path,
        *,
        split: str,
        segment_width: int = 256,
        seed: int = 42,
        random_crop: bool = False,
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = {
            record["file_name"]: record
            for record in manifest["works"]
            if record["split"] == split
        }
        works, _ = load_placements(csv_path)
        self.lines: list[tuple[str, int, tuple[int, ...]]] = []
        for file_name in sorted(records):
            starts_by_y: dict[int, set[int]] = defaultdict(set)
            for item in works[file_name]:
                if item.label < 411:
                    starts_by_y[item.y].add(item.x)
            self.lines.extend(
                (file_name, y, tuple(sorted(starts)))
                for y, starts in sorted(starts_by_y.items())
            )
        self.image_paths = {
            name: manifest_path.parent / record["image"]
            for name, record in records.items()
        }
        self.segment_width = segment_width
        self.seed = seed
        self.random_crop = random_crop
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_name, y, starts = self.lines[index]
        image = _read_image(str(self.image_paths[file_name]))
        maximum_x = max(0, image.shape[1] - self.segment_width)
        if self.random_crop and maximum_x:
            rng = np.random.default_rng(
                self.seed + self.epoch * len(self.lines) + index
            )
            crop_x = int(rng.integers(0, maximum_x + 1))
        else:
            crop_x = maximum_x // 2
        context = extract_line_context(
            image,
            y=y,
            x=crop_x,
            width=self.segment_width,
        )
        labels = np.zeros(self.segment_width, dtype=np.float32)
        for start in starts:
            local = start - crop_x
            if 0 <= local < self.segment_width:
                labels[max(0, local - 1) : min(self.segment_width, local + 2)] = 1.0
        valid_width = min(self.segment_width, max(0, image.shape[1] - crop_x))
        mask = np.zeros(self.segment_width, dtype=bool)
        mask[:valid_width] = True
        return (
            torch.from_numpy(context.copy()).unsqueeze(0).float() / 255.0,
            torch.from_numpy(labels),
            torch.from_numpy(mask),
        )
