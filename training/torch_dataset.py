from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .examples import (
    TrainingExample,
    extract_context,
    negative_start_examples,
)
from .reconstruction import load_placements


@lru_cache(maxsize=512)
def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


class DeepAAMultitaskDataset(Dataset[tuple[torch.Tensor, int, float]]):
    def __init__(
        self,
        csv_path: Path,
        manifest_path: Path,
        *,
        split: str,
        class_count: int = 411,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root = manifest_path.parent
        records = {
            record["file_name"]: record
            for record in manifest["works"]
            if record["split"] == split
        }
        works, _ = load_placements(csv_path)
        positives: list[TrainingExample] = []
        negatives: list[TrainingExample] = []
        for file_name, record in records.items():
            placements = [
                item for item in works[file_name] if item.label < class_count
            ]
            positives.extend(
                TrainingExample(
                    file_name=file_name,
                    x=item.x,
                    y=item.y,
                    char_label=item.label,
                    start_label=1,
                    split=split,
                )
                for item in placements
            )
            negatives.extend(
                negative_start_examples(
                    placements,
                    split=split,
                    seed=seed + sum(ord(char) for char in file_name),
                )
            )
        self.examples = [
            example
            for pair in zip(positives, negatives)
            for example in pair
        ]
        if len(negatives) < len(positives):
            self.examples.extend(positives[len(negatives) :])
        self.image_paths = {
            name: self.root / record["image"] for name, record in records.items()
        }
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        example = self.examples[index]
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        x = example.x
        y = example.y
        if self.augment and example.start_label:
            x += int(rng.integers(-1, 2))
            y += int(rng.integers(-1, 2))
        image = _read_image(str(self.image_paths[example.file_name]))
        window = extract_context(image, x, y)
        tensor = torch.from_numpy(window.copy()).unsqueeze(0).float() / 255.0
        return tensor, example.char_label, float(example.start_label)


def subset_indices(length: int, maximum: int | None, seed: int) -> list[int] | None:
    if maximum is None or maximum >= length:
        return None
    return random.Random(seed).sample(range(length), maximum)
