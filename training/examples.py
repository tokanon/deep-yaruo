from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .reconstruction import Placement, load_placements


CONTEXT_SIZE = 64
CONTEXT_MARGIN = 23


@dataclass(frozen=True)
class TrainingExample:
    file_name: str
    x: int
    y: int
    char_label: int
    start_label: int
    split: str


def load_split_map(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["file_name"]: record["split"] for record in manifest["works"]}


def positive_examples(
    csv_path: Path,
    manifest_path: Path,
    *,
    class_count: int = 411,
) -> list[TrainingExample]:
    works, _ = load_placements(csv_path)
    splits = load_split_map(manifest_path)
    examples: list[TrainingExample] = []
    for file_name, placements in works.items():
        if file_name not in splits:
            continue
        for item in placements:
            if item.label >= class_count:
                continue
            examples.append(
                TrainingExample(
                    file_name=file_name,
                    x=item.x,
                    y=item.y,
                    char_label=item.label,
                    start_label=1,
                    split=splits[file_name],
                )
            )
    return examples


def negative_start_examples(
    placements: list[Placement],
    *,
    split: str,
    seed: int,
) -> Iterator[TrainingExample]:
    """Yield one reproducible non-start location for each positive placement."""
    starts_by_row: dict[int, set[int]] = {}
    for item in placements:
        starts_by_row.setdefault(item.y, set()).add(item.x)

    rng = random.Random(seed)
    for item in placements:
        offsets = [-5, -4, -3, 3, 4, 5]
        rng.shuffle(offsets)
        for offset in offsets:
            candidate_x = item.x + offset
            starts = starts_by_row[item.y]
            if candidate_x < 0:
                continue
            if any(abs(candidate_x - start) <= 1 for start in starts):
                continue
            yield TrainingExample(
                file_name=item.file_name,
                x=candidate_x,
                y=item.y,
                char_label=-1,
                start_label=0,
                split=split,
            )
            break


def extract_context(image: np.ndarray, x: int, y: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Training images must be two-dimensional grayscale arrays.")
    window = np.full((CONTEXT_SIZE, CONTEXT_SIZE), 255, dtype=np.uint8)
    left = x - CONTEXT_MARGIN
    top = y - CONTEXT_MARGIN
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image.shape[1], left + CONTEXT_SIZE)
    source_bottom = min(image.shape[0], top + CONTEXT_SIZE)
    if source_left >= source_right or source_top >= source_bottom:
        return window
    destination_left = source_left - left
    destination_top = source_top - top
    window[
        destination_top : destination_top + source_bottom - source_top,
        destination_left : destination_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]
    return window


def augment_context(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a line-image variant without moving the supervised start position."""
    ink = (window < 128).astype(np.uint8) * 255
    operation = int(rng.integers(0, 3))
    if operation == 1:
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8), iterations=1)
    elif operation == 2:
        ink = cv2.erode(ink, np.ones((2, 2), np.uint8), iterations=1)
    result = 255 - ink
    sigma = float(rng.uniform(0.0, 0.8))
    if sigma >= 0.1:
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
    noise = rng.normal(0.0, rng.uniform(0.0, 7.0), size=result.shape)
    return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
