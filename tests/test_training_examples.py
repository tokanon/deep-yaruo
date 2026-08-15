from __future__ import annotations

import numpy as np

from training.examples import (
    CONTEXT_MARGIN,
    augment_context,
    extract_context,
    negative_start_examples,
)
from training.reconstruction import Placement


def test_context_places_character_start_at_original_deepaa_margin() -> None:
    image = np.full((36, 80), 255, dtype=np.uint8)
    image[18:34, 20] = 0
    window = extract_context(image, x=20, y=18)
    assert window.shape == (64, 64)
    assert np.all(window[CONTEXT_MARGIN : CONTEXT_MARGIN + 16, CONTEXT_MARGIN] == 0)


def test_negative_starts_do_not_overlap_known_starts() -> None:
    placements = [
        Placement("work", 0, 10, "A", 0),
        Placement("work", 0, 15, "B", 1),
    ]
    negatives = list(negative_start_examples(placements, split="train", seed=42))
    starts = {10, 15}
    assert len(negatives) == 2
    assert all(all(abs(item.x - start) > 1 for start in starts) for item in negatives)
    assert all(item.char_label == -1 and item.start_label == 0 for item in negatives)


def test_augmentation_is_reproducible() -> None:
    window = np.full((64, 64), 255, dtype=np.uint8)
    window[23:39, 23] = 0
    first = augment_context(window, np.random.default_rng(7))
    second = augment_context(window, np.random.default_rng(7))
    assert np.array_equal(first, second)
