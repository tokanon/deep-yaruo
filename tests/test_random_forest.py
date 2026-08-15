from __future__ import annotations

import numpy as np

from backend.random_forest import CONTEXT_SIZE, character_digest, random_forest_features
from training.examples import TrainingExample
from training.random_forest import CLASS_COUNT, stratified_examples


def test_random_forest_features_preserve_binary_pixel_positions() -> None:
    windows = np.full((2, CONTEXT_SIZE, CONTEXT_SIZE), 255, dtype=np.uint8)
    windows[0, 5, 7] = 0
    features = random_forest_features(windows)
    assert features.shape == (2, CONTEXT_SIZE * CONTEXT_SIZE)
    assert features.dtype == np.uint8
    assert features[0, 5 * CONTEXT_SIZE + 7] == 1
    assert features[1].sum() == 0


def test_character_digest_depends_on_order() -> None:
    assert character_digest(("A", "B")) != character_digest(("B", "A"))


def test_stratified_examples_caps_every_class_reproducibly() -> None:
    examples = [
        TrainingExample(
            file_name=f"work-{label}-{index}",
            x=index,
            y=0,
            char_label=label,
            start_label=1,
            split="train",
        )
        for label in range(CLASS_COUNT)
        for index in range(3)
    ]
    first = stratified_examples(
        examples,
        split="train",
        maximum_per_class=2,
        seed=42,
    )
    second = stratified_examples(
        examples,
        split="train",
        maximum_per_class=2,
        seed=42,
    )
    assert first == second
    assert len(first) == CLASS_COUNT * 2
    assert set(item.char_label for item in first) == set(range(CLASS_COUNT))


def test_stratified_validation_allows_classes_missing_from_split() -> None:
    examples = [
        TrainingExample("work", 0, 0, 0, 1, "validation"),
        TrainingExample("work", 1, 0, 0, 1, "validation"),
    ]
    selected = stratified_examples(
        examples,
        split="validation",
        maximum_per_class=1,
        seed=42,
        require_all_classes=False,
    )
    assert len(selected) == 1
    assert selected[0].char_label == 0
