from __future__ import annotations

import numpy as np

from training.aa_fill_layers import detect_fill_runs
from training.multichannel_dataset import apply_tone_nuisance
from training.structure_proxy_dataset import VocabularyEntry, text_character_examples


def test_character_indices_can_join_detected_fill_runs() -> None:
    text = "|::::::|\n"
    vocabulary = (
        VocabularyEntry(label=0, char="|", frequency=1000, frequency_band="1000+"),
        VocabularyEntry(label=1, char=":", frequency=1000, frequency_band="1000+"),
    )
    examples, _ = text_character_examples(
        text,
        vocabulary,
        source_width=64,
        target_width=64,
    )
    fill_positions = {
        (run.line_index, index)
        for run in detect_fill_runs(text)
        for index in range(run.start_index, run.end_index)
    }
    flags = [
        (item["line_index"], item["character_index"]) in fill_positions
        for item in examples
    ]
    assert flags == [False, True, True, True, True, True, True, False]


def test_tone_nuisance_is_deterministic_and_preserves_existing_dark_tone() -> None:
    preview = np.full((64, 64), 255, dtype=np.uint8)
    preview[20:30, 20:30] = 0
    first = apply_tone_nuisance(preview, "fixed", probability=1.0)
    second = apply_tone_nuisance(preview, "fixed", probability=1.0)

    assert np.array_equal(first, second)
    assert np.all(first[20:30, 20:30] == 0)
    assert np.count_nonzero(first < 255) > 100
    assert set(np.unique(first)).issubset({0, 85, 170, 255})
