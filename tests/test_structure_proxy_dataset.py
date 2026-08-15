from __future__ import annotations

from pathlib import Path

from training.structure_proxy_dataset import (
    VocabularyEntry,
    frequency_band,
    load_vocabulary,
    text_character_examples,
)


def test_text_character_examples_keep_stable_coordinates_with_p5_augment() -> None:
    # P5 render perturbations are selected after labels are mapped.  This guards
    # the intended invariant: no augmentation-specific coordinate path exists.
    vocabulary = (
        VocabularyEntry(label=0, char=" ", frequency=1000, frequency_band="1000+"),
        VocabularyEntry(label=1, char="/", frequency=1000, frequency_band="1000+"),
    )
    examples, diagnostics = text_character_examples(
        " / /\n",
        vocabulary,
        source_width=64,
        target_width=64,
    )
    assert [item["x"] for item in examples] == sorted(item["x"] for item in examples)
    assert diagnostics["normalized_coordinate_collisions"] == 0


ROOT = Path(__file__).resolve().parents[1]


def test_deepaa_vocabulary_and_frequency_bands_are_explicit() -> None:
    vocabulary = load_vocabulary(ROOT / "models" / "deepaa-charset.csv")

    assert len(vocabulary) == 411
    assert vocabulary[0].frequency >= vocabulary[-1].frequency
    assert {entry.frequency_band for entry in vocabulary} == {
        "1000+",
        "100-999",
        "20-99",
        "10-19",
    }
    assert frequency_band(1000) == "1000+"
    assert frequency_band(100) == "100-999"
    assert frequency_band(20) == "20-99"
    assert frequency_band(10) == "10-19"


def test_text_character_examples_preserve_advances_and_report_oov() -> None:
    vocabulary = load_vocabulary(ROOT / "models" / "deepaa-charset.csv")
    examples, diagnostics = text_character_examples(
        " A█\n/ ",
        vocabulary,
        source_width=128,
        target_width=512,
    )

    assert diagnostics["total_characters"] == 5
    assert diagnostics["out_of_vocabulary_characters"] >= 1
    assert diagnostics["included_characters"] == len(examples)
    first_line = [example for example in examples if example["y"] == 0]
    assert first_line[0]["char"] == " "
    assert first_line[1]["char"] == "A"
    assert first_line[1]["x"] > first_line[0]["x"]
    second_line = [example for example in examples if example["y"] > 0]
    assert second_line[0]["char"] == "/"
