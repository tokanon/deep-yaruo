from __future__ import annotations

import numpy as np

from backend.fill_postprocess import (
    FillStage1Config,
    apply_fill_stage1,
    build_fill_vocabulary,
)
from backend.rendering import glyph_advance, line_pitch


EMPIRICAL_RUNS = {
    ":": 1_835,
    ".": 339,
    ";": 177,
    "'": 93,
    ",": 49,
    '"': 35,
    "@": 2,
}


def _width(text: str) -> int:
    return sum(glyph_advance(character) for character in text)


def test_fill_vocabulary_uses_only_empirically_supported_cp932_runs() -> None:
    vocabulary = build_fill_vocabulary(EMPIRICAL_RUNS)

    characters = {entry.character for entry in vocabulary}
    assert characters == {":", ".", ";", "'", ",", '"'}
    assert "@" not in characters
    assert all(entry.empirical_run_count >= 20 for entry in vocabulary)
    assert all(entry.advance == glyph_advance(entry.character) for entry in vocabulary)


def test_stage1_preserves_boundary_structure_and_exact_row_advance() -> None:
    original = "|" + " " * 18 + "|\n"
    canvas_width = _width(original.rstrip("\n"))
    height = line_pitch(16)
    structure = np.zeros((height, canvas_width), dtype=np.float32)
    tone = np.full((height, canvas_width), 2 / 3, dtype=np.float32)
    fill = np.zeros((height, canvas_width), dtype=np.float32)

    left_boundary = glyph_advance("|")
    right_boundary = canvas_width - glyph_advance("|")
    fill[:, left_boundary:right_boundary] = 1.0

    protected_index = 9
    protected_x0 = left_boundary + protected_index * glyph_advance(" ")
    protected_x1 = protected_x0 + glyph_advance(" ")
    structure[:, protected_x0:protected_x1] = 1.0

    result = apply_fill_stage1(
        original,
        structure=structure,
        tone=tone,
        fill=fill,
        empirical_run_counts=EMPIRICAL_RUNS,
        canvas_width=canvas_width,
        config=FillStage1Config(boundary_margin_px=1),
    )

    output = result.text.rstrip("\n")
    assert output.startswith("|")
    assert output.endswith("|")
    assert _width(output) == canvas_width
    assert result.replacements
    assert all(item.repeated_count >= 3 for item in result.replacements)
    assert all(
        item.replacement.strip(" ")
        == item.fill_character * item.repeated_count
        for item in result.replacements
    )
    assert all(
        _width(item.original) == _width(item.replacement)
        for item in result.replacements
    )
    assert all(
        not (item.x_start <= protected_x0 and item.x_end >= protected_x1)
        for item in result.replacements
    )


def test_stage1_does_not_fill_a_boundary_only_cell() -> None:
    original = "|     |\n"
    canvas_width = _width(original.rstrip("\n"))
    height = line_pitch(16)
    structure = np.zeros((height, canvas_width), dtype=np.float32)
    tone = np.ones((height, canvas_width), dtype=np.float32)
    fill = np.zeros((height, canvas_width), dtype=np.float32)
    middle_start = glyph_advance("|")
    middle_end = canvas_width - glyph_advance("|")
    fill[:, middle_start : middle_start + 1] = 1.0
    fill[:, middle_end - 1 : middle_end] = 1.0

    result = apply_fill_stage1(
        original,
        structure=structure,
        tone=tone,
        fill=fill,
        empirical_run_counts=EMPIRICAL_RUNS,
        canvas_width=canvas_width,
    )

    assert result.text == original
    assert result.replacements == ()
