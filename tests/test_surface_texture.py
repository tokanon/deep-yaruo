from __future__ import annotations

import hashlib

import numpy as np

from backend.rendering import glyph_advance, line_pitch
from backend.surface_texture import (
    _line_separated_graph,
    _replacement_for_motif,
    _surface_pair_distances,
    _texture_cells,
    apply_surface_texture,
    apply_surface_texture_proximity,
    load_texture_catalog,
    resolve_texture_baseline_options,
)


def test_motif_replacement_preserves_natural_advance_and_density() -> None:
    original = ":" * 4
    width = sum(glyph_advance(character) for character in original)

    proposal = _replacement_for_motif(
        width,
        ";",
        original,
        font_size=16,
        left_adjustments=(".",),
        right_adjustments=(".",),
    )

    assert proposal is not None
    replacement, error, repetitions, adjustment_count = proposal
    assert sum(glyph_advance(character) for character in replacement) == width
    assert repetitions >= 2
    assert adjustment_count >= 1
    assert error <= 0.03


def test_frozen_texture_catalog_matches_t1_audit() -> None:
    palette, adjustments, catalog_hash, config_hash = load_texture_catalog()

    assert palette == (";", ":i", ":", ".")
    assert {side: {motif: len(values) for motif, values in by_motif.items()} for side, by_motif in adjustments.items()} == {
        "left": {";": 3, ":i": 7, ":": 12, ".": 4},
        "right": {";": 4, ":i": 8, ":": 13, ".": 6},
    }
    assert catalog_hash == "d8e8efa7c8984812fca86969b0e8edea4a6439a96b097aff84d60207e571dfb9"
    assert config_hash == "9924e77f15aecae4113ff8cd2274bf32382362d9ecc094dc0bd518962a041096"


def _labels_for_lines(text: str, owners: list[list[int]]) -> np.ndarray:
    lines = text.rstrip("\n").split("\n")
    width = max(sum(glyph_advance(character) for character in line) for line in lines)
    pitch = line_pitch(16)
    labels = np.zeros((len(lines) * pitch, width), dtype=np.int32)
    for line_index, (line, line_owners) in enumerate(zip(lines, owners, strict=True)):
        cursor = 0
        for character, owner in zip(line, line_owners, strict=True):
            end = cursor + glyph_advance(character)
            labels[line_index * pitch : (line_index + 1) * pitch, cursor:end] = owner
            cursor = end
    return labels


def test_graph_connects_fill_owners_across_same_row_separator_only() -> None:
    separated = "::::::::X;;;;;;;;\n"
    labels = _labels_for_lines(separated, [[1] * 8 + [0] + [2] * 8])
    rows = _texture_cells(separated, labels, font_size=16)

    assert set(_line_separated_graph(rows)) == {(1, 2)}

    direct = "::::::::;;;;;;;;\n"
    labels = _labels_for_lines(direct, [[1] * 8 + [2] * 8])
    assert _line_separated_graph(_texture_cells(direct, labels, font_size=16)) == {}

    blank = ":::::::: ;;;;;;;;\n"
    labels = _labels_for_lines(blank, [[1] * 8 + [0] + [2] * 8])
    assert _line_separated_graph(_texture_cells(blank, labels, font_size=16)) == {}


def test_graph_connects_fill_above_and_below_separator_advance() -> None:
    text = "::::::::\nXXXXXXXX\n;;;;;;;;\n"
    labels = _labels_for_lines(
        text,
        [
            [1] * 8,
            [0] * 8,
            [2] * 8,
        ],
    )

    graph = _line_separated_graph(_texture_cells(text, labels, font_size=16))

    assert set(graph) == {(1, 2)}
    assert graph[(1, 2)]


def test_texture_changes_only_owned_repeated_fill_run() -> None:
    text = "X::::::::Y\n"
    width = sum(glyph_advance(character) for character in text.rstrip("\n"))
    owner = np.zeros((18, width), dtype=np.int32)
    x0 = glyph_advance("X")
    x1 = x0 + glyph_advance(":") * 4
    owner[:, x0:x1] = 1
    processed = np.full(owner.shape, 255, dtype=np.uint8)

    output, summary = apply_surface_texture(
        text,
        owner=owner,
        processed=processed,
        font_size=16,
        palette=(";", ":"),
        edge_adjustments={"left": {}, "right": {}},
    )

    assert output.startswith("X") and output.endswith("::::Y\n")
    assert sum(glyph_advance(character) for character in output.rstrip("\n")) == width
    assert summary["replacement_count"] == 1
    assert set(summary["style_assignments"]) == {"1"}


def test_texture_preserves_separator_between_two_owned_fill_runs() -> None:
    text = "::::::::X;;;;;;;;\n"
    labels = _labels_for_lines(
        text,
        [[1] * 8 + [0] + [2] * 8],
    )
    processed = np.full(labels.shape, 255, dtype=np.uint8)

    output, summary = apply_surface_texture(
        text,
        owner=labels,
        processed=processed,
        font_size=16,
        palette=(";", ":"),
        edge_adjustments={"left": {}, "right": {}},
    )

    assert output.count("X") == 1
    assert output.index("X") > 0
    assert summary["raw_line_separated_pair_count"] == 1
    assert summary["eligible_pair_count"] == 1
    assert summary["actual_separated_pair_count"] == 1


def test_proximity_distance_crosses_blank_gap_in_normalized_units() -> None:
    text = "::::::::  ::::::::\n"
    labels = _labels_for_lines(
        text,
        [[1] * 8 + [0, 0] + [2] * 8],
    )
    rows = _texture_cells(text, labels, font_size=16)

    distances = _surface_pair_distances(rows, {1, 2})

    expected = 2 * glyph_advance(" ", 16) / 8.0
    assert distances == {(1, 2): expected}


def test_proximity_texture_separates_near_fill_surfaces_across_blank_gap() -> None:
    text = "X::::::::  ::::::::Y\n"
    labels = _labels_for_lines(
        text,
        [[0] + [1] * 8 + [0, 0] + [2] * 8 + [0]],
    )
    processed = np.full(labels.shape, 255, dtype=np.uint8)
    original_width = sum(glyph_advance(character) for character in text.rstrip("\n"))

    output, summary = apply_surface_texture_proximity(
        text,
        owner=labels,
        processed=processed,
        font_size=16,
        palette=(";", ":", "."),
        edge_adjustments={"left": {}, "right": {}},
    )

    assert output.startswith("X") and output.endswith("Y\n")
    assert "  " in output
    assert sum(glyph_advance(character) for character in output.rstrip("\n")) == original_width
    assert summary["nearest_compatible_pairs"] == [[1, 2]]
    assert summary["actual_nearest_separated_pairs"] == [[1, 2]]
    assert summary["nearest_separation_fraction"] == 1.0
    assert summary["style_assignments"]["1"] != summary["style_assignments"]["2"]


def test_proximity_resolves_saved_v4_options_without_current_ui_settings() -> None:
    data = b"fixed-source"
    source = {"artifact": {"sha256": hashlib.sha256(data).hexdigest()}}

    class Store:
        def __init__(self, records: list[dict[str, object]]) -> None:
            self.records = records

        def list(self) -> list[dict[str, object]]:
            return self.records

    v2 = Store(
        [
            {
                "recipe_version": "surface-fill-v2",
                "source": source,
                "none_usable": True,
                "options": {"columns": 48, "profile": "person"},
            }
        ]
    )
    v4 = Store(
        [
            {
                "recipe_version": "information-recovery-v4",
                "source": source,
                "selected_variant": "information-combined-v4",
                "none_usable": False,
                "options": {
                    "columns": 72,
                    "detail": 72,
                    "crop_y": 0.015,
                    "profile": "person",
                },
                "candidates": [
                    {"variant_id": "information-combined-v4"},
                ],
            }
        ]
    )

    options = resolve_texture_baseline_options(data, v2, v4)  # type: ignore[arg-type]

    assert options.columns == 72
    assert options.detail == 72
    assert options.crop_y == 0.015
