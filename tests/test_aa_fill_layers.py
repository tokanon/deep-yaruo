from __future__ import annotations

import numpy as np

from backend.rendering import line_pitch
from training.aa_fill_layers import (
    connect_fill_runs,
    detect_fill_runs,
    render_fill_layers,
    render_fill_surface_labels,
)


def test_fill_runs_accept_unbounded_tone_fields_and_ignore_rule_lines() -> None:
    text = ";;;;;;;;;;\n::::::::::\n│▓▓▓▓│\n│▓▓▓▓│\n│────│\n台詞……"

    runs = detect_fill_runs(text)

    assert [(run.character, run.count, run.line_index) for run in runs] == [
        (";", 10, 0),
        (":", 10, 1),
        ("▓", 4, 2),
        ("▓", 4, 3),
    ]
    assert not runs[0].bounded_left
    assert not runs[0].bounded_right


def test_render_fill_layers_removes_only_detected_fill_characters() -> None:
    text = "┌:::::┐\n│:::::│\n└─────┘"

    original, line, fill, composite, diagnostic, runs = render_fill_layers(text)

    assert len(runs) == 2
    assert original.shape == line.shape == fill.shape == composite.shape
    assert diagnostic.shape == (*original.shape, 3)
    assert np.count_nonzero(fill < 255) > 0
    assert np.count_nonzero(line < 255) > 0
    assert np.all(composite <= line)
    assert np.all(composite <= fill)


def test_fill_surfaces_connect_only_adjacent_overlapping_runs_deterministically() -> None:
    text = "::::::::\n :::::::: \n  ::::::::  \n                    ........"
    runs = detect_fill_runs(text)

    surfaces = connect_fill_runs(reversed(runs))

    assert len(surfaces) == 2
    assert surfaces[0].surface_id == 0
    assert surfaces[0].line_start == 0
    assert surfaces[0].line_end == 3
    assert surfaces[0].line_count == 3
    assert surfaces[0].run_count == 3
    assert surfaces[0].bbox_px == (
        surfaces[0].x_start_px,
        surfaces[0].y_start_px,
        surfaces[0].x_end_px,
        surfaces[0].y_end_px,
    )
    assert dict(surfaces[0].character_counts) == {":": 3}
    assert surfaces[1].line_start == 3
    assert surfaces[1].line_count == 1
    assert surfaces[1].run_count == 1

    repeated = connect_fill_runs(runs)
    assert [surface.to_dict() for surface in surfaces] == [
        surface.to_dict() for surface in repeated
    ]


def test_fill_surface_mask_area_matches_surface_aggregation() -> None:
    text = "::::::;;;;;;\n ::::::;;;;;;\n  ::::::;;;;;;"
    runs = detect_fill_runs(text)
    surfaces = connect_fill_runs(runs)
    width = max(round(run.x_end) for run in runs) + 2
    height = 3 * line_pitch(16)

    labels = render_fill_surface_labels(surfaces, width=width, height=height)

    assert int(np.count_nonzero(labels)) == sum(
        surface.pixel_area for surface in surfaces
    )
    for surface in surfaces:
        assert int(np.count_nonzero(labels == surface.surface_id + 1)) == (
            surface.pixel_area
        )


def test_fill_surface_overlap_ratio_is_validated() -> None:
    with np.testing.assert_raises(ValueError):
        connect_fill_runs([], minimum_overlap_ratio=0.0)
