from __future__ import annotations

import hashlib

import numpy as np
import pytest

from backend.contracts import ConversionOptions
from backend.information_recovery import (
    _filter_small_components,
    _recover_text,
    _selected_v3_baseline,
)
from backend.rendering import glyph_advance


def test_component_filter_removes_fewer_than_three_cells() -> None:
    mask = np.zeros((3, 6), dtype=bool)
    mask[0, :2] = True
    mask[2, 3:] = True

    filtered = _filter_small_components(mask)

    assert not filtered[0].any()
    assert filtered[2, 3:].all()


def test_information_recovery_preserves_width_and_caps_changed_cells() -> None:
    text = "        \n"
    width = sum(glyph_advance(character, 16) for character in text.rstrip("\n"))
    processed = np.full((18, width), 255, dtype=np.uint8)
    rendered = np.full_like(processed, 255)
    relative_darkness = np.full(processed.shape, 0.7, dtype=np.float32)
    target_grid = np.ones((1, width // 8), dtype=bool)
    missing_grid = target_grid.copy()

    recovered, diagnostics = _recover_text(
        text,
        processed=processed,
        rendered=rendered,
        relative_darkness=relative_darkness,
        target_grid=target_grid,
        missing_grid=missing_grid,
        font_size=16,
    )

    assert recovered != text
    assert sum(glyph_advance(char, 16) for char in recovered.rstrip("\n")) == width
    assert diagnostics["source_supported_changed_cells"] <= diagnostics[
        "missing_cell_budget"
    ]
    assert diagnostics["source_supported_changed_fraction"] == 1.0


def test_v3_setting_mismatch_reports_expected_values() -> None:
    source = b"same-source"
    expected = ConversionOptions(
        profile="background",
        columns=120,
        max_rows=46,
        detail=58,
        abstraction=15,
        threshold_low=65,
        threshold_high=170,
        min_component=14,
        crop_x=0.19,
        crop_y=0.0,
        crop_width=0.65,
        crop_height=1.0,
    ).normalized()

    class Store:
        def list(self):
            return [
                {
                    "source": {
                        "artifact": {"sha256": hashlib.sha256(source).hexdigest()}
                    },
                    "options": expected.__dict__,
                }
            ]

    with pytest.raises(ValueError, match=r"max_rows=46.*crop_x=0.19"):
        _selected_v3_baseline(
            source,
            ConversionOptions(profile="background", columns=120, max_rows=42),
            Store(),
        )
