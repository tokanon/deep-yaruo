from __future__ import annotations

import numpy as np

from training.evaluate_tone_windows_p5 import (
    _select_tone_conditions,
    sample_tone_context,
    tone_descriptor,
)


def test_tone_descriptor_is_shape_independent() -> None:
    first = tone_descriptor(np.zeros((32, 64), dtype=np.float32))
    second = tone_descriptor(np.zeros((96, 192), dtype=np.float32))
    assert first.shape == second.shape
    assert first.ndim == 1


def test_tone_context_uses_zero_for_outside_darkness() -> None:
    tone = np.ones((20, 20), dtype=np.float32)
    center = sample_tone_context(tone, 10, 10)
    outside = sample_tone_context(tone, -100, -100)
    assert center.max() == 1.0
    assert center.min() == 0.0
    assert np.count_nonzero(outside) == 0


def test_tone_selection_prefers_smaller_window_when_leakage_is_equal() -> None:
    def condition(width: int, *, levels: float, leak: float, auc: float) -> dict[str, object]:
        return {
            "config": {"tone_window_width": width},
            "domain": {"separability_auc": auc},
            "real_tone_retention": {"mean_spatial_std": 0.1},
            "proxy_tone_retention": {
                "mean_occupied_levels": levels,
                "mean_spatial_std": 0.1,
            },
            "exact_character_leakage": {"nonblank_accuracy": leak},
        }

    reports = {
        "w48-g1": condition(48, levels=1.0, leak=0.2, auc=0.7),
        "w48-g0.5": condition(48, levels=2.0, leak=0.0, auc=0.93),
        "w80-g0.5": condition(80, levels=2.0, leak=0.0, auc=0.90),
    }
    selection = _select_tone_conditions(reports)
    assert selection["selected_condition"] == "w48-g0.5"
    assert "w48-g1" in selection["rejected"]
