from __future__ import annotations

import numpy as np

from training.structure_augment import (
    STRUCTURE_AUGMENT_POLICIES,
    apply_render_augment,
    choose_augment,
    get_augment_policy,
)


def _sample() -> np.ndarray:
    image = np.full((40, 60), 255, dtype=np.uint8)
    image[10:30, 29:31] = 0
    image[19:21, 10:50] = 0
    return image


def test_all_p5_augments_preserve_shape_and_dtype() -> None:
    source = _sample()
    augments = {
        item.augment
        for policy in STRUCTURE_AUGMENT_POLICIES.values()
        for item in policy.variants
    }
    for augment in augments:
        result = apply_render_augment(source, augment)
        assert result.shape == source.shape
        assert result.dtype == np.uint8


def test_p5_policies_never_delete_lines_or_move_coordinates_by_design() -> None:
    for policy in STRUCTURE_AUGMENT_POLICIES.values():
        manifest = policy.manifest()
        assert manifest["destructive_line_dropout"] is False
        assert manifest["coordinate_transform"] is False


def test_weighted_choice_is_stable() -> None:
    policy = get_augment_policy("combined_50")
    first = [choose_augment(policy, str(index), seed=42) for index in range(100)]
    second = [choose_augment(policy, str(index), seed=42) for index in range(100)]
    assert first == second
    assert set(first).issubset({item.augment for item in policy.variants})


def test_unknown_policy_is_rejected() -> None:
    try:
        get_augment_policy("missing")
    except ValueError as exc:
        assert "Unknown augment policy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing policy should fail")
