from __future__ import annotations

import numpy as np
import pytest

from evaluation.surface_metrics import (
    SurfaceSet,
    UnpairedSurfaceEvaluationError,
    evaluate_surface_sets,
)


def _surface_set(
    labels: np.ndarray,
    *,
    correspondence_id: str | None = "synthetic-pair",
    coordinate_space_id: str = "synthetic-12x12",
    line_mask: np.ndarray | None = None,
) -> SurfaceSet:
    return SurfaceSet(
        labels=labels.astype(np.int32),
        coordinate_space_id=coordinate_space_id,
        correspondence_id=correspondence_id,
        line_mask=line_mask,
    )


def test_surface_metrics_report_exact_one_to_one_match() -> None:
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:5, 1:5] = 1
    labels[7:10, 6:11] = 9

    report = evaluate_surface_sets(_surface_set(labels), _surface_set(labels.copy()))

    assert report["matching"]["mean_iou"] == 1.0
    assert report["surface_metrics"] == {
        "reference_count": 2,
        "predicted_count": 2,
        "true_positive_count": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "overfill_count": 0,
        "overfill_ids": [],
        "missing_count": 0,
        "missing_ids": [],
    }
    assert report["pixel_metrics"]["iou"] == 1.0
    assert report["topology"]["split_count"] == 0
    assert report["topology"]["merge_count"] == 0


def test_surface_metrics_distinguish_overfill_and_missing_surfaces() -> None:
    reference = np.zeros((12, 12), dtype=np.int32)
    reference[1:4, 1:4] = 1
    reference[7:10, 1:4] = 2
    predicted = np.zeros_like(reference)
    predicted[1:4, 1:4] = 7
    predicted[1:4, 8:11] = 8

    report = evaluate_surface_sets(_surface_set(reference), _surface_set(predicted))

    assert report["surface_metrics"]["true_positive_count"] == 1
    assert report["surface_metrics"]["precision"] == 0.5
    assert report["surface_metrics"]["recall"] == 0.5
    assert report["surface_metrics"]["overfill_ids"] == [8]
    assert report["surface_metrics"]["missing_ids"] == [2]


def test_surface_metrics_detect_split_topology() -> None:
    reference = np.zeros((10, 12), dtype=np.int32)
    reference[2:8, 2:10] = 1
    predicted = np.zeros_like(reference)
    predicted[2:8, 2:6] = 10
    predicted[2:8, 6:10] = 11

    report = evaluate_surface_sets(_surface_set(reference), _surface_set(predicted))

    assert report["topology"]["split_count"] == 1
    assert report["topology"]["splits"] == [
        {"reference_id": 1, "predicted_ids": [10, 11]}
    ]
    assert report["topology"]["merge_count"] == 0
    assert report["surface_metrics"]["true_positive_count"] == 1
    assert report["surface_metrics"]["overfill_count"] == 1


def test_surface_metrics_detect_merge_topology() -> None:
    reference = np.zeros((10, 12), dtype=np.int32)
    reference[2:8, 2:6] = 1
    reference[2:8, 6:10] = 2
    predicted = np.zeros_like(reference)
    predicted[2:8, 2:10] = 12

    report = evaluate_surface_sets(_surface_set(reference), _surface_set(predicted))

    assert report["topology"]["merge_count"] == 1
    assert report["topology"]["merges"] == [
        {"predicted_id": 12, "reference_ids": [1, 2]}
    ]
    assert report["topology"]["split_count"] == 0
    assert report["surface_metrics"]["true_positive_count"] == 1
    assert report["surface_metrics"]["missing_count"] == 1


def test_surface_metrics_measure_line_loss_inside_reference_surfaces() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:7, 1:7] = 1
    reference_lines = np.zeros_like(labels, dtype=bool)
    reference_lines[2, 2] = True
    reference_lines[5, 5] = True
    predicted_lines = np.zeros_like(labels, dtype=bool)
    predicted_lines[2, 2] = True

    report = evaluate_surface_sets(
        _surface_set(labels, line_mask=reference_lines),
        _surface_set(labels.copy(), line_mask=predicted_lines),
        line_tolerance_px=0,
    )

    assert report["line_metrics"] == {
        "tolerance_px": 0,
        "reference_line_pixels_inside_surfaces": 2,
        "retained_line_pixels": 1,
        "retention": 0.5,
        "loss": 0.5,
    }


def test_surface_metrics_reject_unpaired_or_mismatched_collections() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    unpaired = _surface_set(labels, correspondence_id=None)
    paired = _surface_set(labels)

    with pytest.raises(UnpairedSurfaceEvaluationError):
        evaluate_surface_sets(unpaired, paired)
    with pytest.raises(UnpairedSurfaceEvaluationError):
        evaluate_surface_sets(
            _surface_set(labels, correspondence_id=""),
            _surface_set(labels, correspondence_id=""),
        )
    with pytest.raises(UnpairedSurfaceEvaluationError):
        evaluate_surface_sets(
            paired,
            _surface_set(labels, correspondence_id="another-pair"),
        )
    with pytest.raises(ValueError, match="coordinate spaces"):
        evaluate_surface_sets(
            paired,
            _surface_set(labels, coordinate_space_id="another-space"),
        )
    with pytest.raises(ValueError, match="label-mask shape"):
        evaluate_surface_sets(
            paired,
            _surface_set(
                np.zeros((5, 4), dtype=np.int32),
                coordinate_space_id="synthetic-12x12",
            ),
        )


def test_surface_metrics_require_line_masks_on_both_sides() -> None:
    labels = np.zeros((4, 4), dtype=np.int32)
    line_mask = np.zeros_like(labels, dtype=bool)

    with pytest.raises(ValueError, match="both surface sets"):
        evaluate_surface_sets(
            _surface_set(labels, line_mask=line_mask),
            _surface_set(labels),
        )
