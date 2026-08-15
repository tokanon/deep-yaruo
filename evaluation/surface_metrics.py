from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class UnpairedSurfaceEvaluationError(ValueError):
    """Raised when correctness metrics are requested for unpaired surfaces."""


@dataclass(frozen=True)
class SurfaceSet:
    labels: np.ndarray
    coordinate_space_id: str
    correspondence_id: str | None
    line_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError("surface labels must be a 2D array")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("surface labels must use an integer dtype")
        if np.any(labels < 0):
            raise ValueError("surface labels must be non-negative")
        if not self.coordinate_space_id:
            raise ValueError("coordinate_space_id must not be empty")
        object.__setattr__(self, "labels", labels.astype(np.int32, copy=False))

        if self.line_mask is not None:
            line_mask = np.asarray(self.line_mask)
            if line_mask.shape != labels.shape:
                raise ValueError("line_mask must have the same shape as labels")
            object.__setattr__(self, "line_mask", line_mask.astype(bool, copy=False))


@dataclass(frozen=True)
class SurfaceMatch:
    reference_id: int
    predicted_id: int
    intersection_px: int
    union_px: int
    iou: float
    reference_coverage: float
    predicted_coverage: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "reference_id": self.reference_id,
            "predicted_id": self.predicted_id,
            "intersection_px": self.intersection_px,
            "union_px": self.union_px,
            "iou": round(self.iou, 6),
            "reference_coverage": round(self.reference_coverage, 6),
            "predicted_coverage": round(self.predicted_coverage, 6),
        }


def _positive_label_areas(labels: np.ndarray) -> dict[int, int]:
    identifiers, counts = np.unique(labels[labels > 0], return_counts=True)
    return {
        int(identifier): int(count)
        for identifier, count in zip(identifiers, counts, strict=True)
    }


def _overlap_counts(
    reference_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[tuple[int, int], int]:
    shared = (reference_labels > 0) & (predicted_labels > 0)
    if not np.any(shared):
        return {}
    pairs, counts = np.unique(
        np.stack(
            (reference_labels[shared], predicted_labels[shared]),
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    return {
        (int(pair[0]), int(pair[1])): int(count)
        for pair, count in zip(pairs, counts, strict=True)
    }


def _candidate_matches(
    reference_areas: dict[int, int],
    predicted_areas: dict[int, int],
    overlaps: dict[tuple[int, int], int],
) -> list[SurfaceMatch]:
    matches: list[SurfaceMatch] = []
    for (reference_id, predicted_id), intersection in overlaps.items():
        reference_area = reference_areas[reference_id]
        predicted_area = predicted_areas[predicted_id]
        union = reference_area + predicted_area - intersection
        matches.append(
            SurfaceMatch(
                reference_id=reference_id,
                predicted_id=predicted_id,
                intersection_px=intersection,
                union_px=union,
                iou=intersection / union,
                reference_coverage=intersection / reference_area,
                predicted_coverage=intersection / predicted_area,
            )
        )
    return matches


def _greedy_one_to_one_matches(
    candidates: list[SurfaceMatch],
    *,
    minimum_iou: float,
) -> list[SurfaceMatch]:
    matched_reference: set[int] = set()
    matched_predicted: set[int] = set()
    selected: list[SurfaceMatch] = []
    ordered = sorted(
        candidates,
        key=lambda match: (
            -match.iou,
            -match.intersection_px,
            match.reference_id,
            match.predicted_id,
        ),
    )
    for candidate in ordered:
        if candidate.iou < minimum_iou:
            continue
        if candidate.reference_id in matched_reference:
            continue
        if candidate.predicted_id in matched_predicted:
            continue
        selected.append(candidate)
        matched_reference.add(candidate.reference_id)
        matched_predicted.add(candidate.predicted_id)
    return sorted(selected, key=lambda match: (match.reference_id, match.predicted_id))


def _topology_records(
    candidates: list[SurfaceMatch],
    *,
    minimum_overlap_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reference_edges: dict[int, list[int]] = {}
    predicted_edges: dict[int, list[int]] = {}
    for candidate in candidates:
        # Intersection divided by the smaller surface area equals the larger
        # of the two directional coverages.
        overlap_ratio = max(
            candidate.reference_coverage,
            candidate.predicted_coverage,
        )
        if overlap_ratio < minimum_overlap_ratio:
            continue
        reference_edges.setdefault(candidate.reference_id, []).append(
            candidate.predicted_id
        )
        predicted_edges.setdefault(candidate.predicted_id, []).append(
            candidate.reference_id
        )

    splits = [
        {
            "reference_id": reference_id,
            "predicted_ids": sorted(set(predicted_ids)),
        }
        for reference_id, predicted_ids in sorted(reference_edges.items())
        if len(set(predicted_ids)) > 1
    ]
    merges = [
        {
            "predicted_id": predicted_id,
            "reference_ids": sorted(set(reference_ids)),
        }
        for predicted_id, reference_ids in sorted(predicted_edges.items())
        if len(set(reference_ids)) > 1
    ]
    return splits, merges


def _safe_precision(true_positive: int, predicted_count: int) -> float:
    return true_positive / predicted_count if predicted_count else 1.0


def _safe_recall(true_positive: int, reference_count: int) -> float:
    return true_positive / reference_count if reference_count else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _pixel_metrics(reference_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, object]:
    reference_fill = reference_labels > 0
    predicted_fill = predicted_labels > 0
    intersection = int(np.count_nonzero(reference_fill & predicted_fill))
    union = int(np.count_nonzero(reference_fill | predicted_fill))
    reference_count = int(np.count_nonzero(reference_fill))
    predicted_count = int(np.count_nonzero(predicted_fill))
    precision = intersection / predicted_count if predicted_count else 1.0
    recall = intersection / reference_count if reference_count else 1.0
    return {
        "intersection_px": intersection,
        "union_px": union,
        "reference_fill_px": reference_count,
        "predicted_fill_px": predicted_count,
        "iou": round(intersection / union if union else 1.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(_f1(precision, recall), 6),
    }


def _line_metrics(
    reference: SurfaceSet,
    predicted: SurfaceSet,
    *,
    tolerance_px: int,
) -> dict[str, object] | None:
    if reference.line_mask is None and predicted.line_mask is None:
        return None
    if reference.line_mask is None or predicted.line_mask is None:
        raise ValueError("both surface sets must provide line_mask or neither may")
    if tolerance_px < 0:
        raise ValueError("line_tolerance_px must be non-negative")

    inside_reference_surfaces = reference.labels > 0
    reference_lines = reference.line_mask & inside_reference_surfaces
    reference_line_count = int(np.count_nonzero(reference_lines))
    if tolerance_px:
        size = tolerance_px * 2 + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        predicted_support = cv2.dilate(
            predicted.line_mask.astype(np.uint8),
            kernel,
        ).astype(bool)
    else:
        predicted_support = predicted.line_mask
    retained = int(np.count_nonzero(reference_lines & predicted_support))
    retention = retained / reference_line_count if reference_line_count else 1.0
    return {
        "tolerance_px": tolerance_px,
        "reference_line_pixels_inside_surfaces": reference_line_count,
        "retained_line_pixels": retained,
        "retention": round(retention, 6),
        "loss": round(1.0 - retention, 6),
    }


def evaluate_surface_sets(
    reference: SurfaceSet,
    predicted: SurfaceSet,
    *,
    match_iou_threshold: float = 0.5,
    topology_overlap_threshold: float = 0.1,
    line_tolerance_px: int = 1,
) -> dict[str, object]:
    """Evaluate two explicitly paired surface sets in the same pixel space."""
    if not reference.correspondence_id or not predicted.correspondence_id:
        raise UnpairedSurfaceEvaluationError(
            "surface correctness metrics require an explicit correspondence_id"
        )
    if reference.correspondence_id != predicted.correspondence_id:
        raise UnpairedSurfaceEvaluationError(
            "surface sets belong to different correspondence pairs"
        )
    if reference.coordinate_space_id != predicted.coordinate_space_id:
        raise ValueError("surface sets use different coordinate spaces")
    if reference.labels.shape != predicted.labels.shape:
        raise ValueError("surface sets must have the same label-mask shape")
    if not 0.0 < match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be in (0, 1]")
    if not 0.0 < topology_overlap_threshold <= 1.0:
        raise ValueError("topology_overlap_threshold must be in (0, 1]")

    reference_areas = _positive_label_areas(reference.labels)
    predicted_areas = _positive_label_areas(predicted.labels)
    overlaps = _overlap_counts(reference.labels, predicted.labels)
    candidates = _candidate_matches(reference_areas, predicted_areas, overlaps)
    matches = _greedy_one_to_one_matches(
        candidates,
        minimum_iou=match_iou_threshold,
    )
    matched_reference = {match.reference_id for match in matches}
    matched_predicted = {match.predicted_id for match in matches}
    missing_ids = sorted(set(reference_areas) - matched_reference)
    overfill_ids = sorted(set(predicted_areas) - matched_predicted)
    splits, merges = _topology_records(
        candidates,
        minimum_overlap_ratio=topology_overlap_threshold,
    )

    precision = _safe_precision(len(matches), len(predicted_areas))
    recall = _safe_recall(len(matches), len(reference_areas))
    matched_ious = [match.iou for match in matches]
    reference_coverages = [match.reference_coverage for match in matches]
    predicted_coverages = [match.predicted_coverage for match in matches]
    return {
        "schema_version": 1,
        "correspondence_id": reference.correspondence_id,
        "coordinate_space_id": reference.coordinate_space_id,
        "mask_shape": list(reference.labels.shape),
        "matching": {
            "method": "greedy-descending-IoU one-to-one",
            "iou_threshold": match_iou_threshold,
            "matches": [match.to_dict() for match in matches],
            "mean_iou": round(float(np.mean(matched_ious)), 6)
            if matched_ious
            else None,
            "mean_reference_coverage": round(
                float(np.mean(reference_coverages)), 6
            )
            if reference_coverages
            else None,
            "mean_predicted_coverage": round(
                float(np.mean(predicted_coverages)), 6
            )
            if predicted_coverages
            else None,
        },
        "surface_metrics": {
            "reference_count": len(reference_areas),
            "predicted_count": len(predicted_areas),
            "true_positive_count": len(matches),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(_f1(precision, recall), 6),
            "overfill_count": len(overfill_ids),
            "overfill_ids": overfill_ids,
            "missing_count": len(missing_ids),
            "missing_ids": missing_ids,
        },
        "topology": {
            "overlap_threshold": topology_overlap_threshold,
            "split_count": len(splits),
            "splits": splits,
            "merge_count": len(merges),
            "merges": merges,
        },
        "pixel_metrics": _pixel_metrics(reference.labels, predicted.labels),
        "line_metrics": _line_metrics(
            reference,
            predicted,
            tolerance_px=line_tolerance_px,
        ),
    }
