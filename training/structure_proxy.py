from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from backend.input_channels import (
    ChannelExtractionConfig,
    extract_input_channels,
    normalize_geometry,
)
from backend.line_ops import skeletonize
from training.weak_pairs import render_aa_text


StructureProxyMethod = Literal["raw_raster", "supersampled_blur_downsample"]
STRUCTURE_PROXY_METHODS: tuple[StructureProxyMethod, ...] = (
    "raw_raster",
    "supersampled_blur_downsample",
)


@dataclass(frozen=True)
class StructureProxyConfig:
    font_size: int = 16
    supersample: int = 4
    high_resolution_blur_sigma: float = 2.0

    def validate(self) -> None:
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")
        if self.supersample < 2:
            raise ValueError("supersample must be at least two")
        if self.high_resolution_blur_sigma < 0:
            raise ValueError("high_resolution_blur_sigma cannot be negative")


def generate_structure_proxy(
    text: str,
    method: StructureProxyMethod,
    *,
    config: StructureProxyConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create an AA-derived structure proxy, never an inferred source image."""
    active = config or StructureProxyConfig()
    active.validate()
    if method == "raw_raster":
        image = render_aa_text(text, font_size=active.font_size)
        parameters: dict[str, object] = {
            "method": method,
            "render_scale": 1,
            "blur_sigma": 0.0,
            "downsample": None,
        }
    elif method == "supersampled_blur_downsample":
        scale = active.supersample
        target = render_aa_text(text, font_size=active.font_size)
        high_resolution = render_aa_text(
            text,
            font_size=active.font_size * scale,
        )
        sigma = active.high_resolution_blur_sigma
        blurred = (
            cv2.GaussianBlur(
                high_resolution,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
            if sigma > 0
            else high_resolution
        )
        image = cv2.resize(
            blurred,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        parameters = {
            "method": method,
            "render_scale": scale,
            "blur_sigma_high_resolution_px": sigma,
            "blur_sigma_output_equivalent_px": round(sigma / scale, 6),
            "downsample": "INTER_AREA",
            "output_geometry": "matched_to_1x_render",
        }
    else:
        raise ValueError(f"Unknown structure proxy method: {method}")
    return image, {"config": asdict(active), **parameters}


def canonical_aa_skeleton(
    rendered: np.ndarray,
    *,
    channel_config: ChannelExtractionConfig | None = None,
) -> np.ndarray:
    """Return the canonical AA-ink centreline used only as a P2 geometry target."""
    active = channel_config or ChannelExtractionConfig()
    normalized = normalize_geometry(rendered, active)
    ink = cv2.threshold(normalized, 200, 255, cv2.THRESH_BINARY_INV)[1]
    return (skeletonize(ink).astype(np.float32) / 255.0).astype(np.float32)


def extract_proxy_structure(
    proxy: np.ndarray,
    *,
    channel_config: ChannelExtractionConfig | None = None,
) -> np.ndarray:
    return extract_input_channels(
        proxy,
        source_kind="aa_proxy",
        config=channel_config,
    ).structure


def symmetric_chamfer(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Measure a symmetric centreline distance in pixels on one shared grid."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Chamfer inputs must share a shape: {reference.shape} != {candidate.shape}"
        )
    reference_ink = reference >= 0.5
    candidate_ink = candidate >= 0.5
    reference_count = int(np.count_nonzero(reference_ink))
    candidate_count = int(np.count_nonzero(candidate_ink))
    if reference_count == 0 and candidate_count == 0:
        return {
            "symmetric_px": 0.0,
            "reference_to_candidate_px": 0.0,
            "candidate_to_reference_px": 0.0,
            "reference_pixels": 0.0,
            "candidate_pixels": 0.0,
        }
    if reference_count == 0 or candidate_count == 0:
        return {
            "symmetric_px": math.inf,
            "reference_to_candidate_px": math.inf,
            "candidate_to_reference_px": math.inf,
            "reference_pixels": float(reference_count),
            "candidate_pixels": float(candidate_count),
        }
    distance_to_candidate = cv2.distanceTransform(
        (~candidate_ink).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    distance_to_reference = cv2.distanceTransform(
        (~reference_ink).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    reference_to_candidate = float(np.mean(distance_to_candidate[reference_ink]))
    candidate_to_reference = float(np.mean(distance_to_reference[candidate_ink]))
    return {
        "symmetric_px": (reference_to_candidate + candidate_to_reference) / 2.0,
        "reference_to_candidate_px": reference_to_candidate,
        "candidate_to_reference_px": candidate_to_reference,
        "reference_pixels": float(reference_count),
        "candidate_pixels": float(candidate_count),
    }


def structure_descriptor(structure: np.ndarray) -> np.ndarray:
    """Return an image-level morphology descriptor without source dimensions."""
    ink = (structure >= 0.5).astype(np.uint8)
    normalized = cv2.resize(ink.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    horizontal = normalized.mean(axis=1)
    vertical = normalized.mean(axis=0)

    blurred = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), sigmaX=1.0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    orientation = np.mod(angle, np.pi)
    orientation_histogram, _ = np.histogram(
        orientation,
        bins=8,
        range=(0.0, np.pi),
        weights=magnitude,
    )
    orientation_total = float(orientation_histogram.sum())
    if orientation_total > 0:
        orientation_histogram = orientation_histogram / orientation_total

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if component_areas.size:
        component_features = np.array(
            [
                len(component_areas) / max(1, ink.size) * 10_000.0,
                float(np.mean(component_areas)),
                float(np.median(component_areas)),
                float(np.quantile(component_areas, 0.9)),
                float(np.max(component_areas)),
            ],
            dtype=np.float32,
        )
    else:
        component_features = np.zeros(5, dtype=np.float32)

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbors = cv2.filter2D(ink, cv2.CV_16U, neighbor_kernel)
    ink_count = max(1, int(np.count_nonzero(ink)))
    topology = np.array(
        [
            float(np.count_nonzero(ink)) / max(1, ink.size),
            float(np.count_nonzero((ink > 0) & (neighbors == 1))) / ink_count,
            float(np.count_nonzero((ink > 0) & (neighbors >= 3))) / ink_count,
            float(count - 1) / ink_count,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (
            normalized.ravel(),
            horizontal,
            vertical,
            orientation_histogram.astype(np.float32),
            component_features,
            topology,
        )
    ).astype(np.float32)


def domain_auc(
    real_features: np.ndarray,
    proxy_features: np.ndarray,
    *,
    seed: int = 42,
    folds: int = 5,
    repeats: int = 5,
    bootstrap_samples: int = 1_000,
) -> dict[str, object]:
    """Estimate image-level domain separability with repeated held-out folds."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised by the training environment
        raise RuntimeError(
            "P2 domain AUC requires requirements-training.txt in .venv-training."
        ) from exc

    real = np.asarray(real_features, dtype=np.float32)
    proxy = np.asarray(proxy_features, dtype=np.float32)
    if real.ndim != 2 or proxy.ndim != 2 or real.shape[1] != proxy.shape[1]:
        raise ValueError("Domain features must be two matrices with the same width")
    minimum_class = min(len(real), len(proxy))
    active_folds = min(folds, minimum_class)
    if active_folds < 2:
        raise ValueError("Domain AUC requires at least two images from each domain")
    features = np.concatenate((real, proxy), axis=0)
    labels = np.concatenate(
        (np.zeros(len(real), dtype=np.uint8), np.ones(len(proxy), dtype=np.uint8))
    )
    splitter = RepeatedStratifiedKFold(
        n_splits=active_folds,
        n_repeats=repeats,
        random_state=seed,
    )
    prediction_sum = np.zeros(len(labels), dtype=np.float64)
    prediction_count = np.zeros(len(labels), dtype=np.int32)
    for train_indices, test_indices in splitter.split(features, labels):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
                solver="liblinear",
            ),
        )
        classifier.fit(features[train_indices], labels[train_indices])
        prediction_sum[test_indices] += classifier.predict_proba(features[test_indices])[:, 1]
        prediction_count[test_indices] += 1
    if np.any(prediction_count == 0):
        raise AssertionError("Every image must receive held-out domain predictions")
    predictions = prediction_sum / prediction_count
    raw_auc = float(roc_auc_score(labels, predictions))
    separability_auc = max(raw_auc, 1.0 - raw_auc)

    rng = np.random.default_rng(seed)
    real_indices = np.flatnonzero(labels == 0)
    proxy_indices = np.flatnonzero(labels == 1)
    bootstrap_values: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = np.concatenate(
            (
                rng.choice(real_indices, size=len(real_indices), replace=True),
                rng.choice(proxy_indices, size=len(proxy_indices), replace=True),
            )
        )
        value = float(roc_auc_score(labels[sampled], predictions[sampled]))
        bootstrap_values.append(max(value, 1.0 - value))
    confidence_interval = np.quantile(bootstrap_values, (0.025, 0.975))
    return {
        "raw_auc_proxy_positive": round(raw_auc, 6),
        "separability_auc": round(separability_auc, 6),
        "separability_auc_ci95": [
            round(float(confidence_interval[0]), 6),
            round(float(confidence_interval[1]), 6),
        ],
        "real_images": len(real),
        "proxy_images": len(proxy),
        "descriptor_dimensions": int(features.shape[1]),
        "cross_validation": {
            "unit": "image",
            "splitter": "RepeatedStratifiedKFold",
            "folds": active_folds,
            "repeats": repeats,
            "classifier": "StandardScaler + LogisticRegression",
        },
        "bootstrap_samples": bootstrap_samples,
        "interpretation": "0.5 is indistinguishable; 1.0 is perfectly separable",
    }
