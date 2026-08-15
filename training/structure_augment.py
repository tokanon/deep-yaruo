from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np


RenderAugment = Literal[
    "raw",
    "gaussian_035",
    "jpeg_85",
    "resize_90",
    "ink_dilate_cross_1",
]


@dataclass(frozen=True)
class WeightedAugment:
    augment: RenderAugment
    weight: int

    def validate(self) -> None:
        if self.weight <= 0:
            raise ValueError("augment weight must be positive")


@dataclass(frozen=True)
class StructureAugmentPolicy:
    policy_id: str
    variants: tuple[WeightedAugment, ...]
    description: str

    def validate(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id cannot be empty")
        if not self.variants:
            raise ValueError("an augment policy needs at least one variant")
        seen: set[str] = set()
        for variant in self.variants:
            variant.validate()
            if variant.augment in seen:
                raise ValueError(f"duplicate augment: {variant.augment}")
            seen.add(variant.augment)

    @property
    def total_weight(self) -> int:
        return sum(item.weight for item in self.variants)

    def manifest(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "description": self.description,
            "variants": [asdict(item) for item in self.variants],
            "total_weight": self.total_weight,
            "destructive_line_dropout": False,
            "coordinate_transform": False,
        }


STRUCTURE_AUGMENT_POLICIES: dict[str, StructureAugmentPolicy] = {
    "raw_only": StructureAugmentPolicy(
        policy_id="raw_only",
        variants=(WeightedAugment("raw", 1),),
        description="Unmodified Saitamaar raster baseline.",
    ),
    "processing_25": StructureAugmentPolicy(
        policy_id="processing_25",
        variants=(
            WeightedAugment("raw", 9),
            WeightedAugment("gaussian_035", 1),
            WeightedAugment("jpeg_85", 1),
            WeightedAugment("resize_90", 1),
        ),
        description="75% raw plus 25% alignment-preserving render degradation.",
    ),
    "processing_50": StructureAugmentPolicy(
        policy_id="processing_50",
        variants=(
            WeightedAugment("raw", 3),
            WeightedAugment("gaussian_035", 1),
            WeightedAugment("jpeg_85", 1),
            WeightedAugment("resize_90", 1),
        ),
        description="50% raw plus 50% alignment-preserving render degradation.",
    ),
    "stroke_25": StructureAugmentPolicy(
        policy_id="stroke_25",
        variants=(
            WeightedAugment("raw", 3),
            WeightedAugment("ink_dilate_cross_1", 1),
        ),
        description="75% raw plus 25% one-pixel ink widening; no line deletion.",
    ),
    "combined_50": StructureAugmentPolicy(
        policy_id="combined_50",
        variants=(
            WeightedAugment("raw", 4),
            WeightedAugment("gaussian_035", 1),
            WeightedAugment("jpeg_85", 1),
            WeightedAugment("resize_90", 1),
            WeightedAugment("ink_dilate_cross_1", 1),
        ),
        description="50% raw plus 50% conservative processing/stroke variation.",
    ),
}

for _policy in STRUCTURE_AUGMENT_POLICIES.values():
    _policy.validate()


def get_augment_policy(policy_id: str) -> StructureAugmentPolicy:
    try:
        return STRUCTURE_AUGMENT_POLICIES[policy_id]
    except KeyError as exc:
        choices = ", ".join(sorted(STRUCTURE_AUGMENT_POLICIES))
        raise ValueError(f"Unknown augment policy {policy_id!r}; choose from {choices}") from exc


def apply_render_augment(image: np.ndarray, augment: RenderAugment) -> np.ndarray:
    """Apply a coordinate-preserving AA-raster perturbation for ch0 training."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("render augmentation expects a two-dimensional uint8 image")
    if augment == "raw":
        return image.copy()
    if augment == "gaussian_035":
        return cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=0.35,
            sigmaY=0.35,
            borderType=cv2.BORDER_REPLICATE,
        )
    if augment == "jpeg_85":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("OpenCV failed to encode the P5 JPEG variant")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.shape != image.shape:
            raise RuntimeError("OpenCV failed to decode the P5 JPEG variant")
        return decoded
    if augment == "resize_90":
        height, width = image.shape
        small_width = max(1, int(round(width * 0.9)))
        small_height = max(1, int(round(height * 0.9)))
        smaller = cv2.resize(
            image,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.resize(smaller, (width, height), interpolation=cv2.INTER_CUBIC)
    if augment == "ink_dilate_cross_1":
        ink = 255 - image
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        return 255 - cv2.dilate(ink, kernel, iterations=1)
    raise ValueError(f"Unknown render augment: {augment}")


def choose_augment(policy: StructureAugmentPolicy, key: str, *, seed: int) -> RenderAugment:
    """Choose one weighted variant reproducibly without process-global RNG state."""
    digest = hashlib.sha256(f"{seed}:{policy.policy_id}:{key}".encode("utf-8")).digest()
    slot = int.from_bytes(digest[:8], "big") % policy.total_weight
    cursor = 0
    for variant in policy.variants:
        cursor += variant.weight
        if slot < cursor:
            return variant.augment
    raise AssertionError("weighted augment selection did not resolve a slot")


def grouped_domain_auc(
    real_features: np.ndarray,
    proxy_features: np.ndarray,
    *,
    real_groups: list[str],
    proxy_groups: list[str],
    seed: int = 42,
    folds: int = 5,
    repeats: int = 5,
    bootstrap_samples: int = 1_000,
) -> dict[str, object]:
    """Estimate domain separation while keeping all variants of a work together."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError(
            "P5 grouped domain AUC requires requirements-training.txt."
        ) from exc

    real = np.asarray(real_features, dtype=np.float32)
    proxy = np.asarray(proxy_features, dtype=np.float32)
    if real.ndim != 2 or proxy.ndim != 2 or real.shape[1] != proxy.shape[1]:
        raise ValueError("Domain features must be matrices with the same width")
    if len(real_groups) != len(real) or len(proxy_groups) != len(proxy):
        raise ValueError("Every feature row must have one group")
    if set(real_groups) & set(proxy_groups):
        raise ValueError("Real and proxy group identifiers must not overlap")

    features = np.concatenate((real, proxy), axis=0)
    labels = np.concatenate(
        (np.zeros(len(real), dtype=np.uint8), np.ones(len(proxy), dtype=np.uint8))
    )
    groups = np.asarray(real_groups + proxy_groups, dtype=object)
    real_unique = sorted(set(real_groups))
    proxy_unique = sorted(set(proxy_groups))
    active_folds = min(folds, len(real_unique), len(proxy_unique))
    if active_folds < 2:
        raise ValueError("Grouped domain AUC requires two works in each domain")

    prediction_sum = np.zeros(len(labels), dtype=np.float64)
    prediction_count = np.zeros(len(labels), dtype=np.int32)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        real_order = rng.permutation(real_unique).tolist()
        proxy_order = rng.permutation(proxy_unique).tolist()
        for fold in range(active_folds):
            test_groups = set(real_order[fold::active_folds]) | set(
                proxy_order[fold::active_folds]
            )
            test_mask = np.fromiter(
                (group in test_groups for group in groups),
                dtype=bool,
                count=len(groups),
            )
            train_indices = np.flatnonzero(~test_mask)
            test_indices = np.flatnonzero(test_mask)
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed + repeat,
                    solver="liblinear",
                ),
            )
            classifier.fit(features[train_indices], labels[train_indices])
            prediction_sum[test_indices] += classifier.predict_proba(
                features[test_indices]
            )[:, 1]
            prediction_count[test_indices] += 1
    if np.any(prediction_count != repeats):
        raise AssertionError("Every image variant must be tested once per repeat")
    predictions = prediction_sum / prediction_count
    raw_auc = float(roc_auc_score(labels, predictions))
    separability_auc = max(raw_auc, 1.0 - raw_auc)

    rng = np.random.default_rng(seed)
    group_to_indices = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups))
    }
    bootstrap_values: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_groups = rng.choice(real_unique, len(real_unique), replace=True).tolist()
        sampled_groups += rng.choice(proxy_unique, len(proxy_unique), replace=True).tolist()
        sampled_indices = np.concatenate([group_to_indices[group] for group in sampled_groups])
        value = float(roc_auc_score(labels[sampled_indices], predictions[sampled_indices]))
        bootstrap_values.append(max(value, 1.0 - value))
    interval = np.quantile(bootstrap_values, (0.025, 0.975))
    return {
        "raw_auc_proxy_positive": round(raw_auc, 6),
        "separability_auc": round(separability_auc, 6),
        "separability_auc_ci95": [round(float(value), 6) for value in interval],
        "real_images": len(real),
        "proxy_variants": len(proxy),
        "real_groups": len(real_unique),
        "proxy_work_groups": len(proxy_unique),
        "descriptor_dimensions": int(features.shape[1]),
        "cross_validation": {
            "unit": "work_group",
            "folds": active_folds,
            "repeats": repeats,
            "classifier": "StandardScaler + LogisticRegression",
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "work_group",
        "interpretation": "0.5 is indistinguishable; 1.0 is perfectly separable",
    }
