from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_FORMAT = "yaruo-aa-random-forest-v1"
CLASS_COUNT = 411
CONTEXT_SIZE = 64


def character_digest(characters: Iterable[str]) -> str:
    payload = "\0".join(characters).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def random_forest_features(windows: np.ndarray) -> np.ndarray:
    """Flatten 64x64 binary line contexts without learned feature extraction."""
    array = np.asarray(windows)
    if array.ndim == 2:
        array = array[np.newaxis]
    if array.ndim != 3 or array.shape[1:] != (CONTEXT_SIZE, CONTEXT_SIZE):
        raise ValueError("Random Forest contexts must have shape (N, 64, 64)")
    return (array < 128).reshape(len(array), -1).astype(np.uint8, copy=False)


@lru_cache(maxsize=4)
def load_random_forest(path_value: str, character_sha256: str) -> object:
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "Random Forest inference requires scikit-learn and joblib from "
            "requirements-training.txt"
        ) from error
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Random Forest model is missing: {path}")
    bundle = joblib.load(path)
    if (
        not isinstance(bundle, dict)
        or "metadata" not in bundle
        or "estimator" not in bundle
    ):
        raise ValueError("Random Forest model bundle is invalid")
    metadata = bundle["metadata"]
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError("Unsupported Random Forest model format")
    if metadata.get("class_count") != CLASS_COUNT:
        raise ValueError("Random Forest model has an incompatible class count")
    if metadata.get("character_sha256") != character_sha256:
        raise ValueError("Random Forest model uses a different character order")
    estimator = bundle["estimator"]
    # The decoder already batches many contexts. Recreating a joblib worker
    # pool for every small predict_proba call is slower and noisy on Windows.
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    return estimator


def predict_probabilities(
    windows: np.ndarray,
    characters: tuple[str, ...],
    model_path: Path,
) -> np.ndarray:
    estimator = load_random_forest(
        str(model_path.resolve()),
        character_digest(characters),
    )
    raw = np.asarray(estimator.predict_proba(random_forest_features(windows)))
    classes = np.asarray(estimator.classes_, dtype=np.int32)
    if raw.ndim != 2 or len(classes) != raw.shape[1]:
        raise ValueError("Random Forest probability output is invalid")
    probabilities = np.full((len(windows), CLASS_COUNT), 1e-8, dtype=np.float32)
    probabilities[:, classes] = raw.astype(np.float32, copy=False)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities
