from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.random_forest import (
    CLASS_COUNT,
    CONTEXT_SIZE,
    MODEL_FORMAT,
    character_digest,
    random_forest_features,
)

from .examples import TrainingExample, augment_context, extract_context, positive_examples


def load_characters(path: Path, *, minimum_frequency: int = 10) -> tuple[str, ...]:
    with path.open("r", encoding="cp932", newline="") as handle:
        rows = csv.DictReader(handle)
        characters = tuple(
            row["char"] for row in rows if int(row["frequency"]) >= minimum_frequency
        )
    if len(characters) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} characters, got {len(characters)}")
    return characters


def stratified_examples(
    examples: Iterable[TrainingExample],
    *,
    split: str,
    maximum_per_class: int,
    seed: int,
    require_all_classes: bool = True,
) -> list[TrainingExample]:
    if maximum_per_class < 1:
        raise ValueError("maximum_per_class must be positive")
    grouped: dict[int, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        if example.split == split and 0 <= example.char_label < CLASS_COUNT:
            grouped[example.char_label].append(example)
    rng = random.Random(seed)
    selected: list[TrainingExample] = []
    for label in range(CLASS_COUNT):
        candidates = grouped.get(label, [])
        if not candidates:
            if require_all_classes:
                raise ValueError(f"Split {split!r} has no examples for class {label}")
            continue
        if len(candidates) > maximum_per_class:
            candidates = rng.sample(candidates, maximum_per_class)
        selected.extend(candidates)
    rng.shuffle(selected)
    return selected


def _image_paths(manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        record["file_name"]: manifest_path.parent / record["image"]
        for record in manifest["works"]
    }


def build_feature_matrix(
    examples: list[TrainingExample],
    manifest_path: Path,
    *,
    augmentations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentations < 0:
        raise ValueError("augmentations must be zero or greater")
    paths = _image_paths(manifest_path)
    image_cache: dict[str, np.ndarray] = {}
    windows: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(seed)
    for example in examples:
        if example.file_name not in image_cache:
            image = cv2.imread(str(paths[example.file_name]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(paths[example.file_name])
            image_cache[example.file_name] = image
        window = extract_context(
            image_cache[example.file_name],
            example.x,
            example.y,
        )
        windows.append(window)
        labels.append(example.char_label)
        for _ in range(augmentations):
            windows.append(augment_context(window, rng))
            labels.append(example.char_label)
    return random_forest_features(np.stack(windows)), np.asarray(labels, dtype=np.int16)


def train_random_forest(
    csv_path: Path,
    manifest_path: Path,
    char_list_path: Path,
    output_path: Path,
    *,
    estimators: int = 64,
    maximum_train_per_class: int = 32,
    maximum_validation_per_class: int = 16,
    augmentations: int = 1,
    maximum_leaf_nodes: int = 512,
    seed: int = 42,
) -> dict[str, object]:
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
    except ImportError as error:
        raise RuntimeError(
            "Random Forest training requires requirements-training.txt"
        ) from error

    if estimators < 1 or maximum_leaf_nodes < 2:
        raise ValueError("estimators and maximum_leaf_nodes must be positive")
    all_examples = positive_examples(csv_path, manifest_path, class_count=CLASS_COUNT)
    train_examples = stratified_examples(
        all_examples,
        split="train",
        maximum_per_class=maximum_train_per_class,
        seed=seed,
    )
    validation_examples = stratified_examples(
        all_examples,
        split="validation",
        maximum_per_class=maximum_validation_per_class,
        seed=seed + 1,
        require_all_classes=False,
    )
    train_x, train_y = build_feature_matrix(
        train_examples,
        manifest_path,
        augmentations=augmentations,
        seed=seed,
    )
    validation_x, validation_y = build_feature_matrix(
        validation_examples,
        manifest_path,
        augmentations=0,
        seed=seed + 2,
    )

    estimator = RandomForestClassifier(
        n_estimators=estimators,
        max_features="sqrt",
        max_leaf_nodes=maximum_leaf_nodes,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    estimator.fit(train_x, train_y)
    prediction = estimator.predict(validation_x)
    probabilities = estimator.predict_proba(validation_x)
    characters = load_characters(char_list_path)
    metadata: dict[str, object] = {
        "format": MODEL_FORMAT,
        "class_count": CLASS_COUNT,
        "context_size": CONTEXT_SIZE,
        "feature": "binary-raw-pixels",
        "character_sha256": character_digest(characters),
        "seed": seed,
        "estimators": estimators,
        "maximum_leaf_nodes": maximum_leaf_nodes,
        "maximum_train_per_class": maximum_train_per_class,
        "maximum_validation_per_class": maximum_validation_per_class,
        "augmentations": augmentations,
        "train_samples": int(len(train_y)),
        "validation_samples": int(len(validation_y)),
        "validation_accuracy": float(accuracy_score(validation_y, prediction)),
        "validation_macro_f1": float(
            f1_score(validation_y, prediction, average="macro", zero_division=0)
        ),
        "validation_top5_accuracy": float(
            top_k_accuracy_score(
                validation_y,
                probabilities,
                k=5,
                labels=np.arange(CLASS_COUNT),
            )
        ),
        "scikit_learn_version": sklearn.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"metadata": metadata, "estimator": estimator},
        output_path,
        compress=3,
    )
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata
