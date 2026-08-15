from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .review_corpus import load_snapshot_records


QUALITY_RANKER_FORMAT = "yaruo-aa-quality-ranker-v1"


def _top_group(source_path: str) -> str:
    normalized = source_path.replace("\\", "/")
    return normalized.partition("/")[0] or "unknown"


def text_structure_features(text: str) -> dict[str, float]:
    lines = text.splitlines() or [text]
    visible = [character for character in text if not character.isspace()]
    visible_count = max(1, len(visible))
    nonempty_lines = [line for line in lines if line.strip()]
    widths = [len(line) for line in nonempty_lines] or [0]
    line_counts = Counter(line.strip() for line in nonempty_lines)
    repeated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
    symbols = sum(unicodedata.category(character)[0] in {"P", "S"} for character in visible)
    letters = sum(unicodedata.category(character)[0] == "L" for character in visible)
    numbers = sum(unicodedata.category(character)[0] == "N" for character in visible)
    whitespace = sum(character.isspace() for character in text)
    indented = sum(bool(line) and line[0].isspace() for line in nonempty_lines)
    width_mean = sum(widths) / len(widths)
    width_variance = sum((width - width_mean) ** 2 for width in widths) / len(widths)
    return {
        "visible_characters": float(len(visible)),
        "whitespace_ratio": whitespace / max(1, len(text)),
        "symbol_ratio": symbols / visible_count,
        "letter_ratio": letters / visible_count,
        "number_ratio": numbers / visible_count,
        "unique_visible_ratio": len(set(visible)) / visible_count,
        "nonempty_line_ratio": len(nonempty_lines) / max(1, len(lines)),
        "indented_line_ratio": indented / max(1, len(nonempty_lines)),
        "repeated_line_ratio": repeated_lines / max(1, len(nonempty_lines)),
        "mean_line_width": width_mean,
        "line_width_stddev": math.sqrt(width_variance),
    }


def build_quality_feature_matrix(
    snapshot_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    records = [
        record
        for record in load_snapshot_records(snapshot_dir)
        if record["decision"] in {"accept", "reject"}
    ]
    categories = sorted({str(record["corrected_category"]) for record in records})
    top_groups = sorted({_top_group(str(record["source_path"])) for record in records})
    structure_names = list(text_structure_features("").keys())
    numeric_names = [
        "line_count",
        "max_columns",
        "character_count",
        "art_score",
        "sensitive",
        "category_changed",
        "character_reference_transform",
    ]
    feature_names = (
        numeric_names
        + structure_names
        + [f"category={category}" for category in categories]
        + [f"top_group={group}" for group in top_groups]
    )
    matrix: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for record in records:
        text = (snapshot_dir / str(record["text"])).read_text(encoding="utf-8")
        structure = text_structure_features(text)
        category = str(record["corrected_category"])
        group = _top_group(str(record["source_path"]))
        row = [
            float(record["line_count"]),
            float(record["max_columns"]),
            float(record["character_count"]),
            float(record["art_score"]),
            float(bool(record["sensitive"])),
            float(record["indexed_category"] != record["corrected_category"]),
            float(bool(record["text_transforms"])),
        ]
        row.extend(structure[name] for name in structure_names)
        row.extend(float(category == candidate) for candidate in categories)
        row.extend(float(group == candidate) for candidate in top_groups)
        matrix.append(row)
        labels.append(int(record["decision"] == "accept"))
        groups.append(int(record["source_file_id"]))
    return (
        np.asarray(matrix, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int32),
        feature_names,
        records,
    )


def train_quality_ranker(
    snapshot_dir: Path,
    output_path: Path,
    *,
    estimators: int = 256,
    max_depth: int = 8,
    minimum_leaf_samples: int = 3,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    try:
        import joblib
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as error:
        raise RuntimeError("Quality-ranker training requires requirements-training.txt") from error

    features, labels, groups, feature_names, records = build_quality_feature_matrix(snapshot_dir)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Both accepted and rejected reviews are required")
    if len(set(groups.tolist())) < folds:
        raise ValueError(f"At least {folds} source groups are required")

    def estimator(random_state: int) -> object:
        return RandomForestClassifier(
            n_estimators=estimators,
            max_depth=max_depth,
            min_samples_leaf=minimum_leaf_samples,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    fold_records: list[dict[str, object]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, labels, groups),
        1,
    ):
        model = estimator(seed + fold)
        model.fit(features[train_indices], labels[train_indices])
        fold_probability = model.predict_proba(features[validation_indices])[:, 1]
        probabilities[validation_indices] = fold_probability
        fold_records.append(
            {
                "fold": fold,
                "train_samples": int(len(train_indices)),
                "validation_samples": int(len(validation_indices)),
                "train_source_groups": int(len(set(groups[train_indices].tolist()))),
                "validation_source_groups": int(
                    len(set(groups[validation_indices].tolist()))
                ),
                "validation_accepts": int(labels[validation_indices].sum()),
            }
        )

    accepted_count = int(labels.sum())
    ranked_indices = np.argsort(-probabilities)
    top_precision = float(labels[ranked_indices[:accepted_count]].mean())
    final_model = estimator(seed)
    final_model.fit(features, labels)
    importances = sorted(
        (
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(feature_names, final_model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    report: dict[str, object] = {
        "format": QUALITY_RANKER_FORMAT,
        "snapshot": json.loads(
            (snapshot_dir / "snapshot.json").read_text(encoding="utf-8")
        )["snapshot"],
        "seed": seed,
        "samples": int(len(labels)),
        "accepted_samples": accepted_count,
        "rejected_samples": int(len(labels) - accepted_count),
        "source_groups": int(len(set(groups.tolist()))),
        "features": len(feature_names),
        "folds": folds,
        "grouping": "source_file_id",
        "positive_rate_baseline": float(labels.mean()),
        "oof_roc_auc": float(roc_auc_score(labels, probabilities)),
        "oof_average_precision": float(average_precision_score(labels, probabilities)),
        "oof_brier_score": float(brier_score_loss(labels, probabilities)),
        "oof_top_accepted_count_precision": top_precision,
        "fold_records": fold_records,
        "top_feature_importances": importances[:20],
        "estimators": estimators,
        "max_depth": max_depth,
        "minimum_leaf_samples": minimum_leaf_samples,
        "scikit_learn_version": sklearn.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": report,
            "feature_names": feature_names,
            "estimator": final_model,
        },
        output_path,
        compress=3,
    )
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    predictions_path = output_path.with_suffix(".oof.jsonl")
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record, label, probability in zip(records, labels, probabilities):
            handle.write(
                json.dumps(
                    {
                        "entry_id": record["entry_id"],
                        "source_file_id": record["source_file_id"],
                        "decision": record["decision"],
                        "label": int(label),
                        "oof_accept_probability": float(probability),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return report
