from __future__ import annotations

import base64
import collections
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from backend.deepaa import convert_deepaa, line_match_cost
from backend.contracts import ConversionOptions


def decode_data_url(data_url: str) -> bytes:
    _, payload = data_url.split(",", 1)
    return base64.b64decode(payload)


def decode_png(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode generated PNG.")
    return image


def pad_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = max(first.shape[0], second.shape[0])
    width = max(first.shape[1], second.shape[1])
    padded: list[np.ndarray] = []
    for image in (first, second):
        canvas = np.full((height, width), 255, dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        padded.append(canvas)
    return padded[0], padded[1]


def _component_count(ink: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    return max(0, count - 1)


def geometric_metrics(
    target: np.ndarray,
    rendered: np.ndarray,
    *,
    tolerance: float = 2.0,
) -> dict[str, float | int]:
    target, rendered = pad_pair(target, rendered)
    target_ink = target < 128
    rendered_ink = rendered < 128
    target_count = int(target_ink.sum())
    rendered_count = int(rendered_ink.sum())

    if target_count and rendered_count:
        distance_to_target = cv2.distanceTransform(
            (~target_ink).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_to_rendered = cv2.distanceTransform(
            (~rendered_ink).astype(np.uint8), cv2.DIST_L2, 3
        )
        recall = float((distance_to_rendered[target_ink] <= tolerance).mean())
        precision = float((distance_to_target[rendered_ink] <= tolerance).mean())
    else:
        recall = float(target_count == rendered_count)
        precision = recall
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "distance_cost": round(line_match_cost(target, rendered), 6),
        "ink_precision_at_2px": round(precision, 6),
        "ink_recall_at_2px": round(recall, 6),
        "ink_f1_at_2px": round(f1, 6),
        "target_ink_pixels": target_count,
        "rendered_ink_pixels": rendered_count,
        "target_components": _component_count(target_ink),
        "rendered_components": _component_count(rendered_ink),
    }


def text_usage_metrics(text: str) -> dict[str, object]:
    characters = [char for char in text if not char.isspace()]
    counts = collections.Counter(characters)
    total = len(characters)
    common = counts.most_common(10)
    return {
        "non_whitespace_characters": total,
        "unique_characters": len(counts),
        "top_2_share": round(sum(count for _, count in common[:2]) / total, 6)
        if total
        else 0.0,
        "top_10_share": round(sum(count for _, count in common) / total, 6)
        if total
        else 0.0,
        "most_common": [
            {"character": char, "count": count} for char, count in common
        ],
    }


def evaluate_cases(
    cases_path: Path,
    output_dir: Path,
    *,
    decoder_override: str | None = None,
    classifier_override: str | None = None,
    random_forest_model: Path | None = None,
    structure_strength: float = 0.0,
    selected_ids: set[str] | None = None,
) -> dict[str, object]:
    specification = json.loads(cases_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for case in specification["cases"]:
        if selected_ids is not None and case["id"] not in selected_ids:
            continue
        source_path = (cases_path.parent / case["source"]).resolve()
        source_data = source_path.read_bytes()
        options = ConversionOptions(**case["options"])
        started = time.perf_counter()
        decoder = decoder_override or case.get("decoder", "beam")
        classifier = classifier_override or case.get("classifier", "cnn")
        result = convert_deepaa(
            source_data,
            options,
            decoder=decoder,
            classifier=classifier,
            random_forest_model=random_forest_model,
            structure_strength=structure_strength,
        )
        elapsed = time.perf_counter() - started
        processed_data = decode_data_url(result.processed_png)
        rendered_data = decode_data_url(result.rendered_png)
        target = decode_png(processed_data)
        rendered = decode_png(rendered_data)

        case_id = case["id"]
        (output_dir / f"{case_id}.txt").write_text(
            result.ascii_text, encoding="utf-8", newline="\n"
        )
        (output_dir / f"{case_id}-target.png").write_bytes(processed_data)
        (output_dir / f"{case_id}-rendered.png").write_bytes(rendered_data)
        records.append(
            {
                "id": case_id,
                "kind": case["kind"],
                "source": str(source_path),
                "source_sha256": hashlib.sha256(source_data).hexdigest(),
                "options": case["options"],
                "decoder": decoder,
                "classifier": classifier,
                "structure_strength": structure_strength,
                "rows": result.rows,
                "columns": result.columns,
                "elapsed_seconds": round(elapsed, 3),
                "metrics": geometric_metrics(target, rendered),
                "text_usage": text_usage_metrics(result.ascii_text),
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "metric_scope": (
            "Geometric agreement between the extracted line image and rendered AA. "
            "Semantic recognizability still requires the fixed human acceptance checklist."
        ),
        "cases_file": str(cases_path.resolve()),
        "random_forest_model": (
            str(random_forest_model.resolve()) if random_forest_model else None
        ),
        "cases": records,
    }
    if selected_ids is not None and len(records) != len(selected_ids):
        found = {record["id"] for record in records}
        missing = selected_ids - found
        raise ValueError(f"Unknown evaluation case IDs: {', '.join(sorted(missing))}")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
