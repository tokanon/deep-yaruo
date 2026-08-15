from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.input_channels import (
    ChannelExtractionConfig,
    channel_preview,
    extract_tone,
    normalize_geometry,
)
from training.review_corpus import load_snapshot_records, sha256_bytes
from training.structure_proxy import domain_auc
from training.structure_proxy_dataset import load_vocabulary, text_character_examples
from training.weak_pairs import render_aa_text, select_pilot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_REFERENCES = ROOT / ".tmp" / "input-channels" / "manifest.json"
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "tone-window-p5"
DEFAULT_WIDTHS = (24, 36, 48, 64, 80)
DEFAULT_GAMMAS = (1.0, 0.75, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5 ch1 window sweep: domain AUC, exact-character leakage, tone retention."
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--widths", nargs="*", type=int, default=list(DEFAULT_WIDTHS))
    parser.add_argument("--gammas", nargs="*", type=float, default=list(DEFAULT_GAMMAS))
    parser.add_argument("--domain-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tone_descriptor(tone: np.ndarray, *, levels: int = 4) -> np.ndarray:
    if tone.ndim != 2:
        raise ValueError("tone descriptor expects a two-dimensional channel")
    values = np.clip(tone.astype(np.float32), 0.0, 1.0)
    spatial = cv2.resize(values, (16, 16), interpolation=cv2.INTER_AREA)
    horizontal = spatial.mean(axis=1)
    vertical = spatial.mean(axis=0)
    quantized = np.rint(values * (levels - 1)).astype(np.uint8)
    histogram = np.bincount(quantized.ravel(), minlength=levels).astype(np.float32)
    histogram /= max(1.0, float(histogram.sum()))
    summary = np.array(
        [
            float(values.mean()),
            float(values.std()),
            float(np.mean(np.abs(np.diff(values, axis=0)))) if values.shape[0] > 1 else 0.0,
            float(np.mean(np.abs(np.diff(values, axis=1)))) if values.shape[1] > 1 else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate((spatial.ravel(), horizontal, vertical, histogram, summary))


def sample_tone_context(tone: np.ndarray, x: int, y: int, *, size: int = 8) -> np.ndarray:
    """Sample the same 64px context on a coarse grid without copying a full patch."""
    if tone.ndim != 2 or size <= 0:
        raise ValueError("tone must be 2D and size must be positive")
    margin = 23
    centers = np.floor((np.arange(size, dtype=np.float32) + 0.5) * (64.0 / size)).astype(int)
    xs = x - margin + centers
    ys = y - margin + centers
    valid_x = (xs >= 0) & (xs < tone.shape[1])
    valid_y = (ys >= 0) & (ys < tone.shape[0])
    sampled = np.zeros((size, size), dtype=np.float32)
    if valid_x.any() and valid_y.any():
        sampled[np.ix_(valid_y, valid_x)] = tone[np.ix_(ys[valid_y], xs[valid_x])]
    return sampled.ravel()


def _leakage_metrics(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    blank_label: int,
    seed: int,
) -> dict[str, object]:
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - training environment only
        raise RuntimeError("P5 tone leakage requires requirements-training.txt") from exc
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0, class_weight="balanced"),
    )
    classifier.fit(train_features, train_labels)
    prediction = classifier.predict(test_features)
    nonblank = test_labels != blank_label
    class_totals: Counter[int] = Counter(test_labels.tolist())
    class_correct: Counter[int] = Counter(
        test_labels[prediction == test_labels].tolist()
    )
    return {
        "all_accuracy": float(accuracy_score(test_labels, prediction)),
        "nonblank_accuracy": float(accuracy_score(test_labels[nonblank], prediction[nonblank])),
        "blank_accuracy": float(
            accuracy_score(test_labels[~nonblank], prediction[~nonblank])
        ),
        "macro_accuracy_present_classes": float(
            np.mean(
                [class_correct[label] / count for label, count in class_totals.items()]
            )
        ),
        "train_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "feature_dimensions": int(train_features.shape[1]),
        "classifier": "StandardScaler + class-balanced RidgeClassifier",
        "interpretation": (
            "Higher accuracy means ch1 alone leaks more exact-character information; "
            "this is not final model charAcc."
        ),
    }


def _tone_retention(tones: list[np.ndarray]) -> dict[str, float]:
    spatial_std = np.asarray([float(tone.std()) for tone in tones])
    occupied_levels = np.asarray(
        [len(np.unique(np.rint(tone * 3).astype(np.uint8))) for tone in tones]
    )
    return {
        "mean_spatial_std": float(spatial_std.mean()),
        "median_spatial_std": float(np.median(spatial_std)),
        "mean_occupied_levels": float(occupied_levels.mean()),
    }


def _select_tone_conditions(reports: dict[str, object]) -> dict[str, object]:
    baseline_id = "w48-g1" if "w48-g1" in reports else next(iter(reports))
    baseline = reports[baseline_id]
    baseline_std = float(baseline["real_tone_retention"]["mean_spatial_std"])
    baseline_auc = float(baseline["domain"]["separability_auc"])
    eligible: list[tuple[float, int, float, float, str]] = []
    rejected: dict[str, list[str]] = {}
    for condition_id, report in reports.items():
        reasons: list[str] = []
        std_ratio = float(report["real_tone_retention"]["mean_spatial_std"]) / max(
            baseline_std, 1e-9
        )
        auc = float(report["domain"]["separability_auc"])
        proxy_levels = float(report["proxy_tone_retention"]["mean_occupied_levels"])
        if std_ratio < 0.75:
            reasons.append("real_tone_spatial_std_below_75pct_of_w48")
        if proxy_levels < 1.5:
            reasons.append("aa_proxy_tone_uses_fewer_than_1.5_levels_on_average")
        if reasons:
            rejected[condition_id] = reasons
        else:
            eligible.append(
                (
                    float(report["exact_character_leakage"]["nonblank_accuracy"]),
                    int(report["config"]["tone_window_width"]),
                    auc,
                    -float(report["proxy_tone_retention"]["mean_spatial_std"]),
                    condition_id,
                )
            )
        report["relative_to_w48_g1"] = {
            "real_spatial_std_ratio": std_ratio,
            "domain_auc_delta": auc - baseline_auc,
            "nonblank_leakage_delta": (
                float(report["exact_character_leakage"]["nonblank_accuracy"])
                - float(baseline["exact_character_leakage"]["nonblank_accuracy"])
            ),
        }
        report.pop("relative_to_w48", None)
    eligible.sort()
    selected_id = eligible[0][4] if eligible else baseline_id
    shortlisted_ids = [item[4] for item in eligible[:3]]
    return {
        "selected_condition": selected_id,
        "selected_config": reports[selected_id]["config"],
        "shortlisted_for_p6_c2": shortlisted_ids,
        "baseline_condition": baseline_id,
        "rejected": rejected,
        "gate": {
            "real_tone_spatial_std_ratio_min_vs_w48_g1": 0.75,
            "aa_proxy_mean_occupied_levels_min": 1.5,
            "ranking": (
                "lower exact-character leakage, then smaller tone window to retain "
                "small regions, then lower tone-domain AUC"
            ),
        },
        "status": "provisional shortlist for P6 C=2; no real-image charAcc is claimed",
    }


def main() -> None:
    args = parse_args()
    if not args.widths or any(width <= 0 for width in args.widths):
        raise ValueError("tone widths must be positive")
    if not args.gammas or any(gamma <= 0 for gamma in args.gammas):
        raise ValueError("tone gammas must be positive")
    if args.domain_limit < 2 or args.bootstrap_samples <= 0:
        raise ValueError("domain-limit must be >=2 and bootstrap-samples positive")
    output = args.output.resolve()
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    if args.selection_only:
        report_path = output / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"P5 tone report is missing: {report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["selection"] = _select_tone_conditions(payload["windows"])
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(payload["selection"], ensure_ascii=False, indent=2))
        print(f"saved={report_path.resolve()}")
        return
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"P5 tone output is not empty: {output}; use --force")
        if allowed_root not in output.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    reference_manifest = json.loads(args.references.read_text(encoding="utf-8"))
    if not reference_manifest.get("stratification", {}).get("gate_passed"):
        raise ValueError("P1 reference stratification gate has not passed")
    real_inputs: list[tuple[str, np.ndarray, str]] = []
    for record in reference_manifest["records"]:
        source_path = ROOT / str(record["source"])
        if _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"P1 source hash mismatch: {source_path}")
        gray = np.asarray(Image.open(source_path).convert("L"))
        source_kind = str(record["metadata"]["source_kind"])
        real_inputs.append((str(record["id"]), gray, source_kind))

    snapshot_payload = (args.snapshot / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_payload)
    accepted = [
        record
        for record in load_snapshot_records(args.snapshot)
        if record["decision"] == "accept" and record["split"] in {"train", "validation", "test"}
    ]
    domain_records = select_pilot_records(accepted, limit=args.domain_limit, seed=args.seed)
    domain_ids = {int(record["entry_id"]) for record in domain_records}
    vocabulary = load_vocabulary(args.charset)
    blank_label = next(entry.label for entry in vocabulary if entry.char == " ")
    work_data: list[dict[str, object]] = []
    for record in accepted:
        text_path = args.snapshot / str(record["text"])
        payload = text_path.read_bytes()
        if sha256_bytes(payload) != record["unicode_text_sha256"]:
            raise ValueError(f"Accepted AA text hash mismatch: {text_path}")
        text = payload.decode("utf-8")
        rendered = render_aa_text(text, font_size=16)
        examples, diagnostics = text_character_examples(
            text,
            vocabulary,
            source_width=rendered.shape[1],
            target_width=512,
            font_size=16,
        )
        work_data.append(
            {
                "entry_id": int(record["entry_id"]),
                "split": str(record["split"]),
                "rendered": rendered,
                "examples": examples,
                "diagnostics": diagnostics,
            }
        )

    reports: dict[str, object] = {}
    base_config = ChannelExtractionConfig(target_width=512)
    conditions = (
        (width, gamma)
        for width in sorted(set(args.widths))
        for gamma in sorted(set(args.gammas), reverse=True)
    )
    for width, gamma in conditions:
        height = max(1, int(round(width * 54 / 48)))
        config = replace(
            base_config,
            tone_window_width=width,
            tone_window_height=height,
            tone_sigma=width / 4.0,
            tone_gamma=gamma,
        )
        gamma_text = f"{gamma:g}"
        condition_id = f"w{width}-g{gamma_text}"
        real_tones: list[np.ndarray] = []
        real_features: list[np.ndarray] = []
        preview_dir = output / "previews" / condition_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        for record_id, gray, _source_kind in real_inputs:
            tone = extract_tone(normalize_geometry(gray, config), config)
            real_tones.append(tone)
            real_features.append(tone_descriptor(tone, levels=config.tone_levels))
            Image.fromarray(channel_preview(tone)).save(
                preview_dir / f"real-{record_id}.png",
                optimize=True,
            )

        proxy_features: list[np.ndarray] = []
        train_features: list[np.ndarray] = []
        train_labels: list[int] = []
        test_features: list[np.ndarray] = []
        test_labels: list[int] = []
        proxy_tones: list[np.ndarray] = []
        for work in work_data:
            tone = extract_tone(normalize_geometry(work["rendered"], config), config)
            if int(work["entry_id"]) in domain_ids:
                proxy_tones.append(tone)
                proxy_features.append(tone_descriptor(tone, levels=config.tone_levels))
                Image.fromarray(channel_preview(tone)).save(
                    preview_dir / f"proxy-{int(work['entry_id']):07d}.png",
                    optimize=True,
                )
            split = str(work["split"])
            if split not in {"train", "test"}:
                continue
            target_features = train_features if split == "train" else test_features
            target_labels = train_labels if split == "train" else test_labels
            for example in work["examples"]:
                target_features.append(
                    sample_tone_context(tone, int(example["x"]), int(example["y"]))
                )
                target_labels.append(int(example["label"]))
        leakage = _leakage_metrics(
            np.stack(train_features),
            np.asarray(train_labels, dtype=np.int64),
            np.stack(test_features),
            np.asarray(test_labels, dtype=np.int64),
            blank_label=blank_label,
            seed=args.seed,
        )
        reports[condition_id] = {
            "config": {
                "tone_window_width": width,
                "tone_window_height": height,
                "tone_sigma": width / 4.0,
                "tone_gamma": gamma,
                "tone_levels": config.tone_levels,
            },
            "domain": domain_auc(
                np.stack(real_features),
                np.stack(proxy_features),
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            ),
            "real_tone_retention": _tone_retention(real_tones),
            "proxy_tone_retention": _tone_retention(proxy_tones),
            "exact_character_leakage": leakage,
        }
        print(
            json.dumps(
                {
                    "width": width,
                    "gamma": gamma,
                    "auc": reports[condition_id]["domain"]["separability_auc"],
                    "leak_nonblank": leakage["nonblank_accuracy"],
                },
                ensure_ascii=False,
            )
        )

    selection = _select_tone_conditions(reports)
    report = {
        "schema_version": 1,
        "phase": "P5",
        "dataset": "tone-window-p5",
        "purpose": "choose a P6 ch1 scale without claiming real-image charAcc",
        "seed": args.seed,
        "snapshot": snapshot["snapshot"],
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "reference_manifest": str(args.references.resolve()),
        "reference_manifest_sha256": _sha256(args.references),
        "domain_aa_work_ids": sorted(domain_ids),
        "windows": reports,
        "selection": selection,
        "interpretation": {
            "real_images": (
                "Only tone-domain AUC and tone retention are measured. The reference "
                "images have no gold AA, so real-image charAcc is not computed."
            ),
            "synthetic_leakage": (
                "A linear classifier predicts held-out AA characters from ch1 alone. "
                "Lower is safer, but P6 C=2 downstream and human output still decide utility."
            ),
        },
        "rights_status": "local diagnostics only; images and AA are not tracked",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    print(f"saved={(output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
