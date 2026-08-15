from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from training.examples import extract_context
from training.model import DeepAAMultitask, load_pretrained_backbone
from training.multichannel_dataset import (
    apply_tone_nuisance,
    prepare_c2_character_dataset,
)
from training.train_structure_proxy_p3 import _paired_work_bootstrap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"
DEFAULT_PRETRAINED = ROOT / "models" / "deepaa-light.onnx"
DEFAULT_TONE_REPORT = ROOT / ".tmp" / "training-runs" / "tone-window-p5" / "report.json"
DEFAULT_P3_REPORT = ROOT / ".tmp" / "training-runs" / "structure-proxy-p3" / "report.json"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "multichannel-p6"


@lru_cache(maxsize=2_048)
def _read_channels(path: str) -> np.ndarray:
    with np.load(path) as payload:
        channels = payload["channels"].copy()
    if channels.ndim != 3 or channels.shape[0] != 2:
        raise ValueError(f"P6 C=2 channels have an invalid shape: {channels.shape}")
    return channels


class C2CharacterDataset(Dataset[tuple[torch.Tensor, int, int, bool]]):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        tone_nuisance_probability: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parsed = [
            json.loads(line)
            for line in (self.root / self.manifest["examples"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.examples = [item for item in parsed if item["split"] == split]
        self.split = split
        self.tone_nuisance_probability = tone_nuisance_probability
        self.seed = seed
        self.vocabulary = self.manifest["vocabulary"]
        self.blank_label = next(
            int(entry["label"]) for entry in self.vocabulary if entry["char"] == " "
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, bool]:
        example = self.examples[index]
        channels = _read_channels(str(self.root / str(example["channels"])))
        windows = np.stack(
            [
                extract_context(channel, int(example["x"]), int(example["y"]))
                for channel in channels
            ]
        )
        if self.split == "train" and windows.shape[0] >= 2:
            windows[1] = apply_tone_nuisance(
                windows[1],
                (
                    f"{self.seed}:{example['entry_id']}:{example['x']}:"
                    f"{example['y']}:{example['label']}"
                ),
                probability=self.tone_nuisance_probability,
            )
        tensor = torch.from_numpy(windows.copy()).float() / 255.0
        return (
            tensor,
            int(example["label"]),
            int(example["entry_id"]),
            bool(example["is_fill_instance"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P6 C=2 structure+tone comparison using the P5 shortlist."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--tone-report", type=Path, default=DEFAULT_TONE_REPORT)
    parser.add_argument("--p3-report", type=Path, default=DEFAULT_P3_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--tone-nuisance-probability",
        type=float,
        default=0.0,
        help="Train-only probability of adding unrelated broad tone to each context.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _make_loader(
    dataset: C2CharacterDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    maximum: int | None = None,
) -> DataLoader:
    selected: Dataset = dataset
    if maximum is not None and maximum < len(dataset):
        selected = Subset(dataset, random.Random(seed).sample(range(len(dataset)), maximum))
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def _training_epoch(
    model: DeepAAMultitask,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0
    for inputs, targets, _, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        loss_total += float(loss.detach()) * len(targets)
        correct += int((logits.argmax(1) == targets).sum())
        count += len(targets)
    return {
        "loss": loss_total / max(1, count),
        "character_accuracy": correct / max(1, count),
        "character_count": count,
    }


def _evaluate(
    model: DeepAAMultitask,
    loader: DataLoader,
    device: torch.device,
    vocabulary: list[dict[str, object]],
    blank_label: int,
    *,
    structure_only: bool = False,
) -> dict[str, object]:
    model.eval()
    label_to_band = {
        int(entry["label"]): str(entry["frequency_band"]) for entry in vocabulary
    }
    label_to_char = {int(entry["label"]): str(entry["char"]) for entry in vocabulary}
    totals: Counter[str] = Counter()
    corrects: Counter[str] = Counter()
    band_totals: Counter[str] = Counter()
    band_corrects: Counter[str] = Counter()
    class_totals: Counter[int] = Counter()
    class_corrects: Counter[int] = Counter()
    work_totals: Counter[int] = Counter()
    work_corrects: Counter[int] = Counter()
    work_nonblank_totals: Counter[int] = Counter()
    work_nonblank_corrects: Counter[int] = Counter()
    loss_total = 0.0
    with torch.no_grad():
        for inputs, targets, entry_ids, fill_flags in loader:
            if structure_only:
                inputs = inputs[:, :1]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = model(inputs)
            loss_total += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            predictions = logits.argmax(1)
            top5 = logits.topk(5, dim=1).indices
            matches = predictions == targets
            top5_matches = (top5 == targets.unsqueeze(1)).any(1)
            for label, matched, top5_matched, entry_id, is_fill in zip(
                targets.cpu().tolist(),
                matches.cpu().tolist(),
                top5_matches.cpu().tolist(),
                entry_ids.tolist(),
                fill_flags.tolist(),
            ):
                totals["all"] += 1
                corrects["all"] += int(matched)
                corrects["top5"] += int(top5_matched)
                kind = "blank" if label == blank_label else "nonblank"
                totals[kind] += 1
                corrects[kind] += int(matched)
                fill_kind = "fill_instance" if is_fill else "nonfill_instance"
                totals[fill_kind] += 1
                corrects[fill_kind] += int(matched)
                band = label_to_band[label]
                band_totals[band] += 1
                band_corrects[band] += int(matched)
                class_totals[label] += 1
                class_corrects[label] += int(matched)
                work_totals[entry_id] += 1
                work_corrects[entry_id] += int(matched)
                if label != blank_label:
                    work_nonblank_totals[entry_id] += 1
                    work_nonblank_corrects[entry_id] += int(matched)
    return {
        "loss": loss_total / max(1, totals["all"]),
        "character_accuracy": corrects["all"] / max(1, totals["all"]),
        "top5_accuracy": corrects["top5"] / max(1, totals["all"]),
        "nonblank_accuracy": corrects["nonblank"] / max(1, totals["nonblank"]),
        "blank_accuracy": corrects["blank"] / max(1, totals["blank"]),
        "fill_instance_accuracy": corrects["fill_instance"]
        / max(1, totals["fill_instance"]),
        "nonfill_instance_accuracy": corrects["nonfill_instance"]
        / max(1, totals["nonfill_instance"]),
        "macro_accuracy_present_classes": float(
            np.mean(
                [class_corrects[label] / count for label, count in class_totals.items()]
            )
        ),
        "counts": dict(totals),
        "frequency_bands": {
            band: {
                "count": band_totals[band],
                "accuracy": band_corrects[band] / max(1, band_totals[band]),
            }
            for band in ("1000+", "100-999", "20-99", "10-19")
        },
        "per_class": [
            {
                "label": label,
                "char": label_to_char[label],
                "count": class_totals[label],
                "correct": class_corrects[label],
                "accuracy": class_corrects[label] / class_totals[label],
            }
            for label in sorted(class_totals)
        ],
        "per_work": {
            str(entry_id): {
                "count": work_totals[entry_id],
                "correct": work_corrects[entry_id],
                "accuracy": work_corrects[entry_id] / work_totals[entry_id],
                "nonblank_count": work_nonblank_totals[entry_id],
                "nonblank_correct": work_nonblank_corrects[entry_id],
                "nonblank_accuracy": work_nonblank_corrects[entry_id]
                / max(1, work_nonblank_totals[entry_id]),
            }
            for entry_id in sorted(work_totals)
        },
    }


def _train_condition(
    condition_id: str,
    dataset_manifest: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    train_dataset = C2CharacterDataset(
        dataset_manifest,
        split="train",
        tone_nuisance_probability=args.tone_nuisance_probability,
        seed=args.seed,
    )
    validation_dataset = C2CharacterDataset(dataset_manifest, split="validation")
    test_dataset = C2CharacterDataset(dataset_manifest, split="test")
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        maximum=args.max_train_samples,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        num_workers=args.num_workers,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 2,
        num_workers=args.num_workers,
    )
    torch.manual_seed(args.seed)
    model = DeepAAMultitask(input_channels=2)
    load_pretrained_backbone(model, args.pretrained)
    model.to(device)
    initial_test = _evaluate(
        model,
        test_loader,
        device,
        test_dataset.vocabulary,
        test_dataset.blank_label,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for module in (model.convolutions, model.normalizations, model.character_head)
            for parameter in module.parameters()
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = _training_epoch(model, train_loader, device, optimizer)
        validation = _evaluate(
            model,
            validation_loader,
            device,
            validation_dataset.vocabulary,
            validation_dataset.blank_label,
        )
        item = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": {
                key: value
                for key, value in validation.items()
                if key not in {"per_class", "per_work"}
            },
        }
        history.append(item)
        print(json.dumps({"condition": condition_id, **item}, ensure_ascii=False))
        checkpoint = {
            "condition": condition_id,
            "epoch": epoch + 1,
            "input_channels": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            torch.save(checkpoint, output_dir / "best.pt")
    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    model.to(device)
    final_validation = _evaluate(
        model,
        validation_loader,
        device,
        validation_dataset.vocabulary,
        validation_dataset.blank_label,
    )
    final_test = _evaluate(
        model,
        test_loader,
        device,
        test_dataset.vocabulary,
        test_dataset.blank_label,
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "condition": condition_id,
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset": json.loads(dataset_manifest.read_text(encoding="utf-8")),
        "selected_epoch": int(best["epoch"]),
        "initial_test": initial_test,
        "final": {"validation": final_validation, "test": final_test},
        "history": history,
        "checkpoint": str((output_dir / "best.pt").resolve()),
    }


def main() -> None:
    args = parse_args()
    tone_report = json.loads(args.tone_report.read_text(encoding="utf-8"))
    p3_report = json.loads(args.p3_report.read_text(encoding="utf-8"))
    conditions = args.conditions or list(
        tone_report["selection"]["shortlisted_for_p6_c2"]
    )
    if not conditions:
        raise ValueError("P5 tone report did not shortlist a P6 condition")
    allowed = set(tone_report["selection"]["shortlisted_for_p6_c2"])
    unexpected = sorted(set(conditions) - allowed)
    if unexpected:
        raise ValueError(f"Conditions are not in the P5 shortlist: {', '.join(unexpected)}")
    output = args.output.resolve()
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"P6 output is not empty: {output}; use --force")
        if allowed_root not in output.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    reports: dict[str, object] = {}
    for condition_id in conditions:
        config = tone_report["windows"][condition_id]["config"]
        dataset_dir = output / "datasets" / condition_id
        prepare_c2_character_dataset(
            args.snapshot,
            args.charset,
            dataset_dir,
            tone_window_width=int(config["tone_window_width"]),
            tone_window_height=int(config["tone_window_height"]),
            tone_sigma=float(config["tone_sigma"]),
            tone_gamma=float(config["tone_gamma"]),
            training_augment_policy="processing_50",
            seed=args.seed,
        )
        reports[condition_id] = _train_condition(
            condition_id,
            dataset_dir / "dataset.json",
            output / "models" / condition_id,
            args,
            device,
        )

    # Evaluate the selected P3 raw C=1 checkpoint on the same tagged P6 test examples.
    first_dataset = C2CharacterDataset(
        Path(reports[conditions[0]]["dataset_manifest"]), split="test"
    )
    first_loader = _make_loader(
        first_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 9,
        num_workers=args.num_workers,
    )
    c1 = DeepAAMultitask(input_channels=1)
    c1_checkpoint_path = Path(p3_report["methods"]["raw_raster"]["checkpoint"])
    c1_checkpoint = torch.load(c1_checkpoint_path, map_location="cpu", weights_only=True)
    c1.load_state_dict(c1_checkpoint["model"])
    c1.to(device)
    c1_test = _evaluate(
        c1,
        first_loader,
        device,
        first_dataset.vocabulary,
        first_dataset.blank_label,
        structure_only=True,
    )

    comparisons: dict[str, object] = {}
    passing: list[tuple[float, float, str]] = []
    for condition_id, condition_report in reports.items():
        test = condition_report["final"]["test"]
        deltas = {
            key: float(test[key]) - float(c1_test[key])
            for key in (
                "character_accuracy",
                "nonblank_accuracy",
                "blank_accuracy",
                "top5_accuracy",
                "macro_accuracy_present_classes",
                "fill_instance_accuracy",
                "nonfill_instance_accuracy",
            )
        }
        gate_passed = (
            deltas["character_accuracy"] >= -0.005
            and deltas["nonblank_accuracy"] >= -0.005
            and deltas["top5_accuracy"] >= -0.005
            and deltas["fill_instance_accuracy"] > 0.0
        )
        if gate_passed:
            passing.append(
                (
                    -float(test["fill_instance_accuracy"]),
                    -float(test["nonblank_accuracy"]),
                    condition_id,
                )
            )
        comparisons[condition_id] = {
            "test_minus_c1": deltas,
            "all_paired_work_bootstrap": _paired_work_bootstrap(
                c1_test,
                test,
                seed=args.seed,
                samples=args.bootstrap_samples,
            ),
            "nonblank_paired_work_bootstrap": _paired_work_bootstrap(
                c1_test,
                test,
                seed=args.seed + 1,
                samples=args.bootstrap_samples,
                field_prefix="nonblank_",
            ),
            "gate_passed": gate_passed,
        }
    passing.sort()
    selected = passing[0][2] if passing else "c1"
    report = {
        "schema_version": 1,
        "phase": "P6",
        "dataset": "multichannel-p6-c2",
        "purpose": "C=2 structure+tone comparison with repeated-fill instance metrics",
        "seed": args.seed,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "ch0_training_augment": "processing_50",
            "initialization": "C1 weights on ch0; zero first-layer weights on ch1",
            "tone_nuisance_probability": args.tone_nuisance_probability,
            "tone_nuisance": (
                "train-only label-independent constant, gradient, blob, or band tone"
            ),
        },
        "tone_report": str(args.tone_report.resolve()),
        "p3_report": str(args.p3_report.resolve()),
        "c1_test_on_p6_labels": c1_test,
        "conditions": reports,
        "comparisons": comparisons,
        "selection": {
            "selected": selected,
            "gate": {
                "all_accuracy_delta_min": -0.005,
                "nonblank_accuracy_delta_min": -0.005,
                "top5_accuracy_delta_min": -0.005,
                "fill_instance_accuracy_delta": "must be positive",
                "ranking": "higher fill-instance accuracy, then higher nonblank accuracy",
            },
            "status": "synthetic gate only; fixed real-image human comparison still required",
        },
        "rights_status": "local training only; datasets and checkpoints are not tracked",
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
