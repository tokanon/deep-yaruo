from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from training.examples import extract_context
from training.model import DeepAAMultitask, load_pretrained_backbone
from training.structure_proxy import STRUCTURE_PROXY_METHODS
from training.structure_proxy_dataset import prepare_structure_proxy_character_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"
DEFAULT_PRETRAINED = ROOT / "models" / "deepaa-light.onnx"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "structure-proxy-p3"


@lru_cache(maxsize=2_048)
def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


class ProxyCharacterDataset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, manifest_path: Path, *, split: str) -> None:
        self.root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        examples_path = self.root / str(manifest["examples"])
        parsed_examples = [
            json.loads(line)
            for line in examples_path.read_text(encoding="utf-8").splitlines()
        ]
        self.examples = [
            example for example in parsed_examples if example["split"] == split
        ]
        self.vocabulary = manifest["vocabulary"]
        self.blank_label = next(
            int(entry["label"]) for entry in self.vocabulary if entry["char"] == " "
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        example = self.examples[index]
        image = _read_image(str(self.root / str(example["image"])))
        window = extract_context(image, int(example["x"]), int(example["y"]))
        tensor = torch.from_numpy(window.copy()).unsqueeze(0).float() / 255.0
        return tensor, int(example["label"]), int(example["entry_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3 C=1 downstream character comparison for P2 structure proxies."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _make_loader(
    dataset: ProxyCharacterDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    maximum: int | None = None,
) -> DataLoader:
    selected: Dataset = dataset
    if maximum is not None and maximum < len(dataset):
        indices = random.Random(seed).sample(range(len(dataset)), maximum)
        selected = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _run_training_epoch(
    model: DeepAAMultitask,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0
    for inputs, targets, _ in loader:
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
) -> dict[str, object]:
    model.eval()
    label_to_band = {
        int(entry["label"]): str(entry["frequency_band"]) for entry in vocabulary
    }
    label_to_char = {int(entry["label"]): str(entry["char"]) for entry in vocabulary}
    total_loss = 0.0
    total = 0
    correct = 0
    top5_correct = 0
    band_totals: Counter[str] = Counter()
    band_correct: Counter[str] = Counter()
    class_totals: Counter[int] = Counter()
    class_correct: Counter[int] = Counter()
    work_totals: Counter[int] = Counter()
    work_correct: Counter[int] = Counter()
    work_nonblank_totals: Counter[int] = Counter()
    work_nonblank_correct: Counter[int] = Counter()
    nonblank_total = 0
    nonblank_correct = 0
    blank_total = 0
    blank_correct = 0
    with torch.no_grad():
        for inputs, targets, entry_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = model(inputs)
            total_loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            predictions = logits.argmax(1)
            top5 = logits.topk(5, dim=1).indices
            matches = predictions == targets
            top5_matches = (top5 == targets.unsqueeze(1)).any(1)
            target_values = targets.cpu().tolist()
            match_values = matches.cpu().tolist()
            top5_values = top5_matches.cpu().tolist()
            for label, matched, top5_matched, entry_id in zip(
                target_values,
                match_values,
                top5_values,
                entry_ids.tolist(),
            ):
                total += 1
                correct += int(matched)
                top5_correct += int(top5_matched)
                band = label_to_band[label]
                band_totals[band] += 1
                band_correct[band] += int(matched)
                class_totals[label] += 1
                class_correct[label] += int(matched)
                work_totals[entry_id] += 1
                work_correct[entry_id] += int(matched)
                if label == blank_label:
                    blank_total += 1
                    blank_correct += int(matched)
                else:
                    nonblank_total += 1
                    nonblank_correct += int(matched)
                    work_nonblank_totals[entry_id] += 1
                    work_nonblank_correct[entry_id] += int(matched)
    present_class_accuracies = [
        class_correct[label] / class_totals[label] for label in class_totals
    ]
    return {
        "loss": total_loss / max(1, total),
        "character_accuracy": correct / max(1, total),
        "top5_accuracy": top5_correct / max(1, total),
        "nonblank_accuracy": nonblank_correct / max(1, nonblank_total),
        "blank_accuracy": blank_correct / max(1, blank_total),
        "macro_accuracy_present_classes": float(np.mean(present_class_accuracies)),
        "counts": {
            "all": total,
            "nonblank": nonblank_total,
            "blank": blank_total,
            "present_classes": len(class_totals),
        },
        "frequency_bands": {
            band: {
                "count": band_totals[band],
                "accuracy": band_correct[band] / max(1, band_totals[band]),
            }
            for band in ("1000+", "100-999", "20-99", "10-19")
        },
        "per_class": [
            {
                "label": label,
                "char": label_to_char[label],
                "count": class_totals[label],
                "correct": class_correct[label],
                "accuracy": class_correct[label] / class_totals[label],
            }
            for label in sorted(class_totals)
        ],
        "per_work": {
            str(entry_id): {
                "count": work_totals[entry_id],
                "correct": work_correct[entry_id],
                "accuracy": work_correct[entry_id] / work_totals[entry_id],
                "nonblank_count": work_nonblank_totals[entry_id],
                "nonblank_correct": work_nonblank_correct[entry_id],
                "nonblank_accuracy": work_nonblank_correct[entry_id]
                / max(1, work_nonblank_totals[entry_id]),
            }
            for entry_id in sorted(work_totals)
        },
    }


def _paired_work_bootstrap(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    seed: int,
    samples: int,
    field_prefix: str = "",
) -> dict[str, object]:
    baseline_works = baseline["per_work"]
    candidate_works = candidate["per_work"]
    work_ids = sorted(set(baseline_works) & set(candidate_works))
    if not work_ids:
        raise ValueError("Paired bootstrap requires common test works")
    count_field = f"{field_prefix}count"
    correct_field = f"{field_prefix}correct"

    def aggregate(report: dict[str, object], ids: list[str]) -> float:
        correct = sum(int(report["per_work"][work_id][correct_field]) for work_id in ids)
        count = sum(int(report["per_work"][work_id][count_field]) for work_id in ids)
        return correct / max(1, count)

    observed = aggregate(candidate, work_ids) - aggregate(baseline, work_ids)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choice(work_ids, size=len(work_ids), replace=True).tolist()
        deltas.append(aggregate(candidate, selected) - aggregate(baseline, selected))
    interval = np.quantile(deltas, (0.025, 0.975))
    return {
        "candidate_minus_baseline": observed,
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_unit": "test_work",
        "work_count": len(work_ids),
        "samples": samples,
    }


def _train_method(
    method: str,
    dataset_manifest: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    training_dataset = ProxyCharacterDataset(dataset_manifest, split="train")
    validation_dataset = ProxyCharacterDataset(dataset_manifest, split="validation")
    test_dataset = ProxyCharacterDataset(dataset_manifest, split="test")
    train_loader = _make_loader(
        training_dataset,
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = DeepAAMultitask()
    load_pretrained_backbone(model, args.pretrained)
    model.to(device)
    baseline_validation = _evaluate(
        model,
        validation_loader,
        device,
        validation_dataset.vocabulary,
        validation_dataset.blank_label,
    )
    baseline_test = _evaluate(
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
    best_loss = float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        train_metrics = _run_training_epoch(model, train_loader, device, optimizer)
        validation_metrics = _evaluate(
            model,
            validation_loader,
            device,
            validation_dataset.vocabulary,
            validation_dataset.blank_label,
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": {
                key: value
                for key, value in validation_metrics.items()
                if key not in {"per_class", "per_work"}
            },
        }
        history.append(record)
        print(json.dumps({"method": method, **record}, ensure_ascii=False))
        checkpoint = {
            "method": method,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            torch.save(checkpoint, output_dir / "best.pt")
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
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
        "method": method,
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset": json.loads(dataset_manifest.read_text(encoding="utf-8")),
        "selected_epoch": int(best_checkpoint["epoch"]),
        "baseline": {"validation": baseline_validation, "test": baseline_test},
        "final": {"validation": final_validation, "test": final_test},
        "history": history,
        "checkpoint": str((output_dir / "best.pt").resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("epochs and bootstrap-samples must be positive")
    output = args.output.resolve()
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"P3 output is not empty: {output}; use --force")
        if allowed_root not in output.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    method_reports: dict[str, object] = {}
    for method in STRUCTURE_PROXY_METHODS:
        dataset_dir = output / "datasets" / method
        prepare_structure_proxy_character_dataset(
            args.snapshot,
            args.charset,
            dataset_dir,
            method=method,
            seed=args.seed,
        )
        method_reports[method] = _train_method(
            method,
            dataset_dir / "dataset.json",
            output / "models" / method,
            args,
            device,
        )

    raw_test = method_reports["raw_raster"]["final"]["test"]
    smooth_test = method_reports["supersampled_blur_downsample"]["final"]["test"]
    comparison = {
        "smooth_minus_raw_all": _paired_work_bootstrap(
            raw_test,
            smooth_test,
            seed=args.seed,
            samples=args.bootstrap_samples,
        ),
        "smooth_minus_raw_nonblank": _paired_work_bootstrap(
            raw_test,
            smooth_test,
            seed=args.seed + 1,
            samples=args.bootstrap_samples,
            field_prefix="nonblank_",
        ),
    }
    report = {
        "schema_version": 1,
        "phase": "P3",
        "dataset": "structure-proxy-p3-c1",
        "purpose": "downstream synthetic character utility; not final real-image AA quality",
        "seed": args.seed,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "pretrained": str(args.pretrained.resolve()),
        },
        "methods": method_reports,
        "comparison": comparison,
        "interpretation": (
            "P3 charAcc is measured on held-out accepted AA works. "
            "It complements P2 AUC/Chamfer and is not a final real-image score."
        ),
        "rights_status": "local training only; datasets and checkpoints are not tracked",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"saved={(output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
