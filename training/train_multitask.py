from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .model import (
    DeepAAMultitask,
    export_multitask_onnx,
    load_pretrained_backbone,
)
from .torch_dataset import DeepAAMultitaskDataset, subset_indices


ROOT = Path(__file__).resolve().parents[1]


def augment_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Apply inexpensive line degradation to a complete GPU batch."""
    batch_size = inputs.shape[0]
    ink = 1.0 - inputs
    thickened = nn.functional.max_pool2d(ink, 3, stride=1, padding=1)
    thinned = -nn.functional.max_pool2d(-ink, 3, stride=1, padding=1)
    choices = torch.rand(batch_size, 1, 1, 1, device=inputs.device)
    ink = torch.where(choices < 0.15, thickened, ink)
    ink = torch.where(choices > 0.9, thinned, ink)
    result = 1.0 - ink
    blur_mask = torch.rand(batch_size, 1, 1, 1, device=inputs.device) < 0.25
    blurred = nn.functional.avg_pool2d(
        nn.functional.pad(result, (1, 1, 1, 1), mode="replicate"),
        3,
        stride=1,
    )
    result = torch.where(blur_mask, blurred, result)
    noise_scale = torch.rand(batch_size, 1, 1, 1, device=inputs.device) * 0.025
    return torch.clamp(result + torch.randn_like(result) * noise_scale, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DeepAA with a start head.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "datasets" / "bootstrap" / "deepaa-500.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / ".tmp" / "reconstructed-deepaa" / "manifest.json",
    )
    parser.add_argument(
        "--pretrained", type=Path, default=ROOT / "models" / "deepaa-light.onnx"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".tmp" / "training-runs" / "multitask"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--start-learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-start-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_loader(
    dataset: DeepAAMultitaskDataset,
    *,
    batch_size: int,
    maximum: int | None,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    indices = subset_indices(len(dataset), maximum, seed)
    selected = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: DeepAAMultitask,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    start_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "character_correct": 0.0,
        "character_count": 0.0,
        "start_correct": 0.0,
        "start_true_positive": 0.0,
        "start_predicted_positive": 0.0,
        "start_actual_positive": 0.0,
        "sample_count": 0.0,
    }
    for inputs, character_targets, start_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        character_targets = character_targets.to(device, non_blocking=True)
        start_targets = start_targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
            inputs = augment_batch(inputs)
        with torch.set_grad_enabled(training):
            character_logits, start_logits = model(inputs)
            positives = character_targets >= 0
            character_loss = nn.functional.cross_entropy(
                character_logits[positives], character_targets[positives]
            )
            start_loss = nn.functional.binary_cross_entropy_with_logits(
                start_logits, start_targets
            )
            loss = start_loss if start_only else character_loss + 0.35 * start_loss
            if training:
                loss.backward()
                optimizer.step()
        batch_size = inputs.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["sample_count"] += batch_size
        start_predictions = start_logits >= 0
        start_actual = start_targets >= 0.5
        totals["start_correct"] += float((start_predictions == start_actual).sum())
        totals["start_true_positive"] += float(
            (start_predictions & start_actual).sum()
        )
        totals["start_predicted_positive"] += float(start_predictions.sum())
        totals["start_actual_positive"] += float(start_actual.sum())
        totals["character_correct"] += float(
            (character_logits[positives].argmax(1) == character_targets[positives]).sum()
        )
        totals["character_count"] += int(positives.sum())
    start_precision = totals["start_true_positive"] / max(
        1.0, totals["start_predicted_positive"]
    )
    start_recall = totals["start_true_positive"] / max(
        1.0, totals["start_actual_positive"]
    )
    return {
        "loss": totals["loss"] / totals["sample_count"],
        "character_accuracy": totals["character_correct"]
        / totals["character_count"],
        "start_accuracy": totals["start_correct"] / totals["sample_count"],
        "start_precision": start_precision,
        "start_recall": start_recall,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not args.manifest.exists():
        raise SystemExit(
            "Reconstructed manifest is missing. Run python -m training.reconstruct_dataset first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    training_dataset = DeepAAMultitaskDataset(
        args.csv, args.manifest, split="train", augment=True, seed=args.seed
    )
    validation_dataset = DeepAAMultitaskDataset(
        args.csv, args.manifest, split="validation", augment=False, seed=args.seed
    )
    training_loader = make_loader(
        training_dataset,
        batch_size=args.batch_size,
        maximum=args.max_train_samples,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        maximum=args.max_validation_samples,
        seed=args.seed + 1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = DeepAAMultitask()
    load_pretrained_backbone(model, args.pretrained)
    model.to(device)
    base_parameters = [
        parameter
        for module in (
            model.convolutions,
            model.normalizations,
            model.character_head,
        )
        for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.learning_rate},
            {"params": model.start_head.parameters(), "lr": args.start_learning_rate},
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        start_only = epoch < args.warmup_start_epochs
        for parameter in base_parameters:
            parameter.requires_grad_(not start_only)
        training_dataset.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            training_loader,
            device,
            optimizer,
            start_only=start_only,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                None,
                start_only=start_only,
            )
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "stage": "start_warmup" if start_only else "joint_finetune",
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "history": history,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if (not start_only or args.epochs <= args.warmup_start_epochs) and validation_metrics[
            "loss"
        ] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")

    (args.output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    best_checkpoint = torch.load(
        args.output / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model"])
    export_multitask_onnx(model, args.output / "model.onnx")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
