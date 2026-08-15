from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .start_locator import StartLocator, export_start_locator
from .start_locator_dataset import StartLocatorDataset
from .torch_dataset import subset_indices


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fast line start locator.")
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
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "start-locator",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--segment-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-train-lines", type=int)
    parser.add_argument("--max-validation-lines", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def augment_lines(inputs: torch.Tensor) -> torch.Tensor:
    batch_size = inputs.shape[0]
    ink = 1.0 - inputs
    thickened = nn.functional.max_pool2d(ink, 3, stride=1, padding=1)
    choices = torch.rand(batch_size, 1, 1, 1, device=inputs.device)
    ink = torch.where(choices < 0.15, thickened, ink)
    result = 1.0 - ink
    noise_scale = torch.rand(batch_size, 1, 1, 1, device=inputs.device) * 0.025
    return torch.clamp(result + torch.randn_like(result) * noise_scale, 0.0, 1.0)


def make_loader(
    dataset: StartLocatorDataset,
    *,
    maximum: int | None,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    indices = subset_indices(len(dataset), maximum, seed)
    selected = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: StartLocator,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "positions": 0.0,
        "correct": 0.0,
        "predicted": 0.0,
        "actual": 0.0,
        "true_positive": 0.0,
    }
    for inputs, labels, mask in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if training:
            inputs = augment_lines(inputs)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], labels[mask]
            )
            if training:
                loss.backward()
                optimizer.step()
        predictions = logits >= 0
        actual = labels >= 0.5
        totals["loss"] += float(loss.detach()) * int(mask.sum())
        totals["positions"] += int(mask.sum())
        totals["correct"] += int(((predictions == actual) & mask).sum())
        totals["predicted"] += int((predictions & mask).sum())
        totals["actual"] += int((actual & mask).sum())
        totals["true_positive"] += int((predictions & actual & mask).sum())
    return {
        "loss": totals["loss"] / totals["positions"],
        "accuracy": totals["correct"] / totals["positions"],
        "candidate_rate": totals["predicted"] / totals["positions"],
        "precision": totals["true_positive"] / max(1.0, totals["predicted"]),
        "recall": totals["true_positive"] / max(1.0, totals["actual"]),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    training_dataset = StartLocatorDataset(
        args.csv,
        args.manifest,
        split="train",
        segment_width=args.segment_width,
        seed=args.seed,
        random_crop=True,
    )
    validation_dataset = StartLocatorDataset(
        args.csv,
        args.manifest,
        split="validation",
        segment_width=args.segment_width,
        seed=args.seed,
        random_crop=False,
    )
    training_loader = make_loader(
        training_dataset,
        maximum=args.max_train_lines,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        maximum=args.max_validation_lines,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        shuffle=False,
    )
    model = StartLocator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        training_dataset.set_epoch(epoch)
        train_metrics = run_epoch(model, training_loader, device, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device, None)
        record: dict[str, object] = {
            "epoch": epoch + 1,
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
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")
    best = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    export_start_locator(model, args.output / "model.onnx")
    (args.output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
