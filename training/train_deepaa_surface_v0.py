from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.deepaa_surface_v0 import (
    DeepAASurfaceV0,
    ReverseChannelDataset,
    SUPPORT_LINE,
    SUPPORT_LOW_FILL,
    SUPPORT_SUPPORTED_FILL,
    WorkGroupedBatchSampler,
    load_deepaa_line_encoder,
    parameter_count,
    training_vocabulary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "datasets"
    / "incoming"
    / "yaruyomi"
    / "v32.1"
    / "accepted-v2"
    / "derived"
    / "0500"
    / "reverse-channels-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train approved DeepAA L/LS comparison.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--pretrained", type=Path, default=ROOT / "models" / "deepaa-light.onnx"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "deepaa-surface-v0",
    )
    parser.add_argument("--variants", nargs="+", choices=("L", "LS"), default=("L", "LS"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--role-loss-weight", type=float, default=0.2)
    parser.add_argument("--start-loss-weight", type=float, default=0.35)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _loader(
    dataset: ReverseChannelDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    maximum_samples: int | None,
) -> tuple[DataLoader, WorkGroupedBatchSampler]:
    sampler = WorkGroupedBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        maximum_samples=maximum_samples,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        ),
        sampler,
    )


def _class_weights(
    characters: list[str], counts: dict[str, int], device: torch.device
) -> torch.Tensor:
    values = np.asarray([counts[character] for character in characters], dtype=np.float64)
    weights = np.sqrt(values.sum() / (len(values) * values))
    weights = np.minimum(weights, 10.0)
    weights /= np.average(weights, weights=values)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: DeepAASurfaceV0,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    start_loss_weight: float,
    role_loss_weight: float,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "positive_count": 0.0,
        "seen_count": 0.0,
        "char_correct_seen": 0.0,
        "start_correct": 0.0,
        "start_true_positive": 0.0,
        "start_predicted_positive": 0.0,
        "start_actual_positive": 0.0,
        "role_correct": 0.0,
    }
    strata = {
        "line": [0, 0],
        "fill": [0, 0],
        "supported_fill": [0, 0],
        "low_support_fill": [0, 0],
    }
    for line, surface, character, start, role, support, _codepoint in loader:
        line = line.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        character = character.to(device, non_blocking=True)
        start = start.to(device, non_blocking=True)
        role = role.to(device, non_blocking=True)
        support = support.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            character_logits, start_logits, role_logits = model(line, surface)
            seen = character >= 0
            positives = start >= 0.5
            character_loss = (
                nn.functional.cross_entropy(
                    character_logits[seen], character[seen], weight=class_weights
                )
                if bool(seen.any())
                else character_logits.sum() * 0.0
            )
            start_loss = nn.functional.binary_cross_entropy_with_logits(start_logits, start)
            role_loss = (
                nn.functional.cross_entropy(role_logits[positives], role[positives])
                if bool(positives.any())
                else role_logits.sum() * 0.0
            )
            loss = character_loss + start_loss_weight * start_loss + role_loss_weight * role_loss
            if training:
                loss.backward()
                optimizer.step()
        batch_size = line.shape[0]
        prediction = character_logits.argmax(1)
        correct = (prediction == character) & seen
        start_prediction = start_logits >= 0
        actual_start = positives
        totals["loss_sum"] += float(loss.detach()) * batch_size
        totals["sample_count"] += batch_size
        totals["positive_count"] += int(positives.sum())
        totals["seen_count"] += int(seen.sum())
        totals["char_correct_seen"] += int(correct.sum())
        totals["start_correct"] += int((start_prediction == actual_start).sum())
        totals["start_true_positive"] += int((start_prediction & actual_start).sum())
        totals["start_predicted_positive"] += int(start_prediction.sum())
        totals["start_actual_positive"] += int(actual_start.sum())
        totals["role_correct"] += int(
            (role_logits[positives].argmax(1) == role[positives]).sum()
        )
        masks = {
            "line": positives & (support == SUPPORT_LINE),
            "fill": positives & (support != SUPPORT_LINE),
            "supported_fill": positives & (support == SUPPORT_SUPPORTED_FILL),
            "low_support_fill": positives & (support == SUPPORT_LOW_FILL),
        }
        for name, mask in masks.items():
            strata[name][0] += int((correct & mask).sum())
            strata[name][1] += int(mask.sum())
    sample_count = max(1.0, totals["sample_count"])
    positive_count = max(1.0, totals["positive_count"])
    seen_count = max(1.0, totals["seen_count"])
    metrics: dict[str, object] = {
        "loss": totals["loss_sum"] / sample_count,
        "samples": int(totals["sample_count"]),
        "positive_samples": int(totals["positive_count"]),
        "unseen_target_count": int(totals["positive_count"] - totals["seen_count"]),
        "character_accuracy_all": totals["char_correct_seen"] / positive_count,
        "character_accuracy_seen": totals["char_correct_seen"] / seen_count,
        "start_accuracy": totals["start_correct"] / sample_count,
        "start_precision": totals["start_true_positive"]
        / max(1.0, totals["start_predicted_positive"]),
        "start_recall": totals["start_true_positive"]
        / max(1.0, totals["start_actual_positive"]),
        "role_accuracy": totals["role_correct"] / positive_count,
        "character_strata": {
            name: {
                "correct": values[0],
                "count": values[1],
                "accuracy": values[0] / max(1, values[1]),
            }
            for name, values in strata.items()
        },
    }
    return metrics


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit("Reverse-channel dataset is missing.")
    audit_path = args.data / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("training_candidate_selected") != "b-periodic":
        raise SystemExit("Reverse-channel audit has not selected the B candidate.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    characters, counts = training_vocabulary(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "characters": characters,
        "codepoints": [ord(character) for character in characters],
        "train_counts": counts,
        "class_count": len(characters),
        "frequency_cutoff": None,
        "unseen_policy": "report-separately-no-unk",
    }
    (args.output / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    train_dataset = ReverseChannelDataset(
        args.data,
        split="train",
        characters=characters,
        negative_stride=args.negative_stride,
        seed=args.seed,
    )
    validation_dataset = ReverseChannelDataset(
        args.data,
        split="validation",
        characters=characters,
        negative_stride=args.negative_stride,
        seed=args.seed,
    )
    train_loader, train_sampler = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        maximum_samples=args.max_train_samples,
    )
    validation_loader, validation_sampler = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        maximum_samples=args.max_validation_samples,
    )
    class_weights = _class_weights(characters, counts, device)
    config = {
        "schema_version": 1,
        "experiment": "deepaa-surface-v0",
        "data": str(args.data.resolve()),
        "reverse_candidate": "b-periodic",
        "variants": list(args.variants),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "negative_stride": args.negative_stride,
        "role_loss_weight": args.role_loss_weight,
        "start_loss_weight": args.start_loss_weight,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "seed": args.seed,
        "surface_channels": ["membership", "boundary", "tone"],
        "raw_surface_ids_as_input": False,
        "motif_metadata_as_input": False,
        "normal_generator_changed": False,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
    }
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for variant_index, variant in enumerate(args.variants):
        # Re-seeding makes the comparison reproducible; both line encoders are
        # then overwritten from the exact same historical C1 initialization.
        torch.manual_seed(args.seed + variant_index)
        model = DeepAASurfaceV0(len(characters), variant=variant)
        load_deepaa_line_encoder(model.line_encoder, args.pretrained)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        variant_dir = args.output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, object]] = []
        best_loss = float("inf")
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            validation_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                class_weights,
                optimizer=optimizer,
                start_loss_weight=args.start_loss_weight,
                role_loss_weight=args.role_loss_weight,
            )
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model,
                    validation_loader,
                    device,
                    class_weights,
                    optimizer=None,
                    start_loss_weight=args.start_loss_weight,
                    role_loss_weight=args.role_loss_weight,
                )
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(record)
            print(json.dumps({"variant": variant, **record}, ensure_ascii=False), flush=True)
            checkpoint = {
                "schema_version": 1,
                "variant": variant,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "validation": validation_metrics,
                "class_count": len(characters),
            }
            torch.save(checkpoint, variant_dir / "last.pt")
            if float(validation_metrics["loss"]) < best_loss:
                best_loss = float(validation_metrics["loss"])
                torch.save(checkpoint, variant_dir / "best.pt")
        (variant_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (variant_dir / "model.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "parameter_count": parameter_count(model),
                    "best_validation_loss": best_loss,
                    "line_encoder_initialized_from": str(args.pretrained.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
