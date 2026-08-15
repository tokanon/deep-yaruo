from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch import nn

from .model import DeepAAMultitask
from .reconstruction import LINE_PITCH, load_placements


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense character-start localization on held-out works."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / ".tmp" / "reconstructed-deepaa" / "manifest.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "datasets" / "bootstrap" / "deepaa-500.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "pilot-262k" / "best.pt",
    )
    parser.add_argument("--max-works", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "start-head-evaluation.json",
    )
    return parser.parse_args()


def dense_row_probabilities(
    model: DeepAAMultitask,
    image: torch.Tensor,
    y: int,
    *,
    batch_size: int,
) -> torch.Tensor:
    width = image.shape[1]
    padded = nn.functional.pad(image[None, None], (23, 40, 23, 40), value=1.0)
    strip = padded[:, :, y : y + 64, :]
    windows = nn.functional.unfold(strip, kernel_size=(64, 64), stride=1)
    windows = windows.squeeze(0).T.reshape(width, 1, 64, 64)
    probabilities: list[torch.Tensor] = []
    for start in range(0, width, batch_size):
        _, logits = model(windows[start : start + batch_size])
        probabilities.append(torch.sigmoid(logits))
    return torch.cat(probabilities)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validation_records = [
        record for record in manifest["works"] if record["split"] == "validation"
    ][: args.max_works]
    works, _ = load_placements(args.csv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DeepAAMultitask()
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    thresholds = (0.3, 0.5, 0.7, 0.8, 0.9)
    totals = {
        threshold: defaultdict(float)
        for threshold in thresholds
    }

    with torch.inference_mode():
        for record in validation_records:
            image_path = args.manifest.parent / record["image"]
            image_array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image_array is None:
                raise FileNotFoundError(image_path)
            image = torch.from_numpy(image_array).float().to(device) / 255.0
            starts_by_y: dict[int, set[int]] = defaultdict(set)
            for placement in works[record["file_name"]]:
                if placement.label < 411:
                    starts_by_y[placement.y].add(placement.x)
            for y, true_starts in starts_by_y.items():
                probabilities = dense_row_probabilities(
                    model,
                    image,
                    y,
                    batch_size=args.batch_size,
                )
                truth = torch.zeros(image.shape[1], dtype=torch.bool, device=device)
                truth[list(true_starts)] = True
                truth_nearby = nn.functional.max_pool1d(
                    truth.float()[None, None], 3, stride=1, padding=1
                ).squeeze().bool()
                for threshold in thresholds:
                    candidates = probabilities >= threshold
                    candidate_nearby = nn.functional.max_pool1d(
                        candidates.float()[None, None], 3, stride=1, padding=1
                    ).squeeze().bool()
                    totals[threshold]["positions"] += truth.numel()
                    totals[threshold]["true_starts"] += truth.sum().item()
                    totals[threshold]["candidates"] += candidates.sum().item()
                    totals[threshold]["matched_candidates"] += (
                        candidates & truth_nearby
                    ).sum().item()
                    totals[threshold]["recalled_starts"] += (
                        truth & candidate_nearby
                    ).sum().item()

    results: list[dict[str, float]] = []
    for threshold in thresholds:
        values = totals[threshold]
        results.append(
            {
                "threshold": threshold,
                "candidate_rate": values["candidates"] / values["positions"],
                "precision_at_1px": values["matched_candidates"]
                / max(1.0, values["candidates"]),
                "recall_at_1px": values["recalled_starts"]
                / max(1.0, values["true_starts"]),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "works": len(validation_records),
        "checkpoint_epoch": checkpoint["epoch"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
