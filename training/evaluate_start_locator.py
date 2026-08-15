from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from torch import nn

from .reconstruction import load_placements
from .start_locator import StartLocator
from .start_locator_dataset import extract_line_context


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fast start locator.")
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
        default=ROOT / ".tmp" / "training-runs" / "start-locator" / "best.pt",
    )
    parser.add_argument("--max-works", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / ".tmp"
        / "training-runs"
        / "start-locator"
        / "evaluation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        record for record in manifest["works"] if record["split"] == "validation"
    ]
    if args.max_works is not None:
        records = records[: args.max_works]
    works, _ = load_placements(args.csv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = StartLocator()
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    totals = {threshold: defaultdict(float) for threshold in thresholds}
    line_count = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for record in records:
            image_path = args.manifest.parent / record["image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            starts_by_y: dict[int, set[int]] = defaultdict(set)
            for placement in works[record["file_name"]]:
                if placement.label < 411:
                    starts_by_y[placement.y].add(placement.x)
            for y, starts in starts_by_y.items():
                context = extract_line_context(image, y=y, x=0, width=image.shape[1])
                inputs = (
                    torch.from_numpy(context.copy())
                    .to(device)
                    .float()[None, None]
                    / 255.0
                )
                probabilities = torch.sigmoid(model(inputs).squeeze(0))
                truth = torch.zeros(image.shape[1], dtype=torch.bool, device=device)
                truth[list(starts)] = True
                truth_nearby = nn.functional.max_pool1d(
                    truth.float()[None, None], 3, stride=1, padding=1
                ).squeeze().bool()
                for threshold in thresholds:
                    candidates = probabilities >= threshold
                    candidate_nearby = nn.functional.max_pool1d(
                        candidates.float()[None, None], 3, stride=1, padding=1
                    ).squeeze().bool()
                    values = totals[threshold]
                    values["positions"] += truth.numel()
                    values["true_starts"] += truth.sum().item()
                    values["candidates"] += candidates.sum().item()
                    values["matched_candidates"] += (
                        candidates & truth_nearby
                    ).sum().item()
                    values["recalled_starts"] += (
                        truth & candidate_nearby
                    ).sum().item()
                line_count += 1
    elapsed = time.perf_counter() - started
    results = []
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
        "checkpoint_epoch": checkpoint["epoch"],
        "works": len(records),
        "lines": line_count,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "milliseconds_per_line": elapsed * 1000 / max(1, line_count),
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
