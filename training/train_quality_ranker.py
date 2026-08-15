from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quality_ranker import train_quality_ranker


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a source-grouped baseline for accepted Yaruo AA quality."
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "quality-ranker" / "model.joblib",
    )
    parser.add_argument("--estimators", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-leaf-samples", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_quality_ranker(
        args.snapshot,
        args.output,
        estimators=args.estimators,
        max_depth=args.max_depth,
        minimum_leaf_samples=args.min_leaf_samples,
        folds=args.folds,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"model={args.output.resolve()}")


if __name__ == "__main__":
    main()
