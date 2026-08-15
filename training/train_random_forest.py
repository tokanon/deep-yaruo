from __future__ import annotations

import argparse
import json
from pathlib import Path

from .random_forest import train_random_forest


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an experimental Random Forest DeepAA character classifier."
    )
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
        "--char-list",
        type=Path,
        default=ROOT / "models" / "deepaa-charset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / ".tmp"
            / "training-runs"
            / "random-forest"
            / "deepaa-random-forest.joblib"
        ),
    )
    parser.add_argument("--estimators", type=int, default=64)
    parser.add_argument("--max-train-per-class", type=int, default=32)
    parser.add_argument("--max-validation-per-class", type=int, default=16)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--max-leaf-nodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_random_forest(
        args.csv,
        args.manifest,
        args.char_list,
        args.output,
        estimators=args.estimators,
        maximum_train_per_class=args.max_train_per_class,
        maximum_validation_per_class=args.max_validation_per_class,
        augmentations=args.augmentations,
        maximum_leaf_nodes=args.max_leaf_nodes,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
