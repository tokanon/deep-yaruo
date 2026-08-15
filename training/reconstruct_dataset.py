from __future__ import annotations

import argparse
from pathlib import Path

from .reconstruction import reconstruct_dataset


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct canonical AA images from the DeepAA bootstrap dataset."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "datasets" / "bootstrap" / "deepaa-500.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "reconstructed-deepaa",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=("work-random", "series-holdout"),
        default="work-random",
        help="Use the legacy random work split or keep filename-derived series isolated.",
    )
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="Reconstruct only this work; repeat to select multiple works.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = reconstruct_dataset(
        args.csv,
        args.output,
        seed=args.seed,
        selected_names=set(args.names) if args.names else None,
        split_strategy=args.split_strategy,
    )
    print(
        f"Reconstructed {manifest['generated_work_count']} works in {args.output.resolve()} "
        f"after removing {manifest['exact_duplicate_rows_removed']} duplicate rows."
    )


if __name__ == "__main__":
    main()
