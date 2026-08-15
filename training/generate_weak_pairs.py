from __future__ import annotations

import argparse
import json
from pathlib import Path

from .weak_pairs import DEFAULT_VARIANTS, generate_weak_pair_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic pseudo-line candidates from accepted AA."
    )
    parser.add_argument("--snapshot", type=Path, default=SOURCE_ROOT / "accepted-v1")
    parser.add_argument("--output", type=Path, default=SOURCE_ROOT / "weak-v1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--variant", action="append", choices=DEFAULT_VARIANTS)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_weak_pair_dataset(
        args.snapshot,
        args.output,
        limit=args.limit,
        variants=tuple(args.variant) if args.variant else DEFAULT_VARIANTS,
        font_size=args.font_size,
        seed=args.seed,
    )
    print(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"dataset={args.output.resolve()}")


if __name__ == "__main__":
    main()
