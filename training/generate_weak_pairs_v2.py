from __future__ import annotations

import argparse
import json
from pathlib import Path

from .weak_pairs import V2_VARIANTS, generate_weak_pair_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION_ROOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the weak-v2 blind preference candidate sweep."
    )
    parser.add_argument(
        "--snapshot", type=Path, default=DEFAULT_VERSION_ROOT / "accepted-v1"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_VERSION_ROOT / "weak-v2")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_weak_pair_dataset(
        args.snapshot,
        args.output,
        limit=args.limit,
        variants=V2_VARIANTS,
        font_size=args.font_size,
        seed=args.seed,
        dataset_name="yaruyomi-weak-v2",
        purpose=(
            "blind preference sweep around skeleton baseline; "
            "not a real source-image pairing"
        ),
    )
    print(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
