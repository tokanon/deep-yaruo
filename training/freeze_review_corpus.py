from __future__ import annotations

import argparse
import json
from pathlib import Path

from .review_corpus import freeze_review_corpus


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze reviewed Yaruyomi AA into a provenance-preserving local corpus."
    )
    parser.add_argument("--index", type=Path, default=SOURCE_ROOT / "index.sqlite3")
    parser.add_argument("--archive", type=Path, default=SOURCE_ROOT / "source.zip")
    parser.add_argument("--reviews", type=Path, default=SOURCE_ROOT / "reviews.sqlite3")
    parser.add_argument("--output", type=Path, default=SOURCE_ROOT / "accepted-v1")
    parser.add_argument("--expected-accepted", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-name", default="yaruyomi-accepted-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = freeze_review_corpus(
        args.index,
        args.archive,
        args.reviews,
        args.output,
        expected_accepted=args.expected_accepted,
        seed=args.seed,
        snapshot_name=args.snapshot_name,
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(f"snapshot={args.output.resolve()}")


if __name__ == "__main__":
    main()
