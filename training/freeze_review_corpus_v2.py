from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from backend.review_v2 import ACCEPTED_TARGETS_250

from .review_corpus import freeze_review_corpus


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an accepted-v2 checkpoint without changing earlier snapshots."
    )
    parser.add_argument("--index", type=Path, default=SOURCE_ROOT / "index.sqlite3")
    parser.add_argument("--archive", type=Path, default=SOURCE_ROOT / "source.zip")
    parser.add_argument("--reviews", type=Path, default=SOURCE_ROOT / "reviews-v2.sqlite3")
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCE_ROOT / "accepted-v2" / "checkpoints" / "0750",
    )
    parser.add_argument("--minimum-accepted", type=int, default=750)
    parser.add_argument("--expected-accepted", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-name", default="yaruyomi-accepted-v2-0750")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.reviews) as connection:
        decision_counts = dict(
            connection.execute(
                "SELECT decision, COUNT(*) FROM reviews GROUP BY decision"
            )
        )
        accepted_by_category = dict(
            connection.execute(
                """
                SELECT corrected_category, COUNT(*)
                FROM reviews
                WHERE decision = 'accept'
                GROUP BY corrected_category
                """
            )
        )
    accepted_count = int(decision_counts.get("accept", 0))
    expected_accepted = args.expected_accepted or accepted_count
    if accepted_count < args.minimum_accepted:
        raise ValueError(
            f"Expected at least {args.minimum_accepted} accepted reviews, "
            f"found {accepted_count}"
        )
    multiplier = args.minimum_accepted / 250
    targets = {
        category: round(base_target * multiplier)
        for category, base_target in ACCEPTED_TARGETS_250.items()
    }
    deficits = {
        category: target - int(accepted_by_category.get(category, 0))
        for category, target in targets.items()
        if int(accepted_by_category.get(category, 0)) < target
    }
    if deficits:
        raise ValueError(f"Accepted category floors are not complete: {deficits}")
    snapshot = freeze_review_corpus(
        args.index,
        args.archive,
        args.reviews,
        args.output,
        expected_accepted=expected_accepted,
        seed=args.seed,
        snapshot_name=args.snapshot_name,
        group_near_duplicates=True,
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    print(f"snapshot={args.output.resolve()}")


if __name__ == "__main__":
    main()
