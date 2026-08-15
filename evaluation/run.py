from __future__ import annotations

import argparse
from pathlib import Path

from .quality import evaluate_cases


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed Yaruo AA quality cases.")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "evaluation" / "cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "evaluation-baseline"
    )
    parser.add_argument(
        "--decoder",
        choices=("beam", "viterbi", "viterbi-pruned"),
        help="Override the decoder in every fixed case.",
    )
    parser.add_argument(
        "--classifier",
        choices=("cnn", "random-forest"),
        help="Override the character classifier in every fixed case.",
    )
    parser.add_argument("--random-forest-model", type=Path)
    parser.add_argument(
        "--structure-strength",
        type=float,
        default=0.0,
        help="Direction and glyph-boundary mismatch weight (0-2).",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this case ID; repeat to select multiple cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_cases(
        args.cases,
        args.output,
        decoder_override=args.decoder,
        classifier_override=args.classifier,
        random_forest_model=args.random_forest_model,
        structure_strength=args.structure_strength,
        selected_ids=set(args.case_ids) if args.case_ids else None,
    )
    for case in report["cases"]:
        metrics = case["metrics"]
        print(
            f"{case['id']}: distance={metrics['distance_cost']:.3f}, "
            f"F1@2px={metrics['ink_f1_at_2px']:.3f}, "
            f"time={case['elapsed_seconds']:.3f}s"
        )
    print(f"Report: {(args.output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
