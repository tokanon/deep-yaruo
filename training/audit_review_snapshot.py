from __future__ import annotations

import argparse
import json
from pathlib import Path

from .review_corpus_audit import audit_review_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit accepted AA hashes, near duplicates, and split isolation."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_review_snapshot(args.snapshot)
    output = args.output or args.snapshot / "audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit={output.resolve()}")


if __name__ == "__main__":
    main()
