from __future__ import annotations

import argparse
import json
from pathlib import Path

from .renderer_verification import verify_renderer


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Saitamaar advances, labels, line pitch, and inverse rendering."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "datasets" / "bootstrap" / "deepaa-500.csv",
    )
    parser.add_argument(
        "--charset",
        type=Path,
        default=ROOT / "models" / "deepaa-charset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "renderer-verification" / "report.json",
    )
    parser.add_argument(
        "--no-gdi",
        action="store_true",
        help="Skip the Windows GDI advance comparison.",
    )
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="Verify only this work; repeat to select multiple works.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify_renderer(
        args.dataset,
        args.charset,
        include_gdi=not args.no_gdi,
        selected_names=set(args.names) if args.names else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Renderer verification {'passed' if report['gate_passed'] else 'failed'}: "
        f"{report['totals']['works']} works, {report['totals']['rows']} rows. "
        f"Report: {args.output.resolve()}"
    )
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

