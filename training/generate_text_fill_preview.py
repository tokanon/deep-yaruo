from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .aa_fill_layers import render_fill_layers
from .review_corpus import load_snapshot_records
from .weak_pairs import select_pilot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview text-structural AA fill extraction without blur or dropout."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "text-fill-preview",
    )
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entry-id",
        type=int,
        action="append",
        dest="entry_ids",
        help="Render a specific accepted entry; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Preview output is not empty: {args.output}")
    records = load_snapshot_records(args.snapshot)
    if args.entry_ids:
        by_id = {
            int(record["entry_id"]): record
            for record in records
            if record["decision"] == "accept"
        }
        missing = [entry_id for entry_id in args.entry_ids if entry_id not in by_id]
        if missing:
            raise ValueError(f"Accepted entries not found: {missing}")
        selected = [by_id[entry_id] for entry_id in args.entry_ids]
    else:
        selected = select_pilot_records(records, limit=args.limit, seed=args.seed)
    manifest: list[dict[str, object]] = []
    for record in selected:
        entry_id = int(record["entry_id"])
        text = (args.snapshot / str(record["text"])).read_text(encoding="utf-8")
        original, line, fill, composite, diagnostic, runs = render_fill_layers(
            text,
            font_size=args.font_size,
        )
        entry_root = args.output / f"{entry_id:07d}"
        entry_root.mkdir(parents=True, exist_ok=True)
        for name, image in (
            ("original", original),
            ("line", line),
            ("fill", fill),
            ("composite", composite),
            ("diagnostic", diagnostic),
        ):
            Image.fromarray(image).save(entry_root / f"{name}.png", optimize=True)
        manifest.append(
            {
                "entry_id": entry_id,
                "category": record["corrected_category"],
                "source_text": record["text"],
                "fill_run_count": len(runs),
                "fill_runs": [run.to_dict() for run in runs],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
