from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import ImageFont

from backend.rendering import find_font

from .aa_fill_layers import detect_fill_runs
from .review_corpus import load_snapshot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure repeated-character fill structure in accepted AA text."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "aa-fill-structure.json",
    )
    parser.add_argument("--font-size", type=int, default=16)
    return parser.parse_args()


def analyze_snapshot(snapshot_root: Path, *, font_size: int = 16) -> dict[str, object]:
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record["decision"] == "accept"
    ]
    font = ImageFont.truetype(str(find_font()), font_size)
    entries: list[dict[str, object]] = []
    character_counts: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_with_fill: Counter[str] = Counter()
    category_run_counts: defaultdict[str, int] = defaultdict(int)

    for record in records:
        text = (snapshot_root / str(record["text"])).read_text(encoding="utf-8")
        lines = text.splitlines() or [""]
        runs = detect_fill_runs(text, font_size=font_size)
        max_width = max(float(font.getlength(line)) for line in lines)
        span_area = sum(run.x_end - run.x_start for run in runs)
        canvas_area = max(1.0, max_width * len(lines))
        category = str(record["corrected_category"])
        category_totals[category] += 1
        category_run_counts[category] += len(runs)
        if runs:
            category_with_fill[category] += 1
        character_counts.update(run.character for run in runs)
        entries.append(
            {
                "entry_id": int(record["entry_id"]),
                "category": category,
                "split": str(record["split"]),
                "text": str(record["text"]),
                "fill_run_count": len(runs),
                "fill_line_count": len({run.line_index for run in runs}),
                "fill_span_fraction": round(span_area / canvas_area, 6),
                "fill_characters": dict(sorted(Counter(run.character for run in runs).items())),
            }
        )

    run_counts = [int(entry["fill_run_count"]) for entry in entries]
    entries_with_fill = sum(count > 0 for count in run_counts)
    categories = {
        category: {
            "entry_count": category_totals[category],
            "entries_with_fill": category_with_fill[category],
            "fill_run_count": category_run_counts[category],
        }
        for category in sorted(category_totals)
    }
    return {
        "schema_version": 1,
        "method": "same-character tone runs; no blur; no random line removal",
        "font": str(find_font()),
        "font_size_px": font_size,
        "entry_count": len(entries),
        "entries_with_fill": entries_with_fill,
        "entries_without_fill": len(entries) - entries_with_fill,
        "median_fill_runs": statistics.median(run_counts) if run_counts else 0,
        "maximum_fill_runs": max(run_counts, default=0),
        "fill_character_counts": dict(character_counts.most_common()),
        "categories": categories,
        "entries": sorted(entries, key=lambda entry: int(entry["entry_id"])),
    }


def main() -> None:
    args = parse_args()
    report = analyze_snapshot(args.snapshot, font_size=args.font_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "entry_count",
        "entries_with_fill",
        "entries_without_fill",
        "median_fill_runs",
        "maximum_fill_runs",
        "fill_character_counts",
    )}, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
