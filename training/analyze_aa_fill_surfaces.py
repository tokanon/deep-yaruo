from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import ImageFont

from backend.rendering import find_font, line_pitch

from .aa_fill_layers import (
    FillSurface,
    connect_fill_runs,
    detect_fill_runs,
    render_fill_surface_labels,
)
from .review_corpus import load_snapshot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_OUTPUT = (
    ROOT / ".tmp" / "training-runs" / "aa-fill-surface-p6r1" / "report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect accepted-AA fill runs into 2D P6R-1 surfaces."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--minimum-overlap-ratio", type=float, default=0.25)
    return parser.parse_args()


def _distribution(values: Iterable[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"minimum": 0, "median": 0, "p95": 0, "maximum": 0}
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "maximum": ordered[-1],
    }


def _surface_summary(surface: FillSurface) -> dict[str, object]:
    return surface.to_dict(include_runs=True)


def analyze_surface_snapshot(
    snapshot_root: Path,
    *,
    font_size: int = 16,
    minimum_overlap_ratio: float = 0.25,
) -> dict[str, object]:
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record["decision"] == "accept"
    ]
    font = ImageFont.truetype(str(find_font()), font_size)
    pitch = line_pitch(font_size)
    entries: list[dict[str, object]] = []
    all_surfaces: list[FillSurface] = []
    character_run_counts: Counter[str] = Counter()
    surface_character_presence_counts: Counter[str] = Counter()
    tone_level_run_counts: Counter[int] = Counter()
    total_run_count = 0
    total_connected_run_count = 0

    for record in records:
        text = (snapshot_root / str(record["text"])).read_text(encoding="utf-8")
        lines = text.splitlines() or [""]
        width = max(2, int(math.ceil(max(font.getlength(line) for line in lines))) + 2)
        height = max(pitch, len(lines) * pitch)
        runs = detect_fill_runs(text, font_size=font_size)
        surfaces = connect_fill_runs(
            runs,
            font_size=font_size,
            minimum_overlap_ratio=minimum_overlap_ratio,
        )
        labels = render_fill_surface_labels(surfaces, width=width, height=height)
        mask_area = int(np.count_nonzero(labels))
        aggregate_area = sum(surface.pixel_area for surface in surfaces)
        if mask_area != aggregate_area:
            raise AssertionError(
                f"surface area mismatch for entry {record['entry_id']}: "
                f"mask={mask_area}, aggregate={aggregate_area}"
            )

        connected_run_count = sum(
            surface.run_count for surface in surfaces if surface.line_count >= 2
        )
        total_run_count += len(runs)
        total_connected_run_count += connected_run_count
        character_run_counts.update(run.character for run in runs)
        for surface in surfaces:
            surface_character_presence_counts.update(
                character for character, _ in surface.character_counts
            )
            for level, count in surface.tone_level_counts:
                tone_level_run_counts[level] += count
        all_surfaces.extend(surfaces)
        entries.append(
            {
                "entry_id": int(record["entry_id"]),
                "category": str(record["corrected_category"]),
                "split": str(record["split"]),
                "text": str(record["text"]),
                "canvas_width_px": width,
                "canvas_height_px": height,
                "fill_run_count": len(runs),
                "fill_surface_count": len(surfaces),
                "multi_line_surface_count": sum(
                    surface.line_count >= 2 for surface in surfaces
                ),
                "connected_run_count": connected_run_count,
                "surface_area_px": aggregate_area,
                "surface_area_fraction": round(
                    aggregate_area / max(1, width * height), 6
                ),
                "surfaces": [_surface_summary(surface) for surface in surfaces],
            }
        )

    multi_line_surfaces = [
        surface for surface in all_surfaces if surface.line_count >= 2
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "phase": "P6R-1",
        "method": (
            "same-character fill runs connected only across adjacent rows when "
            "horizontal overlap covers at least the configured fraction of the "
            "shorter run"
        ),
        "font": str(find_font()),
        "font_size_px": font_size,
        "line_pitch_px": pitch,
        "minimum_overlap_ratio": minimum_overlap_ratio,
        "entry_count": len(entries),
        "entries_with_fill_runs": sum(
            int(entry["fill_run_count"]) > 0 for entry in entries
        ),
        "entries_without_fill_runs": sum(
            int(entry["fill_run_count"]) == 0 for entry in entries
        ),
        "entries_with_multi_line_surfaces": sum(
            int(entry["multi_line_surface_count"]) > 0 for entry in entries
        ),
        "fill_run_count": total_run_count,
        "fill_surface_count": len(all_surfaces),
        "multi_line_surface_count": len(multi_line_surfaces),
        "single_line_surface_count": sum(
            surface.line_count == 1 for surface in all_surfaces
        ),
        "isolated_run_surface_count": sum(
            surface.run_count == 1 for surface in all_surfaces
        ),
        "connected_run_count": total_connected_run_count,
        "connected_run_fraction": round(
            total_connected_run_count / max(1, total_run_count), 6
        ),
        "surface_area_px": _distribution(
            surface.pixel_area for surface in all_surfaces
        ),
        "surface_height_lines": _distribution(
            surface.line_count for surface in all_surfaces
        ),
        "surface_run_count": _distribution(
            surface.run_count for surface in all_surfaces
        ),
        "character_run_counts": dict(character_run_counts.most_common()),
        "surface_character_presence_counts": dict(
            surface_character_presence_counts.most_common()
        ),
        "tone_level_run_counts": {
            str(level): tone_level_run_counts[level]
            for level in sorted(tone_level_run_counts)
        },
        "entries": sorted(entries, key=lambda entry: int(entry["entry_id"])),
    }
    return report


def main() -> None:
    args = parse_args()
    report = analyze_surface_snapshot(
        args.snapshot,
        font_size=args.font_size,
        minimum_overlap_ratio=args.minimum_overlap_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_keys = (
        "entry_count",
        "entries_with_fill_runs",
        "entries_without_fill_runs",
        "entries_with_multi_line_surfaces",
        "fill_run_count",
        "fill_surface_count",
        "multi_line_surface_count",
        "connected_run_count",
        "connected_run_fraction",
        "surface_area_px",
        "surface_height_lines",
        "surface_run_count",
        "character_run_counts",
        "tone_level_run_counts",
    )
    print(
        json.dumps(
            {key: report[key] for key in summary_keys},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
