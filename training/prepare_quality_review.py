from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from training.splits import series_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTION = ROOT / ".tmp" / "reconstructed-quality-audit"
DEFAULT_OUTPUT = ROOT / ".tmp" / "bootstrap-quality-review"
REVIEW_FIELDS = (
    "file_name",
    "series",
    "split",
    "width",
    "height",
    "placements",
    "image",
    "subject",
    "quality",
    "usable_for",
    "notes",
)


def load_existing_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def write_review_csv(manifest: dict[str, object], output_path: Path) -> None:
    previous = load_existing_reviews(output_path)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for record in manifest["works"]:
            old = previous.get(record["file_name"], {})
            writer.writerow(
                {
                    "file_name": record["file_name"],
                    "series": series_name(record["file_name"]),
                    "split": record["split"],
                    "width": record["width"],
                    "height": record["height"],
                    "placements": record["placements"],
                    "image": record["image"],
                    "subject": old.get("subject", ""),
                    "quality": old.get("quality", ""),
                    "usable_for": old.get("usable_for", ""),
                    "notes": old.get("notes", ""),
                }
            )


def build_contact_sheets(
    manifest: dict[str, object],
    reconstruction_dir: Path,
    output_dir: Path,
    *,
    page_size: int = 20,
) -> int:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in manifest["works"]:
        groups[series_name(record["file_name"])].append(record)

    sheet_dir = output_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = 5, 4
    cell_width, cell_height = 320, 340
    label_height = 22
    page_count = 0
    for series, records in sorted(groups.items()):
        records.sort(key=lambda item: item["file_name"])
        for offset in range(0, len(records), page_size):
            page = records[offset : offset + page_size]
            canvas = Image.new("L", (columns * cell_width, rows * cell_height), 255)
            draw = ImageDraw.Draw(canvas)
            for index, record in enumerate(page):
                source = Image.open(reconstruction_dir / record["image"]).convert("L")
                thumb = ImageOps.contain(
                    source,
                    (cell_width - 12, cell_height - label_height - 12),
                    method=Image.Resampling.LANCZOS,
                )
                column = index % columns
                row = index // columns
                x = column * cell_width + (cell_width - thumb.width) // 2
                y = row * cell_height + label_height + (
                    cell_height - label_height - thumb.height
                ) // 2
                canvas.paste(thumb, (x, y))
                draw.text(
                    (column * cell_width + 5, row * cell_height + 4),
                    str(record["file_name"]),
                    fill=0,
                )
            page_number = offset // page_size + 1
            canvas.save(sheet_dir / f"{series}-{page_number:02d}.png")
            page_count += 1
    return page_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contact sheets and a durable manual-review CSV for DeepAA works."
    )
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction = args.reconstruction.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((reconstruction / "manifest.json").read_text(encoding="utf-8"))
    review_csv = output / "review.csv"
    write_review_csv(manifest, review_csv)
    page_count = build_contact_sheets(manifest, reconstruction, output)
    print(f"Prepared {page_count} contact sheets and {review_csv}.")


if __name__ == "__main__":
    main()
