from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from training.prepare_quality_review import build_contact_sheets, write_review_csv


def test_quality_review_builds_series_sheets_and_preserves_scores() -> None:
    test_dir = Path(__file__).resolve().parents[1] / ".tmp" / "test-quality-review"
    reconstruction = test_dir / "reconstruction"
    output = test_dir / "review"
    (reconstruction / "train").mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), 255).save(reconstruction / "train" / "Alpha_0001.png")
    Image.new("L", (24, 40), 255).save(reconstruction / "train" / "Beta0001.png")
    manifest = {
        "works": [
            {
                "file_name": "Alpha_0001",
                "split": "train",
                "width": 32,
                "height": 32,
                "placements": 5,
                "image": "train/Alpha_0001.png",
            },
            {
                "file_name": "Beta0001",
                "split": "train",
                "width": 24,
                "height": 40,
                "placements": 7,
                "image": "train/Beta0001.png",
            },
        ]
    }
    review_csv = output / "review.csv"
    write_review_csv(manifest, review_csv)

    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["quality"] = "A"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    write_review_csv(manifest, review_csv)
    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        preserved = list(csv.DictReader(handle))
    assert preserved[0]["quality"] == "A"
    assert build_contact_sheets(manifest, reconstruction, output) == 2
    assert (output / "sheets" / "Alpha-01.png").exists()
    assert (output / "sheets" / "Beta-01.png").exists()
