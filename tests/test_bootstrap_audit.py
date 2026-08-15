from __future__ import annotations

import csv
from pathlib import Path

from training.audit_bootstrap import audit_dataset, markdown_report, series_name


def _write_dataset(path: Path) -> None:
    with path.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_name", "x", "y", "char", "label"))
        writer.writeheader()
        rows = []
        for file_name in ("Alpha_0001", "Alpha_0002", "Beta0001", "Gamma-01"):
            rows.extend(
                [
                    {"file_name": file_name, "x": 0, "y": 0, "char": " ", "label": 0},
                    {"file_name": file_name, "x": 8, "y": 0, "char": "/", "label": 1},
                ]
            )
        rows.append(rows[0].copy())
        writer.writerows(rows)


def _write_charset(path: Path) -> None:
    with path.open("w", encoding="cp932", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "char", "frequency"))
        writer.writeheader()
        writer.writerows(
            [
                {"index": 0, "char": " ", "frequency": 4},
                {"index": 1, "char": "/", "frequency": 3},
            ]
        )


def test_series_name_removes_only_numeric_suffix() -> None:
    assert series_name("Hachi_0035") == "Hachi"
    assert series_name("UltraRobot0303") == "UltraRobot"
    assert series_name("named-work") == "named-work"


def test_audit_reports_duplicates_series_and_coverage() -> None:
    test_dir = Path(__file__).resolve().parents[1] / ".tmp" / "test-bootstrap-audit"
    test_dir.mkdir(parents=True, exist_ok=True)
    dataset = test_dir / "dataset.csv"
    charset = test_dir / "charset.csv"
    _write_dataset(dataset)
    _write_charset(charset)

    report = audit_dataset(dataset, charset)

    assert report["rows"] == {
        "raw": 9,
        "deduplicated": 8,
        "exact_duplicates": 1,
        "duplicate_fraction": 0.11111111,
    }
    assert report["works"]["count"] == 4
    assert report["characters"]["raw_unique"] == 2
    assert report["characters"]["thresholds"]["10"]["character_count"] == 0
    assert report["series"]["works_per_series"] == {"Alpha": 2, "Beta": 1, "Gamma": 1}
    assert report["works"]["exact_duplicate_group_count"] == 1
    assert len(report["near_duplicates"]["pairs"]) == 1
    assert "500作品" in markdown_report(report)
