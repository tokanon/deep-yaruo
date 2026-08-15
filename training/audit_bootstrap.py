from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from training.splits import assign_series_splits, assign_work_splits, series_name


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "bootstrap" / "deepaa-500.csv"
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"
DEFAULT_JSON = ROOT / "datasets" / "bootstrap" / "audit.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "DATA_AUDIT.md"
LINE_PITCH = 18


@dataclass(frozen=True, order=True)
class AuditPlacement:
    file_name: str
    y: int
    x: int
    char: str
    label: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _quantiles(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {key: 0 for key in ("min", "p25", "median", "p75", "p95", "max")}

    def nearest(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p25": nearest(0.25),
        "median": nearest(0.5),
        "p75": nearest(0.75),
        "p95": nearest(0.95),
        "max": ordered[-1],
    }


def _work_fingerprint(placements: Sequence[AuditPlacement]) -> str:
    payload = [[item.x, item.y, item.char, item.label] for item in placements]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _work_shingles(placements: Sequence[AuditPlacement], size: int = 4) -> set[str]:
    rows: dict[int, list[AuditPlacement]] = defaultdict(list)
    for item in placements:
        rows[item.y].append(item)
    lines = ["".join(item.char for item in sorted(rows[y], key=lambda value: value.x)) for y in sorted(rows)]
    text = "\n".join(lines)
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _near_duplicate_pairs(
    works: dict[str, list[AuditPlacement]],
    *,
    threshold: float = 0.9,
) -> list[dict[str, object]]:
    """Find high-overlap text pairs within a named source series using 4-gram Jaccard."""
    by_series: dict[str, list[str]] = defaultdict(list)
    shingles: dict[str, set[str]] = {}
    for name, placements in works.items():
        by_series[series_name(name)].append(name)
        shingles[name] = _work_shingles(placements)

    matches: list[dict[str, object]] = []
    for series, names in sorted(by_series.items()):
        for left, right in itertools.combinations(sorted(names), 2):
            union = shingles[left] | shingles[right]
            if not union:
                continue
            similarity = len(shingles[left] & shingles[right]) / len(union)
            if similarity >= threshold:
                matches.append(
                    {
                        "left": left,
                        "right": right,
                        "series": series,
                        "jaccard": round(similarity, 6),
                    }
                )
    return sorted(matches, key=lambda item: (-float(item["jaccard"]), item["left"], item["right"]))


def audit_dataset(dataset_path: Path, charset_path: Path) -> dict[str, object]:
    raw_character_counts: Counter[str] = Counter()
    deduplicated_character_counts: Counter[str] = Counter()
    works: dict[str, list[AuditPlacement]] = defaultdict(list)
    seen: set[AuditPlacement] = set()
    raw_rows = 0
    duplicate_rows = 0
    label_to_char: dict[int, str] = {}
    inconsistent_labels: list[dict[str, object]] = []

    with dataset_path.open("r", encoding="cp932", newline="") as handle:
        for row in csv.DictReader(handle):
            placement = AuditPlacement(
                file_name=row["file_name"],
                x=int(row["x"]),
                y=int(row["y"]),
                char=row["char"],
                label=int(row["label"]),
            )
            raw_rows += 1
            raw_character_counts[placement.char] += 1
            known = label_to_char.setdefault(placement.label, placement.char)
            if known != placement.char:
                inconsistent_labels.append(
                    {"label": placement.label, "first": known, "other": placement.char}
                )
            if placement in seen:
                duplicate_rows += 1
                continue
            seen.add(placement)
            works[placement.file_name].append(placement)
            deduplicated_character_counts[placement.char] += 1

    for placements in works.values():
        placements.sort(key=lambda item: (item.y, item.x))

    with charset_path.open("r", encoding="cp932", newline="") as handle:
        charset_rows = list(csv.DictReader(handle))
    asset_characters = [row["char"] for row in charset_rows]
    model_characters = [
        row["char"] for row in charset_rows if int(row["frequency"]) >= 10
    ]
    model_set = set(model_characters)
    whitespace = {char for char in raw_character_counts if char.isspace()}
    raw_non_whitespace = sum(
        count for char, count in raw_character_counts.items() if char not in whitespace
    )

    thresholds: dict[str, object] = {}
    for threshold in (1, 5, 10, 20, 50, 100):
        selected = {char for char, count in raw_character_counts.items() if count >= threshold}
        covered = sum(raw_character_counts[char] for char in selected)
        covered_non_whitespace = sum(
            raw_character_counts[char] for char in selected if char not in whitespace
        )
        thresholds[str(threshold)] = {
            "character_count": len(selected),
            "raw_placement_coverage": round(covered / raw_rows, 8),
            "raw_non_whitespace_coverage": round(
                covered_non_whitespace / raw_non_whitespace if raw_non_whitespace else 1.0,
                8,
            ),
        }

    work_placement_counts = [len(items) for items in works.values()]
    work_row_counts = [len({item.y for item in items}) for items in works.values()]
    work_max_start_x = [max(item.x for item in items) for items in works.values()]
    work_heights = [max(item.y for item in items) + LINE_PITCH for items in works.values()]

    fingerprint_groups: dict[str, list[str]] = defaultdict(list)
    for name, placements in works.items():
        fingerprint_groups[_work_fingerprint(placements)].append(name)
    exact_work_duplicates = [
        sorted(names) for names in fingerprint_groups.values() if len(names) > 1
    ]

    series_counts = Counter(series_name(name) for name in works)
    current_splits = assign_work_splits(works)
    current_series_splits = {
        series: sorted({current_splits[name] for name in works if series_name(name) == series})
        for series in sorted(series_counts)
    }
    recommended_work_splits = assign_series_splits(works)
    recommended_series_splits = {
        series_name(name): split for name, split in recommended_work_splits.items()
    }
    recommended_counts = Counter(recommended_work_splits.values())

    most_frequent = []
    for char, count in raw_character_counts.most_common(30):
        most_frequent.append(
            {
                "character": char,
                "codepoints": [f"U+{ord(value):04X}" for value in char],
                "is_whitespace": char.isspace(),
                "raw_count": count,
                "deduplicated_count": deduplicated_character_counts[char],
                "raw_fraction": round(count / raw_rows, 8),
            }
        )

    missing_model_chars = sorted(model_set - set(raw_character_counts))
    expected_at_ten = {char for char, count in raw_character_counts.items() if count >= 10}
    return {
        "schema_version": 1,
        "method": {
            "series": "strip trailing digits and an optional '_' or '-' from file_name",
            "near_duplicates": "within-series character 4-gram Jaccard >= 0.90",
            "split_seed": 42,
            "quantiles": "nearest-rank over works",
        },
        "source": {
            "dataset": display_path(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "charset": display_path(charset_path),
            "charset_sha256": sha256_file(charset_path),
        },
        "rows": {
            "raw": raw_rows,
            "deduplicated": len(seen),
            "exact_duplicates": duplicate_rows,
            "duplicate_fraction": round(duplicate_rows / raw_rows, 8),
        },
        "works": {
            "count": len(works),
            "placements": _quantiles(work_placement_counts),
            "text_rows": _quantiles(work_row_counts),
            "max_character_start_x": _quantiles(work_max_start_x),
            "height_px_at_18px_pitch": _quantiles(work_heights),
            "exact_duplicate_group_count": len(exact_work_duplicates),
            "exact_duplicate_groups": sorted(exact_work_duplicates),
        },
        "characters": {
            "raw_unique": len(raw_character_counts),
            "deduplicated_unique": len(deduplicated_character_counts),
            "whitespace_characters": sorted(whitespace),
            "thresholds": thresholds,
            "asset_character_count": len(asset_characters),
            "model_charset_count": len(model_characters),
            "model_charset_unique_count": len(model_set),
            "model_matches_frequency_at_least_10": model_set == expected_at_ten,
            "model_characters_missing_from_dataset": missing_model_chars,
            "labels_with_multiple_characters": inconsistent_labels,
            "most_frequent": most_frequent,
        },
        "series": {
            "count": len(series_counts),
            "works_per_series": dict(sorted(series_counts.items())),
        },
        "splits": {
            "current_work_random": {
                "counts": dict(sorted(Counter(current_splits.values()).items())),
                "series_crossing_splits": {
                    series: splits
                    for series, splits in current_series_splits.items()
                    if len(splits) > 1
                },
            },
            "recommended_series_holdout": {
                "counts": dict(sorted(recommended_counts.items())),
                "series_assignment": dict(sorted(recommended_series_splits.items())),
                "warning": (
                    "Only five filename-derived series exist. This split prevents direct series "
                    "leakage but cannot provide balanced, representative validation and test sets."
                ),
            },
        },
        "near_duplicates": {
            "threshold": 0.9,
            "pairs": _near_duplicate_pairs(works),
        },
    }


def _percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def markdown_report(report: dict[str, object]) -> str:
    rows = report["rows"]
    works = report["works"]
    characters = report["characters"]
    series = report["series"]
    splits = report["splits"]
    thresholds = characters["thresholds"]
    current = splits["current_work_random"]
    recommended = splits["recommended_series_holdout"]
    near = report["near_duplicates"]

    lines = [
        "# DeepAA初期データ監査",
        "",
        "この文書は `python -m training.audit_bootstrap` で生成します。数値の正本は "
        "`datasets/bootstrap/audit.json` です。ファイル名から推定できない作者、権利元、題材は、"
        "このCSVだけでは監査できません。",
        "",
        "## 結論",
        "",
        f"- 500作品は5系列に集中しています（{', '.join(f'{key}: {value}' for key, value in series['works_per_series'].items())}）。",
        f"- CSV {rows['raw']:,}行のうち完全重複は{rows['exact_duplicates']:,}行（{_percent(rows['duplicate_fraction'])}）で、除去後は{rows['deduplicated']:,}配置です。",
        f"- 文字は{characters['raw_unique']}種類です。頻度10以上は{thresholds['10']['character_count']}種類で、全配置の{_percent(thresholds['10']['raw_placement_coverage'])}、非空白配置の{_percent(thresholds['10']['raw_non_whitespace_coverage'])}を覆います。これは網羅性の説明であり、AA品質に最適な411文字である証明ではありません。",
        f"- `deepaa-charset.csv` は{characters['asset_character_count']}文字の資産表で、実行時の頻度10以上フィルタは{characters['model_charset_count']}文字です。元CSVから再計算した集合との一致は `{characters['model_matches_frequency_at_least_10']}` です。",
        f"- 現行の作品単位ランダムsplitでは{len(current['series_crossing_splits'])}/{series['count']}系列が複数splitへまたがります。系列固有の作風を学習した場合、検証値が過大になります。",
        f"- 近似重複の機械検出（同系列・文字4-gram Jaccard 0.90以上）は{len(near['pairs'])}組です。これは候補抽出であり、人手確認が必要です。",
        "",
        "## 文字頻度しきい値",
        "",
        "| 最小頻度 | 文字数 | 全配置カバー率 | 非空白カバー率 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for threshold in (1, 5, 10, 20, 50, 100):
        item = thresholds[str(threshold)]
        lines.append(
            f"| {threshold} | {item['character_count']} | "
            f"{_percent(item['raw_placement_coverage'])} | "
            f"{_percent(item['raw_non_whitespace_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## 作品分布",
            "",
            "| 指標 | 最小 | p25 | 中央 | p75 | p95 | 最大 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (
        ("重複除去後の配置数", "placements"),
        ("行数", "text_rows"),
        ("文字開始x座標の最大値", "max_character_start_x"),
        ("18px行ピッチ時の高さ", "height_px_at_18px_pitch"),
    ):
        item = works[key]
        lines.append(
            f"| {label} | {item['min']} | {item['p25']} | {item['median']} | "
            f"{item['p75']} | {item['p95']} | {item['max']} |"
        )

    lines.extend(
        [
            "",
            "## split監査",
            "",
            f"現行splitは `{current['counts']}` ですが、系列を分離しません。機械的に最も比率が近い系列holdoutは "
            f"`{recommended['counts']}`、割当は `{recommended['series_assignment']}` です。",
            "",
            "このholdoutは漏洩確認用には使えますが、系列が5個しかないため、最終評価用としては不十分です。"
            "追加作品は既存5系列と混ぜるだけでなく、作者・出典・題材を記録した新しい系列として収集し、"
            "系列単位でtestへ隔離する必要があります。",
            "",
            "## 現時点で監査できない項目",
            "",
            "- 作者、出典URL、利用条件: 元CSVにメタデータがない。",
            "- 人物、背景、小物などの題材: 元画像または人手ラベルがない。",
            "- 学習曲線、macro-F1、希少文字recall: 複数条件の再学習が必要。",
            "- 411・899・拡張文字集合の最終AA品質: 同じ固定画像と人手評価で比較する必要がある。",
            "",
            "## 次の判断",
            "",
            "1. 現行ランダムsplitは回帰テスト用途に限定し、品質の根拠にはしない。",
            "2. 5系列holdoutを漏洩感度テストとして追加する。",
            "3. 新規収集データには `work_id`、`series_id`、作者・出典、題材、権利、gold/weakを必須項目にする。",
            "4. 50・100・200・400作品の学習曲線と頻度5・10・20の比較は、再学習時間を伴う別工程として実施する。",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the DeepAA bootstrap corpus and its 411-character model set."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(args.dataset.resolve(), args.charset.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.markdown.write_text(markdown_report(report), encoding="utf-8", newline="\n")
    print(
        f"Audited {report['works']['count']} works and {report['rows']['raw']} rows; "
        f"wrote {args.json} and {args.markdown}."
    )


if __name__ == "__main__":
    main()
