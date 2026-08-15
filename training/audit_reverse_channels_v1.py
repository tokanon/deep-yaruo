from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from backend.rendering import glyph_advance
from training.prepare_reverse_channels_v1 import DEFAULT_OUTPUT
from training.reverse_channels import METHOD_VERSION, _canonical_json, _sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPEAT = ROOT / ".tmp" / "training-runs" / "reverse-channels-v1-repeat"


def _payload_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path.name == "audit.json":
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _records(root: Path, manifest: dict[str, object]) -> list[dict[str, object]]:
    path = root / str(manifest["records"])
    payload = path.read_bytes()
    if _sha256(payload) != manifest["records_sha256"]:
        raise ValueError("records.jsonl does not match manifest.json")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _counter_summary(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _group_summary(
    group: list[dict[str, object]],
) -> dict[str, int | float]:
    glyphs = sum(int(record["glyph_count"]) for record in group)
    a_fill = sum(int(record["a"]["fill_glyph_count"]) for record in group)
    b_fill = sum(int(record["b"]["fill_glyph_count"]) for record in group)
    disagreement = sum(int(record["ab_disagreement_glyph_count"]) for record in group)
    return {
        "works": len(group),
        "glyphs": glyphs,
        "a_works_with_fill": sum(int(record["a"]["fill_glyph_count"]) > 0 for record in group),
        "b_works_with_fill": sum(int(record["b"]["fill_glyph_count"]) > 0 for record in group),
        "a_fill_glyphs": a_fill,
        "b_fill_glyphs": b_fill,
        "ab_disagreement_glyphs": disagreement,
        "a_fill_fraction": round(a_fill / max(1, glyphs), 8),
        "b_fill_fraction": round(b_fill / max(1, glyphs), 8),
        "ab_disagreement_fraction": round(disagreement / max(1, glyphs), 8),
    }


def audit_reverse_channels(
    root: Path,
    *,
    repeat_root: Path | None = None,
    visual_status: str = "manual-review-required",
    visual_notes: tuple[str, ...] = (),
    selected_candidate: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = json.loads(manifest_payload.decode("utf-8"))
    records = _records(root, manifest)
    failures: list[dict[str, object]] = []
    category_records: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_records: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    split_category_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        entry_id = int(record["entry_id"])
        category = str(record["category"])
        split = str(record["split"])
        category_records[category].append(record)
        split_records[split].append(record)
        split_category_counts[split][category] += 1
        for artifact_name, artifact in record["artifacts"].items():
            artifact_path = root / str(artifact["path"])
            if not artifact_path.is_file():
                failures.append(
                    {"entry_id": entry_id, "kind": "missing-artifact", "artifact": artifact_name}
                )
                continue
            if _sha256(artifact_path.read_bytes()) != artifact["sha256"]:
                failures.append(
                    {"entry_id": entry_id, "kind": "artifact-hash", "artifact": artifact_name}
                )
        text = (root / str(record["artifacts"]["target"]["path"])).read_text(encoding="utf-8")
        lines = text.splitlines() or [""]
        channels_path = root / str(record["artifacts"]["channels"]["path"])
        with np.load(channels_path) as arrays:
            codepoints = arrays["glyph_codepoints"]
            row_offsets = arrays["row_offsets"]
            row_widths = arrays["row_widths"]
            starts = arrays["glyph_x_starts"]
            advances = arrays["glyph_advances"]
            reconstructed = [
                "".join(chr(int(value)) for value in codepoints[int(row_offsets[index]) : int(row_offsets[index + 1])])
                for index in range(len(row_offsets) - 1)
            ]
            if reconstructed != lines:
                failures.append({"entry_id": entry_id, "kind": "text-lattice"})
            cursor = 0
            expected_starts: list[int] = []
            expected_advances: list[int] = []
            expected_widths: list[int] = []
            for line in lines:
                width = 0
                for character in line:
                    expected_starts.append(width)
                    advance = glyph_advance(character)
                    expected_advances.append(advance)
                    width += advance
                expected_widths.append(width)
                cursor += len(line)
            if cursor != len(codepoints):
                failures.append({"entry_id": entry_id, "kind": "glyph-count"})
            if not np.array_equal(starts, np.asarray(expected_starts, dtype=starts.dtype)):
                failures.append({"entry_id": entry_id, "kind": "glyph-starts"})
            if not np.array_equal(advances, np.asarray(expected_advances, dtype=advances.dtype)):
                failures.append({"entry_id": entry_id, "kind": "glyph-advances"})
            if not np.array_equal(row_widths, np.asarray(expected_widths, dtype=row_widths.dtype)):
                failures.append({"entry_id": entry_id, "kind": "row-widths"})
            target = arrays["target_mask"].astype(bool)
            for prefix in ("a", "b"):
                line_mask = arrays[f"{prefix}_line_mask"].astype(bool)
                fill_mask = arrays[f"{prefix}_fill_glyph_mask"].astype(bool)
                labels = arrays[f"{prefix}_surface_ids"]
                tone = arrays[f"{prefix}_surface_tone"]
                if not np.array_equal(target, line_mask | fill_mask):
                    failures.append({"entry_id": entry_id, "kind": f"{prefix}-recomposition"})
                if np.any(line_mask & fill_mask):
                    failures.append({"entry_id": entry_id, "kind": f"{prefix}-overlap"})
                if labels.shape != target.shape or tone.shape != target.shape:
                    failures.append({"entry_id": entry_id, "kind": f"{prefix}-shape"})
            if np.any(arrays["a_fill_owned"] > arrays["b_fill_owned"]):
                failures.append({"entry_id": entry_id, "kind": "b-not-superset"})

    primary_digest = _payload_digest(root)
    repeat_digest = _payload_digest(repeat_root.resolve()) if repeat_root is not None else None
    deterministic = repeat_digest == primary_digest if repeat_digest is not None else None
    if deterministic is False:
        failures.append({"kind": "repeat-payload-digest"})
    if len(records) != int(manifest["record_count"]):
        failures.append({"kind": "record-count"})
    report = {
        "schema_version": 1,
        "method_version": METHOD_VERSION,
        "manifest_sha256": _sha256(manifest_payload),
        "record_count": len(records),
        "artifact_count": sum(len(record["artifacts"]) for record in records),
        "category_counts": _counter_summary(Counter(str(record["category"]) for record in records)),
        "split_counts": _counter_summary(Counter(str(record["split"]) for record in records)),
        "split_category_counts": {
            split: _counter_summary(counts)
            for split, counts in sorted(split_category_counts.items())
        },
        "by_category": {
            category: _group_summary(group)
            for category, group in sorted(category_records.items())
        },
        "by_split": {
            split: _group_summary(group)
            for split, group in sorted(split_records.items())
        },
        "overall": _group_summary(records),
        "integrity_failure_count": len(failures),
        "integrity_failures": failures[:100],
        "primary_payload_sha256": primary_digest,
        "repeat_payload_sha256": repeat_digest,
        "byte_deterministic": deterministic,
        "passed": not failures and deterministic is not False,
        "training_candidate_selected": selected_candidate,
        "visual_review": {
            "sample_count": len(manifest["preview_paths"]),
            "sample_selection": "two highest A/B disagreement works per corrected category",
            "status": visual_status,
            "notes": list(visual_notes),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit accepted-v2 reverse-channel artifacts.")
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat-root", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--visual-status",
        choices=("manual-review-required", "passed", "failed"),
        default="manual-review-required",
    )
    parser.add_argument("--visual-note", action="append", default=[])
    parser.add_argument(
        "--selected-candidate",
        choices=("a-p6r1", "b-periodic"),
    )
    args = parser.parse_args()
    report = audit_reverse_channels(
        args.root,
        repeat_root=args.repeat_root,
        visual_status=args.visual_status,
        visual_notes=tuple(args.visual_note),
        selected_candidate=args.selected_candidate,
    )
    output = args.output or args.root / "audit.json"
    output.write_bytes(_canonical_json(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"] or report["visual_review"]["status"] == "failed":
        raise SystemExit("Reverse-channel audit failed")


if __name__ == "__main__":
    main()
