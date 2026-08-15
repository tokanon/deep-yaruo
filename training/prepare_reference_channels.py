from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from backend.input_channels import (
    ChannelExtractionConfig,
    channel_preview,
    composite_preview,
    extract_input_channels,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    {
        "id": "person-01-source",
        "path": "person-01-source.png",
        "category": "person",
        "modality": "source",
        "source_kind": "grayscale",
        "style": "illustration",
        "tone_range": "high",
        "rights_status": "local-user-provided",
    },
    {
        "id": "person-01-lineart",
        "path": "person-01-lineart.png",
        "category": "person",
        "modality": "lineart",
        "source_kind": "lineart",
        "style": "lineart",
        "tone_range": "low",
        "rights_status": "local-user-provided",
    },
    {
        "id": "background-01-source",
        "path": "background-01-source.png",
        "category": "background",
        "modality": "source",
        "source_kind": "grayscale",
        "style": "illustration",
        "tone_range": "high",
        "rights_status": "local-user-provided",
    },
    {
        "id": "background-01-lineart",
        "path": "background-01-lineart.png",
        "category": "background",
        "modality": "lineart",
        "source_kind": "lineart",
        "style": "lineart",
        "tone_range": "low",
        "rights_status": "local-user-provided",
    },
)
REQUIRED_STRATA = {
    "category": ("person", "background"),
    "modality": ("source", "lineart"),
    "style": ("photo", "illustration", "lineart"),
    "tone_range": ("low", "medium", "high"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract shared structure, tone, and fill channels for P1 references."
    )
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Local JSON manifest; paths are resolved relative to that file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".tmp" / "input-channels",
    )
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def _load_cases(
    samples: Path,
    manifest_path: Path | None,
) -> tuple[str, bool, Path, list[dict[str, str]]]:
    if manifest_path is None:
        local_manifest = samples / "p1-reference.json"
        manifest_path = local_manifest if local_manifest.is_file() else None
    if manifest_path is None:
        return "p1-reference-pilot", False, samples, [dict(item) for item in DEFAULT_CASES]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reference manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Reference manifest schema_version must be 1")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Reference manifest records must be a non-empty list")
    required = {
        "id",
        "path",
        "category",
        "modality",
        "source_kind",
        "style",
        "tone_range",
        "rights_status",
    }
    normalized: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Reference record {index} must be an object")
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"Reference record {index} is missing: {', '.join(missing)}")
        item = {key: str(record[key]).strip() for key in required}
        if item["source_kind"] not in {"grayscale", "lineart", "aa_proxy"}:
            raise ValueError(
                f"Reference record {index} has invalid source_kind: {item['source_kind']}"
            )
        if any(not value for value in item.values()):
            raise ValueError(f"Reference record {index} contains an empty field")
        normalized.append(item)
    return (
        str(payload.get("dataset") or "p1-reference-local"),
        bool(payload.get("training_allowed", False)),
        manifest_path.parent,
        normalized,
    )


def _stratification(records: list[dict[str, object]]) -> dict[str, object]:
    counts = {
        field: dict(sorted(Counter(str(record[field]) for record in records).items()))
        for field in REQUIRED_STRATA
    }
    missing = {
        field: [value for value in required if value not in counts[field]]
        for field, required in REQUIRED_STRATA.items()
    }
    missing = {field: values for field, values in missing.items() if values}
    return {
        "counts": counts,
        "required_strata": REQUIRED_STRATA,
        "missing_strata": missing,
        "gate_passed": 30 <= len(records) <= 50 and not missing,
    }


def main() -> None:
    args = parse_args()
    config = ChannelExtractionConfig(target_width=args.width)
    dataset_name, training_allowed, source_root, cases = _load_cases(
        args.samples,
        args.manifest,
    )
    records: list[dict[str, object]] = []
    missing: list[str] = []
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    for case in cases:
        case_id = case["id"]
        if case_id in seen_ids or Path(case_id).name != case_id:
            raise ValueError(f"Reference id must be unique and path-free: {case_id!r}")
        seen_ids.add(case_id)
        source_path = (source_root / case["path"]).resolve()
        if not source_path.is_file():
            missing.append(str(source_path))
            continue
        source_hash = _sha256(source_path)
        if source_hash in seen_hashes:
            raise ValueError(f"Duplicate reference image content: {source_path}")
        seen_hashes.add(source_hash)
        gray = np.asarray(Image.open(source_path).convert("L"))
        channels = extract_input_channels(
            gray,
            source_kind=case["source_kind"],
            config=config,
        )
        case_dir = args.output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(channels.resized_gray).save(case_dir / "resized.png")
        Image.fromarray(channel_preview(channels.structure)).save(case_dir / "ch0-structure.png")
        Image.fromarray(channel_preview(channels.tone)).save(case_dir / "ch1-tone.png")
        Image.fromarray(channel_preview(channels.fill)).save(case_dir / "ch2-fill.png")
        Image.fromarray(composite_preview(channels)).save(case_dir / "preview-composite.png")
        np.savez_compressed(case_dir / "channels.npz", channels=channels.stacked)
        records.append(
            {
                "id": case_id,
                "source": source_path.relative_to(ROOT).as_posix(),
                "source_sha256": source_hash,
                "category": case["category"],
                "modality": case["modality"],
                "style": case["style"],
                "tone_range": case["tone_range"],
                "rights_status": case["rights_status"],
                "metadata": channels.metadata,
                "output": case_dir.relative_to(args.output).as_posix(),
            }
        )
    if missing:
        raise FileNotFoundError("Missing reference samples: " + ", ".join(missing))
    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "training_allowed": training_allowed,
        "required_reference_count": "30-50",
        "current_reference_count": len(records),
        "stratification": _stratification(records),
        "records": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Prepared {len(records)} P1 reference cases in {args.output.resolve()}")


if __name__ == "__main__":
    main()
