from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from backend.input_channels import ChannelExtractionConfig, channel_preview
from backend.rendering import find_font
from training.review_corpus import load_snapshot_records, sha256_bytes
from training.structure_proxy import (
    STRUCTURE_PROXY_METHODS,
    StructureProxyConfig,
    domain_auc,
    extract_proxy_structure,
    generate_structure_proxy,
    structure_descriptor,
    symmetric_chamfer,
)
from training.weak_pairs import render_aa_text, select_pilot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_REFERENCES = ROOT / ".tmp" / "input-channels" / "manifest.json"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "structure-proxy-p2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare P2 AA structure proxies by image-level domain AUC and Chamfer."
    )
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_reference_features(manifest_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"P1 reference manifest is missing: {manifest_path}. "
            "Run python -m training.prepare_reference_channels first."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stratification = payload.get("stratification", {})
    if not stratification.get("gate_passed"):
        raise ValueError("P1 reference stratification gate has not passed")
    descriptors: list[np.ndarray] = []
    verified_records: list[dict[str, object]] = []
    output_root = manifest_path.parent
    for record in payload["records"]:
        source_path = ROOT / str(record["source"])
        if not source_path.is_file() or _sha256(source_path) != record["source_sha256"]:
            raise ValueError(f"P1 source hash mismatch: {source_path}")
        channels_path = output_root / str(record["output"]) / "channels.npz"
        with np.load(channels_path) as channels_payload:
            channels = channels_payload["channels"]
        if channels.ndim != 3 or channels.shape[0] < 1:
            raise ValueError(f"Invalid P1 channel array: {channels_path}")
        descriptors.append(structure_descriptor(channels[0]))
        verified_records.append(
            {
                "id": record["id"],
                "category": record["category"],
                "modality": record["modality"],
                "source_sha256": record["source_sha256"],
            }
        )
    return np.stack(descriptors), {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "count": len(descriptors),
        "stratification": stratification,
        "records": verified_records,
    }


def _summarize_candidate(records: list[dict[str, object]]) -> dict[str, object]:
    finite = np.array(
        [
            float(record["chamfer"]["symmetric_px"])
            for record in records
            if record["chamfer"]["symmetric_px"] is not None
        ],
        dtype=np.float64,
    )
    catastrophic = [record for record in records if record["catastrophic_flags"]]
    worst = sorted(
        records,
        key=lambda record: (
            float(record["chamfer"]["symmetric_px"])
            if record["chamfer"]["symmetric_px"] is not None
            else math.inf
        ),
        reverse=True,
    )[:5]
    return {
        "chamfer_symmetric_px": {
            "mean": round(float(np.mean(finite)), 6) if finite.size else None,
            "median": round(float(np.median(finite)), 6) if finite.size else None,
            "p95": round(float(np.quantile(finite, 0.95)), 6) if finite.size else None,
            "maximum": round(float(np.max(finite)), 6) if finite.size else None,
        },
        "catastrophic_failure_count": len(catastrophic),
        "catastrophic_failures": [
            {"entry_id": record["entry_id"], "flags": record["catastrophic_flags"]}
            for record in catastrophic
        ],
        "largest_chamfer": [
            {
                "entry_id": record["entry_id"],
                "category": record["category"],
                "symmetric_px": record["chamfer"]["symmetric_px"],
                "output": record["output"],
            }
            for record in worst
        ],
    }


def main() -> None:
    args = parse_args()
    if args.limit < 2:
        raise ValueError("--limit must be at least two")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        raise FileExistsError(f"P2 output is not empty: {args.output}; use --force")
    args.output.mkdir(parents=True, exist_ok=True)

    real_features, reference_report = _load_reference_features(args.references.resolve())
    snapshot_payload = (args.snapshot / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_payload)
    selected = select_pilot_records(
        load_snapshot_records(args.snapshot),
        limit=args.limit,
        seed=args.seed,
    )
    if len(selected) < args.limit:
        raise ValueError(f"Only {len(selected)} accepted records are available")

    channel_config = ChannelExtractionConfig(target_width=args.width)
    proxy_config = StructureProxyConfig()
    all_records: list[dict[str, object]] = []
    candidate_reports: dict[str, object] = {}
    for method in STRUCTURE_PROXY_METHODS:
        method_records: list[dict[str, object]] = []
        method_features: list[np.ndarray] = []
        for record in selected:
            entry_id = int(record["entry_id"])
            text_path = args.snapshot / str(record["text"])
            text_payload = text_path.read_bytes()
            if sha256_bytes(text_payload) != record["unicode_text_sha256"]:
                raise ValueError(f"Accepted AA text hash mismatch: {text_path}")
            text = text_payload.decode("utf-8")
            rendered = render_aa_text(text, font_size=proxy_config.font_size)
            reference = extract_proxy_structure(
                rendered,
                channel_config=channel_config,
            )
            proxy, parameters = generate_structure_proxy(
                text,
                method,
                config=proxy_config,
            )
            structure = extract_proxy_structure(
                proxy,
                channel_config=channel_config,
            )
            chamfer = symmetric_chamfer(reference, structure)
            reference_pixels = float(chamfer["reference_pixels"])
            candidate_pixels = float(chamfer["candidate_pixels"])
            coverage_ratio = (
                candidate_pixels / reference_pixels if reference_pixels else math.inf
            )
            flags: list[str] = []
            if not math.isfinite(float(chamfer["symmetric_px"])):
                flags.append("empty_reference_or_candidate")
            if coverage_ratio < 0.1:
                flags.append("candidate_line_coverage_below_10pct")
            if coverage_ratio > 10.0:
                flags.append("candidate_line_coverage_above_1000pct")

            output_dir = args.output / "cases" / method / f"{entry_id:07d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rendered).save(output_dir / "aa-rendered.png")
            Image.fromarray(proxy).save(output_dir / "proxy.png")
            Image.fromarray(channel_preview(reference)).save(output_dir / "reference-skeleton.png")
            Image.fromarray(channel_preview(structure)).save(output_dir / "ch0-structure.png")
            output_relative = output_dir.relative_to(args.output).as_posix()
            method_record: dict[str, object] = {
                "entry_id": entry_id,
                "category": record["corrected_category"],
                "split": record["split"],
                "method": method,
                "parameters": parameters,
                "source_text": str(record["text"]),
                "source_text_sha256": record["unicode_text_sha256"],
                "chamfer": {
                    key: round(float(value), 6) if math.isfinite(float(value)) else None
                    for key, value in chamfer.items()
                },
                "line_coverage_ratio": round(coverage_ratio, 6),
                "catastrophic_flags": flags,
                "output": output_relative,
                "rights_status": "unknown; local curation only; do not redistribute",
            }
            method_records.append(method_record)
            all_records.append(method_record)
            method_features.append(structure_descriptor(structure))
        domain_report = domain_auc(
            real_features,
            np.stack(method_features),
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        candidate_reports[method] = {
            "domain": domain_report,
            **_summarize_candidate(method_records),
        }

    records_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in all_records
    ).encode("utf-8")
    (args.output / "records.jsonl").write_bytes(records_payload)
    report = {
        "schema_version": 1,
        "phase": "P2",
        "dataset": "structure-proxy-p2-pilot",
        "purpose": "diagnose structure proxies; not source-image reconstruction",
        "chamfer_reference": (
            "ch0 from the raw 1x AA raster through the same common X; "
            "this measures proxy-induced structure displacement, not semantic source truth"
        ),
        "gate_interpretation": (
            "AUC and Chamfer form a diagnostic Pareto comparison. "
            "Neither is an unconditional acceptance threshold; P3 charAcc is required."
        ),
        "seed": args.seed,
        "selected_aa_count": len(selected),
        "methods": list(STRUCTURE_PROXY_METHODS),
        "references": reference_report,
        "snapshot": {
            "path": str(args.snapshot.resolve()),
            "snapshot": snapshot["snapshot"],
            "snapshot_sha256": sha256_bytes(snapshot_payload),
        },
        "font": {
            "path": str(find_font().resolve()),
            "sha256": _sha256(find_font()),
        },
        "channel_config": channel_config.__dict__,
        "proxy_config": proxy_config.__dict__,
        "candidates": candidate_reports,
        "records": "records.jsonl",
        "records_sha256": sha256_bytes(records_payload),
        "rights_status": "local diagnostics only; inputs and derived outputs are not tracked",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Evaluated {len(selected)} accepted AAs with {len(STRUCTURE_PROXY_METHODS)} "
        f"P2 methods. Report: {(args.output / 'report.json').resolve()}"
    )


if __name__ == "__main__":
    main()
