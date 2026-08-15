from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from backend.input_channels import ChannelExtractionConfig, channel_preview
from training.evaluate_structure_proxies import _load_reference_features
from training.review_corpus import load_snapshot_records, sha256_bytes
from training.structure_augment import (
    STRUCTURE_AUGMENT_POLICIES,
    apply_render_augment,
    grouped_domain_auc,
)
from training.structure_proxy import (
    StructureProxyConfig,
    extract_proxy_structure,
    structure_descriptor,
    symmetric_chamfer,
)
from training.weak_pairs import render_aa_text, select_pilot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_REFERENCES = ROOT / ".tmp" / "input-channels" / "manifest.json"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "structure-augment-p5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5 grouped domain/Chamfer evaluation of safe ch0 augmentation mixes."
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


def _weighted_summary(records: list[dict[str, object]]) -> dict[str, object]:
    values: list[float] = []
    catastrophic: list[dict[str, object]] = []
    for record in records:
        value = float(record["chamfer"]["symmetric_px"])
        weight = int(record["weight"])
        if math.isfinite(value):
            values.extend([value] * weight)
        if record["catastrophic_flags"]:
            catastrophic.append(record)
    array = np.asarray(values, dtype=np.float64)
    return {
        "chamfer_symmetric_px": {
            "mean": round(float(np.mean(array)), 6) if array.size else None,
            "median": round(float(np.median(array)), 6) if array.size else None,
            "p95": round(float(np.quantile(array, 0.95)), 6) if array.size else None,
            "maximum": round(float(np.max(array)), 6) if array.size else None,
            "weighting": "policy mixture weights",
        },
        "catastrophic_failure_count": len(catastrophic),
        "catastrophic_failures": [
            {
                "entry_id": item["entry_id"],
                "augment": item["augment"],
                "flags": item["catastrophic_flags"],
            }
            for item in catastrophic
        ],
    }


def _select_shortlist(reports: dict[str, dict[str, object]]) -> dict[str, object]:
    baseline_auc = float(reports["raw_only"]["domain"]["separability_auc"])
    eligible: list[tuple[float, float, str]] = []
    rejected: dict[str, list[str]] = {}
    for policy_id, report in reports.items():
        if policy_id == "raw_only":
            continue
        reasons: list[str] = []
        summary = report["summary"]
        chamfer = summary["chamfer_symmetric_px"]
        mean = float(chamfer["mean"])
        p95 = float(chamfer["p95"])
        auc = float(report["domain"]["separability_auc"])
        if int(summary["catastrophic_failure_count"]) != 0:
            reasons.append("catastrophic_failure")
        if mean > 0.5:
            reasons.append("weighted_mean_chamfer_above_0.5px")
        if p95 > 1.0:
            reasons.append("weighted_p95_chamfer_above_1.0px")
        if auc > baseline_auc + 0.02:
            reasons.append("domain_auc_more_than_0.02_above_raw")
        if reasons:
            rejected[policy_id] = reasons
        else:
            eligible.append((auc, mean, policy_id))
    eligible.sort()
    shortlisted = [policy_id for _, _, policy_id in eligible[:2]]
    return {
        "baseline": "raw_only",
        "shortlisted_for_p3_downstream": shortlisted,
        "rejected_before_downstream": rejected,
        "gate": {
            "weighted_mean_chamfer_max_px": 0.5,
            "weighted_p95_chamfer_max_px": 1.0,
            "catastrophic_failures": 0,
            "domain_auc_max_increase_vs_raw": 0.02,
            "selection_order": "lower AUC, then lower mean Chamfer",
        },
        "note": (
            "AUC is diagnostic only. A shortlisted policy is not selected until "
            "held-out P3 character metrics are compared with raw."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 2 or args.bootstrap_samples <= 0:
        raise ValueError("limit must be >=2 and bootstrap-samples must be positive")
    output = args.output.resolve()
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"P5 output is not empty: {output}; use --force")
        if allowed_root not in output.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    real_features, reference_report = _load_reference_features(args.references.resolve())
    real_groups = [f"real:{item['id']}" for item in reference_report["records"]]
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
    reports: dict[str, dict[str, object]] = {}
    all_records: list[dict[str, object]] = []
    image_cache: dict[tuple[int, str], str] = {}
    rendered_by_entry: dict[int, np.ndarray] = {}
    reference_by_entry: dict[int, np.ndarray] = {}
    category_by_entry: dict[int, str] = {}
    for record in selected:
        entry_id = int(record["entry_id"])
        payload = (args.snapshot / str(record["text"])).read_bytes()
        if sha256_bytes(payload) != record["unicode_text_sha256"]:
            raise ValueError(f"Accepted AA text hash mismatch: {record['text']}")
        rendered = render_aa_text(payload.decode("utf-8"), font_size=proxy_config.font_size)
        rendered_by_entry[entry_id] = rendered
        reference_by_entry[entry_id] = extract_proxy_structure(
            rendered,
            channel_config=channel_config,
        )
        category_by_entry[entry_id] = str(record["corrected_category"])

    for policy_id, policy in STRUCTURE_AUGMENT_POLICIES.items():
        proxy_features: list[np.ndarray] = []
        proxy_groups: list[str] = []
        policy_records: list[dict[str, object]] = []
        for entry_id, rendered in rendered_by_entry.items():
            reference = reference_by_entry[entry_id]
            for weighted in policy.variants:
                augmented = apply_render_augment(rendered, weighted.augment)
                structure = extract_proxy_structure(
                    augmented,
                    channel_config=channel_config,
                )
                descriptor = structure_descriptor(structure)
                proxy_features.extend([descriptor] * weighted.weight)
                proxy_groups.extend([f"proxy:{entry_id}"] * weighted.weight)
                chamfer = symmetric_chamfer(reference, structure)
                reference_pixels = float(chamfer["reference_pixels"])
                candidate_pixels = float(chamfer["candidate_pixels"])
                ratio = candidate_pixels / reference_pixels if reference_pixels else math.inf
                flags: list[str] = []
                if not math.isfinite(float(chamfer["symmetric_px"])):
                    flags.append("empty_reference_or_candidate")
                elif float(chamfer["symmetric_px"]) > 3.0:
                    flags.append("chamfer_above_3px")
                if ratio < 0.1:
                    flags.append("candidate_line_coverage_below_10pct")
                if ratio > 10.0:
                    flags.append("candidate_line_coverage_above_1000pct")
                cache_key = (entry_id, weighted.augment)
                if cache_key not in image_cache:
                    image_relative = (
                        Path("images")
                        / f"{entry_id:07d}"
                        / f"{weighted.augment}.png"
                    )
                    image_path = output / image_relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(channel_preview(structure)).save(image_path, optimize=True)
                    image_cache[cache_key] = image_relative.as_posix()
                item = {
                    "policy_id": policy_id,
                    "entry_id": entry_id,
                    "category": category_by_entry[entry_id],
                    "augment": weighted.augment,
                    "weight": weighted.weight,
                    "chamfer": {
                        key: (round(float(value), 6) if math.isfinite(float(value)) else None)
                        for key, value in chamfer.items()
                    },
                    "line_coverage_ratio": round(ratio, 6) if math.isfinite(ratio) else None,
                    "catastrophic_flags": flags,
                    "output": image_cache[cache_key],
                }
                policy_records.append(item)
                all_records.append(item)
        domain = grouped_domain_auc(
            real_features,
            np.stack(proxy_features),
            real_groups=real_groups,
            proxy_groups=proxy_groups,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        reports[policy_id] = {
            "policy": policy.manifest(),
            "domain": domain,
            "summary": _weighted_summary(policy_records),
            "distinct_variant_records": len(policy_records),
        }

    selection = _select_shortlist(reports)
    records_payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in all_records
    )
    (output / "records.jsonl").write_text(records_payload, encoding="utf-8", newline="\n")
    report = {
        "schema_version": 1,
        "phase": "P5",
        "dataset": "structure-augment-p5",
        "purpose": "safe ch0 mixture diagnostics before downstream character training",
        "seed": args.seed,
        "channel_config": channel_config.__dict__,
        "snapshot": snapshot["snapshot"],
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "reference": reference_report,
        "selected_aa_works": [int(record["entry_id"]) for record in selected],
        "policies": reports,
        "selection": selection,
        "excluded_operations": {
            "line_dropout": "previously removed necessary lines",
            "rotation_scale_aspect": "requires applying the exact transform to labels",
            "translation": "requires shifting supervised start coordinates",
        },
        "rights_status": "local diagnostics only; AA and images are not tracked",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print(f"saved={(output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
