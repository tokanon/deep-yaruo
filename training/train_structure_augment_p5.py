from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from training.structure_proxy_dataset import prepare_structure_proxy_character_dataset
from training.train_structure_proxy_p3 import _paired_work_bootstrap, _train_method


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_CHARSET = ROOT / "models" / "deepaa-charset.csv"
DEFAULT_PRETRAINED = ROOT / "models" / "deepaa-light.onnx"
DEFAULT_DIAGNOSTIC = ROOT / ".tmp" / "training-runs" / "structure-augment-p5" / "report.json"
DEFAULT_P3 = ROOT / ".tmp" / "training-runs" / "structure-proxy-p3" / "report.json"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "structure-augment-p5-downstream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P5-safe ch0 augmentation mixes and compare with P3 raw."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--charset", type=Path, default=DEFAULT_CHARSET)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--diagnostic-report", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--p3-report", type=Path, default=DEFAULT_P3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policies", nargs="*")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _metric_delta(candidate: dict[str, object], baseline: dict[str, object], key: str) -> float:
    return float(candidate[key]) - float(baseline[key])


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("epochs and bootstrap-samples must be positive")
    diagnostic = json.loads(args.diagnostic_report.read_text(encoding="utf-8"))
    p3_report = json.loads(args.p3_report.read_text(encoding="utf-8"))
    policies = args.policies
    if policies is None:
        policies = list(diagnostic["selection"]["shortlisted_for_p3_downstream"])
    if not policies:
        raise ValueError("P5 diagnostic did not shortlist an augmentation policy")
    allowed = set(diagnostic["selection"]["shortlisted_for_p3_downstream"])
    unexpected = sorted(set(policies) - allowed)
    if unexpected:
        raise ValueError(f"Policies did not pass P5 diagnostics: {', '.join(unexpected)}")

    output = args.output.resolve()
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"P5 downstream output is not empty: {output}; use --force")
        if allowed_root not in output.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    reports: dict[str, object] = {}
    for policy_id in policies:
        dataset_dir = output / "datasets" / policy_id
        prepare_structure_proxy_character_dataset(
            args.snapshot,
            args.charset,
            dataset_dir,
            method="raw_raster",
            seed=args.seed,
            training_augment_policy=policy_id,
        )
        reports[policy_id] = _train_method(
            policy_id,
            dataset_dir / "dataset.json",
            output / "models" / policy_id,
            args,
            device,
        )

    raw_test = p3_report["methods"]["raw_raster"]["final"]["test"]
    comparisons: dict[str, object] = {}
    passing: list[tuple[float, float, str]] = []
    for policy_id, policy_report in reports.items():
        test = policy_report["final"]["test"]
        deltas = {
            key: _metric_delta(test, raw_test, key)
            for key in (
                "character_accuracy",
                "nonblank_accuracy",
                "blank_accuracy",
                "top5_accuracy",
                "macro_accuracy_present_classes",
            )
        }
        bootstrap = {
            "all": _paired_work_bootstrap(
                raw_test,
                test,
                seed=args.seed,
                samples=args.bootstrap_samples,
            ),
            "nonblank": _paired_work_bootstrap(
                raw_test,
                test,
                seed=args.seed + 1,
                samples=args.bootstrap_samples,
                field_prefix="nonblank_",
            ),
        }
        gate_passed = (
            deltas["character_accuracy"] >= -0.005
            and deltas["nonblank_accuracy"] >= -0.005
            and deltas["top5_accuracy"] >= -0.005
        )
        if gate_passed:
            passing.append(
                (
                    -float(test["nonblank_accuracy"]),
                    -float(test["character_accuracy"]),
                    policy_id,
                )
            )
        comparisons[policy_id] = {
            "test_minus_p3_raw": deltas,
            "paired_work_bootstrap": bootstrap,
            "preservation_gate_passed": gate_passed,
        }
    passing.sort()
    selected = passing[0][2] if passing else "raw_only"
    report = {
        "schema_version": 1,
        "phase": "P5",
        "dataset": "structure-augment-p5-downstream-c1",
        "purpose": "held-out synthetic character utility of safe ch0 augmentation mixes",
        "seed": args.seed,
        "device": str(device),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_samples": args.max_train_samples,
            "pretrained": str(args.pretrained.resolve()),
        },
        "diagnostic_report": str(args.diagnostic_report.resolve()),
        "p3_baseline_report": str(args.p3_report.resolve()),
        "p3_raw_test": raw_test,
        "policies": reports,
        "comparisons": comparisons,
        "selection": {
            "selected_policy": selected,
            "gate": {
                "character_accuracy_delta_min": -0.005,
                "nonblank_accuracy_delta_min": -0.005,
                "top5_accuracy_delta_min": -0.005,
                "ranking": "higher nonblank accuracy, then higher all-character accuracy",
            },
            "interpretation": (
                "Passing only preserves synthetic downstream utility. Real images have no "
                "gold charAcc; runtime AA quality remains a fixed-case human comparison."
            ),
        },
        "rights_status": "local training only; datasets and checkpoints are not tracked",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    print(f"saved={(output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
