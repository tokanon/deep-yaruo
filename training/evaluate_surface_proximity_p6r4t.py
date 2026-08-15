from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.contracts import ConversionOptions
from backend.surface_fill_preferences import (
    DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT,
    DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
    SurfaceFillPreferenceStore,
)
from backend.surface_texture import (
    MINIMUM_IMAGE_COUNT,
    MINIMUM_PAIR_COUNT,
    MINIMUM_SEPARATION_FRACTION,
    PROXIMITY_COMPARISON_KIND,
    PROXIMITY_RECIPE_VERSION,
    generate_surface_texture_proximity_comparison,
    load_texture_catalog,
)
from training.audit_texture_motifs_p6r4t import _best_baselines


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / ".tmp" / "training-runs" / "surface-proximity-p6r4t" / "report.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def evaluate(
    v2_root: Path = DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
    v4_root: Path = DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    palette, edge_adjustments, catalog_hash, audit_config_hash = (
        load_texture_catalog()
    )
    v2_store = SurfaceFillPreferenceStore(v2_root)
    v4_store = SurfaceFillPreferenceStore(
        v4_root,
        comparison_kind="information-recovery-v4",
        schema_version=4,
        require_common_processed=True,
    )
    cases: list[dict[str, object]] = []
    deterministic = True
    for root, record in _best_baselines(v2_root, v4_root):
        source_artifact = record["source"]["artifact"]
        data = (
            root / str(record["record_id"]) / str(source_artifact["path"])
        ).read_bytes()
        options = ConversionOptions(**record["options"])
        first = generate_surface_texture_proximity_comparison(
            data,
            options,
            v2_store,
            v4_store,
            human_only=False,
            palette=palette,
            edge_adjustments=edge_adjustments,
            catalog_sha256=catalog_hash,
            audit_config_sha256=audit_config_hash,
        )
        second = generate_surface_texture_proximity_comparison(
            data,
            options,
            v2_store,
            v4_store,
            human_only=False,
            palette=palette,
            edge_adjustments=edge_adjustments,
            catalog_sha256=catalog_hash,
            audit_config_sha256=audit_config_hash,
        )
        deterministic &= _canonical(first) == _canonical(second)
        t3 = next(
            candidate
            for candidate in first["candidates"]
            if candidate["variantId"] == "surface-texture-proximity-v6"
        )
        surface = t3["surface"]
        texture = surface["texture"]
        cases.append(
            {
                "source_filename": first["sourceFilename"],
                "baseline_record_id": surface["source_baseline_record_id"],
                "baseline_lineage": surface["source_baseline_lineage"],
                "options": record["options"],
                "crop": record["crop"],
                "stable_surface_count": texture["stable_surface_count"],
                "textureable_surface_count": texture["textureable_surface_count"],
                "proximity_pair_count": texture["proximity_pair_count"],
                "compatible_proximity_pair_count": texture[
                    "compatible_proximity_pair_count"
                ],
                "nearest_compatible_pair_count": texture[
                    "nearest_compatible_pair_count"
                ],
                "actual_nearest_separated_pair_count": texture[
                    "actual_nearest_separated_pair_count"
                ],
                "nearest_separation_fraction": texture[
                    "nearest_separation_fraction"
                ],
                "weighted_separation_fraction": texture[
                    "weighted_separation_fraction"
                ],
                "applied_surface_count": texture["applied_surface_count"],
                "replacement_count": texture["replacement_count"],
                "maximum_density_error": texture["maximum_density_error"],
                "exact_row_widths": surface["width_diagnostics"][
                    "exact_row_widths"
                ],
                "main_line_loss_at_1px": surface["line_diagnostics"][
                    "main_line_loss_at_1px"
                ],
                "case_mechanical_gate": surface["case_mechanical_gate"],
                "comparison_sha256": hashlib.sha256(_canonical(first)).hexdigest(),
            }
        )
    nearest_pairs = sum(int(case["nearest_compatible_pair_count"]) for case in cases)
    separated_pairs = sum(
        int(case["actual_nearest_separated_pair_count"]) for case in cases
    )
    pair_images = sum(
        int(int(case["nearest_compatible_pair_count"]) > 0) for case in cases
    )
    separation_fraction = separated_pairs / nearest_pairs if nearest_pairs else 0.0
    maximum_error = max(float(case["maximum_density_error"]) for case in cases)
    maximum_line_loss = max(float(case["main_line_loss_at_1px"]) for case in cases)
    gate = bool(
        len(cases) == 15
        and nearest_pairs >= MINIMUM_PAIR_COUNT
        and pair_images >= MINIMUM_IMAGE_COUNT
        and separation_fraction >= MINIMUM_SEPARATION_FRACTION
        and maximum_error <= 0.03
        and maximum_line_loss <= 0.02
        and all(bool(case["exact_row_widths"]) for case in cases)
        and all(bool(case["case_mechanical_gate"]) for case in cases)
        and deterministic
    )
    report = {
        "schema_version": 1,
        "phase": "P6R-4T3",
        "comparison_kind": PROXIMITY_COMPARISON_KIND,
        "recipe_version": PROXIMITY_RECIPE_VERSION,
        "motif_catalog_sha256": catalog_hash,
        "motif_audit_config_sha256": audit_config_hash,
        "accepted_v1_reaggregation_skipped": True,
        "accepted_v1_reaggregation_reason": "frozen motif catalog unchanged",
        "palette": list(palette),
        "case_count": len(cases),
        "nearest_compatible_pair_count": nearest_pairs,
        "nearest_pair_image_count": pair_images,
        "minimum_pair_count": MINIMUM_PAIR_COUNT,
        "minimum_image_count": MINIMUM_IMAGE_COUNT,
        "actual_nearest_separated_pair_count": separated_pairs,
        "minimum_separation_fraction": MINIMUM_SEPARATION_FRACTION,
        "nearest_separation_fraction": round(separation_fraction, 6),
        "maximum_density_error": round(maximum_error, 8),
        "maximum_main_line_loss_at_1px": round(maximum_line_loss, 8),
        "deterministic": deterministic,
        "mechanical_gate_passed": gate,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_SURFACE_FILL_PREFERENCE_ROOT)
    parser.add_argument(
        "--v4-root", type=Path, default=DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.v2_root, args.v4_root, args.output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
