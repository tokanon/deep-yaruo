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
    COMPARISON_KIND,
    MINIMUM_ELIGIBLE_FRACTION,
    MINIMUM_IMAGE_COUNT,
    MINIMUM_PAIR_COUNT,
    MINIMUM_SEPARATION_FRACTION,
    RECIPE_VERSION,
    generate_surface_texture_comparison,
    load_texture_catalog,
)
from training.audit_texture_motifs_p6r4t import (
    DEFAULT_OUTPUT as DEFAULT_AUDIT_REPORT,
    _best_baselines,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "surface-texture-p6r4t" / "report.json"


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
    audit_report_path: Path = DEFAULT_AUDIT_REPORT,
) -> dict[str, object]:
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if (
        audit_report.get("method_version") != "periodic-fill-motif-audit-v2"
        or audit_report.get("motif_catalog", {}).get("accepted_entry_count") != 100
        or not audit_report.get("gate_passed")
    ):
        raise ValueError("P6R-4T1 requires a passing full v2 motif audit")
    motif_catalog = audit_report["motif_catalog"]
    palette = tuple(str(value) for value in audit_report["stress_set"]["selected_palette"])
    motif_density = {
        str(item["motif"]): float(item["ink_density"])
        for item in motif_catalog["motifs"]
        if item["supported"]
    }
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]] = {}
    for side in ("left", "right"):
        by_motif: dict[str, list[str]] = {motif: [] for motif in palette}
        for item in motif_catalog["edge_adjustments"][side]:
            motif = str(item["motif"])
            character = str(item["character"])
            if (
                motif not in by_motif
                or character == "\ufffd"
                or int(item["natural_advance_px"]) > 8
                or abs(float(item["ink_density"]) - motif_density[motif]) > 0.03
            ):
                continue
            if character not in by_motif[motif]:
                by_motif[motif].append(character)
        edge_adjustments[side] = {
            motif: tuple(characters) for motif, characters in by_motif.items()
        }
    frozen_palette, frozen_adjustments, frozen_catalog_hash, frozen_config_hash = (
        load_texture_catalog()
    )
    if (
        frozen_palette != palette
        or frozen_adjustments != edge_adjustments
        or frozen_catalog_hash != str(motif_catalog["catalog_sha256"])
        or frozen_config_hash != str(audit_report["config_sha256"])
    ):
        raise ValueError("The frozen P6R-4T catalog does not match the full audit")
    v2_store = SurfaceFillPreferenceStore(v2_root)
    v4_store = SurfaceFillPreferenceStore(
        v4_root,
        comparison_kind="information-recovery-v4",
        schema_version=4,
        require_common_processed=True,
    )
    cases = []
    deterministic = True
    for root, record in _best_baselines(v2_root, v4_root):
        source_artifact = record["source"]["artifact"]
        data = (root / str(record["record_id"]) / str(source_artifact["path"])).read_bytes()
        options = ConversionOptions(**record["options"])
        try:
            first = generate_surface_texture_comparison(
                data,
                options,
                v2_store,
                v4_store,
                human_only=False,
                palette=palette,
                edge_adjustments=edge_adjustments,
                catalog_sha256=str(motif_catalog["catalog_sha256"]),
                audit_config_sha256=str(audit_report["config_sha256"]),
            )
        except Exception as error:
            raise RuntimeError(
                f"P6R-4T failed for {record['source']['filename']}"
            ) from error
        second = generate_surface_texture_comparison(
            data,
            options,
            v2_store,
            v4_store,
            human_only=False,
            palette=palette,
            edge_adjustments=edge_adjustments,
            catalog_sha256=str(motif_catalog["catalog_sha256"]),
            audit_config_sha256=str(audit_report["config_sha256"]),
        )
        deterministic &= _canonical(first) == _canonical(second)
        textured = next(
            item
            for item in first["candidates"]
            if item["variantId"] == "surface-texture-applied-v5"
        )
        surface = textured["surface"]
        texture = surface["texture"]
        cases.append(
            {
                "source_filename": first["sourceFilename"],
                "baseline_record_id": surface["source_baseline_record_id"],
                "baseline_lineage": surface["source_baseline_lineage"],
                "options": record["options"],
                "crop": record["crop"],
                "stable_surface_count": texture["stable_surface_count"],
                "raw_line_separated_pair_count": texture[
                    "raw_line_separated_pair_count"
                ],
                "eligible_pair_count": texture["eligible_pair_count"],
                "eligible_fraction": texture["eligible_fraction"],
                "actual_separated_pair_count": texture["actual_separated_pair_count"],
                "separation_fraction": texture["separation_fraction"],
                "applied_surface_count": texture["applied_surface_count"],
                "replacement_count": texture["replacement_count"],
                "eligibility": texture["eligibility"],
                "maximum_density_error": texture["maximum_density_error"],
                "exact_row_widths": surface["width_diagnostics"]["exact_row_widths"],
                "main_line_loss_at_1px": surface["line_diagnostics"]["main_line_loss_at_1px"],
                "case_mechanical_gate": surface["case_mechanical_gate"],
                "comparison_sha256": hashlib.sha256(_canonical(first)).hexdigest(),
            }
        )
    raw_pairs = sum(int(case["raw_line_separated_pair_count"]) for case in cases)
    eligible = sum(int(case["eligible_pair_count"]) for case in cases)
    separated = sum(int(case["actual_separated_pair_count"]) for case in cases)
    raw_pair_images = sum(
        int(int(case["raw_line_separated_pair_count"]) > 0) for case in cases
    )
    eligible_pair_images = sum(int(int(case["eligible_pair_count"]) > 0) for case in cases)
    eligible_fraction = eligible / raw_pairs if raw_pairs else 0.0
    separation_fraction = separated / eligible if eligible else 0.0
    maximum_error = max(float(case["maximum_density_error"]) for case in cases)
    maximum_line_loss = max(float(case["main_line_loss_at_1px"]) for case in cases)
    gate = bool(
        len(cases) == 15
        and raw_pairs >= MINIMUM_PAIR_COUNT
        and eligible >= MINIMUM_PAIR_COUNT
        and raw_pair_images >= MINIMUM_IMAGE_COUNT
        and eligible_pair_images >= MINIMUM_IMAGE_COUNT
        and eligible_fraction >= MINIMUM_ELIGIBLE_FRACTION
        and separation_fraction >= MINIMUM_SEPARATION_FRACTION
        and maximum_error <= 0.03
        and maximum_line_loss <= 0.02
        and all(bool(case["exact_row_widths"]) for case in cases)
        and all(bool(case["case_mechanical_gate"]) for case in cases)
        and deterministic
    )
    report = {
        "schema_version": 2,
        "phase": "P6R-4T2",
        "comparison_kind": COMPARISON_KIND,
        "recipe_version": RECIPE_VERSION,
        "motif_audit_method_version": audit_report["method_version"],
        "motif_catalog_sha256": motif_catalog["catalog_sha256"],
        "motif_audit_config_sha256": audit_report["config_sha256"],
        "palette": list(palette),
        "edge_adjustment_character_counts": {
            side: {
                motif: len(characters)
                for motif, characters in edge_adjustments[side].items()
            }
            for side in ("left", "right")
        },
        "case_count": len(cases),
        "raw_line_separated_pair_count": raw_pairs,
        "raw_pair_image_count": raw_pair_images,
        "eligible_pair_count": eligible,
        "eligible_pair_image_count": eligible_pair_images,
        "minimum_pair_count": MINIMUM_PAIR_COUNT,
        "minimum_image_count": MINIMUM_IMAGE_COUNT,
        "minimum_eligible_fraction": MINIMUM_ELIGIBLE_FRACTION,
        "eligible_fraction": round(eligible_fraction, 6),
        "actual_separated_pair_count": separated,
        "minimum_separation_fraction": MINIMUM_SEPARATION_FRACTION,
        "separation_fraction": round(separation_fraction, 6),
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
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    args = parser.parse_args()
    report = evaluate(
        args.v2_root,
        args.v4_root,
        args.output,
        args.audit_report,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
