from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.image_io import decode_color_image
from backend.surface_proposals import (
    SurfaceProposalConfig,
    extract_surface_proposals,
    normalize_surface_geometry,
    render_proposal_layer,
)
from training.prepare_reference_channels import _load_cases


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "surface-proposals-p6r3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose P6R-3 color and closed-region surface proposals."
    )
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": round(float(array.min()), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
        "maximum": round(float(array.max()), 6),
    }


def _save_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)


def _contact_sheet(
    records: list[dict[str, object]],
    *,
    source_kind: str,
    output: Path,
) -> None:
    selected = [record for record in records if record["surface_source_kind"] == source_kind]
    if not selected:
        return
    tile_width = 256
    tile_height = 188
    columns = 3
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        preview_path = output / str(record["preview"])
        preview = Image.open(preview_path).convert("RGB")
        preview = ImageOps.contain(preview, (tile_width - 12, tile_height - 32))
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(preview, (x + 6, y + 22))
        draw.text((x + 6, y + 5), str(record["id"]), fill="black")
    sheet.save(output / f"contact-sheet-{source_kind}.png")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {args.output}. Use --force to replace it."
            )
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    dataset, training_allowed, source_root, cases = _load_cases(
        args.samples,
        args.manifest,
    )
    config = SurfaceProposalConfig(target_width=args.width)
    records: list[dict[str, object]] = []
    layer_proposal_counts: dict[str, list[float]] = {}
    layer_area_fractions: dict[str, list[float]] = {}
    layer_coverages: dict[str, list[float]] = {}
    cross_layer_overlaps: dict[str, list[float]] = {}
    cross_layer_splits: dict[str, list[float]] = {}
    cross_layer_merges: dict[str, list[float]] = {}
    lineart_external_fractions: list[float] = []
    source_hashes: set[str] = set()

    for case in cases:
        source_path = (source_root / case["path"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing P1 reference image: {source_path}")
        payload = source_path.read_bytes()
        source_hash = _sha256(payload)
        if source_hash in source_hashes:
            raise ValueError(f"Duplicate P1 reference content: {source_path}")
        source_hashes.add(source_hash)
        image = decode_color_image(payload)
        source_kind = "lineart" if case["source_kind"] == "lineart" else "color"
        coordinate_space_id = f"p1:{case['id']}:surface-v1:w{args.width}"
        proposal_set = extract_surface_proposals(
            image,
            source_kind=source_kind,
            coordinate_space_id=coordinate_space_id,
            config=config,
        )
        case_dir = args.output / case["id"]
        case_dir.mkdir()
        normalized, _, _ = normalize_surface_geometry(image, config)
        _save_bgr(case_dir / "normalized.png", normalized)
        layer_summaries: list[dict[str, object]] = []
        for layer in proposal_set.layers:
            preview_name = f"{layer.layer_id}.png"
            _save_bgr(
                case_dir / preview_name,
                render_proposal_layer(image, proposal_set, layer),
            )
            proposal_count = len(layer.proposals)
            layer_proposal_counts.setdefault(layer.layer_id, []).append(proposal_count)
            layer_coverages.setdefault(layer.layer_id, []).append(
                float(layer.diagnostics["proposal_coverage_fraction"])
            )
            layer_area_fractions.setdefault(layer.layer_id, []).extend(
                proposal.area_fraction for proposal in layer.proposals
            )
            if source_kind == "lineart":
                lineart_external_fractions.append(
                    float(layer.diagnostics["external_background_fraction"])
                )
            layer_summaries.append(
                {
                    "layer_id": layer.layer_id,
                    "method": layer.method,
                    "proposal_count": proposal_count,
                    "preview": preview_name,
                    "diagnostics": layer.diagnostics,
                    "area_fraction": _summary(
                        [proposal.area_fraction for proposal in layer.proposals]
                    ),
                }
            )
        proposal_payload = proposal_set.to_dict(include_label_spans=True)
        relation_summaries: list[dict[str, object]] = []
        for relation in proposal_set.cross_layer_relations:
            pair_id = (
                f"{relation['source_layer_id']}->{relation['target_layer_id']}"
            )
            cross_layer_overlaps.setdefault(pair_id, []).append(
                float(relation["overlap_count"])
            )
            cross_layer_splits.setdefault(pair_id, []).append(
                float(relation["source_split_count"])
            )
            cross_layer_merges.setdefault(pair_id, []).append(
                float(relation["target_merge_count"])
            )
            relation_summaries.append(
                {
                    key: value
                    for key, value in relation.items()
                    if key != "overlaps"
                }
            )
        (case_dir / "proposals.json").write_text(
            json.dumps(proposal_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        preview_relative = Path(case["id"]) / f"{proposal_set.layers[0].layer_id}.png"
        records.append(
            {
                "id": case["id"],
                "source": source_path.relative_to(ROOT).as_posix(),
                "source_sha256": source_hash,
                "category": case["category"],
                "modality": case["modality"],
                "style": case["style"],
                "rights_status": case["rights_status"],
                "surface_source_kind": source_kind,
                "coordinate_space_id": coordinate_space_id,
                "proposal_count": proposal_set.proposal_count,
                "layers": layer_summaries,
                "cross_layer_relations": relation_summaries,
                "proposal_data": (Path(case["id"]) / "proposals.json").as_posix(),
                "preview": preview_relative.as_posix(),
            }
        )

    _contact_sheet(records, source_kind="color", output=args.output)
    _contact_sheet(records, source_kind="lineart", output=args.output)
    kind_counts = Counter(str(record["surface_source_kind"]) for record in records)
    category_counts = Counter(str(record["category"]) for record in records)
    report = {
        "schema_version": 1,
        "phase": "P6R-3",
        "dataset": dataset,
        "training_allowed": training_allowed,
        "reference_count": len(records),
        "source_kind_counts": dict(sorted(kind_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "paired_gold_metrics": None,
        "paired_gold_metrics_status": (
            "not computed: P1 references and accepted-v1 are unpaired"
        ),
        "config": proposal_set.to_dict()["config"] if records else {},
        "aggregate": {
            "records_with_any_proposal": sum(
                int(record["proposal_count"]) > 0 for record in records
            ),
            "layer_proposal_count": {
                layer_id: _summary(values)
                for layer_id, values in sorted(layer_proposal_counts.items())
            },
            "layer_proposal_area_fraction": {
                layer_id: _summary(values)
                for layer_id, values in sorted(layer_area_fractions.items())
            },
            "layer_coverage_fraction": {
                layer_id: _summary(values)
                for layer_id, values in sorted(layer_coverages.items())
            },
            "lineart_external_background_fraction": _summary(
                lineart_external_fractions
            ),
            "cross_layer_overlap_count": {
                pair_id: _summary(values)
                for pair_id, values in sorted(cross_layer_overlaps.items())
            },
            "cross_layer_split_count": {
                pair_id: _summary(values)
                for pair_id, values in sorted(cross_layer_splits.items())
            },
            "cross_layer_merge_count": {
                pair_id: _summary(values)
                for pair_id, values in sorted(cross_layer_merges.items())
            },
        },
        "records": records,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Evaluated {len(records)} unpaired P1 references; "
        f"saved diagnostics to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
