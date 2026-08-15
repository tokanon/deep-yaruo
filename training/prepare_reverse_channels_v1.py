from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training.reverse_channels import (
    EXPANDED_PERIODIC_CONFIG,
    METHOD_VERSION,
    SCHEMA_VERSION,
    _canonical_json,
    _sha256,
    decompose_text,
    decomposition_summary,
    font_sha256,
)
from training.review_corpus import load_snapshot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT
    / "datasets"
    / "incoming"
    / "yaruyomi"
    / "v32.1"
    / "accepted-v2"
    / "checkpoints"
    / "0500"
)
DEFAULT_OUTPUT = (
    ROOT
    / "datasets"
    / "incoming"
    / "yaruyomi"
    / "v32.1"
    / "accepted-v2"
    / "derived"
    / "0500"
    / "reverse-channels-v1"
)
CANDIDATES = ("a-p6r1", "b-periodic")


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


def _write_json(path: Path, value: object) -> str:
    return _write_bytes(path, _canonical_json(value))


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return _sha256(path.read_bytes())


def _candidate_arrays(prefix: str, decomposition: object) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_fill_owned": decomposition.fill_owned.astype(np.uint8),
        f"{prefix}_line_mask": decomposition.line_mask.astype(np.uint8),
        f"{prefix}_fill_glyph_mask": decomposition.fill_glyph_mask.astype(np.uint8),
        f"{prefix}_surface_ids": decomposition.surface_ids.astype(np.int32),
        f"{prefix}_surface_tone": decomposition.surface_tone.astype(np.uint8),
    }


def _motif_catalog(
    counts: Counter[str],
    work_ids: dict[str, set[int]],
) -> list[dict[str, object]]:
    return [
        {
            "motif": motif,
            "run_count": count,
            "work_count": len(work_ids[motif]),
            "supported": (
                count >= EXPANDED_PERIODIC_CONFIG.minimum_run_count
                and len(work_ids[motif]) >= EXPANDED_PERIODIC_CONFIG.minimum_work_count
            ),
        }
        for motif, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def _tone_image(tone: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(tone > 0, 255 - tone, 255).astype(np.uint8), mode="L").convert("RGB")


def _difference_image(mask: np.ndarray) -> Image.Image:
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask.astype(bool)] = (220, 0, 180)
    return Image.fromarray(image, mode="RGB")


def _preview_panel(image: Image.Image, label: str) -> Image.Image:
    active = image.copy()
    active.thumbnail((760, 620), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (780, 660), "white")
    ImageDraw.Draw(panel).text((10, 6), label, fill="black")
    panel.paste(active, (10, 30))
    return panel


def _write_preview(work_dir: Path, output_path: Path, entry_id: int) -> None:
    with np.load(work_dir / "channels.npz") as arrays:
        panels = [
            _preview_panel(_mask_image(arrays["target_mask"]), "target"),
            _preview_panel(_mask_image(arrays["a_line_mask"]), "A line"),
            _preview_panel(_tone_image(arrays["a_surface_tone"]), "A surface tone"),
            _preview_panel(_mask_image(arrays["b_line_mask"]), "B line"),
            _preview_panel(_tone_image(arrays["b_surface_tone"]), "B surface tone"),
            _preview_panel(_difference_image(arrays["ab_disagreement_mask"]), "A/B disagreement"),
        ]
    canvas = Image.new("RGB", (1560, 1980), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * 780, (index // 2) * 660))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _select_preview_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_category: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)
    selected: list[dict[str, object]] = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda record: (
                -int(record["ab_disagreement_glyph_count"]),
                int(record["entry_id"]),
            ),
        )
        selected.extend(ordered[:2])
    return selected


def prepare_reverse_channels(snapshot_root: Path, output_root: Path) -> dict[str, object]:
    snapshot_root = snapshot_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((snapshot_root / "snapshot.json").read_text(encoding="utf-8"))
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record["decision"] == "accept"
    ]
    motif_counts: dict[str, Counter[str]] = {candidate: Counter() for candidate in CANDIDATES}
    motif_work_ids: dict[str, defaultdict[str, set[int]]] = {
        candidate: defaultdict(set) for candidate in CANDIDATES
    }
    aggregate: dict[str, Counter[str]] = {candidate: Counter() for candidate in CANDIDATES}

    temporary = output_root.with_name(f"{output_root.name}.building")
    if temporary.exists():
        raise FileExistsError(f"Build directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        manifest_records: list[dict[str, object]] = []
        for record in sorted(records, key=lambda item: int(item["entry_id"])):
            entry_id = int(record["entry_id"])
            text_path = snapshot_root / str(record["text"])
            text_payload = text_path.read_bytes()
            text = text_payload.decode("utf-8")
            decompositions = {}
            lattice = None
            summaries = {}
            for candidate in CANDIDATES:
                decomposition, candidate_lattice = decompose_text(text, candidate=candidate)
                decompositions[candidate] = decomposition
                if lattice is None:
                    lattice = candidate_lattice
                else:
                    for key in lattice:
                        if not np.array_equal(lattice[key], candidate_lattice[key]):
                            raise AssertionError(f"Candidate lattices differ for entry {entry_id}: {key}")
                summaries[candidate] = decomposition_summary(
                    decomposition,
                    candidate_lattice["target_mask"],
                )
                for event in decomposition.motif_events:
                    motif_counts[candidate][event.motif] += 1
                    motif_work_ids[candidate][event.motif].add(entry_id)
                aggregate[candidate]["works_with_fill"] += bool(decomposition.fill_owned.any())
                aggregate[candidate]["fill_glyphs"] += int(decomposition.fill_owned.sum())
                aggregate[candidate]["runs"] += len(decomposition.runs)
                aggregate[candidate]["surfaces"] += len(decomposition.surfaces)
                aggregate[candidate]["multi_line_surfaces"] += sum(
                    surface.line_end - surface.line_start > 1
                    for surface in decomposition.surfaces
                )
                aggregate[candidate]["recomposition_failures"] += not summaries[candidate]["target_recomposition_exact"]
                aggregate[candidate]["surface_integrity_failures"] += len(summaries[candidate]["surface_integrity_failures"])

            assert lattice is not None
            a = decompositions["a-p6r1"]
            b = decompositions["b-periodic"]
            if np.any(a.fill_owned > b.fill_owned):
                raise AssertionError(f"Expanded candidate is not a superset for entry {entry_id}")
            disagreement_owned = np.logical_xor(a.fill_owned, b.fill_owned).astype(np.uint8)
            disagreement_mask = np.logical_xor(a.surface_ids > 0, b.surface_ids > 0).astype(np.uint8)
            work_relative = Path("works") / f"{entry_id:07d}"
            work_dir = temporary / work_relative
            text_hash = _write_bytes(work_dir / "target.txt", text_payload)
            arrays = {
                **{key: value for key, value in lattice.items()},
                **_candidate_arrays("a", a),
                **_candidate_arrays("b", b),
                "ab_disagreement_owned": disagreement_owned,
                "ab_disagreement_mask": disagreement_mask,
            }
            arrays_hash = _write_npz(work_dir / "channels.npz", arrays)
            surfaces_payload = {
                "schema_version": SCHEMA_VERSION,
                "method_version": METHOD_VERSION,
                "entry_id": entry_id,
                "candidates": {
                    "a-p6r1": {
                        "summary": summaries["a-p6r1"],
                        "runs": [run.to_dict() for run in a.runs],
                        "surfaces": [surface.to_dict() for surface in a.surfaces],
                    },
                    "b-periodic": {
                        "summary": summaries["b-periodic"],
                        "runs": [run.to_dict() for run in b.runs],
                        "surfaces": [surface.to_dict() for surface in b.surfaces],
                    },
                },
            }
            surfaces_hash = _write_json(work_dir / "surfaces.json", surfaces_payload)
            manifest_records.append(
                {
                    "entry_id": entry_id,
                    "source_file_id": int(record["source_file_id"]),
                    "split": record["split"],
                    "category": record["corrected_category"],
                    "source_text_sha256": record["unicode_text_sha256"],
                    "artifacts": {
                        "target": {
                            "path": (work_relative / "target.txt").as_posix(),
                            "sha256": text_hash,
                        },
                        "channels": {
                            "path": (work_relative / "channels.npz").as_posix(),
                            "sha256": arrays_hash,
                        },
                        "surfaces": {
                            "path": (work_relative / "surfaces.json").as_posix(),
                            "sha256": surfaces_hash,
                        },
                    },
                    "rows": int(len(lattice["row_widths"])),
                    "maximum_row_width_px": int(lattice["row_widths"].max(initial=0)),
                    "glyph_count": int(len(lattice["glyph_codepoints"])),
                    "a": summaries["a-p6r1"],
                    "b": summaries["b-periodic"],
                    "ab_disagreement_glyph_count": int(disagreement_owned.sum()),
                    "ab_disagreement_cell_pixel_count": int(disagreement_mask.sum()),
                }
            )

        preview_records = _select_preview_records(manifest_records)
        preview_paths = []
        for record in preview_records:
            entry_id = int(record["entry_id"])
            relative = Path("previews") / f"{record['category']}-{entry_id:07d}.png"
            _write_preview(temporary / "works" / f"{entry_id:07d}", temporary / relative, entry_id)
            preview_paths.append(relative.as_posix())

        catalogs = {
            candidate: _motif_catalog(motif_counts[candidate], motif_work_ids[candidate])
            for candidate in CANDIDATES
        }
        aggregate_payload = {
            candidate: dict(sorted(aggregate[candidate].items()))
            for candidate in CANDIDATES
        }
        integrity_passed = all(
            aggregate[candidate]["recomposition_failures"] == 0
            and aggregate[candidate]["surface_integrity_failures"] == 0
            for candidate in CANDIDATES
        )
        records_payload = b"".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for record in manifest_records
        )
        records_hash = _write_bytes(temporary / "records.jsonl", records_payload)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "method_version": METHOD_VERSION,
            "source_snapshot": snapshot["snapshot"],
            "source_snapshot_json_sha256": _sha256((snapshot_root / "snapshot.json").read_bytes()),
            "source_records_sha256": snapshot["records_sha256"],
            "font": {
                "name": "Saitamaar",
                "size_px": 16,
                "line_pitch_px": 18,
                "sha256": font_sha256(),
            },
            "candidate_contract": {
                "a-p6r1": "P6R-1 single-character repeated fill surfaces",
                "b-periodic": "A plus 1-4 character periodic motifs; natural advances; no glyph-width cap",
                "surface_ids": "int32 labels; zero is not a surface; ordinal values are not model intensities",
                "surface_tone": "uint8 glyph-density attribute broadcast inside each owned run rectangle",
                "target": "original Unicode AA, glyph starts, natural advances, row ends, and positioned render",
            },
            "expanded_periodic_config": asdict(EXPANDED_PERIODIC_CONFIG),
            "record_count": len(manifest_records),
            "aggregate": aggregate_payload,
            "motif_catalogs": catalogs,
            "ab_disagreement": {
                "work_count": sum(record["ab_disagreement_glyph_count"] > 0 for record in manifest_records),
                "glyph_count": sum(int(record["ab_disagreement_glyph_count"]) for record in manifest_records),
                "cell_pixel_count": sum(int(record["ab_disagreement_cell_pixel_count"]) for record in manifest_records),
            },
            "preview_paths": preview_paths,
            "records": "records.jsonl",
            "records_sha256": records_hash,
            "integrity_passed": integrity_passed,
            "training_candidate_selected": None,
            "normal_generator_changed": False,
        }
        _write_json(temporary / "manifest.json", manifest)
        if not integrity_passed:
            raise RuntimeError("Reverse-channel integrity audit failed")
        if output_root.exists():
            output_root.rmdir()
        shutil.move(str(temporary), str(output_root))
    except Exception:
        # Keep a failed build for diagnosis. A later run must never silently
        # overwrite or mix it with a complete immutable derived dataset.
        raise
    return manifest


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build accepted-v2 A/B line and indexed-surface reverse channels."
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare_reverse_channels(args.snapshot, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
