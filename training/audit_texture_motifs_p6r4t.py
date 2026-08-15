from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from backend.aa_semantics import OUTLINE_RUN_CHARACTERS
from backend.rendering import glyph_advance, line_pitch, render_text_mask
from backend.surface_fill import SURFACE_LAYER_ID, SURFACE_PROPOSAL_CONFIG
from backend.surface_fill_preferences import SurfaceFillPreferenceStore
from backend.surface_proposals import extract_surface_proposals
from backend.image_io import decode_color_image
from training.aa_fill_layers import detect_fill_runs
from training.review_corpus import load_snapshot_records


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "accepted-v1"
)
DEFAULT_V2_ROOT = ROOT / "datasets" / "incoming" / "draft-preferences" / "v2"
DEFAULT_V4_ROOT = ROOT / "datasets" / "incoming" / "draft-preferences" / "v4"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "texture-motif-audit-p6r4t" / "report.json"
SCHEMA_VERSION = 1
METHOD_VERSION = "periodic-fill-motif-audit-v2"


@dataclass(frozen=True)
class MotifAuditConfig:
    maximum_period_characters: int = 4
    minimum_run_characters: int = 6
    long_run_characters: int = 12
    minimum_vertical_overlap_fraction: float = 0.25
    minimum_run_count: int = 20
    minimum_work_count: int = 5
    density_tolerance: float = 0.03
    maximum_styles: int = 4
    minimum_evaluation_images: int = 3
    minimum_eligible_pairs: int = 5
    minimum_separation_fraction: float = 0.80
    maximum_individual_glyph_advance_px: int = 8


@dataclass(frozen=True)
class PeriodicRun:
    line_index: int
    start_index: int
    end_index: int
    motif: str
    observed_phase: str
    x_start: int
    x_end: int
    vertically_supported: bool = False

    @property
    def character_count(self) -> int:
        return self.end_index - self.start_index


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_rotation(pattern: str) -> str:
    rotations = [pattern[index:] + pattern[:index] for index in range(len(pattern))]
    return min(rotations)


def _line_periodic_runs(
    line: str,
    *,
    line_index: int,
    config: MotifAuditConfig,
    font_size: int = 16,
) -> list[PeriodicRun]:
    advances = [glyph_advance(character, font_size) for character in line]
    positions = [0]
    for advance in advances:
        positions.append(positions[-1] + advance)
    runs: list[PeriodicRun] = []
    cursor = 0
    while cursor < len(line):
        candidates: list[tuple[int, int, str]] = []
        for period in range(1, config.maximum_period_characters + 1):
            pattern = line[cursor : cursor + period]
            if len(pattern) != period or any(character.isspace() for character in pattern):
                continue
            if any(
                glyph_advance(character, font_size)
                > config.maximum_individual_glyph_advance_px
                for character in pattern
            ):
                continue
            if all(character in OUTLINE_RUN_CHARACTERS for character in pattern):
                continue
            end = cursor
            while end < len(line) and line[end] == pattern[(end - cursor) % period]:
                end += 1
            if end - cursor >= max(config.minimum_run_characters, period * 2):
                candidates.append((end, period, pattern))
        if not candidates:
            cursor += 1
            continue
        end, period, pattern = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
        minimal_pattern = pattern
        for smaller in range(1, period):
            if period % smaller == 0 and pattern == pattern[:smaller] * (period // smaller):
                minimal_pattern = pattern[:smaller]
                break
        runs.append(
            PeriodicRun(
                line_index=line_index,
                start_index=cursor,
                end_index=end,
                motif=_canonical_rotation(minimal_pattern),
                observed_phase=minimal_pattern,
                x_start=positions[cursor],
                x_end=positions[end],
            )
        )
        cursor = end
    return runs


def extract_periodic_runs(
    text: str,
    *,
    config: MotifAuditConfig | None = None,
    font_size: int = 16,
) -> list[PeriodicRun]:
    active = config or MotifAuditConfig()
    by_line = {
        line_index: _line_periodic_runs(
            line,
            line_index=line_index,
            config=active,
            font_size=font_size,
        )
        for line_index, line in enumerate(text.splitlines() or [""])
    }
    result: list[PeriodicRun] = []
    for line_index in sorted(by_line):
        for run in by_line[line_index]:
            vertical = False
            for adjacent_line in (line_index - 1, line_index + 1):
                for other in by_line.get(adjacent_line, []):
                    overlap = max(0, min(run.x_end, other.x_end) - max(run.x_start, other.x_start))
                    shorter = min(run.x_end - run.x_start, other.x_end - other.x_start)
                    if shorter > 0 and overlap >= shorter * active.minimum_vertical_overlap_fraction:
                        vertical = True
                        break
                if vertical:
                    break
            if run.character_count >= active.long_run_characters or vertical:
                result.append(
                    PeriodicRun(
                        **{
                            **asdict(run),
                            "vertically_supported": vertical,
                        }
                    )
                )
    return result


def _motif_density(motif: str, *, font_size: int = 16) -> tuple[int, float]:
    width = sum(glyph_advance(character, font_size) for character in motif)
    mask = render_text_mask(
        motif,
        font_size,
        canvas_height_per_line=line_pitch(font_size),
    )
    if mask.shape[1] != width:
        raise AssertionError("Motif rendering did not preserve measured Saitamaar advance")
    return width, float(mask.mean())


def audit_motif_catalog(
    snapshot_root: Path,
    *,
    config: MotifAuditConfig | None = None,
    font_size: int = 16,
) -> dict[str, object]:
    active = config or MotifAuditConfig()
    records = [
        record
        for record in load_snapshot_records(snapshot_root)
        if record["decision"] == "accept"
    ]
    run_counts: Counter[str] = Counter()
    work_ids: defaultdict[str, set[int]] = defaultdict(set)
    phase_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    edge_counts: dict[str, Counter[str]] = {
        "left": Counter(),
        "right": Counter(),
    }
    edge_work_ids: dict[str, defaultdict[str, set[int]]] = {
        "left": defaultdict(set),
        "right": defaultdict(set),
    }
    for record in records:
        path = snapshot_root / str(record["text"])
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines() or [""]
        for run in extract_periodic_runs(text, config=active, font_size=font_size):
            run_counts[run.motif] += 1
            work_ids[run.motif].add(int(record["entry_id"]))
            phase_counts[run.motif][run.observed_phase] += 1
            line = lines[run.line_index]
            for side, index in (
                ("left", run.start_index - 1),
                ("right", run.end_index),
            ):
                if not 0 <= index < len(line):
                    continue
                character = line[index]
                if (
                    character in OUTLINE_RUN_CHARACTERS
                    or glyph_advance(character, font_size)
                    > active.maximum_individual_glyph_advance_px
                ):
                    continue
                key = f"{run.motif}\0{character}"
                edge_counts[side][key] += 1
                edge_work_ids[side][key].add(int(record["entry_id"]))
    candidates = []
    for motif in sorted(run_counts):
        width, density = _motif_density(motif, font_size=font_size)
        candidates.append(
            {
                "motif": motif,
                "period_characters": len(motif),
                "natural_advance_px": width,
                "ink_density": round(density, 8),
                "run_count": run_counts[motif],
                "work_count": len(work_ids[motif]),
                "observed_phases": dict(phase_counts[motif].most_common()),
                "supported": bool(
                    run_counts[motif] >= active.minimum_run_count
                    and len(work_ids[motif]) >= active.minimum_work_count
                ),
            }
        )
    supported = [item for item in candidates if item["supported"]]
    supported_names = {str(item["motif"]) for item in supported}
    edge_adjustments = {}
    for side in ("left", "right"):
        entries = []
        for key, count in edge_counts[side].items():
            motif, character = key.split("\0", 1)
            if motif not in supported_names:
                continue
            entries.append(
                {
                    "motif": motif,
                    "character": character,
                    "natural_advance_px": glyph_advance(character, font_size),
                    "ink_density": round(_motif_density(character, font_size=font_size)[1], 8),
                    "edge_count": count,
                    "work_count": len(edge_work_ids[side][key]),
                }
            )
        edge_adjustments[side] = sorted(
            entries,
            key=lambda item: (
                -int(item["edge_count"]),
                -int(item["work_count"]),
                str(item["motif"]),
                str(item["character"]),
            ),
        )
    catalog_payload = {
        "method_version": METHOD_VERSION,
        "config": asdict(active),
        "font_size": font_size,
        "motifs": supported,
        "edge_adjustments": edge_adjustments,
    }
    return {
        "accepted_entry_count": len(records),
        "candidate_motif_count": len(candidates),
        "supported_motif_count": len(supported),
        "catalog_sha256": _sha256(_canonical_json(catalog_payload)),
        "motifs": candidates,
        "edge_adjustments": edge_adjustments,
    }


def _decode_png(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode a stored PNG artifact")
    return image


def _selected_candidate(record: dict[str, object]) -> dict[str, object]:
    selected = record.get("selected_variant")
    if not selected or record.get("none_usable"):
        raise ValueError("Evaluation baseline must have one selected candidate")
    matches = [
        candidate
        for candidate in record["candidates"]
        if candidate["variant_id"] == selected
    ]
    if len(matches) != 1:
        raise ValueError("Selected candidate could not be resolved uniquely")
    return matches[0]


def _artifact_bytes(
    root: Path,
    record: dict[str, object],
    artifact: dict[str, object],
) -> bytes:
    return (root / str(record["record_id"]) / str(artifact["path"])).read_bytes()


def _best_baselines(v2_root: Path, v4_root: Path) -> list[tuple[Path, dict[str, object]]]:
    v2_store = SurfaceFillPreferenceStore(v2_root)
    v4_store = SurfaceFillPreferenceStore(
        v4_root,
        comparison_kind="information-recovery-v4",
        schema_version=4,
        require_common_processed=False,
    )
    v2_records = [
        record
        for record in v2_store.list()
        if record.get("recipe_version") == "surface-fill-v2"
    ]
    v4_records = v4_store.list()
    v4_by_source = {
        str(record["source"]["artifact"]["sha256"]): record
        for record in v4_records
    }
    if len(v4_by_source) != len(v4_records):
        raise ValueError("v4 contains duplicate source images")
    selected: list[tuple[Path, dict[str, object]]] = []
    for record in sorted(v2_records, key=lambda item: str(item["source"]["filename"])):
        if not record["none_usable"]:
            selected.append((v2_root, record))
            continue
        source_hash = str(record["source"]["artifact"]["sha256"])
        replacement = v4_by_source.get(source_hash)
        if replacement is None:
            raise ValueError(
                f"No selected v4 replacement for unusable v2 source: {record['source']['filename']}"
            )
        selected.append((v4_root, replacement))
    if len(selected) != 15:
        raise ValueError(f"P6R-4T stress set requires 15 baselines, found {len(selected)}")
    return selected


def _owner_graph(active: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    if active.shape != labels.shape:
        raise ValueError("Fill mask and proposal labels must share one coordinate space")
    owner = np.where(active, labels, 0).astype(np.int32)
    edges: set[tuple[int, int]] = set()
    for first, second in (
        (owner[:, :-1], owner[:, 1:]),
        (owner[:-1, :], owner[1:, :]),
    ):
        boundary = (first > 0) & (second > 0) & (first != second)
        if not np.any(boundary):
            continue
        for left, right in np.unique(
            np.stack((first[boundary], second[boundary]), axis=1), axis=0
        ):
            edges.add(tuple(sorted((int(left), int(right)))))
    return owner, edges


def _surface_target_densities(
    text: str,
    owner: np.ndarray,
    *,
    font_size: int = 16,
) -> dict[int, float]:
    weighted: defaultdict[int, float] = defaultdict(float)
    weights: Counter[int] = Counter()
    pitch = line_pitch(font_size)
    density_cache: dict[str, float] = {}
    for run in detect_fill_runs(text, font_size=font_size):
        y0 = run.line_index * pitch
        y1 = min(owner.shape[0], y0 + pitch)
        x0 = max(0, int(round(run.x_start)))
        x1 = min(owner.shape[1], int(round(run.x_end)))
        if y0 >= y1 or x0 >= x1:
            continue
        values, counts = np.unique(owner[y0:y1, x0:x1], return_counts=True)
        density = density_cache.setdefault(
            run.character,
            _motif_density(run.character, font_size=font_size)[1],
        )
        for label, count in zip(values.tolist(), counts.tolist(), strict=True):
            if int(label) <= 0:
                continue
            weighted[int(label)] += density * int(count)
            weights[int(label)] += int(count)
    return {
        label: weighted[label] / weights[label]
        for label in sorted(weights)
        if weights[label] > 0
    }


def _assign_palette(
    domains: dict[int, tuple[str, ...]],
    errors: dict[tuple[int, str], float],
    edges: set[tuple[int, int]],
    palette: tuple[str, ...],
) -> tuple[dict[int, str], int, float]:
    degrees = Counter(vertex for edge in edges for vertex in edge)
    neighbor_ids: defaultdict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        neighbor_ids[left].add(right)
        neighbor_ids[right].add(left)
    assignment: dict[int, str] = {}
    for vertex in sorted(domains, key=lambda item: (-degrees[item], item)):
        available = [motif for motif in palette if motif in domains[vertex]]
        if not available:
            continue
        assigned_neighbors = [
            assignment[other] for other in neighbor_ids[vertex] if other in assignment
        ]
        assignment[vertex] = min(
            available,
            key=lambda motif: (
                assigned_neighbors.count(motif),
                errors[(vertex, motif)],
                motif,
            ),
        )
    for _ in range(2):
        for vertex in sorted(assignment):
            available = [motif for motif in palette if motif in domains[vertex]]
            neighbors = [
                assignment[other]
                for other in neighbor_ids[vertex]
                if other in assignment
            ]
            assignment[vertex] = min(
                available,
                key=lambda motif: (neighbors.count(motif), errors[(vertex, motif)], motif),
            )
    separated = sum(
        left in assignment
        and right in assignment
        and assignment[left] != assignment[right]
        for left, right in edges
    )
    mean_error = (
        sum(errors[(vertex, motif)] for vertex, motif in assignment.items())
        / max(1, len(assignment))
    )
    return assignment, separated, mean_error


def _best_palette(
    cases: list[dict[str, object]],
    motif_catalog: list[dict[str, object]],
    *,
    config: MotifAuditConfig,
) -> tuple[tuple[str, ...], dict[str, tuple[dict[int, str], int, float]]]:
    supported = [item for item in motif_catalog if item["supported"]]
    motif_density = {str(item["motif"]): float(item["ink_density"]) for item in supported}
    coverage = Counter()
    for case in cases:
        for domain in case["domains"].values():
            coverage.update(domain)
    pool = sorted(
        motif_density,
        key=lambda motif: (-coverage[motif], -next(
            int(item["run_count"]) for item in supported if item["motif"] == motif
        ), motif),
    )[:16]
    best: tuple[tuple[object, ...], tuple[str, ...], dict[str, tuple[dict[int, str], int, float]]] | None = None
    for size in range(1, min(config.maximum_styles, len(pool)) + 1):
        for palette in itertools.combinations(pool, size):
            outcomes: dict[str, tuple[dict[int, str], int, float]] = {}
            total_separated = 0
            total_assigned = 0
            total_error = 0.0
            for case in cases:
                assignment, separated, mean_error = _assign_palette(
                    case["domains"], case["errors"], case["edges"], palette
                )
                outcomes[str(case["filename"])] = (assignment, separated, mean_error)
                total_separated += separated
                total_assigned += len(assignment)
                total_error += mean_error * len(assignment)
            score = (
                -total_separated,
                -total_assigned,
                total_error / max(1, total_assigned),
                len(palette),
                palette,
            )
            if best is None or score < best[0]:
                best = (score, palette, outcomes)
    if best is None:
        return (), {}
    return best[1], best[2]


def audit_stress_set(
    v2_root: Path,
    v4_root: Path,
    motif_catalog: list[dict[str, object]],
    *,
    config: MotifAuditConfig | None = None,
    font_size: int = 16,
) -> dict[str, object]:
    active_config = config or MotifAuditConfig()
    supported = [item for item in motif_catalog if item["supported"]]
    motif_density = {str(item["motif"]): float(item["ink_density"]) for item in supported}
    raw_cases: list[dict[str, object]] = []
    lineage_counts = Counter()
    manifest_hashes = []
    for root, record in _best_baselines(v2_root, v4_root):
        candidate = _selected_candidate(record)
        artifacts = candidate["artifacts"]
        source_payload = _artifact_bytes(root, record, record["source"]["artifact"])
        text = _artifact_bytes(root, record, artifacts["text"]).decode("utf-8")
        fill_mask = _decode_png(_artifact_bytes(root, record, artifacts["fill_mask"])) > 127
        source = decode_color_image(source_payload)
        proposal_set = extract_surface_proposals(
            source,
            source_kind="color",
            coordinate_space_id=f"p6r4t-audit:{record['record_id']}",
            config=SURFACE_PROPOSAL_CONFIG,
            crop_box_xyxy=tuple(int(value) for value in record["crop"]),
            target_shape_hw=fill_mask.shape,
        )
        layer = next(item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID)
        owner, all_edges = _owner_graph(fill_mask, layer.labels)
        targets = _surface_target_densities(text, owner, font_size=font_size)
        vertices = set(targets)
        edges = {edge for edge in all_edges if edge[0] in vertices and edge[1] in vertices}
        domains = {
            vertex: tuple(
                sorted(
                    motif
                    for motif, density in motif_density.items()
                    if abs(density - targets[vertex]) <= active_config.density_tolerance + 1e-12
                )
            )
            for vertex in sorted(vertices)
        }
        errors = {
            (vertex, motif): abs(motif_density[motif] - targets[vertex])
            for vertex, domain in domains.items()
            for motif in domain
        }
        eligible_edges = {
            edge
            for edge in edges
            if any(
                left != right
                for left in domains[edge[0]]
                for right in domains[edge[1]]
            )
        }
        lineage = "v4" if root == v4_root else "v2"
        lineage_counts[lineage] += 1
        manifest_hashes.append(str(record["manifest_sha256"]))
        raw_cases.append(
            {
                "filename": str(record["source"]["filename"]),
                "record_id": str(record["record_id"]),
                "lineage": lineage,
                "selected_variant": str(record["selected_variant"]),
                "surface_count": len(vertices),
                "adjacent_pair_count": len(edges),
                "eligible_pair_count": len(eligible_edges),
                "domains": domains,
                "errors": errors,
                "edges": eligible_edges,
            }
        )
    palette, outcomes = _best_palette(raw_cases, motif_catalog, config=active_config)
    cases = []
    eligible_total = 0
    separated_total = 0
    assigned_errors = []
    images_with_eligible = 0
    for case in raw_cases:
        assignment, separated, mean_error = outcomes.get(
            str(case["filename"]), ({}, 0, 0.0)
        )
        eligible = int(case["eligible_pair_count"])
        eligible_total += eligible
        separated_total += separated
        if eligible:
            images_with_eligible += 1
        if assignment:
            assigned_errors.extend(
                float(case["errors"][(vertex, motif)])
                for vertex, motif in assignment.items()
            )
        cases.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"domains", "errors", "edges"}
            }
            | {
                "assigned_surface_count": len(assignment),
                "separated_pair_count": separated,
                "separation_fraction": round(separated / max(1, eligible), 6),
                "mean_density_error": round(mean_error, 8),
                "style_assignments": {
                    str(vertex): motif for vertex, motif in sorted(assignment.items())
                },
            }
        )
    separation = separated_total / max(1, eligible_total)
    maximum_error = max(assigned_errors, default=0.0)
    gate = bool(
        images_with_eligible >= active_config.minimum_evaluation_images
        and eligible_total >= active_config.minimum_eligible_pairs
        and separation >= active_config.minimum_separation_fraction
        and maximum_error <= active_config.density_tolerance + 1e-12
    )
    return {
        "case_count": len(cases),
        "baseline_lineage_counts": dict(sorted(lineage_counts.items())),
        "baseline_manifest_set_sha256": _sha256(
            _canonical_json(sorted(manifest_hashes))
        ),
        "selected_palette": list(palette),
        "images_with_eligible_pairs": images_with_eligible,
        "eligible_pair_count": eligible_total,
        "separated_pair_count": separated_total,
        "separation_fraction": round(separation, 6),
        "maximum_density_error": round(maximum_error, 8),
        "gate_passed": gate,
        "cases": cases,
    }


def run_audit(
    snapshot_root: Path = DEFAULT_SNAPSHOT,
    v2_root: Path = DEFAULT_V2_ROOT,
    v4_root: Path = DEFAULT_V4_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    config: MotifAuditConfig | None = None,
) -> dict[str, object]:
    active = config or MotifAuditConfig()
    catalog = audit_motif_catalog(snapshot_root, config=active)
    stress = audit_stress_set(v2_root, v4_root, catalog["motifs"], config=active)
    snapshot_payload = (snapshot_root / "snapshot.json").read_bytes()
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "P6R-4T1-catalog",
        "method_version": METHOD_VERSION,
        "config": asdict(active),
        "config_sha256": _sha256(_canonical_json(asdict(active))),
        "accepted_snapshot_sha256": _sha256(snapshot_payload),
        "motif_catalog": catalog,
        "stress_set": stress,
        "gate_passed": bool(stress["gate_passed"]),
        "next_action": (
            "evaluate exact-width edge-adjusted placement on all 15 stress cases"
            if stress["gate_passed"]
            else "stop P6R-4T; do not invent unsupported texture patterns"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_audit(args.snapshot, args.v2_root, args.v4_root, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
