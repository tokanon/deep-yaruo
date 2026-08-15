from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from training.aa_fill_layers import detect_fill_runs

from .contracts import ConversionOptions
from .image_io import decode_color_image, image_to_data_url
from .rendering import glyph_advance, line_pitch, render_text_mask
from .surface_fill import (
    GENERATOR_VERSION,
    SURFACE_LAYER_ID,
    SURFACE_PROPOSAL_CONFIG,
    _line_diagnostics,
    _mask_data_url,
    _render_text,
)
from .surface_fill_preferences import SurfaceFillPreferenceStore
from .surface_proposals import extract_surface_proposals
from .surface_recovery import _width_diagnostics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXTURE_CATALOG = ROOT / "backend" / "surface_texture_catalog.json"
COMPARISON_KIND = "surface-texture-v5"
RECIPE_VERSION = "surface-texture-line-separated-v5"
METHOD_VERSION = 6
PROXIMITY_COMPARISON_KIND = "surface-texture-v6"
PROXIMITY_RECIPE_VERSION = "surface-texture-proximity-v6"
PROXIMITY_METHOD_VERSION = 7
ACCEPTED_SNAPSHOT_SHA256 = "b7142af200d680e107433a2f2c00f50fb5e4167775b474284bdfb6b4f1ba8f3b"
DENSITY_TOLERANCE = 0.03
MINIMUM_MOTIF_REPETITIONS = 2
MINIMUM_PAIR_COUNT = 5
MINIMUM_IMAGE_COUNT = 3
MINIMUM_ELIGIBLE_FRACTION = 0.80
MINIMUM_SEPARATION_FRACTION = 0.80
SAITAMAAR_HALF_WIDTH_PX = 8.0
HUMAN_EVALUATION_FILES = frozenset(
    {
        "background-02-source.png",
        "background-04-source.png",
        "background-06-source.png",
        "person-04-source.png",
        "person-06-source.png",
    }
)


def load_texture_catalog(
    path: Path = DEFAULT_TEXTURE_CATALOG,
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, tuple[str, ...]]],
    str,
    str,
]:
    if not path.is_file():
        raise FileNotFoundError("The frozen P6R-4T texture catalog is missing.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("accepted_snapshot_sha256") != ACCEPTED_SNAPSHOT_SHA256
    ):
        raise ValueError("The frozen P6R-4T texture catalog is incompatible")
    palette = tuple(str(value) for value in report["palette"])
    raw_adjustments = report["edge_adjustments"]
    adjustments = {
        side: {
            motif: tuple(str(character) for character in raw_adjustments[side][motif])
            for motif in palette
        }
        for side in ("left", "right")
    }
    return (
        palette,
        adjustments,
        str(report["catalog_sha256"]),
        str(report["audit_config_sha256"]),
    )


@dataclass(frozen=True)
class TextureBaseline:
    root: Path
    record: dict[str, object]
    candidate: dict[str, object]
    text: str
    processed: np.ndarray
    rendered: np.ndarray
    fill_mask: np.ndarray


@dataclass(frozen=True)
class TextureCell:
    line_index: int
    character_index: int
    character: str
    x_start: int
    x_end: int
    role: str
    owner: int
    fill_run_id: int | None

    @property
    def key(self) -> tuple[int, int]:
        return self.line_index, self.character_index


def _decode_png(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode a stored texture-baseline artifact")
    return image


def _artifact_bytes(
    root: Path,
    record: dict[str, object],
    artifact: dict[str, object],
) -> bytes:
    return (root / str(record["record_id"]) / str(artifact["path"])).read_bytes()


def _selected_candidate(record: dict[str, object]) -> dict[str, object]:
    selected = record.get("selected_variant")
    if not selected or record.get("none_usable"):
        raise ValueError("Texture baseline does not contain one selected candidate")
    matches = [
        item for item in record["candidates"] if item["variant_id"] == selected
    ]
    if len(matches) != 1:
        raise ValueError("Texture baseline selected candidate is not unique")
    return matches[0]


def resolve_texture_baseline_options(
    data: bytes,
    v2_store: SurfaceFillPreferenceStore,
    v4_store: SurfaceFillPreferenceStore,
) -> ConversionOptions:
    source_hash = hashlib.sha256(data).hexdigest()
    v2_matches = [
        record
        for record in v2_store.list()
        if record.get("recipe_version") == "surface-fill-v2"
        and record.get("source", {}).get("artifact", {}).get("sha256")
        == source_hash
    ]
    if len(v2_matches) != 1:
        raise ValueError(
            "Surface texture requires exactly one surface-fill-v2 record for the "
            f"source image; found {len(v2_matches)}."
        )
    record = v2_matches[0]
    if record.get("none_usable"):
        v4_matches = [
            candidate
            for candidate in v4_store.list()
            if candidate.get("recipe_version") == "information-recovery-v4"
            and candidate.get("source", {}).get("artifact", {}).get("sha256")
            == source_hash
            and candidate.get("selected_variant")
            and not candidate.get("none_usable")
        ]
        if len(v4_matches) != 1:
            raise ValueError(
                "Surface texture requires exactly one selected v4 replacement for "
                f"this unusable v2 record; found {len(v4_matches)}."
            )
        _selected_candidate(v4_matches[0])
        record = v4_matches[0]
    return ConversionOptions(**record["options"]).normalized()


def _matching_records(
    data: bytes,
    options: ConversionOptions,
    store: SurfaceFillPreferenceStore,
    *,
    recipe_version: str,
) -> list[dict[str, object]]:
    source_hash = hashlib.sha256(data).hexdigest()
    normalized = options.normalized()
    return [
        record
        for record in store.list()
        if record.get("recipe_version") == recipe_version
        and record.get("source", {}).get("artifact", {}).get("sha256") == source_hash
        and ConversionOptions(**record["options"]).normalized() == normalized
    ]


def load_texture_baseline(
    data: bytes,
    options: ConversionOptions,
    v2_store: SurfaceFillPreferenceStore,
    v4_store: SurfaceFillPreferenceStore,
) -> TextureBaseline:
    source_hash = hashlib.sha256(data).hexdigest()
    v2_source_matches = [
        record
        for record in v2_store.list()
        if record.get("recipe_version") == "surface-fill-v2"
        and record.get("source", {}).get("artifact", {}).get("sha256") == source_hash
    ]
    if len(v2_source_matches) != 1:
        raise ValueError(
            "Surface texture requires exactly one surface-fill-v2 record for the "
            f"source image; found {len(v2_source_matches)}."
        )
    record = v2_source_matches[0]
    root = v2_store.root
    if record.get("none_usable"):
        v4_matches = _matching_records(
            data,
            options,
            v4_store,
            recipe_version="information-recovery-v4",
        )
        if len(v4_matches) != 1:
            raise ValueError(
                "Surface texture requires exactly one selected v4 replacement for "
                "this unusable v2 record."
            )
        record = v4_matches[0]
        root = v4_store.root
    elif ConversionOptions(**record["options"]).normalized() != options.normalized():
        raise ValueError(
            "The source image matches v2, but the current settings differ from its "
            "selected texture baseline."
        )
    candidate = _selected_candidate(record)
    artifacts = candidate["artifacts"]
    processed_artifact = artifacts.get("processed", record.get("processed"))
    if not isinstance(processed_artifact, dict):
        raise ValueError("Texture baseline has no processed-image artifact")
    return TextureBaseline(
        root=root,
        record=record,
        candidate=candidate,
        text=_artifact_bytes(root, record, artifacts["text"]).decode("utf-8"),
        processed=_decode_png(_artifact_bytes(root, record, processed_artifact)),
        rendered=_decode_png(_artifact_bytes(root, record, artifacts["rendered"])),
        fill_mask=_decode_png(_artifact_bytes(root, record, artifacts["fill_mask"])),
    )


def _selected_owner_labels(active: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if active.shape != labels.shape:
        raise ValueError("Fill mask and stable proposal labels must share one shape")
    return np.where(active, labels, 0).astype(np.int32)


def _texture_cells(
    text: str,
    selected_labels: np.ndarray,
    *,
    font_size: int,
) -> list[list[TextureCell]]:
    lines = text.rstrip("\n").split("\n") if text else [""]
    run_ids: dict[tuple[int, int], int] = {}
    for run_id, run in enumerate(detect_fill_runs(text, font_size=font_size)):
        for character_index in range(run.start_index, run.end_index):
            run_ids[(run.line_index, character_index)] = run_id
    pitch = line_pitch(font_size)
    rows: list[list[TextureCell]] = []
    for line_index, line in enumerate(lines):
        positions = [0]
        for character in line:
            positions.append(positions[-1] + glyph_advance(character, font_size))
        row: list[TextureCell] = []
        y_start = min(selected_labels.shape[0], line_index * pitch)
        y_end = min(selected_labels.shape[0], y_start + pitch)
        for character_index, character in enumerate(line):
            key = (line_index, character_index)
            fill_run_id = run_ids.get(key)
            if fill_run_id is not None:
                role = "fill"
            elif character.isspace():
                role = "blank"
            else:
                role = "separator"
            x_start = positions[character_index]
            x_end = positions[character_index + 1]
            owner = 0
            if role == "fill" and y_start < y_end and x_start < x_end:
                clipped_start = min(selected_labels.shape[1], x_start)
                clipped_end = min(selected_labels.shape[1], x_end)
                patch = selected_labels[y_start:y_end, clipped_start:clipped_end]
                values, counts = np.unique(patch[patch > 0], return_counts=True)
                if values.size:
                    owner = min(
                        zip(values.tolist(), counts.tolist(), strict=True),
                        key=lambda item: (-int(item[1]), int(item[0])),
                    )[0]
            row.append(
                TextureCell(
                    line_index=line_index,
                    character_index=character_index,
                    character=character,
                    x_start=x_start,
                    x_end=x_end,
                    role=role,
                    owner=int(owner),
                    fill_run_id=fill_run_id,
                )
            )
        rows.append(row)
    return rows


def _line_separated_graph(
    rows: list[list[TextureCell]],
) -> dict[tuple[int, int], set[tuple[tuple[int, int], tuple[int, int]]]]:
    evidence: defaultdict[
        tuple[int, int], set[tuple[tuple[int, int], tuple[int, int]]]
    ] = defaultdict(set)

    def add(left: TextureCell, right: TextureCell) -> None:
        if left.owner <= 0 or right.owner <= 0 or left.owner == right.owner:
            return
        edge = tuple(sorted((left.owner, right.owner)))
        ordered_cells = (
            (left.key, right.key)
            if left.owner == edge[0]
            else (right.key, left.key)
        )
        evidence[edge].add(ordered_cells)

    for row in rows:
        cursor = 0
        while cursor < len(row):
            if row[cursor].role != "separator":
                cursor += 1
                continue
            end = cursor + 1
            while end < len(row) and row[end].role == "separator":
                end += 1
            if cursor > 0 and end < len(row):
                left = row[cursor - 1]
                right = row[end]
                if left.role == right.role == "fill":
                    add(left, right)
            cursor = end

    for line_index in range(1, len(rows) - 1):
        above = [cell for cell in rows[line_index - 1] if cell.role == "fill" and cell.owner > 0]
        below = [cell for cell in rows[line_index + 1] if cell.role == "fill" and cell.owner > 0]
        for separator in rows[line_index]:
            if separator.role != "separator" or separator.x_start >= separator.x_end:
                continue
            matching_above = [
                cell
                for cell in above
                if min(cell.x_end, separator.x_end) > max(cell.x_start, separator.x_start)
            ]
            matching_below = [
                cell
                for cell in below
                if min(cell.x_end, separator.x_end) > max(cell.x_start, separator.x_start)
            ]
            for upper, lower in itertools.product(matching_above, matching_below):
                add(upper, lower)
    return {edge: cells for edge, cells in sorted(evidence.items())}


@lru_cache(maxsize=16384)
def _ink_density(text: str, font_size: int) -> float:
    return float(
        render_text_mask(
            text,
            font_size,
            canvas_height_per_line=line_pitch(font_size),
        ).mean()
    )


def _surface_target_densities(
    rows: list[list[TextureCell]],
    *,
    font_size: int,
) -> dict[int, float]:
    weighted: defaultdict[int, float] = defaultdict(float)
    weights: Counter[int] = Counter()
    for row in rows:
        for cell in row:
            if cell.role != "fill" or cell.owner <= 0 or cell.x_end <= cell.x_start:
                continue
            width = cell.x_end - cell.x_start
            weighted[cell.owner] += _ink_density(cell.character, font_size) * width
            weights[cell.owner] += width
    return {label: weighted[label] / weights[label] for label in sorted(weights)}


@lru_cache(maxsize=16384)
def _replacement_for_motif(
    width: int,
    motif: str,
    original: str,
    *,
    font_size: int,
    left_adjustments: tuple[str, ...] = (),
    right_adjustments: tuple[str, ...] = (),
) -> tuple[str, float, int, int] | None:
    motif_width = sum(glyph_advance(character, font_size) for character in motif)
    original_density = _ink_density(original, font_size)
    left_candidates = tuple(
        (character, sum(glyph_advance(item, font_size) for item in character))
        for character in ("", *left_adjustments)
    )
    right_candidates = tuple(
        (character, sum(glyph_advance(item, font_size) for item in character))
        for character in ("", *right_adjustments)
    )
    for repetitions in range(
        width // motif_width,
        MINIMUM_MOTIF_REPETITIONS - 1,
        -1,
    ):
        remainder = width - repetitions * motif_width
        candidates: list[tuple[float, int, str, int]] = []
        for (left, left_width), (right, right_width) in itertools.product(
            left_candidates,
            right_candidates,
        ):
            if left_width + right_width != remainder:
                continue
            replacement = left + motif * repetitions + right
            error = abs(_ink_density(replacement, font_size) - original_density)
            if error <= DENSITY_TOLERANCE + 1e-12:
                adjustment_count = int(bool(left)) + int(bool(right))
                candidates.append((error, adjustment_count, replacement, adjustment_count))
        if candidates:
            error, _, replacement, adjustment_count = min(candidates)
            return replacement, error, repetitions, adjustment_count
    return None


def _eligible_intervals(
    text: str,
    rows: list[list[TextureCell]],
    *,
    font_size: int,
    palette: tuple[str, ...],
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lines = text.rstrip("\n").split("\n") if text else [""]
    result: list[dict[str, object]] = []
    fill_surfaces: set[int] = set()
    grouped_surfaces: set[int] = set()
    owned_fill_cell_count = 0
    fill_cell_count = 0
    for line_index, row in enumerate(rows):
        if line_index >= len(lines):
            continue
        line = lines[line_index]
        cursor = 0
        while cursor < len(row):
            cell = row[cursor]
            if cell.role != "fill":
                cursor += 1
                continue
            fill_cell_count += 1
            if cell.owner > 0:
                owned_fill_cell_count += 1
                fill_surfaces.add(cell.owner)
            label = cell.owner
            fill_run_id = cell.fill_run_id
            end = cursor + 1
            while (
                end < len(row)
                and row[end].role == "fill"
                and row[end].owner == label
                and row[end].fill_run_id == fill_run_id
            ):
                fill_cell_count += 1
                if row[end].owner > 0:
                    owned_fill_cell_count += 1
                    fill_surfaces.add(row[end].owner)
                end += 1
            if label > 0:
                grouped_surfaces.add(label)
                start_index = row[cursor].character_index
                end_index = row[end - 1].character_index + 1
                original = line[start_index:end_index]
                width = row[end - 1].x_end - row[cursor].x_start
                replacements = {}
                for motif in palette:
                    proposal = _replacement_for_motif(
                        width,
                        motif,
                        original,
                        font_size=font_size,
                        left_adjustments=edge_adjustments["left"].get(motif, ()),
                        right_adjustments=edge_adjustments["right"].get(motif, ()),
                    )
                    if proposal is not None:
                        replacements[motif] = proposal
                if replacements:
                    result.append(
                        {
                            "interval_id": len(result),
                            "line_index": line_index,
                            "start_index": start_index,
                            "end_index": end_index,
                            "x_start": row[cursor].x_start,
                            "x_end": row[end - 1].x_end,
                            "surface_id": label,
                            "original": original,
                            "replacements": replacements,
                        }
                    )
            cursor = end
    replacement_surfaces = {int(item["surface_id"]) for item in result}
    return result, {
        "fill_cell_count": fill_cell_count,
        "owned_fill_cell_count": owned_fill_cell_count,
        "owned_fill_surface_count": len(fill_surfaces),
        "grouped_surface_count": len(grouped_surfaces),
        "replacement_qualified_surface_count": len(replacement_surfaces),
        "surfaces_rejected_by_width_or_density": sorted(
            grouped_surfaces - replacement_surfaces
        ),
    }


def _surface_pair_distances(
    rows: list[list[TextureCell]],
    surfaces: set[int],
) -> dict[tuple[int, int], float]:
    cells_by_surface: defaultdict[int, list[TextureCell]] = defaultdict(list)
    for row in rows:
        for cell in row:
            if cell.role == "fill" and cell.owner in surfaces:
                cells_by_surface[cell.owner].append(cell)
    result: dict[tuple[int, int], float] = {}
    for left, right in itertools.combinations(sorted(cells_by_surface), 2):
        right_cells = cells_by_surface[right]
        right_x_start = np.asarray(
            [cell.x_start for cell in right_cells], dtype=np.float32
        )
        right_x_end = np.asarray(
            [cell.x_end for cell in right_cells], dtype=np.float32
        )
        right_lines = np.asarray(
            [cell.line_index for cell in right_cells], dtype=np.float32
        )
        minimum = float("inf")
        for first in cells_by_surface[left]:
            horizontal_gap = np.maximum(
                0.0,
                np.maximum(first.x_start, right_x_start)
                - np.minimum(first.x_end, right_x_end),
            ) / SAITAMAAR_HALF_WIDTH_PX
            vertical_gap = np.maximum(
                0.0,
                np.abs(first.line_index - right_lines) - 1.0,
            )
            minimum = min(
                minimum,
                float(np.hypot(horizontal_gap, vertical_gap).min()),
            )
            if minimum == 0.0:
                break
        if minimum != float("inf"):
            result[(left, right)] = minimum
    return result


def _nearest_compatible_pairs(
    pair_distances: dict[tuple[int, int], float],
    compatible: dict[tuple[int, int], set[tuple[str, str]]],
) -> set[tuple[int, int]]:
    nearest: set[tuple[int, int]] = set()
    surfaces = sorted({surface for edge in compatible for surface in edge})
    for surface in surfaces:
        incident = [
            (pair_distances[edge], edge[1] if edge[0] == surface else edge[0], edge)
            for edge in compatible
            if surface in edge
        ]
        if incident:
            nearest.add(min(incident)[2])
    return nearest


def _surface_domains(
    targets: dict[int, float],
    intervals: list[dict[str, object]],
    palette: tuple[str, ...],
) -> tuple[dict[int, tuple[str, ...]], dict[tuple[int, str], float]]:
    by_surface: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for interval in intervals:
        by_surface[int(interval["surface_id"])].append(interval)
    domains: dict[int, tuple[str, ...]] = {}
    errors: dict[tuple[int, str], float] = {}
    for surface in targets:
        available = []
        for motif in palette:
            values = [
                float(interval["replacements"][motif][1])
                for interval in by_surface[surface]
                if motif in interval["replacements"]
            ]
            if values:
                available.append(motif)
                errors[(surface, motif)] = sum(values) / len(values)
        domains[surface] = tuple(available)
    return domains, errors


def _render_texture_assignment(
    text: str,
    intervals: list[dict[str, object]],
    assignment: dict[int, str],
) -> tuple[str, list[dict[str, object]], set[int], set[int]]:
    replacements: list[dict[str, object]] = []
    applied_surfaces: set[int] = set()
    applied_interval_ids: set[int] = set()
    by_line: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for interval in intervals:
        surface = int(interval["surface_id"])
        motif = assignment.get(surface)
        if motif is None or motif not in interval["replacements"]:
            continue
        replacement, error, repetitions, adjustment_count = interval["replacements"][motif]
        item = {
            key: value for key, value in interval.items() if key != "replacements"
        } | {
            "motif": motif,
            "replacement": replacement,
            "density_error": round(float(error), 8),
            "motif_repetitions": int(repetitions),
            "edge_adjustment_count": int(adjustment_count),
        }
        replacements.append(item)
        by_line[int(interval["line_index"])].append(item)
        applied_surfaces.add(surface)
        applied_interval_ids.add(int(interval["interval_id"]))
    lines = text.rstrip("\n").split("\n") if text else [""]
    output_lines = []
    for line_index, line in enumerate(lines):
        cursor = 0
        chunks = []
        for item in sorted(
            by_line[line_index], key=lambda value: int(value["start_index"])
        ):
            start = int(item["start_index"])
            end = int(item["end_index"])
            if start < cursor:
                raise AssertionError("Texture intervals overlap")
            chunks.extend((line[cursor:start], str(item["replacement"])))
            cursor = end
        chunks.append(line[cursor:])
        output_lines.append("".join(chunks))
    output = "\n".join(output_lines) + ("\n" if text.endswith("\n") else "")
    return output, replacements, applied_surfaces, applied_interval_ids


def _assign_styles(
    domains: dict[int, tuple[str, ...]],
    errors: dict[tuple[int, str], float],
    edges: set[tuple[int, int]],
    palette: tuple[str, ...],
    compatible: dict[tuple[int, int], set[tuple[str, str]]],
    weights: dict[tuple[int, int], float] | None = None,
) -> dict[int, str]:
    edge_weights = weights or {edge: 1.0 for edge in edges}
    degrees: defaultdict[int, float] = defaultdict(float)
    for edge in edges:
        degrees[edge[0]] += edge_weights[edge]
        degrees[edge[1]] += edge_weights[edge]
    assignment: dict[int, str] = {}

    def edge_separated(edge: tuple[int, int], candidate: str, vertex: int) -> bool:
        other = edge[1] if edge[0] == vertex else edge[0]
        if other not in assignment:
            return False
        pair = (
            (candidate, assignment[other])
            if edge[0] == vertex
            else (assignment[other], candidate)
        )
        return candidate != assignment[other] and pair in compatible[edge]

    for vertex in sorted(domains, key=lambda value: (-degrees[value], value)):
        available = domains[vertex]
        if not available:
            continue
        assignment[vertex] = min(
            available,
            key=lambda motif: (
                -sum(
                    edge_weights[edge] * edge_separated(edge, motif, vertex)
                    for edge in edges
                    if vertex in edge
                ),
                errors[(vertex, motif)],
                palette.index(motif),
            ),
        )
    for _ in range(4):
        for vertex in sorted(assignment):
            assignment[vertex] = min(
                domains[vertex],
                key=lambda motif: (
                    -sum(
                        edge_weights[edge] * edge_separated(edge, motif, vertex)
                        for edge in edges
                        if vertex in edge
                    ),
                    errors[(vertex, motif)],
                    palette.index(motif),
                ),
            )
    return assignment


def apply_surface_texture(
    text: str,
    *,
    owner: np.ndarray,
    processed: np.ndarray,
    font_size: int,
    palette: tuple[str, ...],
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[str, dict[str, object]]:
    if owner.shape != processed.shape:
        raise ValueError("Owner labels and processed image must share one shape")
    rows = _texture_cells(text, owner, font_size=font_size)
    targets = _surface_target_densities(rows, font_size=font_size)
    edge_evidence = _line_separated_graph(rows)
    all_edges = set(edge_evidence)
    intervals, eligibility = _eligible_intervals(
        text,
        rows,
        font_size=font_size,
        palette=palette,
        edge_adjustments=edge_adjustments,
    )
    intervals_by_id = {int(item["interval_id"]): item for item in intervals}
    cell_intervals: dict[tuple[int, int], int] = {}
    for interval in intervals:
        interval_id = int(interval["interval_id"])
        line_index = int(interval["line_index"])
        for character_index in range(
            int(interval["start_index"]), int(interval["end_index"])
        ):
            cell_intervals[(line_index, character_index)] = interval_id

    compatible: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for edge, evidences in edge_evidence.items():
        motif_pairs: set[tuple[str, str]] = set()
        for left_cell, right_cell in evidences:
            left_interval = intervals_by_id.get(cell_intervals.get(left_cell, -1))
            right_interval = intervals_by_id.get(cell_intervals.get(right_cell, -1))
            if left_interval is None or right_interval is None:
                continue
            for left_motif, right_motif in itertools.product(
                left_interval["replacements"], right_interval["replacements"]
            ):
                if left_motif != right_motif:
                    motif_pairs.add((str(left_motif), str(right_motif)))
        if motif_pairs:
            compatible[edge] = motif_pairs
    eligible_edges = set(compatible)

    domains, errors = _surface_domains(targets, intervals, palette)
    assignment = _assign_styles(
        domains,
        errors,
        eligible_edges,
        palette,
        compatible,
    )
    output, replacements, applied_surfaces, applied_interval_ids = (
        _render_texture_assignment(text, intervals, assignment)
    )
    actual_separated: set[tuple[int, int]] = set()
    for edge in eligible_edges:
        left_motif = assignment.get(edge[0])
        right_motif = assignment.get(edge[1])
        if left_motif is None or right_motif is None or left_motif == right_motif:
            continue
        for left_cell, right_cell in edge_evidence[edge]:
            left_interval_id = cell_intervals.get(left_cell)
            right_interval_id = cell_intervals.get(right_cell)
            if (
                left_interval_id in applied_interval_ids
                and right_interval_id in applied_interval_ids
                and (left_motif, right_motif) in compatible[edge]
            ):
                actual_separated.add(edge)
                break
    raw_pair_count = len(all_edges)
    eligible_pair_count = len(eligible_edges)
    return output, {
        "palette": list(palette),
        "surface_target_densities": {
            str(key): round(value, 8) for key, value in sorted(targets.items())
        },
        "style_assignments": {
            str(key): value for key, value in sorted(assignment.items())
        },
        "stable_surface_count": len(targets),
        "raw_line_separated_pair_count": raw_pair_count,
        "eligible_pair_count": eligible_pair_count,
        "eligible_fraction": round(
            eligible_pair_count / raw_pair_count if raw_pair_count else 0.0,
            8,
        ),
        "applied_surface_count": len(applied_surfaces),
        "actual_separated_pair_count": len(actual_separated),
        "separation_fraction": round(
            len(actual_separated) / eligible_pair_count if eligible_pair_count else 0.0,
            8,
        ),
        "raw_line_separated_pairs": [list(edge) for edge in sorted(all_edges)],
        "eligible_pairs": [list(edge) for edge in sorted(eligible_edges)],
        "actual_separated_pairs": [list(edge) for edge in sorted(actual_separated)],
        "replacement_count": len(replacements),
        "maximum_density_error": round(
            max((float(item["density_error"]) for item in replacements), default=0.0),
            8,
        ),
        "replacements": replacements,
        "eligibility": eligibility,
    }


def apply_surface_texture_proximity(
    text: str,
    *,
    owner: np.ndarray,
    processed: np.ndarray,
    font_size: int,
    palette: tuple[str, ...],
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[str, dict[str, object]]:
    if owner.shape != processed.shape:
        raise ValueError("Owner labels and processed image must share one shape")
    rows = _texture_cells(text, owner, font_size=font_size)
    targets = _surface_target_densities(rows, font_size=font_size)
    intervals, eligibility = _eligible_intervals(
        text,
        rows,
        font_size=font_size,
        palette=palette,
        edge_adjustments=edge_adjustments,
    )
    domains, errors = _surface_domains(targets, intervals, palette)
    textureable = {surface for surface, domain in domains.items() if domain}
    pair_distances = _surface_pair_distances(rows, textureable)
    compatible: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for edge in pair_distances:
        motif_pairs = {
            (left_motif, right_motif)
            for left_motif, right_motif in itertools.product(
                domains[edge[0]], domains[edge[1]]
            )
            if left_motif != right_motif
        }
        if motif_pairs:
            compatible[edge] = motif_pairs
    compatible_edges = set(compatible)
    weights = {
        edge: 1.0 / (1.0 + pair_distances[edge]) ** 2
        for edge in compatible_edges
    }
    assignment = _assign_styles(
        domains,
        errors,
        compatible_edges,
        palette,
        compatible,
        weights,
    )
    output, replacements, applied_surfaces, _ = _render_texture_assignment(
        text, intervals, assignment
    )
    nearest_pairs = _nearest_compatible_pairs(pair_distances, compatible)
    actual_nearest = {
        edge
        for edge in nearest_pairs
        if edge[0] in applied_surfaces
        and edge[1] in applied_surfaces
        and assignment.get(edge[0]) != assignment.get(edge[1])
        and (assignment[edge[0]], assignment[edge[1]]) in compatible[edge]
    }
    total_weight = sum(weights.values())
    same_motif_weight = sum(
        weight
        for edge, weight in weights.items()
        if assignment.get(edge[0]) == assignment.get(edge[1])
    )
    return output, {
        "palette": list(palette),
        "surface_target_densities": {
            str(key): round(value, 8) for key, value in sorted(targets.items())
        },
        "style_assignments": {
            str(key): value for key, value in sorted(assignment.items())
        },
        "stable_surface_count": len(targets),
        "textureable_surface_count": len(textureable),
        "proximity_pair_count": len(pair_distances),
        "compatible_proximity_pair_count": len(compatible_edges),
        "nearest_compatible_pair_count": len(nearest_pairs),
        "actual_nearest_separated_pair_count": len(actual_nearest),
        "nearest_separation_fraction": round(
            len(actual_nearest) / len(nearest_pairs) if nearest_pairs else 0.0,
            8,
        ),
        "nearest_compatible_pairs": [list(edge) for edge in sorted(nearest_pairs)],
        "actual_nearest_separated_pairs": [
            list(edge) for edge in sorted(actual_nearest)
        ],
        "pair_distances": {
            f"{edge[0]}:{edge[1]}": round(distance, 8)
            for edge, distance in sorted(pair_distances.items())
        },
        "compatible_pair_weight": round(total_weight, 8),
        "same_motif_pair_weight": round(same_motif_weight, 8),
        "weighted_separation_fraction": round(
            1.0 - same_motif_weight / total_weight if total_weight else 0.0,
            8,
        ),
        "applied_surface_count": len(applied_surfaces),
        "replacement_count": len(replacements),
        "maximum_density_error": round(
            max((float(item["density_error"]) for item in replacements), default=0.0),
            8,
        ),
        "replacements": replacements,
        "eligibility": eligibility,
    }


def _stable_variant_order(data: bytes) -> list[str]:
    source_hash = hashlib.sha256(data).hexdigest()
    return sorted(
        ("surface-texture-baseline-v5", "surface-texture-applied-v5"),
        key=lambda variant: hashlib.sha256(f"{source_hash}|{variant}".encode()).digest(),
    )


def _stable_proximity_variant_order(data: bytes) -> list[str]:
    source_hash = hashlib.sha256(data).hexdigest()
    return sorted(
        ("surface-texture-t2-v6", "surface-texture-proximity-v6"),
        key=lambda variant: hashlib.sha256(f"{source_hash}|{variant}".encode()).digest(),
    )


def generate_surface_texture_comparison(
    data: bytes,
    options: ConversionOptions,
    v2_store: SurfaceFillPreferenceStore,
    v4_store: SurfaceFillPreferenceStore,
    *,
    human_only: bool = True,
    palette: tuple[str, ...],
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]],
    catalog_sha256: str,
    audit_config_sha256: str,
) -> dict[str, object]:
    normalized = options.normalized()
    baseline = load_texture_baseline(data, normalized, v2_store, v4_store)
    filename = str(baseline.record["source"]["filename"])
    if human_only and filename not in HUMAN_EVALUATION_FILES:
        raise ValueError(
            "Surface texture human evaluation is limited to the fixed five-case set."
        )
    if not (
        baseline.processed.shape == baseline.rendered.shape == baseline.fill_mask.shape
    ):
        raise ValueError("Texture baseline artifacts must share one shape")
    source = decode_color_image(data)
    crop = tuple(int(value) for value in baseline.record["crop"])
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=f"surface-texture:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1",
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=crop,
        target_shape_hw=baseline.fill_mask.shape,
    )
    layer = next(item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID)
    owner = _selected_owner_labels(baseline.fill_mask > 127, layer.labels)
    textured_text, texture = apply_surface_texture(
        baseline.text,
        owner=owner,
        processed=baseline.processed,
        font_size=normalized.font_size,
        palette=palette,
        edge_adjustments=edge_adjustments,
    )
    textured_rendered = (
        baseline.rendered
        if textured_text == baseline.text
        else _render_text(
            textured_text,
            width=baseline.rendered.shape[1],
            height=baseline.rendered.shape[0],
        )
    )
    widths = _width_diagnostics(baseline.text, textured_text, normalized.font_size)
    line_metrics = _line_diagnostics(
        baseline.rendered,
        textured_rendered,
        1.0 - baseline.processed.astype(np.float32) / 255.0,
        (owner > 0).astype(np.float32),
    )
    case_gate = bool(
        widths["exact_row_widths"]
        and line_metrics["main_line_loss_at_1px"] <= 0.02
        and float(texture["maximum_density_error"]) <= DENSITY_TOLERANCE + 1e-12
    )
    common = {
        "source_baseline_record_id": baseline.record["record_id"],
        "source_baseline_variant_id": baseline.candidate["variant_id"],
        "source_baseline_lineage": "v4" if baseline.root == v4_store.root else "v2",
        "source_filename": filename,
        "layer_id": SURFACE_LAYER_ID,
        "proposal_labels_sha256": hashlib.sha256(
            np.ascontiguousarray(layer.labels, dtype="<i4").tobytes()
        ).hexdigest(),
        "catalog_sha256": catalog_sha256,
        "accepted_snapshot_sha256": ACCEPTED_SNAPSHOT_SHA256,
        "audit_config_sha256": audit_config_sha256,
    }
    candidates = {
        "surface-texture-baseline-v5": {
            "variantId": "surface-texture-baseline-v5",
            "methodVersion": METHOD_VERSION,
            "result": {
                "ascii": baseline.text,
                "rows": int(baseline.candidate["rows"]),
                "columns": int(baseline.candidate["columns"]),
                "processedPng": image_to_data_url(baseline.processed),
                "renderedPng": image_to_data_url(baseline.rendered),
                "crop": list(crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(baseline.fill_mask.astype(np.float32) / 255.0),
            "surface": common | {"texture_applied": False},
        },
        "surface-texture-applied-v5": {
            "variantId": "surface-texture-applied-v5",
            "methodVersion": METHOD_VERSION,
            "result": {
                "ascii": textured_text,
                "rows": int(baseline.candidate["rows"]),
                "columns": int(baseline.candidate["columns"]),
                "processedPng": image_to_data_url(baseline.processed),
                "renderedPng": image_to_data_url(textured_rendered),
                "crop": list(crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(baseline.fill_mask.astype(np.float32) / 255.0),
            "surface": common
            | {
                "texture_applied": True,
                "texture": texture,
                "width_diagnostics": widths,
                "line_diagnostics": line_metrics,
                "case_mechanical_gate": case_gate,
            },
        },
    }
    ordered = []
    for index, variant in enumerate(_stable_variant_order(data)):
        item = candidates[variant]
        item["displayLabel"] = f"候補{chr(ord('A') + index)}"
        ordered.append(item)
    return {
        "comparisonKind": COMPARISON_KIND,
        "generatorVersion": GENERATOR_VERSION,
        "recipeVersion": RECIPE_VERSION,
        "coordinateSpaceId": proposal_set.coordinate_space_id,
        "sourceFilename": filename,
        "candidates": ordered,
    }


def generate_surface_texture_proximity_comparison(
    data: bytes,
    options: ConversionOptions,
    v2_store: SurfaceFillPreferenceStore,
    v4_store: SurfaceFillPreferenceStore,
    *,
    human_only: bool = True,
    palette: tuple[str, ...],
    edge_adjustments: dict[str, dict[str, tuple[str, ...]]],
    catalog_sha256: str,
    audit_config_sha256: str,
) -> dict[str, object]:
    normalized = options.normalized()
    baseline = load_texture_baseline(data, normalized, v2_store, v4_store)
    filename = str(baseline.record["source"]["filename"])
    if human_only and filename not in HUMAN_EVALUATION_FILES:
        raise ValueError(
            "Surface texture human evaluation is limited to the fixed five-case set."
        )
    if not (
        baseline.processed.shape == baseline.rendered.shape == baseline.fill_mask.shape
    ):
        raise ValueError("Texture baseline artifacts must share one shape")
    source = decode_color_image(data)
    crop = tuple(int(value) for value in baseline.record["crop"])
    proposal_set = extract_surface_proposals(
        source,
        source_kind="color",
        coordinate_space_id=(
            f"surface-texture:{hashlib.sha256(data).hexdigest()[:16]}:aa-canvas-v1"
        ),
        config=SURFACE_PROPOSAL_CONFIG,
        crop_box_xyxy=crop,
        target_shape_hw=baseline.fill_mask.shape,
    )
    layer = next(item for item in proposal_set.layers if item.layer_id == SURFACE_LAYER_ID)
    owner = _selected_owner_labels(baseline.fill_mask > 127, layer.labels)
    generated: dict[str, tuple[str, dict[str, object]]] = {
        "surface-texture-t2-v6": apply_surface_texture(
            baseline.text,
            owner=owner,
            processed=baseline.processed,
            font_size=normalized.font_size,
            palette=palette,
            edge_adjustments=edge_adjustments,
        ),
        "surface-texture-proximity-v6": apply_surface_texture_proximity(
            baseline.text,
            owner=owner,
            processed=baseline.processed,
            font_size=normalized.font_size,
            palette=palette,
            edge_adjustments=edge_adjustments,
        ),
    }
    common = {
        "source_baseline_record_id": baseline.record["record_id"],
        "source_baseline_variant_id": baseline.candidate["variant_id"],
        "source_baseline_lineage": "v4" if baseline.root == v4_store.root else "v2",
        "source_filename": filename,
        "layer_id": SURFACE_LAYER_ID,
        "proposal_labels_sha256": hashlib.sha256(
            np.ascontiguousarray(layer.labels, dtype="<i4").tobytes()
        ).hexdigest(),
        "catalog_sha256": catalog_sha256,
        "accepted_snapshot_sha256": ACCEPTED_SNAPSHOT_SHA256,
        "audit_config_sha256": audit_config_sha256,
    }
    candidates: dict[str, dict[str, object]] = {}
    for variant, (textured_text, texture) in generated.items():
        rendered = (
            baseline.rendered
            if textured_text == baseline.text
            else _render_text(
                textured_text,
                width=baseline.rendered.shape[1],
                height=baseline.rendered.shape[0],
            )
        )
        widths = _width_diagnostics(baseline.text, textured_text, normalized.font_size)
        line_metrics = _line_diagnostics(
            baseline.rendered,
            rendered,
            1.0 - baseline.processed.astype(np.float32) / 255.0,
            (owner > 0).astype(np.float32),
        )
        case_gate = bool(
            widths["exact_row_widths"]
            and line_metrics["main_line_loss_at_1px"] <= 0.02
            and float(texture["maximum_density_error"])
            <= DENSITY_TOLERANCE + 1e-12
        )
        candidates[variant] = {
            "variantId": variant,
            "methodVersion": (
                PROXIMITY_METHOD_VERSION
                if variant == "surface-texture-proximity-v6"
                else METHOD_VERSION
            ),
            "result": {
                "ascii": textured_text,
                "rows": int(baseline.candidate["rows"]),
                "columns": int(baseline.candidate["columns"]),
                "processedPng": image_to_data_url(baseline.processed),
                "renderedPng": image_to_data_url(rendered),
                "crop": list(crop),
                "options": asdict(normalized),
            },
            "fillMaskPng": _mask_data_url(
                baseline.fill_mask.astype(np.float32) / 255.0
            ),
            "surface": common
            | {
                "texture_assignment": (
                    "proximity-t3"
                    if variant == "surface-texture-proximity-v6"
                    else "line-separated-t2"
                ),
                "texture": texture,
                "width_diagnostics": widths,
                "line_diagnostics": line_metrics,
                "case_mechanical_gate": case_gate,
            },
        }
    ordered = []
    for index, variant in enumerate(_stable_proximity_variant_order(data)):
        item = candidates[variant]
        item["displayLabel"] = f"候補{chr(ord('A') + index)}"
        ordered.append(item)
    return {
        "comparisonKind": PROXIMITY_COMPARISON_KIND,
        "generatorVersion": GENERATOR_VERSION,
        "recipeVersion": PROXIMITY_RECIPE_VERSION,
        "coordinateSpaceId": proposal_set.coordinate_space_id,
        "sourceFilename": filename,
        "candidates": ordered,
    }
