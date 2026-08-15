from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from backend.rendering import (
    REFERENCE_FONT_SIZE,
    find_font,
    glyph_advance,
    line_pitch,
    load_font,
)
from training.aa_fill_layers import detect_fill_runs
from training.audit_texture_motifs_p6r4t import (
    MotifAuditConfig,
    extract_periodic_runs,
)


SCHEMA_VERSION = 1
METHOD_VERSION = "accepted-v2-reverse-channels-v1"
EXPANDED_PERIODIC_CONFIG = MotifAuditConfig(
    maximum_period_characters=4,
    minimum_run_characters=6,
    long_run_characters=12,
    minimum_vertical_overlap_fraction=0.25,
    # Catalog support is reported after extraction. It is not used to erase
    # per-work candidates from the lossless A/B decomposition.
    minimum_run_count=20,
    minimum_work_count=5,
    # Reverse extraction records natural advances and does not inherit the
    # old texture-placement glyph-width cap.
    maximum_individual_glyph_advance_px=1_000_000,
)


@dataclass(frozen=True)
class ReverseRun:
    line_index: int
    start_index: int
    end_index: int
    x_start: int
    x_end: int
    text: str
    motifs: tuple[str, ...]
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "motifs": list(self.motifs),
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class ReverseSurface:
    surface_id: int
    runs: tuple[ReverseRun, ...]
    line_start: int
    line_end: int
    x_start: int
    x_end: int
    pixel_area: int
    glyph_count: int
    motifs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "surface_id": self.surface_id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "bbox_px": [
                self.x_start,
                self.line_start * line_pitch(REFERENCE_FONT_SIZE),
                self.x_end,
                self.line_end * line_pitch(REFERENCE_FONT_SIZE),
            ],
            "pixel_area": self.pixel_area,
            "glyph_count": self.glyph_count,
            "motifs": list(self.motifs),
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class MotifEvent:
    line_index: int
    start_index: int
    end_index: int
    motif: str
    source: str


@dataclass
class Decomposition:
    name: str
    fill_owned: np.ndarray
    runs: list[ReverseRun]
    surfaces: list[ReverseSurface]
    line_mask: np.ndarray
    fill_glyph_mask: np.ndarray
    surface_ids: np.ndarray
    surface_tone: np.ndarray
    motif_events: list[MotifEvent]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _line_positions(line: str, font_size: int) -> list[int]:
    positions = [0]
    for character in line:
        positions.append(positions[-1] + glyph_advance(character, font_size))
    return positions


def _row_offsets(lines: list[str]) -> np.ndarray:
    result = [0]
    for line in lines:
        result.append(result[-1] + len(line))
    return np.asarray(result, dtype=np.uint32)


def _baseline_events(text: str, font_size: int) -> list[MotifEvent]:
    return [
        MotifEvent(
            line_index=run.line_index,
            start_index=run.start_index,
            end_index=run.end_index,
            motif=run.character,
            source="p6r1-single-character",
        )
        for run in detect_fill_runs(text, font_size=font_size)
    ]


def _expanded_events(text: str, font_size: int) -> list[MotifEvent]:
    result = _baseline_events(text, font_size)
    result.extend(
        MotifEvent(
            line_index=run.line_index,
            start_index=run.start_index,
            end_index=run.end_index,
            motif=run.motif,
            source="periodic-mixed-character",
        )
        for run in extract_periodic_runs(
            text,
            config=EXPANDED_PERIODIC_CONFIG,
            font_size=font_size,
        )
    )
    unique = {
        (event.line_index, event.start_index, event.end_index, event.motif, event.source): event
        for event in result
    }
    return [unique[key] for key in sorted(unique)]


def _events_to_runs(
    lines: list[str],
    events: Iterable[MotifEvent],
    *,
    font_size: int,
) -> tuple[np.ndarray, list[ReverseRun]]:
    offsets = _row_offsets(lines)
    owned = np.zeros(int(offsets[-1]), dtype=np.uint8)
    events_by_line: defaultdict[int, list[MotifEvent]] = defaultdict(list)
    for event in events:
        if not 0 <= event.line_index < len(lines):
            raise ValueError("Motif event has an invalid line index")
        if not 0 <= event.start_index < event.end_index <= len(lines[event.line_index]):
            raise ValueError("Motif event has an invalid character interval")
        events_by_line[event.line_index].append(event)
        start = int(offsets[event.line_index]) + event.start_index
        end = int(offsets[event.line_index]) + event.end_index
        owned[start:end] = 1

    runs: list[ReverseRun] = []
    for line_index, line in enumerate(lines):
        line_start = int(offsets[line_index])
        line_owned = owned[line_start : line_start + len(line)]
        positions = _line_positions(line, font_size)
        cursor = 0
        while cursor < len(line):
            if not line_owned[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while end < len(line) and line_owned[end]:
                end += 1
            overlapping = [
                event
                for event in events_by_line.get(line_index, [])
                if event.start_index < end and event.end_index > cursor
            ]
            runs.append(
                ReverseRun(
                    line_index=line_index,
                    start_index=cursor,
                    end_index=end,
                    x_start=positions[cursor],
                    x_end=positions[end],
                    text=line[cursor:end],
                    motifs=tuple(sorted({event.motif for event in overlapping})),
                    sources=tuple(sorted({event.source for event in overlapping})),
                )
            )
            cursor = end
    return owned, runs


def _overlap(left: ReverseRun, right: ReverseRun) -> int:
    return max(0, min(left.x_end, right.x_end) - max(left.x_start, right.x_start))


def connect_reverse_runs(
    runs: Iterable[ReverseRun],
    *,
    minimum_overlap_ratio: float = 0.25,
    font_size: int = REFERENCE_FONT_SIZE,
) -> list[ReverseSurface]:
    if not 0.0 < minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be in (0, 1]")
    ordered = sorted(
        runs,
        key=lambda run: (
            run.line_index,
            run.x_start,
            run.x_end,
            run.start_index,
            run.end_index,
        ),
    )
    if not ordered:
        return []
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_line: defaultdict[int, list[int]] = defaultdict(list)
    for index, run in enumerate(ordered):
        by_line[run.line_index].append(index)
    for line_index, indices in by_line.items():
        for left_index in indices:
            left = ordered[left_index]
            for right_index in by_line.get(line_index + 1, []):
                right = ordered[right_index]
                shorter = min(left.x_end - left.x_start, right.x_end - right.x_start)
                if shorter and _overlap(left, right) >= shorter * minimum_overlap_ratio:
                    union(left_index, right_index)

    groups: defaultdict[int, list[ReverseRun]] = defaultdict(list)
    for index, run in enumerate(ordered):
        groups[find(index)].append(run)
    components = sorted(
        (tuple(group) for group in groups.values()),
        key=lambda group: (
            group[0].line_index,
            group[0].x_start,
            group[0].start_index,
        ),
    )
    pitch = line_pitch(font_size)
    return [
        ReverseSurface(
            surface_id=surface_id,
            runs=component,
            line_start=min(run.line_index for run in component),
            line_end=max(run.line_index for run in component) + 1,
            x_start=min(run.x_start for run in component),
            x_end=max(run.x_end for run in component),
            pixel_area=sum((run.x_end - run.x_start) * pitch for run in component),
            glyph_count=sum(run.end_index - run.start_index for run in component),
            motifs=tuple(sorted({motif for run in component for motif in run.motifs})),
        )
        for surface_id, component in enumerate(components)
    ]


def _render_partition(
    lines: list[str],
    fill_owned: np.ndarray,
    *,
    font_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pitch = line_pitch(font_size)
    row_widths = np.asarray(
        [sum(glyph_advance(character, font_size) for character in line) for line in lines],
        dtype=np.uint32,
    )
    width = max(2, int(row_widths.max(initial=0)))
    height = max(pitch, len(lines) * pitch)
    target_image = Image.new("L", (width, height), 255)
    target_draw = ImageDraw.Draw(target_image)
    font = load_font(font_size)
    ascent, _ = font.getmetrics()
    offsets = _row_offsets(lines)
    fill_cells = np.zeros((height, width), dtype=bool)

    for line_index, line in enumerate(lines):
        baseline = line_index * pitch + ascent
        target_draw.text((0, baseline), line, font=font, fill=0, anchor="ls")
        positions = _line_positions(line, font_size)
        ownership = fill_owned[int(offsets[line_index]) : int(offsets[line_index + 1])]
        for index, is_fill in enumerate(ownership):
            if is_fill:
                fill_cells[
                    line_index * pitch : (line_index + 1) * pitch,
                    positions[index] : positions[index + 1],
                ] = True

    target = np.asarray(target_image) < 128
    # Partition pixels from the canonical whole-line rendering by the owning
    # natural-advance cell. This stays exact for combining marks and the few
    # glyph pairs whose whole-line raster differs at a substring boundary.
    fill_mask = target & fill_cells
    line_mask = target & ~fill_cells
    return target, line_mask, fill_mask, row_widths, np.asarray(offsets, dtype=np.uint32)


def _character_density(character: str, font_size: int, cache: dict[str, float]) -> float:
    if character in cache:
        return cache[character]
    width = glyph_advance(character, font_size)
    pitch = line_pitch(font_size)
    font = load_font(font_size)
    ascent, _ = font.getmetrics()
    image = Image.new("L", (width, pitch), 255)
    ImageDraw.Draw(image).text((0, ascent), character, font=font, fill=0, anchor="ls")
    density = float(np.mean(np.asarray(image) < 128))
    cache[character] = density
    return density


def render_surface_arrays(
    surfaces: Iterable[ReverseSurface],
    *,
    width: int,
    height: int,
    font_size: int = REFERENCE_FONT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((height, width), dtype=np.int32)
    tone = np.zeros((height, width), dtype=np.uint8)
    pitch = line_pitch(font_size)
    density_cache: dict[str, float] = {}
    for surface in surfaces:
        label = surface.surface_id + 1
        for run in surface.runs:
            x_start = max(0, min(width, run.x_start))
            x_end = max(0, min(width, run.x_end))
            y_start = run.line_index * pitch
            y_end = min(height, y_start + pitch)
            labels[y_start:y_end, x_start:x_end] = label
            advances = [glyph_advance(character, font_size) for character in run.text]
            weighted_density = sum(
                _character_density(character, font_size, density_cache) * advance
                for character, advance in zip(run.text, advances)
            ) / max(1, sum(advances))
            tone[y_start:y_end, x_start:x_end] = round(weighted_density * 255)
    return labels, tone


def decompose_text(
    text: str,
    *,
    candidate: str,
    font_size: int = REFERENCE_FONT_SIZE,
) -> tuple[Decomposition, dict[str, np.ndarray]]:
    lines = text.splitlines() or [""]
    if candidate == "a-p6r1":
        events = _baseline_events(text, font_size)
    elif candidate == "b-periodic":
        events = _expanded_events(text, font_size)
    else:
        raise ValueError(f"Unknown reverse-channel candidate: {candidate}")
    fill_owned, runs = _events_to_runs(lines, events, font_size=font_size)
    surfaces = connect_reverse_runs(runs, font_size=font_size)
    target, line_mask, fill_mask, row_widths, row_offsets = _render_partition(
        lines,
        fill_owned,
        font_size=font_size,
    )
    surface_ids, surface_tone = render_surface_arrays(
        surfaces,
        width=target.shape[1],
        height=target.shape[0],
        font_size=font_size,
    )
    codepoints = np.asarray([ord(character) for line in lines for character in line], dtype=np.uint32)
    rows = np.asarray(
        [line_index for line_index, line in enumerate(lines) for _ in line],
        dtype=np.uint16 if len(lines) <= np.iinfo(np.uint16).max else np.uint32,
    )
    starts = np.asarray(
        [
            x
            for line in lines
            for x in _line_positions(line, font_size)[:-1]
        ],
        dtype=np.uint32,
    )
    advances = np.asarray(
        [glyph_advance(character, font_size) for line in lines for character in line],
        dtype=np.uint16,
    )
    return (
        Decomposition(
            name=candidate,
            fill_owned=fill_owned,
            runs=runs,
            surfaces=surfaces,
            line_mask=line_mask,
            fill_glyph_mask=fill_mask,
            surface_ids=surface_ids,
            surface_tone=surface_tone,
            motif_events=events,
        ),
        {
            "target_mask": target,
            "glyph_codepoints": codepoints,
            "glyph_rows": rows,
            "glyph_x_starts": starts,
            "glyph_advances": advances,
            "row_offsets": row_offsets,
            "row_widths": row_widths,
        },
    )


def surface_integrity_failures(decomposition: Decomposition) -> list[str]:
    failures: list[str] = []
    labels = decomposition.surface_ids
    for surface in decomposition.surfaces:
        mask = (labels == surface.surface_id + 1).astype(np.uint8)
        area = int(mask.sum())
        if area != surface.pixel_area:
            failures.append(f"surface-{surface.surface_id}-area")
        component_count, _ = cv2.connectedComponents(mask, connectivity=4)
        if component_count != 2:
            failures.append(f"surface-{surface.surface_id}-connectedness")
    expected_labels = len(decomposition.surfaces) + 1
    if int(labels.max(initial=0)) >= expected_labels:
        failures.append("unexpected-surface-label")
    return failures


def decomposition_summary(
    decomposition: Decomposition,
    target: np.ndarray,
) -> dict[str, object]:
    recomposed = decomposition.line_mask | decomposition.fill_glyph_mask
    return {
        "fill_glyph_count": int(decomposition.fill_owned.sum()),
        "run_count": len(decomposition.runs),
        "surface_count": len(decomposition.surfaces),
        "multi_line_surface_count": sum(
            surface.line_end - surface.line_start > 1
            for surface in decomposition.surfaces
        ),
        "fill_cell_pixel_count": int(np.count_nonzero(decomposition.surface_ids)),
        "line_ink_pixel_count": int(decomposition.line_mask.sum()),
        "fill_ink_pixel_count": int(decomposition.fill_glyph_mask.sum()),
        "target_recomposition_exact": bool(np.array_equal(target, recomposed)),
        "surface_integrity_failures": surface_integrity_failures(decomposition),
    }


def font_sha256() -> str:
    return _sha256(find_font().read_bytes())
