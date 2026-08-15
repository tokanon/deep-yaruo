from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.aa_semantics import (
    DENSE_FILL_CHARACTERS,
    OUTLINE_RUN_CHARACTERS,
    TONE_FILL_CHARACTERS,
)
from backend.rendering import find_font, line_pitch


@dataclass(frozen=True)
class FillRun:
    line_index: int
    start_index: int
    end_index: int
    character: str
    count: int
    x_start: float
    x_end: float
    bounded_left: bool
    bounded_right: bool
    vertical_support: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FillSurface:
    surface_id: int
    runs: tuple[FillRun, ...] = field(repr=False)
    line_start: int
    line_end: int
    x_start: float
    x_end: float
    x_start_px: int
    x_end_px: int
    y_start_px: int
    y_end_px: int
    pixel_area: int
    run_count: int
    line_count: int
    character_counts: tuple[tuple[str, int], ...]
    tone_level_counts: tuple[tuple[int, int], ...]

    @property
    def bbox_px(self) -> tuple[int, int, int, int]:
        return (
            self.x_start_px,
            self.y_start_px,
            self.x_end_px,
            self.y_end_px,
        )

    def to_dict(self, *, include_runs: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "surface_id": self.surface_id,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "x_start_px": self.x_start_px,
            "x_end_px": self.x_end_px,
            "y_start_px": self.y_start_px,
            "y_end_px": self.y_end_px,
            "bbox_px": list(self.bbox_px),
            "pixel_area": self.pixel_area,
            "run_count": self.run_count,
            "line_count": self.line_count,
            "character_counts": dict(self.character_counts),
            "tone_level_counts": {
                str(level): count for level, count in self.tone_level_counts
            },
        }
        if include_runs:
            result["runs"] = [run.to_dict() for run in self.runs]
        return result


def _non_space_before(line: str, index: int) -> bool:
    return any(not character.isspace() for character in line[:index])


def _non_space_after(line: str, index: int) -> bool:
    return any(not character.isspace() for character in line[index:])


def _overlap(left: FillRun, right: FillRun) -> float:
    return max(0.0, min(left.x_end, right.x_end) - max(left.x_start, right.x_start))


def _run_sort_key(run: FillRun) -> tuple[object, ...]:
    return (
        run.line_index,
        run.x_start,
        run.x_end,
        run.start_index,
        run.end_index,
        run.character,
    )


def detect_fill_runs(
    text: str,
    *,
    font_size: int = 16,
    minimum_run: int = 3,
) -> list[FillRun]:
    font = ImageFont.truetype(str(find_font()), font_size)
    lines = text.splitlines() or [""]
    candidates: list[FillRun] = []
    for line_index, line in enumerate(lines):
        index = 0
        while index < len(line):
            end = index + 1
            while end < len(line) and line[end] == line[index]:
                end += 1
            character = line[index]
            count = end - index
            bounded_left = _non_space_before(line, index)
            bounded_right = _non_space_after(line, end)
            if (
                count >= minimum_run
                and not character.isspace()
                and character not in OUTLINE_RUN_CHARACTERS
                and character in TONE_FILL_CHARACTERS | DENSE_FILL_CHARACTERS
            ):
                candidates.append(
                    FillRun(
                        line_index=line_index,
                        start_index=index,
                        end_index=end,
                        character=character,
                        count=count,
                        x_start=float(font.getlength(line[:index])),
                        x_end=float(font.getlength(line[:end])),
                        bounded_left=bounded_left,
                        bounded_right=bounded_right,
                    )
                )
            index = end

    supported: list[FillRun] = []
    for candidate in candidates:
        vertical_support = any(
            abs(other.line_index - candidate.line_index) == 1
            and _overlap(candidate, other) >= min(
                candidate.x_end - candidate.x_start,
                other.x_end - other.x_start,
            )
            * 0.25
            for other in candidates
        )
        is_supported_tone = (
            candidate.character in TONE_FILL_CHARACTERS
            and (vertical_support or candidate.count >= 6)
        )
        is_dense_fill = candidate.character in DENSE_FILL_CHARACTERS
        if is_supported_tone or is_dense_fill:
            supported.append(
                FillRun(
                    **{
                        **candidate.to_dict(),
                        "vertical_support": vertical_support,
                    }
                )
            )
    return supported


def _glyph_ink_coverage(font: ImageFont.FreeTypeFont, character: str, pitch: int) -> float:
    width = max(2, int(math.ceil(font.getlength(character))) + 2)
    canvas = Image.new("L", (width, pitch), 255)
    draw = ImageDraw.Draw(canvas)
    ascent, _ = font.getmetrics()
    draw.text((0, ascent), character, font=font, fill=0, anchor="ls")
    pixels = np.asarray(canvas, dtype=np.float32)
    return float(np.mean((255.0 - pixels) / 255.0))


def _tone_level(font: ImageFont.FreeTypeFont, character: str, pitch: int) -> int:
    """Map the rendered fill darkness used by previews to a stable 0..3 level."""
    coverage = _glyph_ink_coverage(font, character, pitch)
    darkness = min(0.9, coverage * 4.5)
    return min(3, max(0, int(round(darkness * 3))))


def connect_fill_runs(
    runs: Iterable[FillRun],
    *,
    font_size: int = 16,
    minimum_overlap_ratio: float = 0.25,
) -> list[FillSurface]:
    """Connect horizontally overlapping runs on adjacent rows into 2D surfaces.

    Connections are deliberately limited to a one-row gap. Split/merge topology
    is retained through the run list instead of being flattened into one box.
    """
    if not 0.0 < minimum_overlap_ratio <= 1.0:
        raise ValueError("minimum_overlap_ratio must be in (0, 1]")

    ordered_runs = sorted(runs, key=_run_sort_key)
    if not ordered_runs:
        return []

    parent = list(range(len(ordered_runs)))

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

    by_line: dict[int, list[int]] = {}
    for index, run in enumerate(ordered_runs):
        by_line.setdefault(run.line_index, []).append(index)
    for line_index in sorted(by_line):
        next_indices = by_line.get(line_index + 1, [])
        for left_index in by_line[line_index]:
            left = ordered_runs[left_index]
            for right_index in next_indices:
                right = ordered_runs[right_index]
                shorter_width = min(
                    left.x_end - left.x_start,
                    right.x_end - right.x_start,
                )
                if (
                    shorter_width > 0.0
                    and _overlap(left, right)
                    >= shorter_width * minimum_overlap_ratio
                ):
                    union(left_index, right_index)

    components: dict[int, list[FillRun]] = {}
    for index, run in enumerate(ordered_runs):
        components.setdefault(find(index), []).append(run)
    ordered_components = sorted(
        (tuple(sorted(component, key=_run_sort_key)) for component in components.values()),
        key=lambda component: _run_sort_key(component[0]),
    )

    pitch = line_pitch(font_size)
    font = ImageFont.truetype(str(find_font()), font_size)
    tone_cache: dict[str, int] = {}
    surfaces: list[FillSurface] = []
    for surface_id, component in enumerate(ordered_components):
        line_start = min(run.line_index for run in component)
        line_end = max(run.line_index for run in component) + 1
        x_start = min(run.x_start for run in component)
        x_end = max(run.x_end for run in component)
        x_start_px = min(round(run.x_start) for run in component)
        x_end_px = max(round(run.x_end) for run in component)
        pixel_area = sum(
            max(0, round(run.x_end) - round(run.x_start)) * pitch
            for run in component
        )
        character_counts = Counter(run.character for run in component)
        tone_level_counts = Counter(
            tone_cache.setdefault(
                run.character,
                _tone_level(font, run.character, pitch),
            )
            for run in component
        )
        surfaces.append(
            FillSurface(
                surface_id=surface_id,
                runs=component,
                line_start=line_start,
                line_end=line_end,
                x_start=x_start,
                x_end=x_end,
                x_start_px=x_start_px,
                x_end_px=x_end_px,
                y_start_px=line_start * pitch,
                y_end_px=line_end * pitch,
                pixel_area=pixel_area,
                run_count=len(component),
                line_count=len({run.line_index for run in component}),
                character_counts=tuple(sorted(character_counts.items())),
                tone_level_counts=tuple(sorted(tone_level_counts.items())),
            )
        )
    return surfaces


def render_fill_surface_labels(
    surfaces: Iterable[FillSurface],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Render surface IDs as an integer mask; zero is background."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    labels = np.zeros((height, width), dtype=np.int32)
    for surface in surfaces:
        label = surface.surface_id + 1
        pitch = (surface.y_end_px - surface.y_start_px) // max(
            1, surface.line_end - surface.line_start
        )
        for run in surface.runs:
            x_start = max(0, min(width, round(run.x_start)))
            x_end = max(0, min(width, round(run.x_end)))
            y_start = max(0, min(height, run.line_index * pitch))
            y_end = max(0, min(height, y_start + pitch))
            labels[y_start:y_end, x_start:x_end] = label
    return labels


def render_fill_layers(
    text: str,
    *,
    font_size: int = 16,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[FillRun],
]:
    font = ImageFont.truetype(str(find_font()), font_size)
    pitch = line_pitch(font_size)
    ascent, _ = font.getmetrics()
    lines = text.splitlines() or [""]
    width = max(2, int(math.ceil(max(font.getlength(line) for line in lines))) + 2)
    height = max(pitch, len(lines) * pitch)
    runs = detect_fill_runs(text, font_size=font_size)
    runs_by_line: dict[int, list[FillRun]] = {}
    for run in runs:
        runs_by_line.setdefault(run.line_index, []).append(run)

    original = Image.new("L", (width, height), 255)
    line_layer = Image.new("L", (width, height), 255)
    fill_layer = Image.new("L", (width, height), 255)
    original_draw = ImageDraw.Draw(original)
    line_draw = ImageDraw.Draw(line_layer)
    fill_draw = ImageDraw.Draw(fill_layer)
    coverage_cache: dict[str, float] = {}

    for line_index, line in enumerate(lines):
        baseline = line_index * pitch + ascent
        original_draw.text((0, baseline), line, font=font, fill=0, anchor="ls")
        excluded = {
            index
            for run in runs_by_line.get(line_index, [])
            for index in range(run.start_index, run.end_index)
        }
        for index, character in enumerate(line):
            if index not in excluded:
                line_draw.text(
                    (float(font.getlength(line[:index])), baseline),
                    character,
                    font=font,
                    fill=0,
                    anchor="ls",
                )
        for run in runs_by_line.get(line_index, []):
            coverage = coverage_cache.setdefault(
                run.character,
                _glyph_ink_coverage(font, run.character, pitch),
            )
            darkness = min(0.9, coverage * 4.5)
            shade = max(36, 255 - round(darkness * 230))
            fill_draw.rectangle(
                (
                    round(run.x_start),
                    line_index * pitch,
                    max(round(run.x_start), round(run.x_end) - 1),
                    (line_index + 1) * pitch - 1,
                ),
                fill=shade,
            )

    original_array = np.asarray(original)
    line_array = np.asarray(line_layer)
    fill_array = np.asarray(fill_layer)
    composite = np.minimum(line_array, fill_array)
    diagnostic = np.repeat(original_array[:, :, None], 3, axis=2)
    diagnostic_image = Image.fromarray(diagnostic)
    diagnostic_draw = ImageDraw.Draw(diagnostic_image)
    for run in runs:
        diagnostic_draw.rectangle(
            (
                round(run.x_start),
                run.line_index * pitch,
                max(round(run.x_start), round(run.x_end) - 1),
                (run.line_index + 1) * pitch - 1,
            ),
            outline=(255, 32, 32),
            width=2,
        )
    return original_array, line_array, fill_array, composite, np.asarray(diagnostic_image), runs
