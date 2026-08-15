from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .aa_semantics import FILL_REPEAT_CHARACTERS
from .rendering import glyph_advance, line_pitch, render_glyph_mask


@dataclass(frozen=True)
class FillStage1Config:
    """Conservative P4 thresholds for post-decoder fill replacement."""

    minimum_run: int = 3
    minimum_empirical_runs: int = 20
    fill_fraction_minimum: float = 0.85
    safe_fill_fraction_minimum: float = 0.65
    structure_fraction_maximum: float = 0.16
    boundary_margin_px: int = 2
    coverage_scale: float = 4.5

    def validate(self) -> None:
        if self.minimum_run < 3:
            raise ValueError("minimum_run must be at least three")
        if self.minimum_empirical_runs <= 0:
            raise ValueError("minimum_empirical_runs must be positive")
        for name, value in (
            ("fill_fraction_minimum", self.fill_fraction_minimum),
            ("safe_fill_fraction_minimum", self.safe_fill_fraction_minimum),
            ("structure_fraction_maximum", self.structure_fraction_maximum),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.boundary_margin_px < 0:
            raise ValueError("boundary_margin_px cannot be negative")
        if self.coverage_scale <= 0:
            raise ValueError("coverage_scale must be positive")


@dataclass(frozen=True)
class FillVocabularyEntry:
    character: str
    empirical_run_count: int
    advance: int
    ink_coverage: float
    effective_darkness: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CharacterCell:
    index: int
    character: str
    x_start: int
    x_end: int
    fill_fraction: float
    safe_fill_fraction: float
    structure_fraction: float
    tone: float
    eligible: bool


@dataclass(frozen=True)
class FillReplacement:
    line_index: int
    start_index: int
    end_index: int
    x_start: int
    x_end: int
    original: str
    replacement: str
    fill_character: str
    repeated_count: int
    padding_spaces: int
    target_tone: float
    effective_darkness: float
    mean_fill_fraction: float
    mean_safe_fill_fraction: float
    mean_structure_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FillStage1Result:
    text: str
    replacements: tuple[FillReplacement, ...]
    vocabulary: tuple[FillVocabularyEntry, ...]
    cell_count: int
    eligible_cell_count: int
    replaced_source_cell_count: int
    replaced_pixel_width: int

    def summary(self) -> dict[str, object]:
        return {
            "cell_count": self.cell_count,
            "eligible_cell_count": self.eligible_cell_count,
            "replaced_source_cell_count": self.replaced_source_cell_count,
            "replacement_count": len(self.replacements),
            "replaced_pixel_width": self.replaced_pixel_width,
            "vocabulary": [entry.to_dict() for entry in self.vocabulary],
            "replacements": [item.to_dict() for item in self.replacements],
        }


def build_fill_vocabulary(
    empirical_run_counts: Mapping[str, int],
    *,
    font_size: int = 16,
    config: FillStage1Config | None = None,
) -> tuple[FillVocabularyEntry, ...]:
    """Build P4 candidates only from repeatedly observed CP932 fill runs."""
    active = config or FillStage1Config()
    active.validate()
    entries: list[FillVocabularyEntry] = []
    for character, count_value in empirical_run_counts.items():
        count = int(count_value)
        if (
            len(character) != 1
            or character not in FILL_REPEAT_CHARACTERS
            or count < active.minimum_empirical_runs
        ):
            continue
        try:
            character.encode("cp932")
        except UnicodeEncodeError:
            continue
        coverage = float(render_glyph_mask(character, font_size).mean())
        entries.append(
            FillVocabularyEntry(
                character=character,
                empirical_run_count=count,
                advance=glyph_advance(character, font_size),
                ink_coverage=coverage,
                effective_darkness=min(1.0, coverage * active.coverage_scale),
            )
        )
    if not entries:
        raise ValueError("No empirical fill characters passed the P4 vocabulary gate")
    return tuple(
        sorted(
            entries,
            key=lambda item: (
                item.effective_darkness,
                -item.empirical_run_count,
                item.character,
            ),
        )
    )


def _slice_mean(channel: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> float:
    patch = channel[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size else 0.0


def _mapped_rectangle(
    *,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    x0 = min(target_width - 1, max(0, int(math.floor(x_start * target_width / source_width))))
    x1 = min(target_width, max(x0 + 1, int(math.ceil(x_end * target_width / source_width))))
    y0 = min(target_height - 1, max(0, int(math.floor(y_start * target_height / source_height))))
    y1 = min(target_height, max(y0 + 1, int(math.ceil(y_end * target_height / source_height))))
    return x0, y0, x1, y1


def _cells_for_line(
    line: str,
    line_index: int,
    *,
    structure: np.ndarray,
    tone: np.ndarray,
    fill: np.ndarray,
    safe_fill: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    font_size: int,
    config: FillStage1Config,
) -> list[CharacterCell]:
    cells: list[CharacterCell] = []
    x = 0
    pitch = line_pitch(font_size)
    channel_height, channel_width = fill.shape
    for index, character in enumerate(line):
        width = glyph_advance(character, font_size)
        x_end = x + width
        x0, y0, x1, y1 = _mapped_rectangle(
            x_start=x,
            x_end=x_end,
            y_start=line_index * pitch,
            y_end=(line_index + 1) * pitch,
            source_width=canvas_width,
            source_height=canvas_height,
            target_width=channel_width,
            target_height=channel_height,
        )
        fill_fraction = _slice_mean(fill, y0, y1, x0, x1)
        safe_fraction = _slice_mean(safe_fill, y0, y1, x0, x1)
        structure_fraction = _slice_mean(structure, y0, y1, x0, x1)
        local_tone = _slice_mean(tone, y0, y1, x0, x1)
        eligible = (
            fill_fraction >= config.fill_fraction_minimum
            and safe_fraction >= config.safe_fill_fraction_minimum
            and structure_fraction <= config.structure_fraction_maximum
        )
        cells.append(
            CharacterCell(
                index=index,
                character=character,
                x_start=x,
                x_end=x_end,
                fill_fraction=fill_fraction,
                safe_fill_fraction=safe_fraction,
                structure_fraction=structure_fraction,
                tone=local_tone,
                eligible=eligible,
            )
        )
        x = x_end
    return cells


def _replacement_for_width(
    width: int,
    target_tone: float,
    vocabulary: Sequence[FillVocabularyEntry],
    *,
    space_advance: int,
    minimum_run: int,
    font_size: int,
) -> tuple[str, FillVocabularyEntry, int, int] | None:
    candidates: list[tuple[float, int, str, FillVocabularyEntry, int, int]] = []
    maximum_frequency = max(entry.empirical_run_count for entry in vocabulary)
    for entry in vocabulary:
        maximum_count = width // entry.advance
        for repeated_count in range(minimum_run, maximum_count + 1):
            remainder = width - repeated_count * entry.advance
            if remainder < 0 or remainder % space_advance:
                continue
            padding_spaces = remainder // space_advance
            left_padding = padding_spaces // 2
            right_padding = padding_spaces - left_padding
            replacement = (
                " " * left_padding
                + entry.character * repeated_count
                + " " * right_padding
            )
            frequency_penalty = 0.025 * math.log(
                maximum_frequency / entry.empirical_run_count
            )
            padding_penalty = 0.08 * remainder / max(1, width)
            score = (
                abs(entry.effective_darkness - target_tone)
                + frequency_penalty
                + padding_penalty
            )
            candidates.append(
                (
                    score,
                    -repeated_count,
                    entry.character,
                    entry,
                    repeated_count,
                    padding_spaces,
                )
            )
    if not candidates:
        return None
    _, _, _, entry, repeated_count, padding_spaces = min(candidates)
    left_padding = padding_spaces // 2
    right_padding = padding_spaces - left_padding
    replacement = (
        " " * left_padding + entry.character * repeated_count + " " * right_padding
    )
    if sum(glyph_advance(character, font_size) for character in replacement) != width:
        raise AssertionError("P4 replacement changed the proportional line width")
    return replacement, entry, repeated_count, padding_spaces


def apply_fill_stage1(
    text: str,
    *,
    structure: np.ndarray,
    tone: np.ndarray,
    fill: np.ndarray,
    empirical_run_counts: Mapping[str, int],
    canvas_width: int,
    font_size: int = 16,
    config: FillStage1Config | None = None,
) -> FillStage1Result:
    """Overwrite only safe ch2 interiors while preserving every row advance."""
    active = config or FillStage1Config()
    active.validate()
    if structure.shape != tone.shape or tone.shape != fill.shape:
        raise ValueError("P4 structure, tone, and fill channels must share one shape")
    if structure.ndim != 2 or canvas_width <= 0:
        raise ValueError("P4 expects two-dimensional channels and a positive canvas width")
    vocabulary = build_fill_vocabulary(
        empirical_run_counts,
        font_size=font_size,
        config=active,
    )
    binary_fill = (fill >= 0.5).astype(np.uint8)
    if active.boundary_margin_px:
        radius = active.boundary_margin_px
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        safe_fill = cv2.erode(binary_fill, kernel).astype(np.float32)
    else:
        safe_fill = binary_fill.astype(np.float32)
    normalized_structure = np.clip(structure.astype(np.float32), 0.0, 1.0)
    normalized_tone = np.clip(tone.astype(np.float32), 0.0, 1.0)
    normalized_fill = np.clip(fill.astype(np.float32), 0.0, 1.0)
    lines = text.rstrip("\n").split("\n") if text else [""]
    canvas_height = max(1, len(lines) * line_pitch(font_size))
    space_width = glyph_advance(" ", font_size)
    replacements: list[FillReplacement] = []
    output_lines: list[str] = []
    total_cells = 0
    eligible_cells = 0
    replaced_cells = 0
    replaced_width = 0

    for line_index, line in enumerate(lines):
        cells = _cells_for_line(
            line,
            line_index,
            structure=normalized_structure,
            tone=normalized_tone,
            fill=normalized_fill,
            safe_fill=safe_fill,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            font_size=font_size,
            config=active,
        )
        total_cells += len(cells)
        eligible_cells += sum(cell.eligible for cell in cells)
        chunks: list[str] = []
        cursor = 0
        while cursor < len(cells):
            if not cells[cursor].eligible:
                chunks.append(cells[cursor].character)
                cursor += 1
                continue
            end = cursor + 1
            while end < len(cells) and cells[end].eligible:
                end += 1
            segment = cells[cursor:end]
            width = segment[-1].x_end - segment[0].x_start
            target_tone = float(np.mean([cell.tone for cell in segment]))
            proposal = _replacement_for_width(
                width,
                target_tone,
                vocabulary,
                space_advance=space_width,
                minimum_run=active.minimum_run,
                font_size=font_size,
            )
            if proposal is None:
                chunks.extend(cell.character for cell in segment)
                cursor = end
                continue
            replacement, entry, repeated_count, padding_spaces = proposal
            original = "".join(cell.character for cell in segment)
            original_width = sum(glyph_advance(character, font_size) for character in original)
            replacement_width = sum(
                glyph_advance(character, font_size) for character in replacement
            )
            if original_width != replacement_width or original_width != width:
                raise AssertionError("P4 must preserve every replaced interval advance")
            chunks.append(replacement)
            replacements.append(
                FillReplacement(
                    line_index=line_index,
                    start_index=cursor,
                    end_index=end,
                    x_start=segment[0].x_start,
                    x_end=segment[-1].x_end,
                    original=original,
                    replacement=replacement,
                    fill_character=entry.character,
                    repeated_count=repeated_count,
                    padding_spaces=padding_spaces,
                    target_tone=round(target_tone, 6),
                    effective_darkness=round(entry.effective_darkness, 6),
                    mean_fill_fraction=round(
                        float(np.mean([cell.fill_fraction for cell in segment])), 6
                    ),
                    mean_safe_fill_fraction=round(
                        float(np.mean([cell.safe_fill_fraction for cell in segment])), 6
                    ),
                    mean_structure_fraction=round(
                        float(np.mean([cell.structure_fraction for cell in segment])), 6
                    ),
                )
            )
            replaced_cells += len(segment)
            replaced_width += width
            cursor = end
        output_line = "".join(chunks)
        before_width = sum(glyph_advance(character, font_size) for character in line)
        after_width = sum(glyph_advance(character, font_size) for character in output_line)
        if before_width != after_width:
            raise AssertionError(
                f"P4 changed row {line_index} width from {before_width} to {after_width}"
            )
        output_lines.append(output_line)

    return FillStage1Result(
        text="\n".join(output_lines) + ("\n" if text.endswith("\n") else ""),
        replacements=tuple(replacements),
        vocabulary=vocabulary,
        cell_count=total_cells,
        eligible_cell_count=eligible_cells,
        replaced_source_cell_count=replaced_cells,
        replaced_pixel_width=replaced_width,
    )
