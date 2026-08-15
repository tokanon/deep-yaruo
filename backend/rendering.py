from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "assets" / "fonts" / "Saitamaar.ttf"
REFERENCE_FONT_SIZE = 16
REFERENCE_LINE_PITCH = 18


def find_font() -> Path:
    if FONT_PATH.exists():
        return FONT_PATH
    raise FileNotFoundError(f"Bundled Saitamaar font was not found: {FONT_PATH}")


def line_pitch(font_size: int) -> int:
    return max(
        font_size,
        round(font_size * REFERENCE_LINE_PITCH / REFERENCE_FONT_SIZE),
    )


@lru_cache(maxsize=8)
def load_font(font_size: int = REFERENCE_FONT_SIZE) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(find_font()), font_size)


def glyph_advance(char: str, font_size: int = REFERENCE_FONT_SIZE) -> int:
    """Return the integer cell advance used by the AA renderer."""
    if len(char) != 1:
        raise ValueError(f"Expected one character, got {char!r}")
    return max(2, int(math.ceil(load_font(font_size).getlength(char))))


def render_glyph_mask(
    char: str,
    font_size: int = REFERENCE_FONT_SIZE,
    canvas_height: int | None = None,
) -> np.ndarray:
    """Rasterize one Saitamaar glyph as a black-pixel boolean mask."""
    font = load_font(font_size)
    width = glyph_advance(char, font_size)
    height = canvas_height if canvas_height is not None else line_pitch(font_size)
    canvas = Image.new("L", (width, height), 255)
    ascent, _ = font.getmetrics()
    ImageDraw.Draw(canvas).text((0, ascent), char, font=font, fill=0, anchor="ls")
    return np.asarray(canvas) < 128


def render_text_mask(
    text: str,
    font_size: int = REFERENCE_FONT_SIZE,
    *,
    canvas_height_per_line: int | None = None,
) -> np.ndarray:
    """Render complete text runs, independently of per-glyph composition."""
    rows = text.splitlines() or [""]
    pitch = line_pitch(font_size)
    line_canvas_height = canvas_height_per_line or pitch
    width = max(
        2,
        *(sum(glyph_advance(char, font_size) for char in row) for row in rows),
    )
    height = (len(rows) - 1) * pitch + line_canvas_height
    font = load_font(font_size)
    ascent, _ = font.getmetrics()
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        draw.text(
            (0, row_index * pitch + ascent),
            row,
            font=font,
            fill=0,
            anchor="ls",
        )
    return np.asarray(canvas) < 128
