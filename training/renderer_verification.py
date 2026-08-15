from __future__ import annotations

import csv
import ctypes
import hashlib
import math
import os
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from backend.rendering import (
    FONT_PATH,
    glyph_advance,
    load_font,
    render_glyph_mask,
    render_text_mask,
)
from training.reconstruction import (
    GLYPH_HEIGHT,
    LINE_PITCH,
    Placement,
    load_placements,
    render_work,
)


SCHEMA_VERSION = 1
MAX_RECORDED_FAILURES = 25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_characters(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="cp932", newline="") as handle:
        rows = csv.DictReader(handle)
        return tuple(row["char"] for row in rows if int(row["frequency"]) >= 10)


def placements_from_text(
    text: str,
    glyphs: dict[str, np.ndarray],
    *,
    file_name: str,
) -> list[Placement]:
    """Rebuild coordinates using only text order and the glyph advance table."""
    placements: list[Placement] = []
    for row_index, row in enumerate(text.splitlines()):
        x = 0
        for char in row:
            placements.append(
                Placement(
                    file_name=file_name,
                    y=row_index * LINE_PITCH,
                    x=x,
                    char=char,
                    label=-1,
                )
            )
            x += glyphs[char].shape[1]
    return placements


@dataclass(frozen=True)
class GdiTextMetrics:
    width: int
    height: int


class WindowsGdiMeasurer(AbstractContextManager["WindowsGdiMeasurer"]):
    """Measure Saitamaar with the classic Windows GDI text-width path."""

    FR_PRIVATE = 0x10
    SHIFTJIS_CHARSET = 128
    OUT_TT_PRECIS = 4
    NONANTIALIASED_QUALITY = 3

    class _Size(ctypes.Structure):
        _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]

    def __init__(self, font_path: Path, font_size_px: int = GLYPH_HEIGHT) -> None:
        if os.name != "nt":
            raise RuntimeError("Windows GDI metrics are only available on Windows.")
        self.font_path = font_path.resolve()
        self.font_size_px = font_size_px
        self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._configure_signatures()
        self._dc: int | None = None
        self._font: int | None = None
        self._previous: int | None = None
        self._registered = False

    def _configure_signatures(self) -> None:
        gdi32 = self._gdi32
        handle = wintypes.HANDLE
        gdi32.AddFontResourceExW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        gdi32.AddFontResourceExW.restype = ctypes.c_int
        gdi32.RemoveFontResourceExW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        gdi32.RemoveFontResourceExW.restype = wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateFontW.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPCWSTR,
        ]
        gdi32.CreateFontW.restype = handle
        gdi32.SelectObject.argtypes = [wintypes.HDC, handle]
        gdi32.SelectObject.restype = handle
        gdi32.DeleteObject.argtypes = [handle]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL
        gdi32.GetTextExtentPoint32W.argtypes = [
            wintypes.HDC,
            wintypes.LPCWSTR,
            ctypes.c_int,
            ctypes.POINTER(self._Size),
        ]
        gdi32.GetTextExtentPoint32W.restype = wintypes.BOOL

    def __enter__(self) -> "WindowsGdiMeasurer":
        path = str(self.font_path)
        if self._gdi32.AddFontResourceExW(path, self.FR_PRIVATE, None) <= 0:
            raise OSError(ctypes.get_last_error(), f"Could not register {path}")
        self._registered = True
        self._dc = self._gdi32.CreateCompatibleDC(None)
        if not self._dc:
            raise OSError(ctypes.get_last_error(), "Could not create a GDI DC")
        family = load_font(self.font_size_px).getname()[0]
        self._font = self._gdi32.CreateFontW(
            -self.font_size_px,
            0,
            0,
            0,
            400,
            0,
            0,
            0,
            self.SHIFTJIS_CHARSET,
            self.OUT_TT_PRECIS,
            0,
            self.NONANTIALIASED_QUALITY,
            0,
            family,
        )
        if not self._font:
            raise OSError(ctypes.get_last_error(), f"Could not create GDI font {family}")
        self._previous = self._gdi32.SelectObject(self._dc, self._font)
        if not self._previous:
            raise OSError(ctypes.get_last_error(), "Could not select the GDI font")
        return self

    def measure(self, text: str) -> GdiTextMetrics:
        if self._dc is None:
            raise RuntimeError("GDI measurer is not open.")
        size = self._Size()
        if not self._gdi32.GetTextExtentPoint32W(
            self._dc,
            text,
            len(text),
            ctypes.byref(size),
        ):
            raise OSError(ctypes.get_last_error(), f"Could not measure {text!r}")
        return GdiTextMetrics(width=int(size.cx), height=int(size.cy))

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._dc and self._previous:
            self._gdi32.SelectObject(self._dc, self._previous)
        if self._font:
            self._gdi32.DeleteObject(self._font)
        if self._dc:
            self._gdi32.DeleteDC(self._dc)
        if self._registered:
            self._gdi32.RemoveFontResourceExW(
                str(self.font_path), self.FR_PRIVATE, None
            )
        self._dc = None
        self._font = None
        self._previous = None
        self._registered = False


def _record_failure(
    failures: list[dict[str, object]],
    counts: Counter[str],
    kind: str,
    **details: object,
) -> None:
    counts[kind] += 1
    if len(failures) < MAX_RECORDED_FAILURES:
        failures.append({"kind": kind, **details})


def _first_pixel_difference(
    expected: np.ndarray,
    actual: np.ndarray,
) -> tuple[int, int] | None:
    if expected.shape != actual.shape:
        return None
    positions = np.argwhere(expected != actual)
    if not len(positions):
        return None
    y, x = positions[0]
    return int(x), int(y)


def _placement_at_pixel(
    placements: Iterable[Placement],
    pixel: tuple[int, int] | None,
) -> dict[str, object] | None:
    if pixel is None:
        return None
    x, y = pixel
    row_y = (y // LINE_PITCH) * LINE_PITCH
    row = sorted(
        (item for item in placements if item.y == row_y),
        key=lambda item: item.x,
    )
    for index, item in enumerate(row):
        if item.x <= x < item.x + glyph_advance(item.char):
            return {
                "index": index,
                "char": item.char,
                "codepoint": f"U+{ord(item.char):04X}",
                "x": item.x,
                "y": item.y,
            }
    return None


def _render_positioned_placements(placements: Iterable[Placement]) -> np.ndarray:
    ordered = sorted(placements, key=lambda item: (item.y, item.x))
    if not ordered:
        raise ValueError("Cannot render a work without placements.")
    width = max(item.x + glyph_advance(item.char) for item in ordered)
    height = max(item.y for item in ordered) + LINE_PITCH
    font = load_font(GLYPH_HEIGHT)
    ascent, _ = font.getmetrics()
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    for item in ordered:
        draw.text(
            (item.x, item.y + ascent),
            item.char,
            font=font,
            fill=0,
            anchor="ls",
        )
    return np.asarray(canvas) < 128


def verify_renderer(
    dataset_path: Path,
    charset_path: Path,
    *,
    include_gdi: bool = True,
    selected_names: set[str] | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    charset_path = charset_path.resolve()
    works, duplicate_count = load_placements(dataset_path)
    if selected_names is not None:
        missing = selected_names - works.keys()
        if missing:
            raise ValueError(f"Unknown work names: {', '.join(sorted(missing))}")
        works = {name: works[name] for name in sorted(selected_names)}

    model_characters = load_model_characters(charset_path)
    characters = {
        placement.char for placements in works.values() for placement in placements
    }
    glyphs = {
        char: render_glyph_mask(char, GLYPH_HEIGHT, GLYPH_HEIGHT)
        for char in characters
    }
    failures: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    right_edge_differences: Counter[int] = Counter()
    full_run_differences: Counter[int] = Counter()
    gdi_character_differences: list[dict[str, object]] = []
    gdi_row_differences: Counter[int] = Counter()
    gdi_extent_heights: Counter[int] = Counter()

    gdi: WindowsGdiMeasurer | None = None
    if include_gdi and os.name == "nt":
        gdi = WindowsGdiMeasurer(FONT_PATH)
        gdi.__enter__()

    try:
        if gdi is not None:
            for char in sorted(characters):
                metrics = gdi.measure(char)
                gdi_width = metrics.width
                gdi_extent_heights[metrics.height] += 1
                pillow_width = glyph_advance(char)
                if gdi_width != pillow_width:
                    gdi_character_differences.append(
                        {
                            "char": char,
                            "codepoint": f"U+{ord(char):04X}",
                            "pillow": pillow_width,
                            "gdi": gdi_width,
                        }
                    )

        for file_name, placements in works.items():
            totals["works"] += 1
            totals["placements"] += len(placements)
            rows: dict[int, list[Placement]] = defaultdict(list)
            for placement in placements:
                rows[placement.y].append(placement)
                if placement.y % LINE_PITCH:
                    _record_failure(
                        failures,
                        failure_counts,
                        "line_pitch",
                        file_name=file_name,
                        char=placement.char,
                        x=placement.x,
                        y=placement.y,
                    )
                if placement.label < len(model_characters):
                    totals["model_labeled_placements"] += 1
                    expected_char = model_characters[placement.label]
                    if placement.char != expected_char:
                        _record_failure(
                            failures,
                            failure_counts,
                            "label_character",
                            file_name=file_name,
                            label=placement.label,
                            expected=expected_char,
                            actual=placement.char,
                            x=placement.x,
                            y=placement.y,
                        )

            for y, row in rows.items():
                totals["rows"] += 1
                row.sort(key=lambda item: item.x)
                expected_x = 0
                advance_failure_recorded = False
                for item in row:
                    if item.x != expected_x and not advance_failure_recorded:
                        _record_failure(
                            failures,
                            failure_counts,
                            "character_advance",
                            file_name=file_name,
                            char=item.char,
                            x=item.x,
                            expected_x=expected_x,
                            y=y,
                        )
                        advance_failure_recorded = True
                    expected_x += glyphs[item.char].shape[1]
                actual_right = row[-1].x + glyphs[row[-1].char].shape[1]
                right_edge_differences[actual_right - expected_x] += 1
                row_text = "".join(item.char for item in row)
                full_run_width = int(math.ceil(load_font().getlength(row_text)))
                full_run_differences[full_run_width - expected_x] += 1
                if gdi is not None:
                    metrics = gdi.measure(row_text)
                    gdi_row_differences[metrics.width - expected_x] += 1
                    gdi_extent_heights[metrics.height] += 1

            _, text = render_work(placements, glyphs)
            placement_image = _render_positioned_placements(placements)
            inverse_placements = placements_from_text(
                text,
                glyphs,
                file_name=file_name,
            )
            _, inverse_text = render_work(inverse_placements, glyphs)
            inverse_image = _render_positioned_placements(inverse_placements)
            if text != inverse_text or not np.array_equal(placement_image, inverse_image):
                first = _first_pixel_difference(placement_image, inverse_image)
                _record_failure(
                    failures,
                    failure_counts,
                    "inverse_render",
                    file_name=file_name,
                    expected_shape=list(placement_image.shape),
                    actual_shape=list(inverse_image.shape),
                    first_pixel=list(first) if first else None,
                    first_character=_placement_at_pixel(placements, first),
                )

            direct_image = render_text_mask(
                text,
                GLYPH_HEIGHT,
                canvas_height_per_line=LINE_PITCH,
            )
            if not np.array_equal(placement_image, direct_image):
                first = _first_pixel_difference(placement_image, direct_image)
                _record_failure(
                    failures,
                    failure_counts,
                    "full_text_render",
                    file_name=file_name,
                    expected_shape=list(placement_image.shape),
                    actual_shape=list(direct_image.shape),
                    first_pixel=list(first) if first else None,
                    first_character=_placement_at_pixel(inverse_placements, first),
                )
    finally:
        if gdi is not None:
            gdi.__exit__(None, None, None)

    blank_glyphs = sorted(
        char for char, glyph in glyphs.items() if not bool(np.any(glyph))
    )
    nonzero_failures = dict(sorted(failure_counts.items()))
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "gate_passed": not nonzero_failures and not gdi_character_differences,
        "source": {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "charset": str(charset_path),
            "charset_sha256": _sha256(charset_path),
            "font": str(FONT_PATH.resolve()),
            "font_sha256": _sha256(FONT_PATH),
        },
        "constants": {
            "glyph_height": GLYPH_HEIGHT,
            "line_pitch": LINE_PITCH,
            "model_character_count": len(model_characters),
        },
        "totals": {
            **dict(sorted(totals.items())),
            "characters": len(characters),
            "exact_duplicate_rows_removed": duplicate_count,
            "blank_glyph_characters": blank_glyphs,
        },
        "differences": {
            "dataset_right_edge_minus_sequential": {
                str(key): value for key, value in sorted(right_edge_differences.items())
            },
            "full_run_advance_minus_sequential": {
                str(key): value for key, value in sorted(full_run_differences.items())
            },
            "gdi_advance_minus_sequential": {
                str(key): value for key, value in sorted(gdi_row_differences.items())
            },
            "gdi_character_mismatches": gdi_character_differences,
            "gdi_text_extent_heights": {
                str(key): value for key, value in sorted(gdi_extent_heights.items())
            },
        },
        "failure_counts": nonzero_failures,
        "first_failures": failures,
    }
    return report
