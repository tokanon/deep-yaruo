from __future__ import annotations

import numpy as np


CONTEXT_SIZE = 64
CONTEXT_MARGIN = 23


def surface_input_channels(
    surface_ids: np.ndarray,
    surface_tone: np.ndarray,
) -> np.ndarray:
    """Return membership, ID-boundary, and tone without encoding ID order."""
    ids = np.asarray(surface_ids)
    tone = np.asarray(surface_tone)
    if ids.ndim != 2 or tone.shape != ids.shape:
        raise ValueError("surface IDs and tone must be equally shaped 2D arrays")
    membership = ids > 0
    boundary = np.zeros_like(membership)
    boundary[1:] |= membership[1:] & (ids[1:] != ids[:-1])
    boundary[:-1] |= membership[:-1] & (ids[:-1] != ids[1:])
    boundary[:, 1:] |= membership[:, 1:] & (ids[:, 1:] != ids[:, :-1])
    boundary[:, :-1] |= membership[:, :-1] & (ids[:, :-1] != ids[:, 1:])
    normalized_tone = np.where(membership, tone.astype(np.float32) / 255.0, 0.0)
    return np.stack(
        (membership.astype(np.float32), boundary.astype(np.float32), normalized_tone),
        axis=0,
    )


def real_image_input_channels(
    line_image: np.ndarray,
    surface_ids: np.ndarray,
    surface_tone: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map P6R line/proposal outputs to the exact L/LS tensor contract."""
    line = np.asarray(line_image)
    ids = np.asarray(surface_ids)
    tone = np.asarray(surface_tone)
    if line.ndim != 2 or ids.shape != line.shape or tone.shape != line.shape:
        raise ValueError("real line, surface IDs, and tone must share one 2D shape")
    normalized_line = np.clip(line.astype(np.float32) / 255.0, 0.0, 1.0)
    return normalized_line[np.newaxis], surface_input_channels(ids, tone)


def extract_channel_window(
    channels: np.ndarray,
    *,
    x: int,
    y: int,
    fill_value: float,
) -> np.ndarray:
    value = np.asarray(channels)
    if value.ndim == 2:
        value = value[np.newaxis]
    if value.ndim != 3:
        raise ValueError("channels must be a 2D or CHW array")
    window = np.full(
        (value.shape[0], CONTEXT_SIZE, CONTEXT_SIZE),
        fill_value,
        dtype=np.float32,
    )
    left = int(x) - CONTEXT_MARGIN
    top = int(y) - CONTEXT_MARGIN
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(value.shape[2], left + CONTEXT_SIZE)
    source_bottom = min(value.shape[1], top + CONTEXT_SIZE)
    if source_left >= source_right or source_top >= source_bottom:
        return window
    destination_left = source_left - left
    destination_top = source_top - top
    window[
        :,
        destination_top : destination_top + source_bottom - source_top,
        destination_left : destination_left + source_right - source_left,
    ] = value[:, source_top:source_bottom, source_left:source_right]
    return window
