from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from backend.line_ops import remove_small_components, skeletonize


SourceKind = Literal["grayscale", "lineart", "aa_proxy"]


@dataclass(frozen=True)
class ChannelExtractionConfig:
    target_width: int = 512
    line_pitch: int = 18
    target_line_width: int = 1
    line_min_area: int = 4
    gap_kernel: int = 3
    tone_window_width: int = 48
    tone_window_height: int = 54
    tone_sigma: float = 12.0
    tone_gamma: float = 1.0
    tone_levels: int = 4
    dark_threshold: int | None = None
    fill_erosion_radius: int = 2
    fill_min_area: int = 16
    fill_max_hole: int = 16

    def validate(self) -> None:
        if self.target_width <= 0:
            raise ValueError("target_width must be positive")
        if self.line_pitch <= 0:
            raise ValueError("line_pitch must be positive")
        if self.target_line_width <= 0:
            raise ValueError("target_line_width must be positive")
        if self.line_min_area <= 0 or self.fill_min_area <= 0:
            raise ValueError("component areas must be positive")
        if self.gap_kernel <= 0 or self.gap_kernel % 2 == 0:
            raise ValueError("gap_kernel must be a positive odd number")
        if self.tone_window_width <= 0 or self.tone_window_height <= 0:
            raise ValueError("tone windows must be positive")
        if self.tone_sigma < 0:
            raise ValueError("tone_sigma cannot be negative")
        if self.tone_gamma <= 0:
            raise ValueError("tone_gamma must be positive")
        if self.tone_levels < 2:
            raise ValueError("tone_levels must be at least two")
        if self.dark_threshold is not None and not 0 <= self.dark_threshold <= 255:
            raise ValueError("dark_threshold must be in the 0-255 range")
        if self.fill_erosion_radius <= 0:
            raise ValueError("fill_erosion_radius must be positive")
        if self.fill_max_hole < 0:
            raise ValueError("fill_max_hole cannot be negative")


@dataclass(frozen=True)
class ExtractedChannels:
    resized_gray: np.ndarray
    structure: np.ndarray
    tone: np.ndarray
    fill: np.ndarray
    metadata: dict[str, object]

    @property
    def stacked(self) -> np.ndarray:
        return np.stack((self.structure, self.tone, self.fill)).astype(
            np.float32,
            copy=False,
        )


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {image.shape}")
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale or color image, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    if values.size and float(values.max()) <= 1.0:
        values *= 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def normalize_geometry(
    image: np.ndarray,
    config: ChannelExtractionConfig,
) -> np.ndarray:
    """Resize once and pad to the shared AA line grid for every channel."""
    gray = _as_gray_u8(image)
    source_height, source_width = gray.shape
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Input image cannot be empty")
    scale = config.target_width / source_width
    target_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        gray,
        (config.target_width, target_height),
        interpolation=interpolation,
    )
    padded_height = int(math.ceil(target_height / config.line_pitch) * config.line_pitch)
    if padded_height == target_height:
        return resized
    padded = np.full((padded_height, config.target_width), 255, dtype=np.uint8)
    padded[:target_height] = resized
    return padded


def _otsu_foreground(strength: np.ndarray) -> np.ndarray:
    if not np.any(strength):
        return np.zeros_like(strength, dtype=np.uint8)
    return cv2.threshold(
        strength,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]


def estimate_stroke_width(binary_ink: np.ndarray) -> float:
    if not np.any(binary_ink):
        return 0.0
    distance = cv2.distanceTransform(binary_ink, cv2.DIST_L2, 5)
    ridges = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    widths = distance[ridges] * 2.0
    return float(np.median(widths)) if widths.size else 0.0


def _raw_structure(gray: np.ndarray, source_kind: SourceKind) -> np.ndarray:
    if source_kind in {"lineart", "aa_proxy"}:
        return 255 - gray
    if source_kind != "grayscale":
        raise ValueError(f"Unknown source kind: {source_kind}")
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8, sigmaY=0.8)
    laplacian = np.abs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
    maximum = float(laplacian.max())
    if maximum <= 0:
        return np.zeros_like(gray)
    return np.clip(laplacian * (255.0 / maximum), 0, 255).astype(np.uint8)


def normalize_structure(
    raw_structure: np.ndarray,
    config: ChannelExtractionConfig,
) -> tuple[np.ndarray, float]:
    """Shared final line normalization N for AA proxies and real images."""
    binary = _otsu_foreground(_as_gray_u8(raw_structure))
    measured_width = estimate_stroke_width(binary)
    kernel = np.ones((config.gap_kernel, config.gap_kernel), np.uint8)
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    connected = remove_small_components(connected, config.line_min_area)
    normalized = skeletonize(connected)
    if config.target_line_width > 1:
        size = config.target_line_width if config.target_line_width % 2 else config.target_line_width + 1
        normalized = cv2.dilate(
            normalized,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
    return normalized.astype(np.float32) / 255.0, measured_width


def extract_tone(
    gray: np.ndarray,
    config: ChannelExtractionConfig,
) -> np.ndarray:
    inverted = (255 - gray).astype(np.float32) / 255.0
    density = cv2.boxFilter(
        inverted,
        cv2.CV_32F,
        (config.tone_window_width, config.tone_window_height),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    if config.tone_sigma > 0:
        density = cv2.GaussianBlur(
            density,
            (0, 0),
            sigmaX=config.tone_sigma,
            sigmaY=config.tone_sigma,
        )
    if config.tone_gamma != 1.0:
        density = np.power(np.clip(density, 0.0, 1.0), config.tone_gamma)
    steps = config.tone_levels - 1
    return (np.rint(np.clip(density, 0.0, 1.0) * steps) / steps).astype(np.float32)


def _reconstruct_by_dilation(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    marker = seed.copy()
    kernel = np.ones((3, 3), np.uint8)
    while True:
        expanded = cv2.bitwise_and(cv2.dilate(marker, kernel), mask)
        if np.array_equal(expanded, marker):
            return marker
        marker = expanded


def _remove_small_holes(binary: np.ndarray, maximum_area: int) -> np.ndarray:
    if maximum_area <= 0:
        return binary
    background = cv2.bitwise_not(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)
    result = binary.copy()
    height, width = binary.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border and int(area) <= maximum_area:
            result[labels == label] = 255
    return result


def extract_fill(
    gray: np.ndarray,
    config: ChannelExtractionConfig,
) -> tuple[np.ndarray, int]:
    if config.dark_threshold is None:
        threshold, dark = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        threshold_value = int(round(float(threshold)))
    else:
        threshold_value = config.dark_threshold
        dark = (gray < threshold_value).astype(np.uint8) * 255
    radius = config.fill_erosion_radius
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
    )
    seed = cv2.erode(dark, erosion_kernel)
    reconstructed = _reconstruct_by_dilation(seed, dark)
    # Unbounded geodesic reconstruction restores every thin filament connected to
    # a valid dark core.  Such filaments are linework, not fill.  Restrict the
    # restored mask to one erosion-radius around the core so that we recover the
    # boundary of a thick region without flooding an attached stroke.
    core_support = cv2.dilate(seed, erosion_kernel)
    reconstructed = cv2.bitwise_and(reconstructed, core_support)
    reconstructed = remove_small_components(reconstructed, config.fill_min_area)
    reconstructed = _remove_small_holes(reconstructed, config.fill_max_hole)
    return reconstructed.astype(np.float32) / 255.0, threshold_value


def extract_input_channels(
    image: np.ndarray,
    *,
    source_kind: SourceKind,
    config: ChannelExtractionConfig | None = None,
) -> ExtractedChannels:
    active = config or ChannelExtractionConfig()
    active.validate()
    gray = normalize_geometry(image, active)
    raw_structure = _raw_structure(gray, source_kind)
    structure, measured_width = normalize_structure(raw_structure, active)
    tone = extract_tone(gray, active)
    fill, dark_threshold = extract_fill(gray, active)
    if structure.shape != tone.shape or tone.shape != fill.shape:
        raise AssertionError("X channels must share one geometry")
    tone_counts = np.bincount(
        np.rint(tone * (active.tone_levels - 1)).astype(np.uint8).ravel(),
        minlength=active.tone_levels,
    )
    pixel_count = int(gray.size)
    return ExtractedChannels(
        resized_gray=gray,
        structure=structure,
        tone=tone,
        fill=fill,
        metadata={
            "source_kind": source_kind,
            "shape": [int(gray.shape[0]), int(gray.shape[1])],
            "measured_input_stroke_width": round(measured_width, 4),
            "dark_threshold": dark_threshold,
            "channel_statistics": {
                "structure_fraction": round(float(np.count_nonzero(structure)) / pixel_count, 6),
                "tone_level_fractions": [
                    round(float(count) / pixel_count, 6) for count in tone_counts
                ],
                "fill_fraction": round(float(np.count_nonzero(fill)) / pixel_count, 6),
            },
            "config": asdict(active),
        },
    )


def channel_preview(channel: np.ndarray) -> np.ndarray:
    return np.clip(255.0 - channel * 255.0, 0, 255).astype(np.uint8)


def composite_preview(channels: ExtractedChannels) -> np.ndarray:
    structure = channels.structure >= 0.5
    fill = channels.fill >= 0.5
    visible = np.logical_xor(structure, fill).astype(np.float32)
    return channel_preview(visible)
