from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np


SurfaceSourceKind = Literal["color", "lineart"]


@dataclass(frozen=True)
class SurfaceProposalConfig:
    target_width: int = 512
    line_pitch: int = 18
    minimum_area: int = 64
    minimum_area_fraction: float = 0.0005
    maximum_area_fraction: float = 0.95
    max_proposals_per_layer: int = 96
    color_blur_sigma: float = 1.0
    color_quantizations: tuple[tuple[int, int, int], ...] = (
        (4, 4, 4),
        (6, 6, 6),
    )
    lineart_closing_kernels: tuple[int, ...] = (3, 5)
    cross_layer_min_overlap_fraction: float = 0.1

    def validate(self) -> None:
        if self.target_width <= 0 or self.line_pitch <= 0:
            raise ValueError("Surface proposal geometry must be positive")
        if self.minimum_area <= 0:
            raise ValueError("minimum_area must be positive")
        if not 0.0 <= self.minimum_area_fraction < 1.0:
            raise ValueError("minimum_area_fraction must be in [0, 1)")
        if not 0.0 < self.maximum_area_fraction <= 1.0:
            raise ValueError("maximum_area_fraction must be in (0, 1]")
        if self.minimum_area_fraction >= self.maximum_area_fraction:
            raise ValueError("minimum area fraction must be below maximum")
        if self.max_proposals_per_layer <= 0:
            raise ValueError("max_proposals_per_layer must be positive")
        if self.color_blur_sigma < 0:
            raise ValueError("color_blur_sigma cannot be negative")
        if not self.color_quantizations:
            raise ValueError("At least one Lab quantization is required")
        if any(any(level < 2 for level in levels) for levels in self.color_quantizations):
            raise ValueError("Lab quantization levels must be at least two")
        if not self.lineart_closing_kernels:
            raise ValueError("At least one line-art closing kernel is required")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.lineart_closing_kernels):
            raise ValueError("Line-art closing kernels must be positive odd numbers")
        if not 0.0 < self.cross_layer_min_overlap_fraction <= 1.0:
            raise ValueError("cross_layer_min_overlap_fraction must be in (0, 1]")


@dataclass(frozen=True)
class SurfaceProposal:
    proposal_id: str
    layer_id: str
    region_label: int
    method: str
    area_px: int
    area_fraction: float
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    perimeter_px: int
    compactness: float
    touches_domain_border: bool
    domain_border_contact_px: int
    gray_mean: float
    gray_std: float
    gray_quantiles: tuple[float, float, float]
    surrounding_gray_contrast: float | None
    surrounding_gray_delta: float | None
    lab_mean: tuple[float, float, float] | None
    surrounding_lab_delta: float | None
    adjacent_proposal_ids: tuple[str, ...]
    parameters: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceProposalLayer:
    layer_id: str
    method: str
    labels: np.ndarray
    proposals: tuple[SurfaceProposal, ...]
    diagnostics: dict[str, object]

    def to_dict(self, *, include_label_spans: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "layer_id": self.layer_id,
            "method": self.method,
            "labels_sha256": _label_sha256(self.labels),
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "diagnostics": self.diagnostics,
        }
        if include_label_spans:
            result["label_spans_yx0x1"] = encode_label_spans(self.labels)
        return result


@dataclass(frozen=True)
class SurfaceProposalSet:
    schema_version: int
    coordinate_space_id: str
    source_kind: SurfaceSourceKind
    source_image_shape_hw: tuple[int, int]
    source_crop_box_xyxy: tuple[int, int, int, int]
    image_shape_hw: tuple[int, int]
    content_shape_hw: tuple[int, int]
    layers: tuple[SurfaceProposalLayer, ...]
    cross_layer_relations: tuple[dict[str, object], ...]
    config: SurfaceProposalConfig

    @property
    def proposal_count(self) -> int:
        return sum(len(layer.proposals) for layer in self.layers)

    def to_dict(self, *, include_label_spans: bool = False) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coordinate_space_id": self.coordinate_space_id,
            "source_kind": self.source_kind,
            "source_image_shape_hw": list(self.source_image_shape_hw),
            "source_crop_box_xyxy": list(self.source_crop_box_xyxy),
            "image_shape_hw": list(self.image_shape_hw),
            "content_shape_hw": list(self.content_shape_hw),
            "proposal_count": self.proposal_count,
            "config": asdict(self.config),
            "layers": [
                layer.to_dict(include_label_spans=include_label_spans)
                for layer in self.layers
            ],
            "cross_layer_relations": list(self.cross_layer_relations),
        }


def _label_sha256(labels: np.ndarray) -> str:
    payload = np.ascontiguousarray(labels, dtype="<i4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def encode_label_spans(labels: np.ndarray) -> list[list[int]]:
    if labels.ndim != 2:
        raise ValueError("Surface labels must be two-dimensional")
    spans: list[list[int]] = []
    for y, row in enumerate(labels):
        nonzero = np.flatnonzero(row)
        if nonzero.size == 0:
            continue
        starts = nonzero[np.r_[True, np.diff(nonzero) > 1]]
        ends = nonzero[np.r_[np.diff(nonzero) > 1, True]] + 1
        for x0, x1 in zip(starts.tolist(), ends.tolist(), strict=True):
            cursor = x0
            while cursor < x1:
                label = int(row[cursor])
                end = cursor + 1
                while end < x1 and int(row[end]) == label:
                    end += 1
                spans.append([label, y, cursor, end])
                cursor = end
    return spans


def decode_label_spans(
    shape: tuple[int, int],
    spans: list[list[int]],
) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    height, width = shape
    for span in spans:
        if len(span) != 4:
            raise ValueError("Each label span must contain label, y, x0, x1")
        label, y, x0, x1 = (int(value) for value in span)
        if label <= 0 or not (0 <= y < height and 0 <= x0 < x1 <= width):
            raise ValueError("A label span is outside the target geometry")
        if np.any(labels[y, x0:x1] != 0):
            raise ValueError("Label spans overlap")
        labels[y, x0:x1] = label
    return labels


def _as_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        values = image.astype(np.float32)
        if values.size and float(values.max()) <= 1.0:
            values *= 255.0
        image = np.clip(values, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return np.rint(color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Expected a grayscale, BGR, or BGRA image, got {image.shape}")


def normalize_surface_geometry(
    image: np.ndarray,
    config: SurfaceProposalConfig,
    *,
    crop_box_xyxy: tuple[int, int, int, int] | None = None,
    target_shape_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    bgr = _as_bgr_u8(image)
    source_height, source_width = bgr.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Input image cannot be empty")
    crop = crop_box_xyxy or (0, 0, source_width, source_height)
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
        raise ValueError("Surface proposal crop must be inside the source image")
    cropped = bgr[y0:y1, x0:x1]
    crop_height, crop_width = cropped.shape[:2]
    if target_shape_hw is not None:
        target_height, target_width = target_shape_hw
        if target_height <= 0 or target_width <= 0:
            raise ValueError("Surface proposal target shape must be positive")
        scale = target_width / crop_width
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(
            cropped,
            (target_width, target_height),
            interpolation=interpolation,
        )
        valid = np.ones((target_height, target_width), dtype=bool)
        return resized, valid, (target_height, target_width)
    scale = config.target_width / crop_width
    content_height = max(1, int(round(crop_height * scale)))
    padded_height = int(math.ceil(content_height / config.line_pitch) * config.line_pitch)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        cropped,
        (config.target_width, content_height),
        interpolation=interpolation,
    )
    canvas = np.full((padded_height, config.target_width, 3), 255, dtype=np.uint8)
    canvas[:content_height] = resized
    valid = np.zeros((padded_height, config.target_width), dtype=bool)
    valid[:content_height] = True
    return canvas, valid, (content_height, config.target_width)


def _domain_border(valid: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(
        valid.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return valid & ~eroded


def _minimum_area(valid_pixels: int, config: SurfaceProposalConfig) -> int:
    return max(
        config.minimum_area,
        int(math.ceil(valid_pixels * config.minimum_area_fraction)),
    )


def _select_components(
    raw_labels: np.ndarray,
    components: list[dict[str, object]],
    *,
    max_count: int,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    selected = sorted(
        components,
        key=lambda item: (
            -int(item["area"]),
            int(item["y"]),
            int(item["x"]),
            int(item["raw_label"]),
        ),
    )[:max_count]
    selected.sort(
        key=lambda item: (
            int(item["y"]),
            int(item["x"]),
            -int(item["area"]),
            int(item["raw_label"]),
        )
    )
    labels = np.zeros(raw_labels.shape, dtype=np.int32)
    for region_label, component in enumerate(selected, start=1):
        labels[raw_labels == int(component["raw_label"])] = region_label
        component["region_label"] = region_label
    return labels, selected, max(0, len(components) - len(selected))


def _adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    result = {int(label): set() for label in np.unique(labels) if label > 0}
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        mask = (first > 0) & (second > 0) & (first != second)
        if not np.any(mask):
            continue
        pairs = np.unique(np.stack((first[mask], second[mask]), axis=1), axis=0)
        for left, right in pairs:
            result[int(left)].add(int(right))
            result[int(right)].add(int(left))
    return result


def _make_layer(
    *,
    layer_id: str,
    method: str,
    raw_labels: np.ndarray,
    components: list[dict[str, object]],
    bgr: np.ndarray,
    valid: np.ndarray,
    lab: np.ndarray | None,
    config: SurfaceProposalConfig,
    diagnostics: dict[str, object],
) -> SurfaceProposalLayer:
    labels, selected, discarded_by_limit = _select_components(
        raw_labels,
        components,
        max_count=config.max_proposals_per_layer,
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    border = _domain_border(valid)
    valid_pixels = max(1, int(np.count_nonzero(valid)))
    adjacency = _adjacency(labels)
    provisional: list[SurfaceProposal] = []
    for component in selected:
        region_label = int(component["region_label"])
        mask = labels == region_label
        ys, xs = np.nonzero(mask)
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        area = int(mask.sum())
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        perimeter = int(boundary.sum())
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ring &= valid & ~mask
        gray_values = gray[mask].astype(np.float32)
        ring_values = gray[ring].astype(np.float32)
        contrast = (
            abs(float(gray_values.mean()) - float(ring_values.mean()))
            if ring_values.size
            else None
        )
        gray_delta = (
            float(ring_values.mean()) - float(gray_values.mean())
            if ring_values.size
            else None
        )
        lab_mean = (
            tuple(float(value) for value in lab[mask].mean(axis=0))
            if lab is not None
            else None
        )
        surrounding_lab_delta = None
        if lab is not None and ring_values.size:
            ring_lab_mean = lab[ring].mean(axis=0)
            surrounding_lab_delta = float(
                np.linalg.norm(np.asarray(lab_mean) - ring_lab_mean)
            )
        proposal_id = f"{layer_id}:{region_label:04d}"
        adjacent_ids = tuple(
            f"{layer_id}:{adjacent:04d}"
            for adjacent in sorted(adjacency.get(region_label, set()))
        )
        provisional.append(
            SurfaceProposal(
                proposal_id=proposal_id,
                layer_id=layer_id,
                region_label=region_label,
                method=method,
                area_px=area,
                area_fraction=round(area / valid_pixels, 8),
                bbox_xyxy=(x0, y0, x1, y1),
                centroid_xy=(round(float(xs.mean()), 4), round(float(ys.mean()), 4)),
                perimeter_px=perimeter,
                compactness=round(
                    4.0 * math.pi * area / max(1, perimeter * perimeter),
                    6,
                ),
                touches_domain_border=bool(np.any(mask & border)),
                domain_border_contact_px=int(np.count_nonzero(mask & border)),
                gray_mean=round(float(gray_values.mean()), 4),
                gray_std=round(float(gray_values.std()), 4),
                gray_quantiles=tuple(
                    round(float(value), 4)
                    for value in np.quantile(gray_values, (0.1, 0.5, 0.9))
                ),
                surrounding_gray_contrast=(
                    round(float(contrast), 4) if contrast is not None else None
                ),
                surrounding_gray_delta=(
                    round(float(gray_delta), 4) if gray_delta is not None else None
                ),
                lab_mean=(
                    tuple(round(value, 4) for value in lab_mean)
                    if lab_mean is not None
                    else None
                ),
                surrounding_lab_delta=(
                    round(surrounding_lab_delta, 4)
                    if surrounding_lab_delta is not None
                    else None
                ),
                adjacent_proposal_ids=adjacent_ids,
                parameters=dict(component.get("parameters", {})),
            )
        )
    coverage = int(np.count_nonzero(labels))
    layer_diagnostics = {
        **diagnostics,
        "proposal_count": len(provisional),
        "discarded_by_count_limit": discarded_by_limit,
        "proposal_coverage_fraction": round(coverage / valid_pixels, 8),
        "border_touching_proposals": sum(
            proposal.touches_domain_border for proposal in provisional
        ),
    }
    return SurfaceProposalLayer(
        layer_id=layer_id,
        method=method,
        labels=labels,
        proposals=tuple(provisional),
        diagnostics=layer_diagnostics,
    )


def _color_layers(
    bgr: np.ndarray,
    valid: np.ndarray,
    config: SurfaceProposalConfig,
) -> tuple[SurfaceProposalLayer, ...]:
    if config.color_blur_sigma > 0:
        working = cv2.GaussianBlur(
            bgr,
            (0, 0),
            sigmaX=config.color_blur_sigma,
            sigmaY=config.color_blur_sigma,
        )
    else:
        working = bgr
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    valid_pixels = int(np.count_nonzero(valid))
    minimum_area = _minimum_area(valid_pixels, config)
    maximum_area = int(math.floor(valid_pixels * config.maximum_area_fraction))
    layers: list[SurfaceProposalLayer] = []
    for l_levels, a_levels, b_levels in config.color_quantizations:
        layer_id = f"lab-l{l_levels}-a{a_levels}-b{b_levels}"
        channels = lab.astype(np.int32)
        l_bin = channels[:, :, 0] * l_levels // 256
        a_bin = channels[:, :, 1] * a_levels // 256
        b_bin = channels[:, :, 2] * b_levels // 256
        quantized = (l_bin * a_levels + a_bin) * b_levels + b_bin
        raw_labels = np.zeros(valid.shape, dtype=np.int32)
        components: list[dict[str, object]] = []
        next_raw_label = 1
        rejected_small = 0
        rejected_large = 0
        for color_bin in np.unique(quantized[valid]):
            binary = ((quantized == color_bin) & valid).astype(np.uint8)
            count, local_labels, stats, _ = cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
            for local_label in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[local_label])
                if area < minimum_area:
                    rejected_small += 1
                    continue
                if area > maximum_area:
                    rejected_large += 1
                    continue
                raw_labels[local_labels == local_label] = next_raw_label
                components.append(
                    {
                        "raw_label": next_raw_label,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "area": area,
                        "parameters": {
                            "lab_quantized_bin": int(color_bin),
                            "lab_levels": [l_levels, a_levels, b_levels],
                        },
                    }
                )
                next_raw_label += 1
        layers.append(
            _make_layer(
                layer_id=layer_id,
                method="lab_connected_region",
                raw_labels=raw_labels,
                components=components,
                bgr=bgr,
                valid=valid,
                lab=lab,
                config=config,
                diagnostics={
                    "minimum_area_px": minimum_area,
                    "maximum_area_px": maximum_area,
                    "rejected_small_components": rejected_small,
                    "rejected_large_components": rejected_large,
                    "color_blur_sigma": config.color_blur_sigma,
                },
            )
        )
    return tuple(layers)


def _cross_layer_relations(
    layers: tuple[SurfaceProposalLayer, ...],
    *,
    minimum_overlap_fraction: float,
) -> tuple[dict[str, object], ...]:
    relations: list[dict[str, object]] = []
    for source, target in zip(layers, layers[1:], strict=False):
        source_areas = {
            proposal.region_label: proposal.area_px for proposal in source.proposals
        }
        target_areas = {
            proposal.region_label: proposal.area_px for proposal in target.proposals
        }
        joint = (source.labels > 0) & (target.labels > 0)
        overlaps: list[dict[str, object]] = []
        if np.any(joint):
            pairs, counts = np.unique(
                np.stack((source.labels[joint], target.labels[joint]), axis=1),
                axis=0,
                return_counts=True,
            )
            for pair, intersection in zip(pairs, counts, strict=True):
                source_label, target_label = (int(value) for value in pair)
                source_area = source_areas[source_label]
                target_area = target_areas[target_label]
                overlap_fraction = int(intersection) / min(source_area, target_area)
                if overlap_fraction < minimum_overlap_fraction:
                    continue
                overlaps.append(
                    {
                        "source_proposal_id": (
                            f"{source.layer_id}:{source_label:04d}"
                        ),
                        "target_proposal_id": (
                            f"{target.layer_id}:{target_label:04d}"
                        ),
                        "intersection_px": int(intersection),
                        "source_coverage": round(int(intersection) / source_area, 8),
                        "target_coverage": round(int(intersection) / target_area, 8),
                        "iou": round(
                            int(intersection)
                            / (source_area + target_area - int(intersection)),
                            8,
                        ),
                    }
                )
        source_degree = Counter(
            str(overlap["source_proposal_id"]) for overlap in overlaps
        )
        target_degree = Counter(
            str(overlap["target_proposal_id"]) for overlap in overlaps
        )
        relations.append(
            {
                "source_layer_id": source.layer_id,
                "target_layer_id": target.layer_id,
                "minimum_overlap_fraction": minimum_overlap_fraction,
                "overlap_count": len(overlaps),
                "source_split_count": sum(degree > 1 for degree in source_degree.values()),
                "target_merge_count": sum(degree > 1 for degree in target_degree.values()),
                "overlaps": overlaps,
            }
        )
    return tuple(relations)


def _lineart_layers(
    bgr: np.ndarray,
    valid: np.ndarray,
    config: SurfaceProposalConfig,
) -> tuple[SurfaceProposalLayer, ...]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    valid_pixels = int(np.count_nonzero(valid))
    minimum_area = _minimum_area(valid_pixels, config)
    maximum_area = int(math.floor(valid_pixels * config.maximum_area_fraction))
    _, ink = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ink[~valid] = 0
    domain_border = _domain_border(valid)
    layers: list[SurfaceProposalLayer] = []
    for kernel_size in config.lineart_closing_kernels:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        detection_ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        free = ((detection_ink == 0) & valid).astype(np.uint8)
        count, local_labels, stats, _ = cv2.connectedComponentsWithStats(
            free,
            connectivity=8,
        )
        exterior_labels = set(int(value) for value in np.unique(local_labels[domain_border]))
        exterior_labels.discard(0)
        exterior_area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in exterior_labels)
        raw_labels = np.zeros(valid.shape, dtype=np.int32)
        components: list[dict[str, object]] = []
        next_raw_label = 1
        rejected_small = 0
        rejected_large = 0
        for local_label in range(1, count):
            if local_label in exterior_labels:
                continue
            x, y, width, height, area = (int(value) for value in stats[local_label])
            if area < minimum_area:
                rejected_small += 1
                continue
            if area > maximum_area:
                rejected_large += 1
                continue
            raw_labels[local_labels == local_label] = next_raw_label
            components.append(
                {
                    "raw_label": next_raw_label,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "area": area,
                    "parameters": {"closing_kernel_px": kernel_size},
                }
            )
            next_raw_label += 1
        layer_id = f"closed-k{kernel_size}"
        layers.append(
            _make_layer(
                layer_id=layer_id,
                method="lineart_closed_region",
                raw_labels=raw_labels,
                components=components,
                bgr=bgr,
                valid=valid,
                lab=None,
                config=config,
                diagnostics={
                    "minimum_area_px": minimum_area,
                    "maximum_area_px": maximum_area,
                    "closing_kernel_px": kernel_size,
                    "external_background_components": len(exterior_labels),
                    "external_background_fraction": round(
                        exterior_area / max(1, valid_pixels),
                        8,
                    ),
                    "rejected_small_components": rejected_small,
                    "rejected_large_components": rejected_large,
                    "detection_ink_fraction": round(
                        float(np.count_nonzero(detection_ink & valid.astype(np.uint8)))
                        / max(1, valid_pixels),
                        8,
                    ),
                },
            )
        )
    return tuple(layers)


def extract_surface_proposals(
    image: np.ndarray,
    *,
    source_kind: SurfaceSourceKind,
    coordinate_space_id: str,
    config: SurfaceProposalConfig | None = None,
    crop_box_xyxy: tuple[int, int, int, int] | None = None,
    target_shape_hw: tuple[int, int] | None = None,
) -> SurfaceProposalSet:
    if not coordinate_space_id.strip():
        raise ValueError("coordinate_space_id must not be empty")
    if source_kind not in {"color", "lineart"}:
        raise ValueError(f"Unknown surface proposal source kind: {source_kind}")
    active = config or SurfaceProposalConfig()
    active.validate()
    source_height, source_width = _as_bgr_u8(image).shape[:2]
    crop = crop_box_xyxy or (0, 0, source_width, source_height)
    bgr, valid, content_shape = normalize_surface_geometry(
        image,
        active,
        crop_box_xyxy=crop,
        target_shape_hw=target_shape_hw,
    )
    layers = (
        _color_layers(bgr, valid, active)
        if source_kind == "color"
        else _lineart_layers(bgr, valid, active)
    )
    cross_layer_relations = _cross_layer_relations(
        layers,
        minimum_overlap_fraction=active.cross_layer_min_overlap_fraction,
    )
    return SurfaceProposalSet(
        schema_version=1,
        coordinate_space_id=coordinate_space_id,
        source_kind=source_kind,
        source_image_shape_hw=(source_height, source_width),
        source_crop_box_xyxy=crop,
        image_shape_hw=valid.shape,
        content_shape_hw=content_shape,
        layers=layers,
        cross_layer_relations=cross_layer_relations,
        config=active,
    )


def render_proposal_layer(
    image: np.ndarray,
    proposal_set: SurfaceProposalSet,
    layer: SurfaceProposalLayer,
) -> np.ndarray:
    bgr, _, _ = normalize_surface_geometry(
        image,
        proposal_set.config,
        crop_box_xyxy=proposal_set.source_crop_box_xyxy,
        target_shape_hw=proposal_set.image_shape_hw,
    )
    if bgr.shape[:2] != layer.labels.shape:
        raise ValueError("Proposal layer and preview image have different geometry")
    overlay = bgr.copy()
    for proposal in layer.proposals:
        label = proposal.region_label
        color = np.array(
            [
                48 + (label * 71) % 176,
                48 + (label * 113) % 176,
                48 + (label * 151) % 176,
            ],
            dtype=np.uint8,
        )
        mask = layer.labels == label
        overlay[mask] = np.rint(
            overlay[mask].astype(np.float32) * 0.45 + color.astype(np.float32) * 0.55
        ).astype(np.uint8)
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        overlay[boundary] = color
    return overlay
