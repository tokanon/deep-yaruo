from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum:
            cleaned[labels == label] = 255
    return cleaned


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Reduce connected white line regions to one-pixel representative strokes."""
    work = binary > 0

    def deletion_mask(image: np.ndarray, second_step: bool) -> np.ndarray:
        padded = np.pad(image, 1, mode="constant")
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]
        neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
        neighbor_count = sum(item.astype(np.uint8) for item in neighbors)
        transitions = sum(
            np.logical_and(~current, following).astype(np.uint8)
            for current, following in zip(neighbors, neighbors[1:] + neighbors[:1])
        )
        if second_step:
            corner_a = p2 & p4 & p8
            corner_b = p2 & p6 & p8
        else:
            corner_a = p2 & p4 & p6
            corner_b = p4 & p6 & p8
        return (
            image
            & (neighbor_count >= 2)
            & (neighbor_count <= 6)
            & (transitions == 1)
            & ~corner_a
            & ~corner_b
        )

    while True:
        first = deletion_mask(work, False)
        work = work & ~first
        second = deletion_mask(work, True)
        work = work & ~second
        if not first.any() and not second.any():
            break
    return work.astype(np.uint8) * 255

