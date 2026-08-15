from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from typing import Iterable


SERIES_SUFFIX = re.compile(r"(?:[_-]?\d+)+$")


def series_name(file_name: str) -> str:
    """Return the source-series prefix encoded in a DeepAA file name."""
    prefix = SERIES_SUFFIX.sub("", file_name).rstrip("_-")
    return prefix or file_name


def assign_work_splits(
    file_names: Iterable[str],
    *,
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    names = sorted(file_names)
    random.Random(seed).shuffle(names)
    test_count = round(len(names) * test_fraction)
    validation_count = round(len(names) * validation_fraction)
    if test_count + validation_count >= len(names):
        raise ValueError("Validation and test splits leave no training works.")
    test_names = set(names[:test_count])
    validation_names = set(names[test_count : test_count + validation_count])
    return {
        name: (
            "test"
            if name in test_names
            else "validation"
            if name in validation_names
            else "train"
        )
        for name in sorted(names)
    }


def assign_series_splits(
    file_names: Iterable[str],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign complete filename-derived series to the least imbalanced split."""
    names = sorted(file_names)
    series_sizes = Counter(series_name(name) for name in names)
    series = sorted(series_sizes)
    if len(series) < 3:
        raise ValueError("At least three source series are required for a three-way split.")

    train_fraction = 1.0 - validation_fraction - test_fraction
    if train_fraction <= 0:
        raise ValueError("Validation and test fractions leave no training works.")
    split_names = ("train", "validation", "test")
    total = len(names)
    targets = {
        "train": total * train_fraction,
        "validation": total * validation_fraction,
        "test": total * test_fraction,
    }
    best: tuple[float, tuple[str, ...]] | None = None
    for assignment in itertools.product(split_names, repeat=len(series)):
        if set(assignment) != set(split_names):
            continue
        counts = Counter()
        for group, split in zip(series, assignment):
            counts[split] += series_sizes[group]
        score = sum((counts[split] - targets[split]) ** 2 for split in split_names)
        candidate = (score, assignment)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("Could not assign source series to all three splits.")

    series_assignment = dict(zip(series, best[1]))
    return {name: series_assignment[series_name(name)] for name in names}
