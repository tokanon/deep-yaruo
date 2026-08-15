from __future__ import annotations

import bisect
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from onnx import numpy_helper
import onnx
from torch import nn
from torch.utils.data import Dataset, Sampler

from backend.surface_channels import (
    extract_channel_window,
    real_image_input_channels,
    surface_input_channels,
)


LINE_PITCH = 18
ROLE_LINE = 0
ROLE_FILL = 1
SUPPORT_LINE = 0
SUPPORT_SUPPORTED_FILL = 1
SUPPORT_LOW_FILL = 2


@dataclass(frozen=True)
class ReverseWork:
    entry_id: int
    split: str
    category: str
    channels_path: Path
    surfaces_path: Path
    glyph_count: int


@dataclass
class LoadedWork:
    line: np.ndarray
    surface: np.ndarray
    codepoints: np.ndarray
    rows: np.ndarray
    starts: np.ndarray
    row_offsets: np.ndarray
    fill_owned: np.ndarray
    support: np.ndarray


def load_reverse_records(root: Path, split: str) -> list[ReverseWork]:
    result: list[ReverseWork] = []
    records_path = root / "records.jsonl"
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] != split:
            continue
        artifacts = record["artifacts"]
        result.append(
            ReverseWork(
                entry_id=int(record["entry_id"]),
                split=record["split"],
                category=record["category"],
                channels_path=root / artifacts["channels"]["path"],
                surfaces_path=root / artifacts["surfaces"]["path"],
                glyph_count=int(record["glyph_count"]),
            )
        )
    return result


def training_vocabulary(root: Path) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for work in load_reverse_records(root, "train"):
        with np.load(work.channels_path) as arrays:
            for codepoint in arrays["glyph_codepoints"]:
                character = chr(int(codepoint))
                counts[character] = counts.get(character, 0) + 1
    # Codepoint order is deterministic and does not promote frequent classes.
    characters = sorted(counts)
    return characters, counts


def _supported_motifs(root: Path) -> set[str]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return {
        item["motif"]
        for item in manifest["motif_catalogs"]["b-periodic"]
        if item["supported"]
    }


def _support_from_runs(
    glyph_count: int,
    row_offsets: np.ndarray,
    surfaces_payload: dict[str, object],
    supported_motifs: set[str],
) -> np.ndarray:
    labels = np.full(glyph_count, SUPPORT_LINE, dtype=np.int8)
    for run in surfaces_payload["candidates"]["b-periodic"]["runs"]:
        row = int(run["line_index"])
        start = int(row_offsets[row]) + int(run["start_index"])
        end = int(row_offsets[row]) + int(run["end_index"])
        if any(motif in supported_motifs for motif in run["motifs"]):
            labels[start:end] = SUPPORT_SUPPORTED_FILL
        else:
            unresolved = labels[start:end] == SUPPORT_LINE
            labels[start:end][unresolved] = SUPPORT_LOW_FILL
    return labels


class ReverseChannelDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Map-style glyph/start dataset backed by the audited B decomposition."""

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        characters: Sequence[str],
        negative_stride: int = 4,
        seed: int = 42,
        cache_size: int = 4,
    ) -> None:
        if negative_stride < 1:
            raise ValueError("negative_stride must be positive")
        self.root = root
        self.works = load_reverse_records(root, split)
        self.characters = tuple(characters)
        self.class_by_character = {character: index for index, character in enumerate(characters)}
        self.negative_stride = negative_stride
        self.seed = seed
        self.cache_size = cache_size
        self.supported_motifs = _supported_motifs(root)
        self.offsets = [0]
        for work in self.works:
            self.offsets.append(self.offsets[-1] + work.glyph_count)
        self.positive_count = self.offsets[-1]
        self.negative_count = math.ceil(self.positive_count / negative_stride)
        self._cache: OrderedDict[int, LoadedWork] = OrderedDict()

    def __len__(self) -> int:
        return self.positive_count + self.negative_count

    def work_index_for_positive(self, positive_index: int) -> int:
        if not 0 <= positive_index < self.positive_count:
            raise IndexError(positive_index)
        return bisect.bisect_right(self.offsets, positive_index) - 1

    def _load_work(self, work_index: int) -> LoadedWork:
        cached = self._cache.pop(work_index, None)
        if cached is not None:
            self._cache[work_index] = cached
            return cached
        work = self.works[work_index]
        with np.load(work.channels_path) as arrays:
            line = 1.0 - arrays["b_line_mask"].astype(np.float32)
            surface = surface_input_channels(
                arrays["b_surface_ids"], arrays["b_surface_tone"]
            )
            codepoints = arrays["glyph_codepoints"].copy()
            rows = arrays["glyph_rows"].copy()
            starts = arrays["glyph_x_starts"].copy()
            row_offsets = arrays["row_offsets"].copy()
            fill_owned = arrays["b_fill_owned"].copy()
        surfaces = json.loads(work.surfaces_path.read_text(encoding="utf-8"))
        loaded = LoadedWork(
            line=line,
            surface=surface,
            codepoints=codepoints,
            rows=rows,
            starts=starts,
            row_offsets=row_offsets,
            fill_owned=fill_owned,
            support=_support_from_runs(
                len(codepoints), row_offsets, surfaces, self.supported_motifs
            ),
        )
        if not np.array_equal(loaded.fill_owned > 0, loaded.support > SUPPORT_LINE):
            raise ValueError(f"B fill/support labels disagree for entry {work.entry_id}")
        self._cache[work_index] = loaded
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def _negative_x(self, loaded: LoadedWork, local_index: int) -> int:
        row = int(loaded.rows[local_index])
        row_start = int(loaded.row_offsets[row])
        row_end = int(loaded.row_offsets[row + 1])
        starts = loaded.starts[row_start:row_end]
        origin = int(loaded.starts[local_index])
        offsets = [-5, -4, -3, 3, 4, 5]
        rng = random.Random(self.seed * 1_000_003 + local_index)
        rng.shuffle(offsets)
        for delta in offsets:
            candidate = origin + delta
            if candidate >= 0 and not np.any(np.abs(starts.astype(np.int64) - candidate) <= 1):
                return candidate
        row_limit = max(origin + 16, int(starts.max(initial=0)) + 6)
        for candidate in range(row_limit + 1):
            if not np.any(np.abs(starts.astype(np.int64) - candidate) <= 1):
                return candidate
        raise RuntimeError("row has no valid negative start location")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        positive = index < self.positive_count
        positive_index = index if positive else (index - self.positive_count) * self.negative_stride
        positive_index = min(positive_index, self.positive_count - 1)
        work_index = self.work_index_for_positive(positive_index)
        local = positive_index - self.offsets[work_index]
        loaded = self._load_work(work_index)
        x = int(loaded.starts[local]) if positive else self._negative_x(loaded, local)
        y = int(loaded.rows[local]) * LINE_PITCH
        line = extract_channel_window(loaded.line, x=x, y=y, fill_value=1.0)
        surface = extract_channel_window(loaded.surface, x=x, y=y, fill_value=0.0)
        character = chr(int(loaded.codepoints[local]))
        character_target = self.class_by_character.get(character, -1) if positive else -1
        role = int(loaded.fill_owned[local]) if positive else -1
        support = int(loaded.support[local]) if positive else -1
        return (
            torch.from_numpy(line),
            torch.from_numpy(surface),
            torch.tensor(character_target, dtype=torch.long),
            torch.tensor(float(positive), dtype=torch.float32),
            torch.tensor(role, dtype=torch.long),
            torch.tensor(support, dtype=torch.long),
            torch.tensor(int(loaded.codepoints[local]) if positive else -1, dtype=torch.long),
        )


class WorkGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle works and examples while keeping NPZ access work-local."""

    def __init__(
        self,
        dataset: ReverseChannelDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        maximum_samples: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.maximum_samples = maximum_samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        work_indices = list(range(len(self.dataset.works)))
        if self.shuffle:
            rng.shuffle(work_indices)
        emitted = 0
        for work_index in work_indices:
            start = self.dataset.offsets[work_index]
            end = self.dataset.offsets[work_index + 1]
            indices = list(range(start, end))
            first_negative = math.ceil(start / self.dataset.negative_stride)
            last_negative = math.ceil(end / self.dataset.negative_stride)
            indices.extend(
                self.dataset.positive_count + negative_index
                for negative_index in range(first_negative, last_negative)
                if negative_index * self.dataset.negative_stride < end
            )
            if self.shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset : offset + self.batch_size]
                if self.maximum_samples is not None:
                    remaining = self.maximum_samples - emitted
                    if remaining <= 0:
                        return
                    batch = batch[:remaining]
                if batch:
                    emitted += len(batch)
                    yield batch

    def __len__(self) -> int:
        count = len(self.dataset)
        if self.maximum_samples is not None:
            count = min(count, self.maximum_samples)
        return math.ceil(count / self.batch_size)


class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        channels = (input_channels, 16, 32, 64, 128)
        self.convolutions = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], 3, padding=1) for i in range(4)
        )
        self.normalizations = nn.ModuleList(
            nn.BatchNorm2d(channel, eps=0.001) for channel in channels[1:]
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            value = self.pool(torch.relu(normalization(convolution(value))))
        return value.permute(0, 2, 3, 1).contiguous().view(value.shape[0], -1)


class DeepAASurfaceV0(nn.Module):
    def __init__(self, class_count: int, *, variant: str) -> None:
        super().__init__()
        if variant not in {"L", "LS"}:
            raise ValueError("variant must be L or LS")
        self.variant = variant
        self.line_encoder = ConvEncoder(1)
        self.surface_encoder = ConvEncoder(3) if variant == "LS" else None
        feature_count = 2048 * (2 if variant == "LS" else 1)
        self.character_head = nn.Linear(feature_count, class_count)
        self.start_head = nn.Linear(feature_count, 1)
        self.role_head = nn.Linear(feature_count, 2)

    def forward(
        self,
        line: torch.Tensor,
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.line_encoder(line)
        if self.surface_encoder is not None:
            features = torch.cat((features, self.surface_encoder(surface)), dim=1)
        return (
            self.character_head(features),
            self.start_head(features).squeeze(1),
            self.role_head(features),
        )


class DeepAASurfaceDenseScanner(nn.Module):
    """Reuse LS convolutional features over 16 offset phases of a whole row."""

    def __init__(self, model: DeepAASurfaceV0) -> None:
        super().__init__()
        if model.variant != "LS" or model.surface_encoder is None:
            raise ValueError("dense scanner requires the LS variant")
        self.line_encoder = model.line_encoder
        self.surface_encoder = model.surface_encoder
        self.character_scan = nn.Conv2d(256, model.character_head.out_features, 4)
        self.start_scan = nn.Conv2d(256, 1, 4)
        self.role_scan = nn.Conv2d(256, 2, 4)
        with torch.no_grad():
            for source, target in (
                (model.character_head, self.character_scan),
                (model.start_head, self.start_scan),
                (model.role_head, self.role_scan),
            ):
                output_count = source.out_features
                weight = (
                    source.weight.reshape(output_count, 2, 4, 4, 128)
                    .permute(0, 1, 4, 2, 3)
                    .reshape(output_count, 256, 4, 4)
                )
                target.weight.copy_(weight)
                target.bias.copy_(source.bias)

    @staticmethod
    def _encode(encoder: ConvEncoder, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for convolution, normalization in zip(
            encoder.convolutions, encoder.normalizations
        ):
            value = encoder.pool(torch.relu(normalization(convolution(value))))
        return value

    def forward(
        self,
        line: torch.Tensor,
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat(
            (self._encode(self.line_encoder, line), self._encode(self.surface_encoder, surface)),
            dim=1,
        )
        return (
            self.character_scan(features).squeeze(2).permute(0, 2, 1),
            self.start_scan(features).squeeze(1).squeeze(1),
            self.role_scan(features).squeeze(2).permute(0, 2, 1),
        )


def load_deepaa_line_encoder(encoder: ConvEncoder, onnx_path: Path) -> None:
    tensors = {
        tensor.name: numpy_helper.to_array(tensor).copy()
        for tensor in onnx.load(str(onnx_path)).graph.initializer
    }
    with torch.no_grad():
        for index, (convolution, normalization) in enumerate(
            zip(encoder.convolutions, encoder.normalizations), start=1
        ):
            prefix = f"conv2d_{index}"
            convolution.weight.copy_(torch.from_numpy(tensors[f"{prefix}.kernel"]))
            convolution.bias.copy_(torch.from_numpy(tensors[f"{prefix}.bias"]))
            prefix = f"batch_normalization_{index}"
            normalization.weight.copy_(torch.from_numpy(tensors[f"{prefix}.gamma"]))
            normalization.bias.copy_(torch.from_numpy(tensors[f"{prefix}.beta"]))
            normalization.running_mean.copy_(torch.from_numpy(tensors[f"{prefix}.mean"]))
            normalization.running_var.copy_(torch.from_numpy(tensors[f"{prefix}.variance"]))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
