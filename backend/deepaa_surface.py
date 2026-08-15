from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .rendering import REFERENCE_FONT_SIZE, glyph_advance
from .surface_channels import CONTEXT_MARGIN, CONTEXT_SIZE


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "deepaa-surface-v0-ls.onnx"
VOCABULARY_PATH = ROOT / "models" / "deepaa-surface-v0-vocabulary.json"
DECODER_VERSION = "dense-learned-start-exact-width-dp-v1"
DEFAULT_TOP_K = 8
LINE_PITCH = 18
PHASE_COUNT = 16


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@dataclass(frozen=True)
class SurfaceModelAssets:
    session: ort.InferenceSession
    characters: tuple[str, ...]
    advances: np.ndarray
    advance_groups: dict[int, np.ndarray]
    model_sha256: str
    vocabulary_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class DenseDecodeResult:
    text: str
    rows: int
    target_width: int
    row_widths: tuple[int, ...]
    start_counts: tuple[int, ...]
    all_whitespace_rows: tuple[int, ...]
    model_sha256: str
    vocabulary_sha256: str
    checkpoint_sha256: str


@lru_cache(maxsize=1)
def load_surface_model() -> SurfaceModelAssets:
    missing = [str(path) for path in (MODEL_PATH, VOCABULARY_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError("DeepAA surface assets are missing: " + ", ".join(missing))
    vocabulary = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    characters = tuple(str(character) for character in vocabulary["characters"])
    if int(vocabulary["class_count"]) != len(characters):
        raise ValueError("DeepAA surface vocabulary class count is inconsistent")
    if int(vocabulary["reference_font_size"]) != REFERENCE_FONT_SIZE:
        raise ValueError("DeepAA surface vocabulary uses an unsupported font size")
    advances = np.asarray(
        [glyph_advance(character, REFERENCE_FONT_SIZE) for character in characters],
        dtype=np.int16,
    )
    if np.any(advances <= 0):
        raise ValueError("DeepAA surface vocabulary contains a non-advancing glyph")
    advance_groups = {
        int(advance): np.flatnonzero(advances == advance)
        for advance in np.unique(advances)
    }
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_names = tuple(item.name for item in session.get_inputs())
    output_names = tuple(item.name for item in session.get_outputs())
    if input_names != ("line", "surface") or output_names != (
        "character_logits",
        "start_logits",
        "role_logits",
    ):
        raise ValueError("DeepAA surface ONNX input/output contract is inconsistent")
    return SurfaceModelAssets(
        session=session,
        characters=characters,
        advances=advances,
        advance_groups=advance_groups,
        model_sha256=_sha256(MODEL_PATH),
        vocabulary_sha256=_sha256(VOCABULARY_PATH),
        checkpoint_sha256=str(vocabulary["checkpoint_sha256"]),
    )


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    shifted = logits.astype(np.float64) - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def decode_dense_row(
    character_logits: np.ndarray,
    start_logits: np.ndarray,
    *,
    characters: tuple[str, ...],
    advances: np.ndarray,
    advance_groups: dict[int, np.ndarray],
    target_width: int,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, tuple[int, ...]] | None:
    """Decode one row without target starts, using start/non-start likelihoods."""
    char_logits = np.asarray(character_logits)
    starts = np.asarray(start_logits).reshape(-1)
    if char_logits.shape != (target_width, len(characters)) or starts.shape != (
        target_width,
    ):
        raise ValueError("Dense row logits do not match target geometry/vocabulary")
    log_probabilities = _log_softmax(char_logits)
    start_cost = np.logaddexp(0.0, -starts.astype(np.float64))
    nonstart_cost = np.logaddexp(0.0, starts.astype(np.float64))
    nonstart_prefix = np.zeros(target_width + 1, dtype=np.float64)
    nonstart_prefix[1:] = np.cumsum(nonstart_cost)

    costs = np.full(target_width + 1, np.inf, dtype=np.float64)
    costs[0] = 0.0
    previous_position = np.full(target_width + 1, -1, dtype=np.int32)
    previous_class = np.full(target_width + 1, -1, dtype=np.int32)
    for position in range(target_width):
        if not np.isfinite(costs[position]):
            continue
        scores = log_probabilities[position]
        keep = min(max(1, top_k), len(scores))
        top = np.argpartition(scores, len(scores) - keep)[-keep:]
        candidates = {int(index) for index in top}
        for indices in advance_groups.values():
            candidates.add(int(indices[np.argmax(scores[indices])]))
        for class_index in sorted(candidates):
            next_position = position + int(advances[class_index])
            if next_position > target_width:
                continue
            interior_cost = nonstart_prefix[next_position] - nonstart_prefix[position + 1]
            candidate_cost = (
                costs[position]
                - float(scores[class_index])
                + float(start_cost[position])
                + float(interior_cost)
            )
            if candidate_cost < costs[next_position]:
                costs[next_position] = candidate_cost
                previous_position[next_position] = position
                previous_class[next_position] = class_index
    if not np.isfinite(costs[target_width]):
        return None
    classes: list[int] = []
    starts_used: list[int] = []
    position = target_width
    while position > 0:
        class_index = int(previous_class[position])
        prior = int(previous_position[position])
        if class_index < 0 or prior < 0:
            return None
        classes.append(class_index)
        starts_used.append(prior)
        position = prior
    classes.reverse()
    starts_used.reverse()
    return "".join(characters[index] for index in classes), tuple(starts_used)


def _row_logits(
    line_channels: np.ndarray,
    surface_channels: np.ndarray,
    *,
    row: int,
    assets: SurfaceModelAssets,
) -> tuple[np.ndarray, np.ndarray]:
    width = int(line_channels.shape[2])
    bottom_padding = CONTEXT_SIZE - CONTEXT_MARGIN
    phase_padding = PHASE_COUNT - 1
    line_padded = np.pad(
        line_channels,
        (
            (0, 0),
            (CONTEXT_MARGIN, bottom_padding),
            (CONTEXT_MARGIN, bottom_padding + phase_padding),
        ),
        constant_values=1.0,
    )
    surface_padded = np.pad(
        surface_channels,
        (
            (0, 0),
            (CONTEXT_MARGIN, bottom_padding),
            (CONTEXT_MARGIN, bottom_padding + phase_padding),
        ),
        constant_values=0.0,
    )
    y = row * LINE_PITCH
    scan_width = width + CONTEXT_SIZE - 1
    line_batch = np.stack(
        [
            line_padded[
                :, y : y + CONTEXT_SIZE, phase : phase + scan_width
            ]
            for phase in range(PHASE_COUNT)
        ]
    ).astype(np.float32, copy=False)
    surface_batch = np.stack(
        [
            surface_padded[
                :, y : y + CONTEXT_SIZE, phase : phase + scan_width
            ]
            for phase in range(PHASE_COUNT)
        ]
    ).astype(np.float32, copy=False)
    phased_characters, phased_starts, _roles = assets.session.run(
        None, {"line": line_batch, "surface": surface_batch}
    )
    characters = np.empty((width, len(assets.characters)), dtype=np.float32)
    starts = np.empty(width, dtype=np.float32)
    for phase in range(PHASE_COUNT):
        positions = np.arange(phase, width, PHASE_COUNT)
        count = len(positions)
        characters[positions] = phased_characters[phase, :count]
        starts[positions] = phased_starts[phase, :count]
    return characters, starts


def decode_surface_text(
    line_channels: np.ndarray,
    surface_channels: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> DenseDecodeResult:
    line = np.asarray(line_channels, dtype=np.float32)
    surface = np.asarray(surface_channels, dtype=np.float32)
    if line.ndim != 3 or line.shape[0] != 1:
        raise ValueError("line input must be 1xHxW")
    if surface.shape != (3, line.shape[1], line.shape[2]):
        raise ValueError("surface input must be 3xHxW and match the line input")
    if line.shape[1] % LINE_PITCH:
        raise ValueError("line input height must be a whole number of 18px rows")
    if line.shape[2] < 1:
        raise ValueError("line input width must be positive")
    assets = load_surface_model()
    rows = line.shape[1] // LINE_PITCH
    target_width = int(line.shape[2])
    decoded_lines: list[str] = []
    widths: list[int] = []
    counts: list[int] = []
    whitespace_rows: list[int] = []
    for row in range(rows):
        character_logits, start_logits = _row_logits(
            line,
            surface,
            row=row,
            assets=assets,
        )
        decoded = decode_dense_row(
            character_logits,
            start_logits,
            characters=assets.characters,
            advances=assets.advances,
            advance_groups=assets.advance_groups,
            target_width=target_width,
            top_k=top_k,
        )
        if decoded is None:
            raise ValueError(f"dense exact-width decoding failed on row {row}")
        text, starts_used = decoded
        width = sum(glyph_advance(character, REFERENCE_FONT_SIZE) for character in text)
        if width != target_width:
            raise ValueError(f"dense decoder produced width {width} on row {row}")
        decoded_lines.append(text)
        widths.append(width)
        counts.append(len(starts_used))
        row_top = row * LINE_PITCH
        row_bottom = row_top + LINE_PITCH
        input_nonempty = bool(
            np.any(line[0, row_top:row_bottom] < 0.98)
            or np.any(surface[0, row_top:row_bottom] > 0.5)
        )
        if input_nonempty and not any(not character.isspace() for character in text):
            whitespace_rows.append(row)
    return DenseDecodeResult(
        text="\n".join(decoded_lines) + "\n",
        rows=rows,
        target_width=target_width,
        row_widths=tuple(widths),
        start_counts=tuple(counts),
        all_whitespace_rows=tuple(whitespace_rows),
        model_sha256=assets.model_sha256,
        vocabulary_sha256=assets.vocabulary_sha256,
        checkpoint_sha256=assets.checkpoint_sha256,
    )
