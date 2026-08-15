from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.contracts import ConversionOptions
from backend.deepaa import (
    LINE_PITCH,
    ROW_REPEAT_WEIGHT,
    _crop_image,
    _predictions_cost,
    load_assets,
    render_predictions,
)
from backend.image_io import decode_image
from backend.input_channels import (
    ChannelExtractionConfig,
    channel_preview,
    extract_input_channels,
)
from training.aa_fill_layers import detect_fill_runs
from training.examples import extract_context
from training.model import DeepAAMultitask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "cases.json"
DEFAULT_P3_REPORT = ROOT / ".tmp" / "training-runs" / "structure-proxy-p3" / "report.json"
DEFAULT_P6_REPORT = ROOT / ".tmp" / "training-runs" / "multichannel-p6" / "report.json"
DEFAULT_OUTPUT = ROOT / ".tmp" / "training-runs" / "multichannel-p6-real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed real-image C1/C2 candidates for the P6 human gate."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--p3-report", type=Path, default=DEFAULT_P3_REPORT)
    parser.add_argument("--p6-report", type=Path, default=DEFAULT_P6_REPORT)
    parser.add_argument(
        "--condition",
        help="C=2 condition to inspect even when the synthetic gate selected C1.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="*",
        default=list(range(LINE_PITCH)),
        help="Vertical offsets to compare; defaults to all 18 legal offsets.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_model(path: Path, input_channels: int, device: torch.device) -> DeepAAMultitask:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DeepAAMultitask(input_channels=input_channels)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    return model


def _greedy_decode(
    previews: np.ndarray,
    model: DeepAAMultitask,
    device: torch.device,
    *,
    offset: int,
) -> tuple[str, list[list[str]]]:
    if previews.ndim != 3 or previews.shape[0] not in {1, 2}:
        raise ValueError(f"Expected CxHxW preview channels, got {previews.shape}")
    if previews.shape[1] % LINE_PITCH:
        raise ValueError("P6 preview height must be a multiple of the 18px line pitch")
    if not 0 <= offset < LINE_PITCH:
        raise ValueError("P6 vertical offset must be between 0 and 17")
    assets = load_assets()
    rows = previews.shape[1] // LINE_PITCH
    target_width = previews.shape[2]
    positions = np.zeros(rows, dtype=np.int32)
    previous_half_space = np.zeros(rows, dtype=bool)
    predictions: list[list[str]] = [[] for _ in range(rows)]
    active = list(range(rows))
    with torch.no_grad():
        while active:
            windows = np.stack(
                [
                    np.stack(
                        [
                            extract_context(
                                channel,
                                int(positions[row]),
                                row * LINE_PITCH - offset,
                            )
                            for channel in previews
                        ]
                    )
                    for row in active
                ]
            )
            inputs = torch.from_numpy(windows.copy()).float().div_(255.0).to(device)
            logits, _ = model(inputs)
            scores = torch.softmax(logits, dim=1).cpu().numpy()
            next_active: list[int] = []
            for batch_index, row in enumerate(active):
                row_scores = scores[batch_index]
                if previous_half_space[row]:
                    row_scores = row_scores.copy()
                    row_scores[assets.half_space_index] = -1.0
                class_index = int(np.argmax(row_scores))
                char = assets.characters[class_index]
                predictions[row].append(char)
                previous_half_space[row] = class_index == assets.half_space_index
                positions[row] += assets.glyphs[char].shape[1]
                if positions[row] < target_width:
                    next_active.append(row)
            active = next_active
    lines = ["".join(line).rstrip() for line in predictions]
    return "\n".join(lines).rstrip() + "\n", predictions


def _best_offset_decode(
    previews: np.ndarray,
    model: DeepAAMultitask,
    device: torch.device,
    offsets: list[int],
    *,
    repeat_weight: float,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for offset in offsets:
        text, predictions = _greedy_decode(previews, model, device, offset=offset)
        cost = _predictions_cost(
            previews[0],
            predictions,
            load_assets().glyphs,
            repeat_weight=repeat_weight,
        )
        candidate = {
            "offset": offset,
            "text": text,
            "predictions": predictions,
            "render_cost": float(cost),
        }
        if best is None or float(candidate["render_cost"]) < float(best["render_cost"]):
            best = candidate
    assert best is not None
    return best


def _text_diagnostics(text: str) -> dict[str, object]:
    nonblank = [char for char in text if char not in {" ", "　", "\n", "\r"}]
    fill_runs = detect_fill_runs(text, font_size=16)
    fill_characters = sum(run.end_index - run.start_index for run in fill_runs)
    counts = Counter(nonblank)
    return {
        "nonblank_characters": len(nonblank),
        "unique_nonblank_characters": len(counts),
        "top_nonblank_characters": counts.most_common(10),
        "fill_run_count": len(fill_runs),
        "fill_run_character_count": fill_characters,
    }


def _write_gray(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _prepare_output(path: Path, *, force: bool) -> None:
    allowed_root = (ROOT / ".tmp" / "training-runs").resolve()
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not force:
            raise FileExistsError(f"P6 real-image output is not empty: {resolved}; use --force")
        if allowed_root not in resolved.parents:
            raise ValueError(f"Refusing to replace output outside {allowed_root}: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _dataset_tone_statistics(dataset_manifest: Path) -> dict[str, object]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    root = dataset_manifest.parent
    split_counts: dict[str, np.ndarray] = {}
    split_images: Counter[str] = Counter()
    for work in manifest["works"]:
        split = str(work["split"])
        with np.load(root / str(work["raw_channels"])) as payload:
            preview = payload["channels"][1]
        levels = np.rint((1.0 - preview.astype(np.float32) / 255.0) * 3).astype(
            np.uint8
        )
        counts = np.bincount(levels.ravel(), minlength=4)
        split_counts.setdefault(split, np.zeros(4, dtype=np.int64))
        split_counts[split] += counts
        split_images[split] += 1
    return {
        split: {
            "image_count": split_images[split],
            "tone_level_fractions": (counts / max(1, counts.sum())).tolist(),
        }
        for split, counts in sorted(split_counts.items())
    }


def main() -> None:
    args = parse_args()
    offsets = sorted(set(args.offsets))
    if not offsets or any(offset < 0 or offset >= LINE_PITCH for offset in offsets):
        raise ValueError("--offsets must contain values from 0 through 17")
    _prepare_output(args.output, force=args.force)
    output = args.output.resolve()
    p3_report = json.loads(args.p3_report.read_text(encoding="utf-8"))
    p6_report = json.loads(args.p6_report.read_text(encoding="utf-8"))
    selected = args.condition or str(p6_report["selection"]["selected"])
    if selected == "c1":
        raise ValueError("P6 did not select a C=2 condition")
    condition = p6_report["conditions"][selected]
    config_payload = condition["dataset"]["channel_config"]
    channel_config = ChannelExtractionConfig(**config_payload)
    channel_config.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c1_path = Path(p3_report["methods"]["raw_raster"]["checkpoint"])
    c2_path = Path(condition["checkpoint"])
    c1 = _load_model(c1_path, 1, device)
    c2 = _load_model(c2_path, 2, device)
    dataset_manifest = Path(condition["dataset_manifest"])
    cases_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    case_root = args.cases.resolve().parent
    records: list[dict[str, object]] = []
    for case in cases_payload["cases"]:
        case_id = str(case["id"])
        print(f"case={case_id}")
        options = ConversionOptions(**case["options"]).normalized()
        source_path = (case_root / str(case["source"])).resolve()
        source = decode_image(source_path.read_bytes())
        cropped, crop_box = _crop_image(source, options)
        source_kind = (
            "lineart"
            if options.profile in {"lineart", "background_lineart"}
            else "grayscale"
        )
        channels = extract_input_channels(
            cropped,
            source_kind=source_kind,
            config=channel_config,
        )
        previews = np.stack(
            (channel_preview(channels.structure), channel_preview(channels.tone))
        )
        repeat_weight = 0.0 if options.profile == "background" else ROW_REPEAT_WEIGHT
        c1_result = _best_offset_decode(
            previews[:1], c1, device, offsets, repeat_weight=repeat_weight
        )
        c2_result = _best_offset_decode(
            previews, c2, device, offsets, repeat_weight=repeat_weight
        )
        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_gray(case_dir / "01-input.png", channels.resized_gray)
        _write_gray(case_dir / "02-structure.png", previews[0])
        _write_gray(case_dir / "03-tone.png", previews[1])
        rendered_arrays: list[np.ndarray] = []
        model_records: dict[str, object] = {}
        for model_id, result in (("c1", c1_result), ("c2", c2_result)):
            text = str(result["text"])
            (case_dir / f"{model_id}.txt").write_text(
                text, encoding="utf-8", newline="\n"
            )
            rendered = render_predictions(result["predictions"], previews.shape[2])
            rendered.save(case_dir / f"{model_id}.png")
            rendered_arrays.append(np.asarray(rendered.convert("L")))
            model_records[model_id] = {
                "offset": result["offset"],
                "render_cost": result["render_cost"],
                "text": f"{model_id}.txt",
                "rendered": f"{model_id}.png",
                "diagnostics": _text_diagnostics(text),
            }
        comparison = np.concatenate(
            [channels.resized_gray, previews[0], previews[1], *rendered_arrays], axis=1
        )
        _write_gray(case_dir / "comparison.png", comparison)
        records.append(
            {
                "id": case_id,
                "kind": case["kind"],
                "source": str(source_path),
                "crop": list(crop_box),
                "source_kind": source_kind,
                "channel_metadata": channels.metadata,
                "comparison": str((case_dir / "comparison.png").relative_to(output)),
                "models": model_records,
            }
        )
    report = {
        "schema_version": 1,
        "phase": "P6-real",
        "purpose": "fixed real-image C1/C2 candidate generation for human comparison",
        "device": str(device),
        "selected_c2_condition": selected,
        "c1_checkpoint": str(c1_path.resolve()),
        "c2_checkpoint": str(c2_path.resolve()),
        "offsets_evaluated": offsets,
        "synthetic_aa_tone_statistics": _dataset_tone_statistics(dataset_manifest),
        "cases": records,
        "selection_status": "human review required; real images have no gold AA charAcc",
        "rights_status": "local evaluation only; sources, texts, and rendered candidates are not tracked",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"saved={(output / 'report.json').resolve()}")


if __name__ == "__main__":
    main()
