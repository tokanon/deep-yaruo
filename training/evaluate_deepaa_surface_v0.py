from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from backend.rendering import REFERENCE_FONT_SIZE, glyph_advance, load_font
from training.deepaa_surface_v0 import (
    DeepAASurfaceV0,
    ReverseChannelDataset,
    WorkGroupedBatchSampler,
)
from training.train_deepaa_surface_v0 import (
    DEFAULT_DATA,
    _class_weights,
    run_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepAA L/LS without tuning test.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / ".tmp" / "training-runs" / "deepaa-surface-v0",
    )
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--decoder-top-k", type=int, default=8)
    parser.add_argument("--negative-stride", type=int, default=4)
    parser.add_argument("--max-works", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _advance_groups(characters: list[str]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    advances = np.asarray(
        [glyph_advance(character, REFERENCE_FONT_SIZE) for character in characters],
        dtype=np.int16,
    )
    groups = {
        int(advance): np.flatnonzero(advances == advance)
        for advance in np.unique(advances)
    }
    return advances, groups


def _decode_exact_width(
    log_probabilities: np.ndarray,
    *,
    characters: list[str],
    advances: np.ndarray,
    advance_groups: dict[int, np.ndarray],
    target_width: int,
    top_k: int,
) -> list[str] | None:
    states: dict[int, tuple[float, tuple[int, int] | None]] = {0: (0.0, None)}
    history: list[dict[int, tuple[float, tuple[int, int] | None]]] = []
    total = len(log_probabilities)
    minimum_advance = int(advances.min())
    maximum_advance = int(advances.max())
    for position, scores in enumerate(log_probabilities):
        keep = min(top_k, len(scores))
        top = np.argpartition(scores, len(scores) - keep)[-keep:]
        candidates = set(int(index) for index in top)
        for indices in advance_groups.values():
            candidates.add(int(indices[np.argmax(scores[indices])]))
        remaining = total - position - 1
        next_states: dict[int, tuple[float, tuple[int, int]]] = {}
        for width, (cost, _previous) in states.items():
            for class_index in candidates:
                next_width = width + int(advances[class_index])
                if next_width + remaining * minimum_advance > target_width:
                    continue
                if next_width + remaining * maximum_advance < target_width:
                    continue
                next_cost = cost - float(scores[class_index])
                prior = next_states.get(next_width)
                if prior is None or next_cost < prior[0]:
                    next_states[next_width] = (next_cost, (width, class_index))
        history.append(next_states)
        states = next_states
        if not states:
            return None
    if target_width not in states:
        return None
    result: list[str] = []
    width = target_width
    for layer in range(len(history) - 1, -1, -1):
        previous = history[layer][width][1]
        assert previous is not None
        width, class_index = previous
        result.append(characters[class_index])
    result.reverse()
    return result


def _render_row(text: str, width: int) -> np.ndarray:
    if width == 0:
        return np.zeros((18, 0), dtype=bool)
    image = Image.new("L", (max(1, width), 18), 255)
    font = load_font(REFERENCE_FONT_SIZE)
    ascent, _ = font.getmetrics()
    ImageDraw.Draw(image).text((0, ascent), text, font=font, fill=0, anchor="ls")
    return np.asarray(image) < 128


def _frequency_bucket(count: int) -> str:
    if count < 10:
        return "1-9"
    if count < 100:
        return "10-99"
    return "100+"


def evaluate_decode(
    model: DeepAASurfaceV0,
    dataset: ReverseChannelDataset,
    device: torch.device,
    *,
    characters: list[str],
    train_counts: dict[str, int],
    batch_size: int,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    advances, advance_groups = _advance_groups(characters)
    totals = defaultdict(int)
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decoded_records: list[dict[str, object]] = []
    model.eval()
    for work_index, work in enumerate(dataset.works):
        loaded = dataset._load_work(work_index)
        probabilities: list[np.ndarray] = []
        start_global = dataset.offsets[work_index]
        end_global = dataset.offsets[work_index + 1]
        for batch_start in range(start_global, end_global, batch_size):
            items = [
                dataset[index]
                for index in range(batch_start, min(end_global, batch_start + batch_size))
            ]
            line = torch.stack([item[0] for item in items]).to(device)
            surface = torch.stack([item[1] for item in items]).to(device)
            with torch.no_grad():
                character_logits, _start, _role = model(line, surface)
                batch_log_probabilities = torch.log_softmax(character_logits, dim=1)
            probabilities.append(batch_log_probabilities.cpu().numpy())
        log_probabilities = np.concatenate(probabilities, axis=0)
        predicted_lines: list[str] = []
        work_failures = 0
        target_mask_path = work.channels_path
        with np.load(target_mask_path) as arrays:
            target_mask = arrays["target_mask"].astype(bool)
            surface_ids = arrays["b_surface_ids"]
            row_widths = arrays["row_widths"]
        for row in range(len(loaded.row_offsets) - 1):
            start = int(loaded.row_offsets[row])
            end = int(loaded.row_offsets[row + 1])
            target_width = int(row_widths[row])
            prediction = _decode_exact_width(
                log_probabilities[start:end],
                characters=characters,
                advances=advances,
                advance_groups=advance_groups,
                target_width=target_width,
                top_k=top_k,
            )
            if prediction is None:
                predicted_lines.append("")
                work_failures += 1
                totals["row_width_failure"] += 1
                totals["row_count"] += 1
                continue
            predicted_text = "".join(prediction)
            predicted_lines.append(predicted_text)
            predicted_width = sum(
                glyph_advance(character, REFERENCE_FONT_SIZE) for character in prediction
            )
            totals["row_count"] += 1
            totals["row_width_exact"] += int(predicted_width == target_width)
            rendered = _render_row(predicted_text, target_width)
            target = target_mask[row * 18 : (row + 1) * 18, :target_width]
            fill_region = surface_ids[row * 18 : (row + 1) * 18, :target_width] > 0
            line_region = ~fill_region
            mismatch = rendered != target
            totals["pixel_count"] += target.size
            totals["pixel_mismatch"] += int(mismatch.sum())
            totals["fill_pixel_count"] += int(fill_region.sum())
            totals["fill_pixel_mismatch"] += int((mismatch & fill_region).sum())
            totals["line_pixel_count"] += int(line_region.sum())
            totals["line_pixel_mismatch"] += int((mismatch & line_region).sum())
            totals["target_ink"] += int(target.sum())
            totals["missed_ink"] += int((target & ~rendered).sum())
            totals["rendered_ink"] += int(rendered.sum())
            totals["extra_ink"] += int((rendered & ~target).sum())
        predictions = log_probabilities.argmax(axis=1)
        for index, codepoint in enumerate(loaded.codepoints):
            character = chr(int(codepoint))
            bucket = _frequency_bucket(train_counts.get(character, 0)) if character in train_counts else "unseen"
            buckets[bucket][1] += 1
            expected = dataset.class_by_character.get(character)
            if expected is not None and int(predictions[index]) == expected:
                buckets[bucket][0] += 1
        decoded_records.append(
            {
                "entry_id": work.entry_id,
                "category": work.category,
                "row_count": len(loaded.row_offsets) - 1,
                "row_width_failures": work_failures,
                "prediction": "\n".join(predicted_lines),
            }
        )
    metrics = {
        "decoder": {
            "name": "true-start-sequence-width-constrained-dp-v1",
            "top_k": top_k,
            "advance_fallback": "best-class-per-observed-train-advance",
            "uses_target_character_identity": False,
            "uses_target_start_count": True,
        },
        "rows": totals["row_count"],
        "exact_width_rows": totals["row_width_exact"],
        "exact_width_fraction": totals["row_width_exact"] / max(1, totals["row_count"]),
        "undecodable_rows": totals["row_width_failure"],
        "render_pixel_mismatch": totals["pixel_mismatch"] / max(1, totals["pixel_count"]),
        "fill_render_pixel_mismatch": totals["fill_pixel_mismatch"]
        / max(1, totals["fill_pixel_count"]),
        "line_render_pixel_mismatch": totals["line_pixel_mismatch"]
        / max(1, totals["line_pixel_count"]),
        "target_ink_omission": totals["missed_ink"] / max(1, totals["target_ink"]),
        "rendered_ink_excess": totals["extra_ink"] / max(1, totals["rendered_ink"]),
        "frequency_buckets": {
            name: {"correct": values[0], "count": values[1], "accuracy": values[0] / max(1, values[1])}
            for name, values in sorted(buckets.items())
        },
    }
    return metrics, decoded_records


def main() -> None:
    args = parse_args()
    validation_report_path = args.run / "evaluation-validation.json"
    if args.split == "test":
        if not validation_report_path.exists():
            raise SystemExit("Validation evaluation must be fixed before test is opened.")
        validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
        if validation_report.get("partial_max_works") is not None:
            raise SystemExit("A partial validation run cannot authorize test evaluation.")
        validation_decoder = validation_report["variants"]["L"]["decode"]["decoder"]
        if int(validation_decoder["top_k"]) != args.decoder_top_k:
            raise SystemExit("Test decoder settings differ from the fixed validation decoder.")
    vocabulary = json.loads((args.run / "vocabulary.json").read_text(encoding="utf-8"))
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    characters = vocabulary["characters"]
    train_counts = {key: int(value) for key, value in vocabulary["train_counts"].items()}
    device = torch.device(
        "cuda" if args.device == "cuda" else "cpu"
        if args.device == "cpu"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    dataset = ReverseChannelDataset(
        args.data,
        split=args.split,
        characters=characters,
        negative_stride=args.negative_stride,
        seed=args.seed,
    )
    if args.max_works is not None:
        if args.max_works < 1:
            raise SystemExit("--max-works must be positive")
        dataset.works = dataset.works[: args.max_works]
        dataset.offsets = [0]
        for work in dataset.works:
            dataset.offsets.append(dataset.offsets[-1] + work.glyph_count)
        dataset.positive_count = dataset.offsets[-1]
        dataset.negative_count = math.ceil(dataset.positive_count / dataset.negative_stride)
    sampler = WorkGroupedBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    class_weights = _class_weights(characters, train_counts, device)
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "deepaa-surface-v0",
        "split": args.split,
        "test_was_tuned": False,
        "normal_generator_changed": False,
        "partial_max_works": args.max_works,
        "variants": {},
    }
    for variant in ("L", "LS"):
        model = DeepAASurfaceV0(len(characters), variant=variant)
        checkpoint = torch.load(
            args.run / variant / "best.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        with torch.no_grad():
            classification = run_epoch(
                model,
                loader,
                device,
                class_weights,
                optimizer=None,
                start_loss_weight=float(config["start_loss_weight"]),
                role_loss_weight=float(config["role_loss_weight"]),
            )
        decode, records = evaluate_decode(
            model,
            dataset,
            device,
            characters=characters,
            train_counts=train_counts,
            batch_size=args.batch_size,
            top_k=args.decoder_top_k,
        )
        report["variants"][variant] = {
            "selected_epoch": int(checkpoint["epoch"]),
            "classification": classification,
            "decode": decode,
        }
        decoded_path = args.run / variant / f"decoded-{args.split}.jsonl"
        decoded_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps({"variant": variant, "classification": classification, "decode": decode}, ensure_ascii=False), flush=True)
    output = args.run / f"evaluation-{args.split}.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.split == "test":
        l_metrics = report["variants"]["L"]
        ls_metrics = report["variants"]["LS"]
        l_classification = l_metrics["classification"]
        ls_classification = ls_metrics["classification"]
        l_decode = l_metrics["decode"]
        ls_decode = ls_metrics["decode"]
        line_delta = (
            ls_classification["character_strata"]["line"]["accuracy"]
            - l_classification["character_strata"]["line"]["accuracy"]
        )
        comparison = {
            "fill_character_accuracy_delta_ls_minus_l": (
                ls_classification["character_strata"]["fill"]["accuracy"]
                - l_classification["character_strata"]["fill"]["accuracy"]
            ),
            "supported_fill_character_accuracy_delta_ls_minus_l": (
                ls_classification["character_strata"]["supported_fill"]["accuracy"]
                - l_classification["character_strata"]["supported_fill"]["accuracy"]
            ),
            "low_support_fill_character_accuracy_delta_ls_minus_l": (
                ls_classification["character_strata"]["low_support_fill"]["accuracy"]
                - l_classification["character_strata"]["low_support_fill"]["accuracy"]
            ),
            "line_character_accuracy_delta_ls_minus_l": line_delta,
            "fill_render_mismatch_delta_ls_minus_l": (
                ls_decode["fill_render_pixel_mismatch"]
                - l_decode["fill_render_pixel_mismatch"]
            ),
            "line_render_mismatch_delta_ls_minus_l": (
                ls_decode["line_render_pixel_mismatch"]
                - l_decode["line_render_pixel_mismatch"]
            ),
            "exact_width_both": (
                l_decode["exact_width_fraction"] == 1.0
                and ls_decode["exact_width_fraction"] == 1.0
            ),
            "synthetic_stop_checks": {
                "fill_character_improved": (
                    ls_classification["character_strata"]["fill"]["accuracy"]
                    > l_classification["character_strata"]["fill"]["accuracy"]
                ),
                "fill_render_improved": (
                    ls_decode["fill_render_pixel_mismatch"]
                    < l_decode["fill_render_pixel_mismatch"]
                ),
                "line_character_not_materially_degraded": line_delta >= -0.01,
                "line_render_not_materially_degraded": (
                    ls_decode["line_render_pixel_mismatch"]
                    - l_decode["line_render_pixel_mismatch"]
                )
                <= 0.01,
                "exact_width_preserved": (
                    l_decode["exact_width_fraction"] == 1.0
                    and ls_decode["exact_width_fraction"] == 1.0
                ),
            },
            "material_line_degradation_threshold_absolute": 0.01,
            "real_image_gate_evaluated": False,
            "normal_generator_changed": False,
        }
        checks = comparison["synthetic_stop_checks"]
        comparison["synthetic_gate_passed"] = all(checks.values())
        comparison["next_if_passed"] = "unseen-real-image comparison against P6R-G1"
        comparison["normal_generation_authorized"] = False
        (args.run / "comparison-test.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
