from __future__ import annotations

import argparse
import base64
from pathlib import Path

from .contracts import ConversionOptions
from .deepaa import convert_deepaa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a structure-based Yaruo AA draft.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--classifier",
        choices=("cnn", "random-forest"),
        default="cnn",
        help="Experimental DeepAA character classifier.",
    )
    parser.add_argument(
        "--random-forest-model",
        type=Path,
        help="Path to a trained Random Forest joblib bundle.",
    )
    parser.add_argument(
        "--structure-strength",
        type=float,
        default=0.0,
        help="Direction and glyph-boundary mismatch weight (0-2).",
    )
    parser.add_argument(
        "--decoder",
        choices=("beam", "viterbi", "viterbi-pruned"),
        default="beam",
        help="DeepAA decoder; Viterbi modes are intended for quality experiments.",
    )
    parser.add_argument("--columns", type=int, default=72)
    parser.add_argument("--detail", type=int, default=72)
    parser.add_argument("--low", type=int, default=55)
    parser.add_argument("--high", type=int, default=150)
    parser.add_argument("--min-component", type=int, default=10)
    parser.add_argument(
        "--abstraction",
        type=int,
        default=35,
        help="Merge fine detail into representative AA-scale strokes (0-100).",
    )
    parser.add_argument("--max-rows", type=int, default=58)
    parser.add_argument(
        "--profile",
        choices=("auto", "person", "background", "lineart", "background_lineart"),
        default="auto",
    )
    parser.add_argument("--crop", type=float, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"), default=(0, 0, 1, 1))
    return parser.parse_args()


def save_data_url(data_url: str, path: Path) -> None:
    _, payload = data_url.split(",", 1)
    path.write_bytes(base64.b64decode(payload))


def main() -> None:
    args = parse_args()
    crop_x, crop_y, crop_width, crop_height = args.crop
    options = ConversionOptions(
        columns=args.columns,
        detail=args.detail,
        threshold_low=args.low,
        threshold_high=args.high,
        min_component=args.min_component,
        abstraction=args.abstraction,
        max_rows=args.max_rows,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        profile=args.profile,
    )
    result = convert_deepaa(
        args.image.read_bytes(),
        options,
        decoder=args.decoder,
        classifier=args.classifier,
        random_forest_model=args.random_forest_model,
        structure_strength=args.structure_strength,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    (args.output / f"{stem}.txt").write_text(result.ascii_text, encoding="utf-8")
    save_data_url(result.processed_png, args.output / f"{stem}-edges.png")
    save_data_url(result.rendered_png, args.output / f"{stem}-aa.png")
    print(f"Saved {result.columns} columns x {result.rows} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
