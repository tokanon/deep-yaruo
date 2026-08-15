from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from training.deepaa_surface_v0 import DeepAASurfaceDenseScanner, DeepAASurfaceV0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".tmp" / "training-runs" / "deepaa-surface-v0"
DEFAULT_MODEL = ROOT / "models" / "deepaa-surface-v0-ls.onnx"
DEFAULT_VOCABULARY = ROOT / "models" / "deepaa-surface-v0-vocabulary.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the accepted-v2 LS checkpoint.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    args = parser.parse_args()

    source_vocabulary = json.loads(
        (args.run / "vocabulary.json").read_text(encoding="utf-8")
    )
    characters = list(source_vocabulary["characters"])
    checkpoint_path = args.run / "LS" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "LS" or int(checkpoint["class_count"]) != len(characters):
        raise ValueError("LS checkpoint and vocabulary do not match")

    model = DeepAASurfaceV0(len(characters), variant="LS")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scanner = DeepAASurfaceDenseScanner(model)
    scanner.eval()
    line = torch.zeros((2, 1, 64, 95), dtype=torch.float32)
    surface = torch.zeros((2, 3, 64, 95), dtype=torch.float32)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = args.model.with_name(args.model.name + ".export")
    torch.onnx.export(
        scanner,
        (line, surface),
        temporary_model,
        input_names=["line", "surface"],
        output_names=["character_logits", "start_logits", "role_logits"],
        dynamic_axes={
            "line": {0: "batch", 3: "scan_width"},
            "surface": {0: "batch", 3: "scan_width"},
            "character_logits": {0: "batch", 1: "position"},
            "start_logits": {0: "batch", 1: "position"},
            "role_logits": {0: "batch", 1: "position"},
        },
        opset_version=17,
        dynamo=False,
    )

    session = ort.InferenceSession(
        str(temporary_model), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(42)
    verify_line = rng.random((3, 1, 64, 64), dtype=np.float32)
    verify_surface = rng.random((3, 3, 64, 64), dtype=np.float32)
    with torch.no_grad():
        torch_outputs = model(
            torch.from_numpy(verify_line), torch.from_numpy(verify_surface)
        )
    onnx_outputs = session.run(
        None, {"line": verify_line, "surface": verify_surface}
    )
    maximum_error = max(
        float(np.max(np.abs(expected.numpy() - actual[:, 0])))
        for expected, actual in zip(torch_outputs, onnx_outputs, strict=True)
    )
    if maximum_error > 1e-4:
        raise ValueError(f"ONNX verification error is too large: {maximum_error}")
    temporary_model.replace(args.model)

    exported_vocabulary = {
        "schema_version": 1,
        "model_variant": "LS",
        "characters": characters,
        "codepoints": [ord(character) for character in characters],
        "class_count": len(characters),
        "reference_font_size": 16,
        "line_pitch": 18,
        "context_size": 64,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_vocabulary_sha256": _sha256(args.run / "vocabulary.json"),
    }
    args.vocabulary.write_text(
        json.dumps(exported_vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "model_sha256": _sha256(args.model),
                "vocabulary": str(args.vocabulary.resolve()),
                "vocabulary_sha256": _sha256(args.vocabulary),
                "checkpoint_sha256": exported_vocabulary["checkpoint_sha256"],
                "maximum_onnx_error": maximum_error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
