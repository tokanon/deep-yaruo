from __future__ import annotations

import base64
import binascii
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from .contracts import ConversionOptions, ConversionResult
from .image_io import image_to_data_url
from .rendering import FONT_PATH, line_pitch, render_text_mask
from .deepaa_surface import (
    MODEL_PATH as SURFACE_MODEL_PATH,
    VOCABULARY_PATH as SURFACE_VOCABULARY_PATH,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORRECTION_ROOT = ROOT / "datasets" / "incoming" / "correction-pairs" / "v1"
MODEL_PATH = ROOT / "models" / "deepaa-light.onnx"
CHARSET_PATH = ROOT / "models" / "deepaa-charset.csv"
START_LOCATOR_PATH = ROOT / "models" / "deepaa-start-locator.onnx"
SCHEMA_VERSION = 1
EXTENSIONS_SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARACTERS = 2_000_000
MAX_EXTENSION_BYTES = 10 * 1024 * 1024
EXTENSION_SLOTS = (
    "surface_proposals",
    "surface_correspondence",
    "surface_metrics",
)
_RECORD_ID = re.compile(r"[0-9a-f]{32}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _artifact(path: str, payload: bytes, media_type: str) -> dict[str, object]:
    return {
        "path": path,
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "media_type": media_type,
    }


def decode_png_data_url(data_url: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError("Expected a PNG data URL")
    try:
        payload = base64.b64decode(data_url[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("PNG data URL has invalid base64 data") from error
    _decode_image_shape(payload, grayscale=True)
    return payload


def _decode_image_shape(payload: bytes, *, grayscale: bool = False) -> tuple[int, int]:
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), mode)
    if image is None:
        raise ValueError("An image artifact could not be decoded")
    return int(image.shape[1]), int(image.shape[0])


def _render_corrected_text(
    text: str,
    *,
    font_size: int,
    canvas_width: int,
    canvas_height: int,
) -> bytes:
    mask = render_text_mask(
        text,
        font_size,
        canvas_height_per_line=line_pitch(font_size),
    )
    canvas = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    visible_height = min(canvas_height, mask.shape[0])
    visible_width = min(canvas_width, mask.shape[1])
    canvas[:visible_height, :visible_width][
        mask[:visible_height, :visible_width]
    ] = 0
    return decode_png_data_url(image_to_data_url(canvas))


def character_diff(draft_text: str, corrected_text: str) -> dict[str, object]:
    matcher = difflib.SequenceMatcher(
        None,
        draft_text,
        corrected_text,
        autojunk=False,
    )
    operations: list[dict[str, object]] = []
    inserted = 0
    deleted = 0
    replaced_from = 0
    replaced_to = 0
    for operation, draft_start, draft_end, corrected_start, corrected_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        draft_fragment = draft_text[draft_start:draft_end]
        corrected_fragment = corrected_text[corrected_start:corrected_end]
        if operation == "insert":
            inserted += len(corrected_fragment)
        elif operation == "delete":
            deleted += len(draft_fragment)
        elif operation == "replace":
            replaced_from += len(draft_fragment)
            replaced_to += len(corrected_fragment)
        operations.append(
            {
                "operation": operation,
                "draft_start": draft_start,
                "draft_end": draft_end,
                "corrected_start": corrected_start,
                "corrected_end": corrected_end,
                "draft_text": draft_fragment,
                "corrected_text": corrected_fragment,
            }
        )
    return {
        "operation_count": len(operations),
        "inserted_characters": inserted,
        "deleted_characters": deleted,
        "replaced_draft_characters": replaced_from,
        "replaced_corrected_characters": replaced_to,
        "operations": operations,
    }


@dataclass(frozen=True)
class CorrectionPairSubmission:
    source_bytes: bytes
    source_filename: str
    source_media_type: str
    source_origin: str
    rights_status: str
    options: ConversionOptions
    crop: tuple[int, int, int, int]
    draft_text: str
    corrected_text: str
    processed_png: bytes
    draft_rendered_png: bytes
    generation: dict[str, object] | None = None


class CorrectionPairStore:
    def __init__(self, root: Path = DEFAULT_CORRECTION_ROOT) -> None:
        self.root = root

    def _record_dir(self, record_id: str) -> Path:
        if _RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("Invalid correction-pair record ID")
        return self.root / record_id

    def _validate_submission(
        self,
        submission: CorrectionPairSubmission,
    ) -> tuple[ConversionOptions, tuple[int, int], tuple[int, int]]:
        if not submission.source_bytes:
            raise ValueError("source image is empty")
        if len(submission.source_bytes) > MAX_SOURCE_BYTES:
            raise ValueError("source image exceeds 20 MB")
        if not submission.source_filename.strip():
            raise ValueError("source_filename must not be empty")
        if not submission.source_origin.strip():
            raise ValueError("source_origin must not be empty")
        if not submission.rights_status.strip():
            raise ValueError("rights_status must not be empty")
        if len(submission.draft_text) > MAX_TEXT_CHARACTERS:
            raise ValueError("draft text is too large")
        if len(submission.corrected_text) > MAX_TEXT_CHARACTERS:
            raise ValueError("corrected text is too large")
        if submission.draft_text == submission.corrected_text:
            raise ValueError("corrected text must differ from the draft")

        options = submission.options.normalized()
        source_size = _decode_image_shape(submission.source_bytes)
        processed_size = _decode_image_shape(submission.processed_png, grayscale=True)
        rendered_size = _decode_image_shape(
            submission.draft_rendered_png,
            grayscale=True,
        )
        if rendered_size != processed_size:
            raise ValueError("draft rendering and processed image sizes differ")
        x0, y0, x1, y1 = submission.crop
        source_width, source_height = source_size
        if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
            raise ValueError("crop must be a non-empty box inside the source image")
        return options, source_size, processed_size

    def save(self, submission: CorrectionPairSubmission) -> dict[str, object]:
        options, source_size, processed_size = self._validate_submission(submission)
        source_payload = bytes(submission.source_bytes)
        processed_payload = bytes(submission.processed_png)
        draft_rendered_payload = bytes(submission.draft_rendered_png)
        draft_payload = submission.draft_text.encode("utf-8")
        corrected_payload = submission.corrected_text.encode("utf-8")
        corrected_rendered_payload = _render_corrected_text(
            submission.corrected_text,
            font_size=options.font_size,
            canvas_width=processed_size[0],
            canvas_height=processed_size[1],
        )
        option_values = asdict(options)
        generation = submission.generation or {
            "generator_version": "deepaa-beam-v0.2",
            "recipe_version": "legacy-v0.2",
            "enabled": False,
        }
        if not isinstance(generation, dict):
            raise ValueError("generation metadata must be an object")
        _canonical_json(generation)
        crop = tuple(int(value) for value in submission.crop)
        asset_paths = {
            "model": MODEL_PATH,
            "charset": CHARSET_PATH,
            "start_locator": START_LOCATOR_PATH,
            "font": FONT_PATH,
        }
        if generation.get("recipe_version") == "deepaa-surface-v0-provisional-v1":
            asset_paths.update(
                {
                    "surface_model": SURFACE_MODEL_PATH,
                    "surface_vocabulary": SURFACE_VOCABULARY_PATH,
                }
            )
        generator_assets = {}
        for name, path in asset_paths.items():
            generator_assets[name] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path.read_bytes()),
            }
        decoder = (
            "dense-learned-start-exact-width-dp-v1"
            if generation.get("recipe_version") == "deepaa-surface-v0-provisional-v1"
            else "beam"
        )
        classifier = (
            "deepaa-surface-v0-ls"
            if generation.get("recipe_version") == "deepaa-surface-v0-provisional-v1"
            else "cnn"
        )
        identity = {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": _sha256(source_payload),
            "processed_sha256": _sha256(processed_payload),
            "draft_rendered_sha256": _sha256(draft_rendered_payload),
            "draft_text_sha256": _sha256(draft_payload),
            "corrected_text_sha256": _sha256(corrected_payload),
            "options": option_values,
            "crop": list(crop),
            "source_filename": Path(submission.source_filename).name,
            "source_media_type": submission.source_media_type,
            "source_origin": submission.source_origin.strip(),
            "rights_status": submission.rights_status.strip(),
            "decoder": decoder,
            "classifier": classifier,
            "generator_assets": generator_assets,
            "generation": generation,
        }
        record_id = _sha256(_canonical_json(identity))[:32]
        record_dir = self._record_dir(record_id)
        if record_dir.exists():
            return self.get(record_id)

        source_width, source_height = source_size
        canvas_width, canvas_height = processed_size
        x0, y0, x1, y1 = crop
        scale_x = canvas_width / (x1 - x0)
        scale_y = canvas_height / (y1 - y0)
        coordinate_transform = {
            "source_size_px": [source_width, source_height],
            "source_crop_box_px": list(crop),
            "aa_canvas_size_px": [canvas_width, canvas_height],
            "source_to_aa_canvas": {
                "scale_x": scale_x,
                "scale_y": scale_y,
                "offset_x": -x0 * scale_x,
                "offset_y": -y0 * scale_y,
            },
            "aa_canvas_to_source": {
                "scale_x": 1.0 / scale_x,
                "scale_y": 1.0 / scale_y,
                "offset_x": x0,
                "offset_y": y0,
            },
            "font_size_px": options.font_size,
            "line_pitch_px": line_pitch(options.font_size),
        }
        artifacts = {
            "source": _artifact(
                "source-image.bin",
                source_payload,
                submission.source_media_type or "application/octet-stream",
            ),
            "processed": _artifact("processed.png", processed_payload, "image/png"),
            "draft_text": _artifact(
                "draft.txt",
                draft_payload,
                "text/plain; charset=utf-8",
            ),
            "corrected_text": _artifact(
                "corrected.txt",
                corrected_payload,
                "text/plain; charset=utf-8",
            ),
            "draft_rendered": _artifact(
                "draft-rendered.png",
                draft_rendered_payload,
                "image/png",
            ),
            "corrected_rendered": _artifact(
                "corrected-rendered.png",
                corrected_rendered_payload,
                "image/png",
            ),
        }
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "source_filename": Path(submission.source_filename).name,
                "source_media_type": submission.source_media_type,
                "source_origin": submission.source_origin.strip(),
                "rights_status": submission.rights_status.strip(),
            },
            "conversion": {
                "options": option_values,
                "decoder": decoder,
                "classifier": classifier,
                "crop": list(crop),
                "generator_assets": generator_assets,
                "generation": generation,
            },
            "coordinate_transform": coordinate_transform,
            "text_diff": character_diff(
                submission.draft_text,
                submission.corrected_text,
            ),
            "artifacts": artifacts,
            "extensions": {
                "path": "extensions.json",
                "schema_version": EXTENSIONS_SCHEMA_VERSION,
                "supported_slots": list(EXTENSION_SLOTS),
            },
        }
        manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))
        extension_data = {slot: None for slot in EXTENSION_SLOTS}
        extensions = {
            "schema_version": EXTENSIONS_SCHEMA_VERSION,
            "record_id": record_id,
            "revision": 0,
            "data": extension_data,
            "data_sha256": _sha256(_canonical_json(extension_data)),
        }

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            payloads = {
                "source-image.bin": source_payload,
                "processed.png": processed_payload,
                "draft.txt": draft_payload,
                "corrected.txt": corrected_payload,
                "draft-rendered.png": draft_rendered_payload,
                "corrected-rendered.png": corrected_rendered_payload,
                "record.json": (
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8"),
                "extensions.json": (
                    json.dumps(extensions, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8"),
            }
            for relative, payload in payloads.items():
                (temporary / relative).write_bytes(payload)
            try:
                os.replace(temporary, record_dir)
            except OSError:
                if not record_dir.exists():
                    raise
                shutil.rmtree(temporary)
                return self.get(record_id)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self.get(record_id)

    def _load_manifest(self, record_id: str) -> tuple[Path, dict[str, object]]:
        record_dir = self._record_dir(record_id)
        manifest_path = record_dir / "record.json"
        if not manifest_path.is_file():
            raise KeyError(f"Correction pair {record_id} was not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported correction-pair schema version")
        if manifest.get("record_id") != record_id:
            raise ValueError("Correction-pair record ID does not match its directory")
        expected_manifest_hash = manifest.get("manifest_sha256")
        manifest_without_hash = dict(manifest)
        manifest_without_hash.pop("manifest_sha256", None)
        if _sha256(_canonical_json(manifest_without_hash)) != expected_manifest_hash:
            raise ValueError("Correction-pair manifest hash mismatch")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Correction-pair artifact table is missing")
        for value in artifacts.values():
            if not isinstance(value, dict):
                raise ValueError("Correction-pair artifact entry is invalid")
            relative = Path(str(value["path"]))
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("Correction-pair artifact path escapes its record")
            path = record_dir / relative
            payload = path.read_bytes()
            if len(payload) != value.get("bytes"):
                raise ValueError(f"Correction-pair artifact size mismatch: {relative}")
            if _sha256(payload) != value.get("sha256"):
                raise ValueError(f"Correction-pair artifact hash mismatch: {relative}")
        return record_dir, manifest

    def _load_extensions(self, record_dir: Path, record_id: str) -> dict[str, object]:
        extensions = json.loads(
            (record_dir / "extensions.json").read_text(encoding="utf-8")
        )
        if extensions.get("schema_version") != EXTENSIONS_SCHEMA_VERSION:
            raise ValueError("Unsupported correction-pair extensions schema")
        if extensions.get("record_id") != record_id:
            raise ValueError("Correction-pair extensions belong to another record")
        data = extensions.get("data")
        if not isinstance(data, dict) or set(data) != set(EXTENSION_SLOTS):
            raise ValueError("Correction-pair extension slots are invalid")
        if _sha256(_canonical_json(data)) != extensions.get("data_sha256"):
            raise ValueError("Correction-pair extensions hash mismatch")
        return extensions

    def get(self, record_id: str) -> dict[str, object]:
        record_dir, manifest = self._load_manifest(record_id)
        artifacts = manifest["artifacts"]
        draft_path = record_dir / str(artifacts["draft_text"]["path"])
        corrected_path = record_dir / str(artifacts["corrected_text"]["path"])
        return {
            "manifest": manifest,
            "draft_text": draft_path.read_text(encoding="utf-8"),
            "corrected_text": corrected_path.read_text(encoding="utf-8"),
            "extensions": self._load_extensions(record_dir, record_id),
        }

    def list(self) -> list[dict[str, object]]:
        if not self.root.exists():
            return []
        records: list[dict[str, object]] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or _RECORD_ID.fullmatch(path.name) is None:
                continue
            record = self.get(path.name)
            manifest = record["manifest"]
            records.append(
                {
                    "record_id": path.name,
                    "created_at_utc": manifest["created_at_utc"],
                    "provenance": manifest["provenance"],
                    "text_diff": {
                        key: value
                        for key, value in manifest["text_diff"].items()
                        if key != "operations"
                    },
                }
            )
        return sorted(
            records,
            key=lambda record: (str(record["created_at_utc"]), str(record["record_id"])),
            reverse=True,
        )

    def update_extensions(
        self,
        record_id: str,
        updates: dict[str, object],
    ) -> dict[str, object]:
        if not updates:
            raise ValueError("At least one extension slot must be provided")
        unknown = sorted(set(updates) - set(EXTENSION_SLOTS))
        if unknown:
            raise ValueError(f"Unsupported correction-pair extension slots: {unknown}")
        record_dir, _ = self._load_manifest(record_id)
        extensions = self._load_extensions(record_dir, record_id)
        data = dict(extensions["data"])
        data.update(updates)
        if len(_canonical_json(data)) > MAX_EXTENSION_BYTES:
            raise ValueError("Correction-pair extension data exceeds 10 MB")
        updated = {
            "schema_version": EXTENSIONS_SCHEMA_VERSION,
            "record_id": record_id,
            "revision": int(extensions["revision"]) + 1,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "data_sha256": _sha256(_canonical_json(data)),
        }
        payload = (json.dumps(updated, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        temporary = record_dir / f".extensions-{os.getpid()}-{uuid4().hex}.tmp"
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, record_dir / "extensions.json")
        finally:
            temporary.unlink(missing_ok=True)
        return self._load_extensions(record_dir, record_id)

    def asset(self, record_id: str, asset_name: str) -> tuple[Path, str]:
        record_dir, manifest = self._load_manifest(record_id)
        artifacts = manifest["artifacts"]
        if asset_name not in artifacts:
            raise KeyError(f"Unknown correction-pair asset: {asset_name}")
        artifact = artifacts[asset_name]
        return record_dir / str(artifact["path"]), str(artifact["media_type"])

    def verify_regeneration(
        self,
        record_id: str,
        converter: Callable[[bytes, ConversionOptions], ConversionResult],
    ) -> dict[str, object]:
        record_dir, manifest = self._load_manifest(record_id)
        artifacts = manifest["artifacts"]
        source_payload = (record_dir / str(artifacts["source"]["path"])).read_bytes()
        options = ConversionOptions(**manifest["conversion"]["options"])
        result = converter(source_payload, options)
        regenerated_processed = decode_png_data_url(result.processed_png)
        regenerated_rendered = decode_png_data_url(result.rendered_png)
        checks = {
            "draft_text": _sha256(result.ascii_text.encode("utf-8"))
            == artifacts["draft_text"]["sha256"],
            "processed": _sha256(regenerated_processed)
            == artifacts["processed"]["sha256"],
            "draft_rendered": _sha256(regenerated_rendered)
            == artifacts["draft_rendered"]["sha256"],
            "crop": list(result.crop) == manifest["conversion"]["crop"],
        }
        for name, asset in manifest["conversion"]["generator_assets"].items():
            relative = Path(str(asset["path"]))
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("Generator asset path escapes the project root")
            current_path = ROOT / relative
            checks[f"generator_asset_{name}"] = (
                current_path.is_file()
                and _sha256(current_path.read_bytes()) == asset["sha256"]
            )
        return {"record_id": record_id, "matches": all(checks.values()), "checks": checks}
