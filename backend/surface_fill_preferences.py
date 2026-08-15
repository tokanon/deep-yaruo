from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .contracts import ConversionOptions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURFACE_FILL_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v2"
)
DEFAULT_SURFACE_RECOVERY_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v3"
)
DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v4"
)
DEFAULT_SURFACE_TEXTURE_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v5"
)
DEFAULT_SURFACE_TEXTURE_PROXIMITY_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v6"
)
SCHEMA_VERSION = 2
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARACTERS = 2_000_000
MAX_TOTAL_ARTIFACT_BYTES = 100 * 1024 * 1024
_RECORD_ID = re.compile(r"[0-9a-f]{32}")
_VARIANT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


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


def _image_size(payload: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("An image artifact could not be decoded")
    return int(image.shape[1]), int(image.shape[0])


@dataclass(frozen=True)
class SurfaceFillCandidateSubmission:
    variant_id: str
    display_label: str
    method_version: int
    options: ConversionOptions
    crop: tuple[int, int, int, int]
    ascii_text: str
    rows: int
    columns: int
    processed_png: bytes
    rendered_png: bytes
    fill_mask_png: bytes
    surface: dict[str, object]


@dataclass(frozen=True)
class SurfaceFillPreferenceSubmission:
    source_bytes: bytes
    source_filename: str
    source_media_type: str
    source_origin: str
    rights_status: str
    comparison_kind: str
    generator_version: str
    recipe_version: str
    candidates: tuple[SurfaceFillCandidateSubmission, ...]
    selected_variant: str | None
    none_usable: bool


class SurfaceFillPreferenceStore:
    def __init__(
        self,
        root: Path = DEFAULT_SURFACE_FILL_PREFERENCE_ROOT,
        *,
        comparison_kind: str = "surface-fill-v2",
        schema_version: int = SCHEMA_VERSION,
        require_common_processed: bool = True,
        candidate_count: int = 4,
        allow_no_preference: bool = False,
    ) -> None:
        self.root = root
        self.comparison_kind = comparison_kind
        self.schema_version = schema_version
        self.require_common_processed = require_common_processed
        self.candidate_count = candidate_count
        self.allow_no_preference = allow_no_preference

    def _record_dir(self, record_id: str) -> Path:
        if _RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("Invalid surface-fill preference record ID")
        return self.root / record_id

    def _validate(self, submission: SurfaceFillPreferenceSubmission) -> None:
        if not submission.source_bytes:
            raise ValueError("source image is empty")
        if len(submission.source_bytes) > MAX_SOURCE_BYTES:
            raise ValueError("source image exceeds 20 MB")
        if not submission.source_filename.strip():
            raise ValueError("source_filename must not be empty")
        if not submission.source_origin.strip() or not submission.rights_status.strip():
            raise ValueError("source provenance must not be empty")
        if submission.comparison_kind != self.comparison_kind:
            raise ValueError(f"comparison_kind must be {self.comparison_kind}")
        if not submission.generator_version.strip() or not submission.recipe_version.strip():
            raise ValueError("generator and recipe versions must not be empty")
        if len(submission.candidates) != self.candidate_count:
            raise ValueError(
                f"{self.comparison_kind} preference requires exactly "
                f"{self.candidate_count} candidates"
            )
        variant_ids = [candidate.variant_id for candidate in submission.candidates]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("candidate variant IDs must be unique")
        if submission.none_usable and submission.selected_variant is not None:
            raise ValueError("a selected candidate cannot also be marked unusable")
        if (
            submission.selected_variant is None
            and not submission.none_usable
            and not self.allow_no_preference
        ):
            raise ValueError("select exactly one candidate or mark all candidates unusable")
        if (
            submission.selected_variant is not None
            and submission.selected_variant not in variant_ids
        ):
            raise ValueError("selected candidate is not part of this comparison")

        source_width, source_height = _image_size(submission.source_bytes)
        common_processed: bytes | None = None
        common_options: ConversionOptions | None = None
        common_crop: tuple[int, int, int, int] | None = None
        total_bytes = len(submission.source_bytes)
        for candidate in submission.candidates:
            if _VARIANT_ID.fullmatch(candidate.variant_id) is None:
                raise ValueError("candidate variant ID is invalid")
            if not candidate.display_label.strip() or candidate.method_version < 1:
                raise ValueError("candidate label and method version are required")
            if len(candidate.ascii_text) > MAX_TEXT_CHARACTERS:
                raise ValueError("candidate AA text is too large")
            if candidate.rows < 1 or candidate.columns < 1:
                raise ValueError("candidate dimensions must be positive")
            options = candidate.options.normalized()
            if options.profile in {"lineart", "background_lineart"}:
                raise ValueError("line-art profiles cannot be surface-fill preferences")
            if options.columns != candidate.columns:
                raise ValueError("candidate columns and options disagree")
            processed_size = _image_size(candidate.processed_png)
            if _image_size(candidate.rendered_png) != processed_size:
                raise ValueError("candidate processed and rendered image sizes differ")
            if _image_size(candidate.fill_mask_png) != processed_size:
                raise ValueError("candidate fill mask and processed image sizes differ")
            x0, y0, x1, y1 = candidate.crop
            if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
                raise ValueError("candidate crop is outside the source image")
            if not isinstance(candidate.surface, dict):
                raise ValueError("candidate surface metadata must be an object")
            _canonical_json(candidate.surface)
            if common_processed is None:
                common_processed = candidate.processed_png
                common_options = options
                common_crop = candidate.crop
            elif self.require_common_processed and candidate.processed_png != common_processed:
                raise ValueError("surface-fill candidates must share extracted lines")
            elif options != common_options or candidate.crop != common_crop:
                raise ValueError("surface-fill candidates must share options and crop")
            total_bytes += (
                len(candidate.ascii_text.encode("utf-8"))
                + len(candidate.rendered_png)
                + len(candidate.fill_mask_png)
                + len(_canonical_json(candidate.surface))
            )
            if not self.require_common_processed:
                total_bytes += len(candidate.processed_png)
        assert common_processed is not None
        if self.require_common_processed:
            total_bytes += len(common_processed)
        if total_bytes > MAX_TOTAL_ARTIFACT_BYTES:
            raise ValueError("surface-fill preference artifacts exceed 100 MB")

    def save(self, submission: SurfaceFillPreferenceSubmission) -> dict[str, object]:
        self._validate(submission)
        source_payload = bytes(submission.source_bytes)
        processed_payload = bytes(submission.candidates[0].processed_png)
        processed_artifact = _artifact("processed.png", processed_payload, "image/png")
        payload_groups: list[dict[str, object]] = []
        identity_candidates: list[dict[str, object]] = []
        for index, candidate in enumerate(submission.candidates, start=1):
            prefix = f"candidate-{index:02d}"
            payloads = {
                "text": candidate.ascii_text.encode("utf-8"),
                "rendered": bytes(candidate.rendered_png),
                "fill_mask": bytes(candidate.fill_mask_png),
            }
            artifacts = {
                "text": _artifact(
                    f"{prefix}.txt", payloads["text"], "text/plain; charset=utf-8"
                ),
                "rendered": _artifact(
                    f"{prefix}-rendered.png", payloads["rendered"], "image/png"
                ),
                "fill_mask": _artifact(
                    f"{prefix}-fill-mask.png", payloads["fill_mask"], "image/png"
                ),
            }
            if not self.require_common_processed:
                payloads["processed"] = bytes(candidate.processed_png)
                artifacts["processed"] = _artifact(
                    f"{prefix}-processed.png", payloads["processed"], "image/png"
                )
            item = {
                "variant_id": candidate.variant_id,
                "display_label": candidate.display_label,
                "method_version": candidate.method_version,
                "rows": candidate.rows,
                "columns": candidate.columns,
                "surface": candidate.surface,
                "artifacts": artifacts,
            }
            identity_candidates.append(item)
            payload_groups.append({"payloads": payloads, "artifacts": artifacts})

        options = submission.candidates[0].options.normalized()
        crop = list(submission.candidates[0].crop)
        identity = {
            "schema_version": self.schema_version,
            "comparison_kind": submission.comparison_kind,
            "generator_version": submission.generator_version,
            "recipe_version": submission.recipe_version,
            "source_sha256": _sha256(source_payload),
            "options": asdict(options),
            "crop": crop,
            "candidates": identity_candidates,
            "selected_variant": submission.selected_variant,
            "none_usable": submission.none_usable,
        }
        if self.require_common_processed:
            identity["processed"] = processed_artifact
        record_id = _sha256(_canonical_json(identity))[:32]
        record_dir = self._record_dir(record_id)
        if record_dir.exists():
            return self.get(record_id)

        source_artifact = _artifact(
            "source-image.bin",
            source_payload,
            submission.source_media_type or "application/octet-stream",
        )
        manifest: dict[str, object] = {
            **identity,
            "record_id": record_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "filename": submission.source_filename,
                "origin": submission.source_origin,
                "rights_status": submission.rights_status,
                "artifact": source_artifact,
            },
        }
        manifest.pop("source_sha256")
        manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".surface-fill-", dir=self.root))
        try:
            (temporary / source_artifact["path"]).write_bytes(source_payload)
            if self.require_common_processed:
                (temporary / processed_artifact["path"]).write_bytes(processed_payload)
            for group in payload_groups:
                payloads = group["payloads"]
                artifacts = group["artifacts"]
                assert isinstance(payloads, dict) and isinstance(artifacts, dict)
                for name in artifacts:
                    (temporary / artifacts[name]["path"]).write_bytes(payloads[name])
            (temporary / "record.json").write_bytes(_canonical_json(manifest))
            try:
                os.replace(temporary, record_dir)
            except FileExistsError:
                shutil.rmtree(temporary, ignore_errors=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self.get(record_id)

    def get(self, record_id: str) -> dict[str, object]:
        record_dir = self._record_dir(record_id)
        manifest_path = record_dir / "record.json"
        if not manifest_path.is_file():
            raise KeyError(record_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_hash = manifest.get("manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        if expected_manifest_hash != _sha256(_canonical_json(unsigned)):
            raise ValueError("surface-fill preference manifest hash mismatch")

        artifacts = [manifest["source"]["artifact"]]
        if "processed" in manifest:
            artifacts.append(manifest["processed"])
        for candidate in manifest["candidates"]:
            artifacts.extend(candidate["artifacts"].values())
        for artifact in artifacts:
            path = record_dir / artifact["path"]
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or _sha256(path.read_bytes()) != artifact["sha256"]
            ):
                raise ValueError(
                    f"surface-fill preference artifact mismatch: {artifact['path']}"
                )
        return manifest

    def list(self) -> list[dict[str, object]]:
        if not self.root.is_dir():
            return []
        records = []
        for path in self.root.iterdir():
            if path.is_dir() and _RECORD_ID.fullmatch(path.name):
                records.append(self.get(path.name))
        records.sort(key=lambda item: str(item["created_at_utc"]), reverse=True)
        return records
