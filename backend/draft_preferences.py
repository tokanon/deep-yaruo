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
DEFAULT_DRAFT_PREFERENCE_ROOT = (
    ROOT / "datasets" / "incoming" / "draft-preferences" / "v1"
)
SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARACTERS = 2_000_000
MAX_TOTAL_ARTIFACT_BYTES = 80 * 1024 * 1024
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
class DraftCandidateSubmission:
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


@dataclass(frozen=True)
class DraftPreferenceSubmission:
    source_bytes: bytes
    source_filename: str
    source_media_type: str
    source_origin: str
    rights_status: str
    candidates: tuple[DraftCandidateSubmission, ...]
    selected_variant: str | None
    none_usable: bool


class DraftPreferenceStore:
    def __init__(self, root: Path = DEFAULT_DRAFT_PREFERENCE_ROOT) -> None:
        self.root = root

    def _record_dir(self, record_id: str) -> Path:
        if _RECORD_ID.fullmatch(record_id) is None:
            raise ValueError("Invalid draft-preference record ID")
        return self.root / record_id

    def _validate(self, submission: DraftPreferenceSubmission) -> None:
        if not submission.source_bytes:
            raise ValueError("source image is empty")
        if len(submission.source_bytes) > MAX_SOURCE_BYTES:
            raise ValueError("source image exceeds 20 MB")
        if not submission.source_filename.strip():
            raise ValueError("source_filename must not be empty")
        if not submission.source_origin.strip() or not submission.rights_status.strip():
            raise ValueError("source provenance must not be empty")
        if not 2 <= len(submission.candidates) <= 8:
            raise ValueError("draft preference requires 2 to 8 candidates")
        variant_ids = [candidate.variant_id for candidate in submission.candidates]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("candidate variant IDs must be unique")
        if submission.none_usable == (submission.selected_variant is not None):
            raise ValueError("select exactly one candidate or mark all candidates unusable")
        if (
            submission.selected_variant is not None
            and submission.selected_variant not in variant_ids
        ):
            raise ValueError("selected candidate is not part of this comparison")

        source_width, source_height = _image_size(submission.source_bytes)
        total_bytes = len(submission.source_bytes)
        for candidate in submission.candidates:
            if _VARIANT_ID.fullmatch(candidate.variant_id) is None:
                raise ValueError("candidate variant ID is invalid")
            if not candidate.display_label.strip():
                raise ValueError("candidate display label must not be empty")
            if candidate.method_version < 1:
                raise ValueError("candidate method version must be positive")
            if len(candidate.ascii_text) > MAX_TEXT_CHARACTERS:
                raise ValueError("candidate AA text is too large")
            if candidate.rows < 1 or candidate.columns < 1:
                raise ValueError("candidate dimensions must be positive")
            options = candidate.options.normalized()
            if options.columns != candidate.columns:
                raise ValueError("candidate columns and options disagree")
            processed_size = _image_size(candidate.processed_png)
            if _image_size(candidate.rendered_png) != processed_size:
                raise ValueError("candidate processed and rendered image sizes differ")
            x0, y0, x1, y1 = candidate.crop
            if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
                raise ValueError("candidate crop is outside the source image")
            total_bytes += (
                len(candidate.ascii_text.encode("utf-8"))
                + len(candidate.processed_png)
                + len(candidate.rendered_png)
            )
        if total_bytes > MAX_TOTAL_ARTIFACT_BYTES:
            raise ValueError("draft-preference artifacts exceed 80 MB")

    def save(self, submission: DraftPreferenceSubmission) -> dict[str, object]:
        self._validate(submission)
        source_payload = bytes(submission.source_bytes)
        candidate_payloads: list[dict[str, object]] = []
        identity_candidates: list[dict[str, object]] = []
        for index, candidate in enumerate(submission.candidates, start=1):
            prefix = f"candidate-{index:02d}"
            text_payload = candidate.ascii_text.encode("utf-8")
            processed_payload = bytes(candidate.processed_png)
            rendered_payload = bytes(candidate.rendered_png)
            artifacts = {
                "text": _artifact(f"{prefix}.txt", text_payload, "text/plain; charset=utf-8"),
                "processed": _artifact(f"{prefix}-processed.png", processed_payload, "image/png"),
                "rendered": _artifact(f"{prefix}-rendered.png", rendered_payload, "image/png"),
            }
            candidate_payloads.append(
                {
                    "candidate": candidate,
                    "artifacts": artifacts,
                    "payloads": {
                        "text": text_payload,
                        "processed": processed_payload,
                        "rendered": rendered_payload,
                    },
                }
            )
            identity_candidates.append(
                {
                    "variant_id": candidate.variant_id,
                    "method_version": candidate.method_version,
                    "options": asdict(candidate.options.normalized()),
                    "crop": list(candidate.crop),
                    "rows": candidate.rows,
                    "columns": candidate.columns,
                    "artifacts": artifacts,
                }
            )

        identity = {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": _sha256(source_payload),
            "candidates": identity_candidates,
            "selected_variant": submission.selected_variant,
            "none_usable": submission.none_usable,
        }
        record_id = _sha256(_canonical_json(identity))[:32]
        record_dir = self._record_dir(record_id)
        if record_dir.exists():
            return self.get(record_id)

        source_artifact = _artifact(
            "source-image.bin",
            source_payload,
            submission.source_media_type or "application/octet-stream",
        )
        manifest_candidates = []
        for payload_group in candidate_payloads:
            candidate = payload_group["candidate"]
            assert isinstance(candidate, DraftCandidateSubmission)
            manifest_candidates.append(
                {
                    "variant_id": candidate.variant_id,
                    "display_label": candidate.display_label,
                    "method_version": candidate.method_version,
                    "options": asdict(candidate.options.normalized()),
                    "crop": list(candidate.crop),
                    "rows": candidate.rows,
                    "columns": candidate.columns,
                    "artifacts": payload_group["artifacts"],
                }
            )
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "filename": submission.source_filename,
                "origin": submission.source_origin,
                "rights_status": submission.rights_status,
                "artifact": source_artifact,
            },
            "selection": {
                "selected_variant": submission.selected_variant,
                "none_usable": submission.none_usable,
            },
            "candidates": manifest_candidates,
        }
        manifest["manifest_sha256"] = _sha256(_canonical_json(manifest))

        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".draft-preference-", dir=self.root))
        try:
            (temporary / source_artifact["path"]).write_bytes(source_payload)
            for payload_group in candidate_payloads:
                artifacts = payload_group["artifacts"]
                payloads = payload_group["payloads"]
                assert isinstance(artifacts, dict) and isinstance(payloads, dict)
                for name in ("text", "processed", "rendered"):
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
        expected_manifest_hash = manifest.pop("manifest_sha256", None)
        actual_manifest_hash = _sha256(_canonical_json(manifest))
        manifest["manifest_sha256"] = expected_manifest_hash
        if expected_manifest_hash != actual_manifest_hash:
            raise ValueError("draft-preference manifest hash mismatch")
        artifact_groups = [manifest["source"]["artifact"]]
        for candidate in manifest["candidates"]:
            artifact_groups.extend(candidate["artifacts"].values())
        for artifact in artifact_groups:
            path = record_dir / artifact["path"]
            payload = path.read_bytes()
            if len(payload) != artifact["bytes"] or _sha256(payload) != artifact["sha256"]:
                raise ValueError(f"draft-preference artifact mismatch: {artifact['path']}")
        return manifest

    def list(self) -> list[dict[str, object]]:
        if not self.root.exists():
            return []
        records = []
        for path in self.root.iterdir():
            if path.is_dir() and _RECORD_ID.fullmatch(path.name):
                records.append(self.get(path.name))
        records.sort(key=lambda record: str(record["created_at_utc"]), reverse=True)
        return records
