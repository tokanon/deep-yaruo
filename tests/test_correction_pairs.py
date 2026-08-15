from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

import backend.correction_pairs as correction_pair_module
from backend.contracts import ConversionOptions, ConversionResult
from backend.correction_pairs import (
    EXTENSION_SLOTS,
    CorrectionPairStore,
    CorrectionPairSubmission,
    character_diff,
)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def png_data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def make_submission() -> CorrectionPairSubmission:
    source = np.full((30, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(source, (5, 4), (34, 23), (0, 0, 0), 1)
    processed = np.full((18, 32), 255, dtype=np.uint8)
    cv2.line(processed, (2, 15), (29, 2), 0, 1)
    draft_rendered = np.full((18, 32), 255, dtype=np.uint8)
    cv2.line(draft_rendered, (1, 16), (30, 1), 0, 1)
    return CorrectionPairSubmission(
        source_bytes=encode_png(source),
        source_filename="example.png",
        source_media_type="image/png",
        source_origin="test-generated",
        rights_status="test-only",
        options=ConversionOptions(
            columns=32,
            detail=63,
            crop_x=0.125,
            crop_y=0.1,
            crop_width=0.75,
            crop_height=2 / 3,
            profile="lineart",
        ),
        crop=(5, 4, 35, 24),
        draft_text="／＼\n",
        corrected_text="／｜\n",
        processed_png=encode_png(processed),
        draft_rendered_png=encode_png(draft_rendered),
    )


def test_character_diff_records_replacements_insertions_and_deletions() -> None:
    diff = character_diff("abc\ndef", "axc\ndef!")

    assert diff["operation_count"] == 2
    assert diff["replaced_draft_characters"] == 1
    assert diff["replaced_corrected_characters"] == 1
    assert diff["inserted_characters"] == 1
    assert diff["deleted_characters"] == 0


def test_correction_pair_round_trip_is_hashed_and_extensible(tmp_path: Path) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    submission = make_submission()

    saved = store.save(submission)
    manifest = saved["manifest"]
    record_id = manifest["record_id"]
    record_dir = store.root / record_id
    manifest_payload = (record_dir / "record.json").read_bytes()

    assert manifest["schema_version"] == 1
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["conversion"]["options"] == asdict(submission.options.normalized())
    assert manifest["conversion"]["crop"] == list(submission.crop)
    assert manifest["coordinate_transform"]["source_size_px"] == [40, 30]
    assert manifest["coordinate_transform"]["aa_canvas_size_px"] == [32, 18]
    assert manifest["coordinate_transform"]["source_crop_box_px"] == [5, 4, 35, 24]
    assert saved["draft_text"] == submission.draft_text
    assert saved["corrected_text"] == submission.corrected_text
    assert set(saved["extensions"]["data"]) == set(EXTENSION_SLOTS)
    assert all(value is None for value in saved["extensions"]["data"].values())

    corrected_rendered = record_dir / manifest["artifacts"]["corrected_rendered"]["path"]
    rendered = cv2.imread(str(corrected_rendered), cv2.IMREAD_GRAYSCALE)
    assert rendered is not None
    assert rendered.shape == (18, 32)

    duplicate = store.save(submission)
    assert duplicate["manifest"]["record_id"] == record_id
    assert [path.name for path in store.root.iterdir() if path.is_dir()] == [record_id]
    assert store.list()[0]["record_id"] == record_id

    extensions = store.update_extensions(
        record_id,
        {"surface_proposals": {"schema_version": 1, "surfaces": []}},
    )
    assert extensions["revision"] == 1
    assert extensions["data"]["surface_proposals"]["surfaces"] == []
    assert (record_dir / "record.json").read_bytes() == manifest_payload

    source_path, media_type = store.asset(record_id, "source")
    assert source_path.read_bytes() == submission.source_bytes
    assert media_type == "image/png"


def test_correction_pair_regeneration_verifies_every_draft_artifact(tmp_path: Path) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    submission = make_submission()
    record = store.save(submission)
    record_id = record["manifest"]["record_id"]

    def converter(payload: bytes, options: ConversionOptions) -> ConversionResult:
        assert payload == submission.source_bytes
        assert options == submission.options.normalized()
        return ConversionResult(
            ascii_text=submission.draft_text,
            rows=1,
            columns=options.columns,
            processed_png=png_data_url(submission.processed_png),
            rendered_png=png_data_url(submission.draft_rendered_png),
            crop=submission.crop,
        )

    verified = store.verify_regeneration(record_id, converter)
    assert verified["matches"] is True
    assert all(verified["checks"].values())

    def changed_converter(payload: bytes, options: ConversionOptions) -> ConversionResult:
        result = converter(payload, options)
        result.ascii_text = "changed\n"
        return result

    changed = store.verify_regeneration(record_id, changed_converter)
    assert changed["matches"] is False
    assert changed["checks"]["draft_text"] is False


def test_surface_model_correction_pair_pins_ls_model_and_vocabulary(tmp_path: Path) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    submission = make_submission()
    surface_submission = CorrectionPairSubmission(
        **{
            **submission.__dict__,
            "generation": {
                "generator_version": "deepaa-surface-v0-ls",
                "recipe_version": "deepaa-surface-v0-provisional-v1",
                "fallback_used": False,
                "stages": [
                    {
                        "model_sha256": "model-hash",
                        "checkpoint_sha256": "checkpoint-hash",
                    }
                ],
            },
        }
    )

    saved = store.save(surface_submission)
    conversion = saved["manifest"]["conversion"]

    assert conversion["decoder"] == "dense-learned-start-exact-width-dp-v1"
    assert conversion["classifier"] == "deepaa-surface-v0-ls"
    assert "surface_model" in conversion["generator_assets"]
    assert "surface_vocabulary" in conversion["generator_assets"]


def test_correction_pair_detects_artifact_and_manifest_tampering(tmp_path: Path) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    record = store.save(make_submission())
    record_id = record["manifest"]["record_id"]
    record_dir = store.root / record_id

    corrected_path = record_dir / "corrected.txt"
    original_corrected = corrected_path.read_bytes()
    corrected_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        store.get(record_id)
    corrected_path.write_bytes(original_corrected)

    manifest_path = record_dir / "record.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["conversion"]["options"]["columns"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        store.get(record_id)


def test_correction_pair_rejects_unmodified_text_and_unknown_extension(tmp_path: Path) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    submission = make_submission()
    unchanged = CorrectionPairSubmission(
        **{
            **submission.__dict__,
            "corrected_text": submission.draft_text,
        }
    )
    with pytest.raises(ValueError, match="must differ"):
        store.save(unchanged)

    record = store.save(submission)
    with pytest.raises(ValueError, match="Unsupported"):
        store.update_extensions(
            record["manifest"]["record_id"],
            {"future_unknown_slot": {}},
        )


def test_correction_pair_extension_limit_applies_to_the_combined_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    record = store.save(make_submission())
    record_id = record["manifest"]["record_id"]
    first_value = {"payload": "x" * 100}
    combined = dict(record["extensions"]["data"])
    combined["surface_proposals"] = first_value
    monkeypatch.setattr(
        correction_pair_module,
        "MAX_EXTENSION_BYTES",
        len(correction_pair_module._canonical_json(combined)) + 10,
    )

    store.update_extensions(record_id, {"surface_proposals": first_value})
    with pytest.raises(ValueError, match="extension data exceeds"):
        store.update_extensions(
            record_id,
            {"surface_metrics": {"payload": "y" * 100}},
        )
