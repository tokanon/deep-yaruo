from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.contracts import ConversionOptions
from backend.correction_pairs import decode_png_data_url
from backend.draft_preferences import (
    DraftCandidateSubmission,
    DraftPreferenceStore,
    DraftPreferenceSubmission,
)
from backend.image_io import image_to_data_url


def png(width: int, height: int, value: int = 255) -> bytes:
    return decode_png_data_url(
        image_to_data_url(np.full((height, width), value, dtype=np.uint8))
    )


def candidate(variant_id: str, value: int) -> DraftCandidateSubmission:
    return DraftCandidateSubmission(
        variant_id=variant_id,
        display_label=variant_id.upper(),
        method_version=1,
        options=ConversionOptions(columns=24, profile="person"),
        crop=(0, 0, 32, 24),
        ascii_text=f"{variant_id}\n",
        rows=1,
        columns=24,
        processed_png=png(24, 18, value),
        rendered_png=png(24, 18, value),
    )


def submission() -> DraftPreferenceSubmission:
    return DraftPreferenceSubmission(
        source_bytes=png(32, 24),
        source_filename="source.png",
        source_media_type="image/png",
        source_origin="test-generated",
        rights_status="test-only",
        candidates=(candidate("baseline-v1", 255), candidate("detail-v1", 220)),
        selected_variant="detail-v1",
        none_usable=False,
    )


def test_draft_preference_store_saves_reproducible_candidates(tmp_path: Path) -> None:
    store = DraftPreferenceStore(tmp_path / "preferences")
    first = store.save(submission())
    second = store.save(submission())

    assert first["record_id"] == second["record_id"]
    assert first["selection"] == {
        "selected_variant": "detail-v1",
        "none_usable": False,
    }
    assert [item["variant_id"] for item in first["candidates"]] == [
        "baseline-v1",
        "detail-v1",
    ]
    assert len(store.list()) == 1


def test_draft_preference_requires_one_choice_or_all_unusable(tmp_path: Path) -> None:
    store = DraftPreferenceStore(tmp_path / "preferences")
    invalid = DraftPreferenceSubmission(
        **{
            **submission().__dict__,
            "selected_variant": None,
            "none_usable": False,
        }
    )

    with pytest.raises(ValueError, match="select exactly one"):
        store.save(invalid)


def test_draft_preference_accepts_all_candidates_unusable(tmp_path: Path) -> None:
    store = DraftPreferenceStore(tmp_path / "preferences")
    all_unusable = DraftPreferenceSubmission(
        **{
            **submission().__dict__,
            "selected_variant": None,
            "none_usable": True,
        }
    )

    record = store.save(all_unusable)

    assert record["selection"] == {
        "selected_variant": None,
        "none_usable": True,
    }


def test_draft_preference_detects_artifact_tampering(tmp_path: Path) -> None:
    store = DraftPreferenceStore(tmp_path / "preferences")
    record = store.save(submission())
    record_dir = store.root / str(record["record_id"])
    (record_dir / "candidate-01.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact mismatch"):
        store.get(str(record["record_id"]))
