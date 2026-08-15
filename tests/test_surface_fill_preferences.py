from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.contracts import ConversionOptions
from backend.surface_fill_preferences import (
    SurfaceFillCandidateSubmission,
    SurfaceFillPreferenceStore,
    SurfaceFillPreferenceSubmission,
)


def _png(value: int, *, width: int = 120, height: int = 18) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", np.full((height, width), value, dtype=np.uint8)
    )
    assert ok
    return encoded.tobytes()


def _submission() -> SurfaceFillPreferenceSubmission:
    options = ConversionOptions(columns=24, max_rows=12, profile="person")
    processed = _png(255)
    variant_ids = (
        "surface-fill-none-v2",
        "surface-fill-conservative-v2",
        "surface-fill-standard-v2",
        "surface-fill-expanded-v2",
    )
    candidates = tuple(
        SurfaceFillCandidateSubmission(
            variant_id=variant_id,
            display_label=f"候補{chr(ord('A') + index)}",
            method_version=1,
            options=options,
            crop=(0, 0, 120, 18),
            ascii_text=f"candidate-{index}\n",
            rows=1,
            columns=24,
            processed_png=processed,
            rendered_png=_png(250 - index),
            fill_mask_png=_png(255 if index else 0),
            surface={
                "layer_id": "lab-l4-a4-b4",
                "minimum_fill_score": None if index == 0 else 0.30,
                "fill_summary": {"replacement_count": index},
            },
        )
        for index, variant_id in enumerate(variant_ids)
    )
    return SurfaceFillPreferenceSubmission(
        source_bytes=_png(220),
        source_filename="source.png",
        source_media_type="image/png",
        source_origin="test-generated",
        rights_status="test-only",
        comparison_kind="surface-fill-v2",
        generator_version="deepaa-beam-v0.2",
        recipe_version="surface-fill-v2",
        candidates=candidates,
        selected_variant="surface-fill-standard-v2",
        none_usable=False,
    )


def test_surface_fill_preference_store_is_separate_idempotent_and_verifiable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "draft-preferences" / "v2"
    store = SurfaceFillPreferenceStore(root)

    first = store.save(_submission())
    second = store.save(_submission())

    assert first == second
    assert first["schema_version"] == 2
    assert first["comparison_kind"] == "surface-fill-v2"
    assert first["generator_version"] == "deepaa-beam-v0.2"
    assert first["recipe_version"] == "surface-fill-v2"
    assert first["selected_variant"] == "surface-fill-standard-v2"
    assert first["none_usable"] is False
    assert len(first["candidates"]) == 4
    assert len(list(root.iterdir())) == 1


def test_surface_fill_preference_detects_tampered_fill_mask(tmp_path: Path) -> None:
    store = SurfaceFillPreferenceStore(tmp_path / "v2")
    manifest = store.save(_submission())
    record_id = str(manifest["record_id"])
    mask_path = store.root / record_id / "candidate-02-fill-mask.png"
    mask_path.write_bytes(mask_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="artifact mismatch"):
        store.get(record_id)


def test_surface_fill_preference_rejects_lineart_profiles(tmp_path: Path) -> None:
    submission = _submission()
    first = submission.candidates[0]
    invalid = SurfaceFillCandidateSubmission(
        **{**first.__dict__, "options": ConversionOptions(columns=24, profile="lineart")}
    )
    submission = SurfaceFillPreferenceSubmission(
        **{**submission.__dict__, "candidates": (invalid, *submission.candidates[1:])}
    )

    with pytest.raises(ValueError, match="line-art profiles"):
        SurfaceFillPreferenceStore(tmp_path / "v2").save(submission)


def test_surface_recovery_store_keeps_candidate_specific_processed_images(
    tmp_path: Path,
) -> None:
    submission = _submission()
    candidates = tuple(
        SurfaceFillCandidateSubmission(
            **{
                **candidate.__dict__,
                "variant_id": f"recovery-{index}-v3",
                "method_version": 3,
                "processed_png": _png(255 - index),
            }
        )
        for index, candidate in enumerate(submission.candidates, start=1)
    )
    recovery = SurfaceFillPreferenceSubmission(
        **{
            **submission.__dict__,
            "comparison_kind": "surface-recovery-v3",
            "recipe_version": "surface-recovery-v3",
            "candidates": candidates,
            "selected_variant": "recovery-2-v3",
        }
    )
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v3",
        comparison_kind="surface-recovery-v3",
        schema_version=3,
        require_common_processed=False,
    )

    record = store.save(recovery)

    assert record["schema_version"] == 3
    assert "processed" not in record
    assert all(
        "processed" in candidate["artifacts"] for candidate in record["candidates"]
    )


def test_surface_texture_store_accepts_two_candidate_no_preference(tmp_path: Path) -> None:
    original = _submission()
    candidates = tuple(
        SurfaceFillCandidateSubmission(
            **{
                **candidate.__dict__,
                "variant_id": variant,
                "method_version": 6,
            }
        )
        for candidate, variant in zip(
            original.candidates[:2],
            ("surface-texture-baseline-v5", "surface-texture-applied-v5"),
            strict=True,
        )
    )
    submission = SurfaceFillPreferenceSubmission(
        **{
            **original.__dict__,
            "comparison_kind": "surface-texture-v5",
            "recipe_version": "surface-texture-line-separated-v5",
            "candidates": candidates,
            "selected_variant": None,
            "none_usable": False,
        }
    )
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v5",
        comparison_kind="surface-texture-v5",
        schema_version=5,
        candidate_count=2,
        allow_no_preference=True,
    )

    record = store.save(submission)

    assert record["schema_version"] == 5
    assert len(record["candidates"]) == 2
    assert record["selected_variant"] is None
    assert record["none_usable"] is False
