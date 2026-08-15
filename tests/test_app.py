from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import numpy as np
from fastapi import UploadFile
from starlette.datastructures import Headers

import backend.app as app_module
from backend.app import FONT_FILE, serialize_cp932_text
from backend.contracts import ConversionOptions, ConversionResult, TextTransportRequest
from backend.correction_pairs import CorrectionPairStore
from backend.cumulative_generator import CumulativeGeneration
from backend.draft_preferences import DraftPreferenceStore
from backend.image_io import image_to_data_url
from backend.surface_fill_preferences import SurfaceFillPreferenceStore


def test_runtime_assets_exist_in_structured_directories() -> None:
    assert FONT_FILE.as_posix().endswith("assets/fonts/Saitamaar.ttf")
    assert FONT_FILE.is_file()


def test_cp932_transport_endpoint_preserves_supported_characters() -> None:
    result = serialize_cp932_text(TextTransportRequest(text="～∥█&<>"))

    assert result == {"text": "～∥&#9608;&<>", "encoding": "cp932-ncr"}


def upload(payload: bytes, filename: str = "input.png") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(payload),
        filename=filename,
        headers=Headers({"content-type": "image/png"}),
    )


def test_convert_endpoint_returns_the_normalized_options(monkeypatch) -> None:
    source_png = image_to_data_url(np.full((12, 16), 255, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_png)
    captured = {}

    def fake_convert(payload, options):
        captured["payload"] = payload
        captured["options"] = options
        return ConversionResult(
            ascii_text="draft\n",
            rows=1,
            columns=options.columns,
            processed_png=source_png,
            rendered_png=source_png,
            crop=(0, 0, 16, 12),
        )

    monkeypatch.setattr(app_module, "convert_deepaa", fake_convert)
    response = asyncio.run(
        app_module.convert_image(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="auto",
            cumulative=False,
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 140
    assert response["options"]["columns"] == 140
    assert response["options"]["font_size"] == 16
    assert response["pipeline"]["enabled"] is False


def test_convert_endpoint_uses_cumulative_generation_by_default(monkeypatch) -> None:
    source_png = image_to_data_url(np.full((12, 16), 255, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_png)
    captured = {}

    def fake_cumulative(payload, options):
        captured["payload"] = payload
        captured["options"] = options
        return CumulativeGeneration(
            result=ConversionResult(
                ascii_text="cumulative\n",
                rows=1,
                columns=options.columns,
                processed_png=source_png,
                rendered_png=source_png,
                crop=(0, 0, 16, 12),
            ),
            pipeline={
                "generator_version": "deepaa-surface-v0-ls",
                "recipe_version": "deepaa-surface-v0-provisional-v1",
                "enabled": True,
                "applied": True,
                "stages": [],
            },
        )

    monkeypatch.setattr(app_module, "generate_cumulative", fake_cumulative)
    response = asyncio.run(
        app_module.convert_image(
            image=upload(source_payload),
            columns=48,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=32,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 48
    assert response["ascii"] == "cumulative\n"
    assert (
        response["pipeline"]["recipe_version"]
        == "deepaa-surface-v0-provisional-v1"
    )


def test_correction_pair_api_saves_and_reads_a_local_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = CorrectionPairStore(tmp_path / "pairs")
    monkeypatch.setattr(app_module, "CORRECTION_PAIR_STORE", store)
    source_data_url = image_to_data_url(np.full((24, 32), 255, dtype=np.uint8))
    processed_data_url = image_to_data_url(np.full((18, 24), 255, dtype=np.uint8))

    response = asyncio.run(
        app_module.save_correction_pair(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            options_json=json.dumps({"columns": 24, "profile": "lineart"}),
            crop_json=json.dumps([0, 0, 32, 24]),
            draft_text="draft\n",
            corrected_text="corrected\n",
            processed_png=processed_data_url,
            draft_rendered_png=processed_data_url,
            generation_json=json.dumps(
                {
                    "generator_version": "deepaa-cumulative-g1",
                    "recipe_version": "p6r-cumulative-g1-v1",
                    "enabled": True,
                    "stages": [],
                }
            ),
            source_origin="test-generated",
            rights_status="test-only",
        )
    )
    record_id = response["record"]["manifest"]["record_id"]

    loaded = app_module.get_correction_pair(record_id)
    assert loaded["draft_text"] == "draft\n"
    assert loaded["corrected_text"] == "corrected\n"
    assert (
        loaded["manifest"]["conversion"]["generation"]["recipe_version"]
        == "p6r-cumulative-g1-v1"
    )
    assert app_module.list_correction_pairs()["records"][0]["record_id"] == record_id

    analysis = app_module.analyze_correction_pair_surface_data(record_id)
    assert analysis["record_id"] == record_id
    assert analysis["extension_revision"] == 1
    assert all(
        value is not None
        for value in store.get(record_id)["extensions"]["data"].values()
    )


def test_draft_preference_api_saves_a_selected_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = DraftPreferenceStore(tmp_path / "draft-preferences")
    monkeypatch.setattr(app_module, "DRAFT_PREFERENCE_STORE", store)
    source_data_url = image_to_data_url(np.full((24, 32), 255, dtype=np.uint8))
    candidate_data_url = image_to_data_url(np.full((18, 24), 255, dtype=np.uint8))
    candidates = [
        {
            "variant_id": variant,
            "display_label": label,
            "method_version": 1,
            "options": {"columns": 24, "profile": "person"},
            "crop": [0, 0, 32, 24],
            "ascii": f"{label}\n",
            "rows": 1,
            "columns": 24,
            "processed_png": candidate_data_url,
            "rendered_png": candidate_data_url,
        }
        for variant, label in (("baseline-v1", "A"), ("detail-v1", "B"))
    ]

    response = asyncio.run(
        app_module.save_draft_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            candidates_json=json.dumps(candidates),
            selected_variant="detail-v1",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["selection"]["selected_variant"] == "detail-v1"
    assert app_module.list_draft_preferences()["records"][0]["record_id"] == record["record_id"]


def test_surface_fill_comparison_endpoint_uses_normalized_options(monkeypatch) -> None:
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_data_url)
    captured = {}

    def fake_comparison(payload, options):
        captured["payload"] = payload
        captured["options"] = options
        return {
            "comparisonKind": "surface-fill-v2",
            "generatorVersion": "deepaa-beam-v0.2",
            "recipeVersion": "surface-fill-v2",
            "coordinateSpaceId": "test-space",
            "candidates": [],
        }

    monkeypatch.setattr(app_module, "generate_surface_fill_comparison", fake_comparison)
    response = asyncio.run(
        app_module.compare_surface_fill(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 140
    assert response["comparisonKind"] == "surface-fill-v2"


def test_surface_fill_preference_api_saves_v2_without_touching_v1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    v1_store = DraftPreferenceStore(tmp_path / "draft-preferences" / "v1")
    v2_store = SurfaceFillPreferenceStore(tmp_path / "draft-preferences" / "v2")
    monkeypatch.setattr(app_module, "DRAFT_PREFERENCE_STORE", v1_store)
    monkeypatch.setattr(app_module, "SURFACE_FILL_PREFERENCE_STORE", v2_store)
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    artifact_data_url = image_to_data_url(np.full((18, 24), 255, dtype=np.uint8))
    variants = (
        "surface-fill-none-v2",
        "surface-fill-conservative-v2",
        "surface-fill-standard-v2",
        "surface-fill-expanded-v2",
    )
    comparison = {
        "comparisonKind": "surface-fill-v2",
        "generatorVersion": "deepaa-beam-v0.2",
        "recipeVersion": "surface-fill-v2",
        "coordinateSpaceId": "test-space",
        "candidates": [
            {
                "variantId": variant,
                "displayLabel": f"候補{chr(ord('A') + index)}",
                "methodVersion": 1,
                "result": {
                    "ascii": f"candidate-{index}\n",
                    "rows": 1,
                    "columns": 24,
                    "processedPng": artifact_data_url,
                    "renderedPng": artifact_data_url,
                    "crop": [0, 0, 32, 24],
                    "options": {
                        "columns": 24,
                        "profile": "person",
                    },
                },
                "fillMaskPng": artifact_data_url,
                "surface": {
                    "layer_id": "lab-l4-a4-b4",
                    "minimum_fill_score": None if index == 0 else 0.30,
                },
            }
            for index, variant in enumerate(variants)
        ],
    }

    response = asyncio.run(
        app_module.save_surface_fill_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            comparison_json=json.dumps(comparison),
            selected_variant="surface-fill-standard-v2",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["schema_version"] == 2
    assert record["selected_variant"] == "surface-fill-standard-v2"
    assert app_module.list_surface_fill_preferences()["records"][0]["record_id"] == record["record_id"]
    assert app_module.list_draft_preferences()["records"] == []


def test_surface_recovery_comparison_endpoint_uses_normalized_options(
    monkeypatch,
) -> None:
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_data_url)
    captured = {}

    def fake_comparison(payload, options):
        captured["payload"] = payload
        captured["options"] = options
        return {
            "comparisonKind": "surface-recovery-v3",
            "generatorVersion": "deepaa-beam-v0.2",
            "recipeVersion": "surface-recovery-v3",
            "coordinateSpaceId": "test-space",
            "candidates": [],
        }

    monkeypatch.setattr(
        app_module, "generate_surface_recovery_comparison", fake_comparison
    )
    response = asyncio.run(
        app_module.compare_surface_recovery(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 140
    assert response["comparisonKind"] == "surface-recovery-v3"


def test_surface_recovery_preference_api_saves_candidate_specific_lines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v3",
        comparison_kind="surface-recovery-v3",
        schema_version=3,
        require_common_processed=False,
    )
    monkeypatch.setattr(app_module, "SURFACE_RECOVERY_PREFERENCE_STORE", store)
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    variants = (
        "recovery-current-v3",
        "recovery-lines-v3",
        "recovery-surfaces-v3",
        "recovery-combined-v3",
    )
    candidates = []
    for index, variant in enumerate(variants):
        processed = image_to_data_url(
            np.full((18, 192), 255 - index, dtype=np.uint8)
        )
        candidates.append(
            {
                "variantId": variant,
                "displayLabel": f"候補{chr(ord('A') + index)}",
                "methodVersion": 3,
                "result": {
                    "ascii": f"candidate-{index}\n",
                    "rows": 1,
                    "columns": 24,
                    "processedPng": processed,
                    "renderedPng": processed,
                    "crop": [0, 0, 32, 24],
                    "options": {"columns": 24, "profile": "person"},
                },
                "fillMaskPng": processed,
                "surface": {"mechanical_gate": True},
            }
        )
    comparison = {
        "comparisonKind": "surface-recovery-v3",
        "generatorVersion": "deepaa-beam-v0.2",
        "recipeVersion": "surface-recovery-v3",
        "coordinateSpaceId": "test-space",
        "candidates": candidates,
    }

    response = asyncio.run(
        app_module.save_surface_recovery_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            comparison_json=json.dumps(comparison),
            selected_variant="recovery-combined-v3",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["schema_version"] == 3
    assert record["selected_variant"] == "recovery-combined-v3"
    assert all("processed" in item["artifacts"] for item in record["candidates"])


def test_information_recovery_comparison_uses_v3_store_and_normalized_options(
    monkeypatch,
) -> None:
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_data_url)
    captured = {}

    def fake_comparison(payload, options, store):
        captured["payload"] = payload
        captured["options"] = options
        captured["store"] = store
        return {
            "comparisonKind": "information-recovery-v4",
            "generatorVersion": "deepaa-beam-v0.2",
            "recipeVersion": "information-recovery-v4",
            "coordinateSpaceId": "test-space",
            "candidates": [],
        }

    monkeypatch.setattr(
        app_module, "generate_information_recovery_comparison", fake_comparison
    )
    response = asyncio.run(
        app_module.compare_information_recovery(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 140
    assert captured["store"] is app_module.SURFACE_RECOVERY_PREFERENCE_STORE
    assert response["comparisonKind"] == "information-recovery-v4"


def test_information_recovery_preference_api_saves_v4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v4",
        comparison_kind="information-recovery-v4",
        schema_version=4,
        require_common_processed=True,
    )
    monkeypatch.setattr(app_module, "INFORMATION_RECOVERY_PREFERENCE_STORE", store)
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    artifact_data_url = image_to_data_url(np.full((18, 192), 255, dtype=np.uint8))
    variants = (
        "information-current-v4",
        "information-surfaces-v4",
        "information-placement-v4",
        "information-combined-v4",
    )
    candidates = [
        {
            "variantId": variant,
            "displayLabel": f"候補{chr(ord('A') + index)}",
            "methodVersion": 4,
            "result": {
                "ascii": f"candidate-{index}\n",
                "rows": 1,
                "columns": 24,
                "processedPng": artifact_data_url,
                "renderedPng": artifact_data_url,
                "crop": [0, 0, 32, 24],
                "options": {"columns": 24, "profile": "person"},
            },
            "fillMaskPng": artifact_data_url,
            "surface": {
                "source_v3_record_id": "a" * 32,
                "mechanical_safety_gate": True,
            },
        }
        for index, variant in enumerate(variants)
    ]
    comparison = {
        "comparisonKind": "information-recovery-v4",
        "generatorVersion": "deepaa-beam-v0.2",
        "recipeVersion": "information-recovery-v4",
        "coordinateSpaceId": "test-space",
        "sourceV3RecordId": "a" * 32,
        "candidates": candidates,
    }

    response = asyncio.run(
        app_module.save_information_recovery_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            comparison_json=json.dumps(comparison),
            selected_variant="information-combined-v4",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["schema_version"] == 4
    assert record["comparison_kind"] == "information-recovery-v4"
    assert record["selected_variant"] == "information-combined-v4"
    assert (
        app_module.list_information_recovery_preferences()["records"][0][
            "record_id"
        ]
        == record["record_id"]
    )


def test_surface_texture_comparison_uses_catalog_stores_and_normalized_options(
    monkeypatch,
) -> None:
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_data_url)
    captured = {}
    catalog = ((";", ":"), {"left": {}, "right": {}}, "catalog", "config")

    def fake_comparison(payload, options, v2_store, v4_store, **kwargs):
        captured.update(
            payload=payload,
            options=options,
            v2_store=v2_store,
            v4_store=v4_store,
            kwargs=kwargs,
        )
        return {
            "comparisonKind": "surface-texture-v5",
            "generatorVersion": "deepaa-beam-v0.2",
            "recipeVersion": "surface-texture-line-separated-v5",
            "candidates": [],
        }

    monkeypatch.setattr(app_module, "load_texture_catalog", lambda: catalog)
    monkeypatch.setattr(app_module, "generate_surface_texture_comparison", fake_comparison)
    response = asyncio.run(
        app_module.compare_surface_texture(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"].columns == 140
    assert captured["v2_store"] is app_module.SURFACE_FILL_PREFERENCE_STORE
    assert captured["v4_store"] is app_module.INFORMATION_RECOVERY_PREFERENCE_STORE
    assert captured["kwargs"]["catalog_sha256"] == "catalog"
    assert response["comparisonKind"] == "surface-texture-v5"


def test_surface_texture_proximity_comparison_uses_v6_generator(
    monkeypatch,
) -> None:
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    source_payload = app_module.decode_png_data_url(source_data_url)
    captured = {}
    catalog = ((";", ":"), {"left": {}, "right": {}}, "catalog", "config")

    def fake_comparison(payload, options, v2_store, v4_store, **kwargs):
        captured.update(
            payload=payload,
            options=options,
            v2_store=v2_store,
            v4_store=v4_store,
            kwargs=kwargs,
        )
        return {
            "comparisonKind": "surface-texture-v6",
            "generatorVersion": "deepaa-beam-v0.2",
            "recipeVersion": "surface-texture-proximity-v6",
            "candidates": [],
        }

    monkeypatch.setattr(app_module, "load_texture_catalog", lambda: catalog)
    stored_options = ConversionOptions(columns=88, profile="person")
    monkeypatch.setattr(
        app_module,
        "resolve_texture_baseline_options",
        lambda payload, v2_store, v4_store: stored_options,
    )
    monkeypatch.setattr(
        app_module,
        "generate_surface_texture_proximity_comparison",
        fake_comparison,
    )
    response = asyncio.run(
        app_module.compare_surface_texture_proximity(
            image=upload(source_payload),
            columns=999,
            detail=72,
            threshold_low=45,
            threshold_high=135,
            min_component=8,
            abstraction=35,
            crop_x=0.0,
            crop_y=0.0,
            crop_width=1.0,
            crop_height=1.0,
            max_rows=64,
            profile="person",
        )
    )

    assert captured["payload"] == source_payload
    assert captured["options"] == stored_options
    assert captured["v2_store"] is app_module.SURFACE_FILL_PREFERENCE_STORE
    assert captured["v4_store"] is app_module.INFORMATION_RECOVERY_PREFERENCE_STORE
    assert captured["kwargs"]["catalog_sha256"] == "catalog"
    assert response["comparisonKind"] == "surface-texture-v6"


def test_surface_texture_preference_api_saves_v5_no_preference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v5",
        comparison_kind="surface-texture-v5",
        schema_version=5,
        candidate_count=2,
        allow_no_preference=True,
    )
    monkeypatch.setattr(app_module, "SURFACE_TEXTURE_PREFERENCE_STORE", store)
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    artifact_data_url = image_to_data_url(np.full((18, 192), 255, dtype=np.uint8))
    candidates = [
        {
            "variantId": variant,
            "displayLabel": f"候補{chr(ord('A') + index)}",
            "methodVersion": 6,
            "result": {
                "ascii": f"candidate-{index}\n",
                "rows": 1,
                "columns": 24,
                "processedPng": artifact_data_url,
                "renderedPng": artifact_data_url,
                "crop": [0, 0, 32, 24],
                "options": {"columns": 24, "profile": "person"},
            },
            "fillMaskPng": artifact_data_url,
            "surface": {"case_mechanical_gate": True},
        }
        for index, variant in enumerate(
            ("surface-texture-baseline-v5", "surface-texture-applied-v5")
        )
    ]
    comparison = {
        "comparisonKind": "surface-texture-v5",
        "generatorVersion": "deepaa-beam-v0.2",
        "recipeVersion": "surface-texture-line-separated-v5",
        "candidates": candidates,
    }

    response = asyncio.run(
        app_module.save_surface_texture_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            comparison_json=json.dumps(comparison),
            selected_variant="",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["schema_version"] == 5
    assert record["selected_variant"] is None
    assert record["none_usable"] is False
    assert app_module.list_surface_texture_preferences()["records"][0]["record_id"] == record["record_id"]


def test_surface_texture_proximity_preference_api_saves_v6_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = SurfaceFillPreferenceStore(
        tmp_path / "draft-preferences" / "v6",
        comparison_kind="surface-texture-v6",
        schema_version=6,
        candidate_count=2,
        allow_no_preference=True,
    )
    monkeypatch.setattr(
        app_module,
        "SURFACE_TEXTURE_PROXIMITY_PREFERENCE_STORE",
        store,
    )
    source_data_url = image_to_data_url(np.full((24, 32), 220, dtype=np.uint8))
    artifact_data_url = image_to_data_url(np.full((18, 192), 255, dtype=np.uint8))
    variants = ("surface-texture-t2-v6", "surface-texture-proximity-v6")
    candidates = [
        {
            "variantId": variant,
            "displayLabel": f"候補{chr(ord('A') + index)}",
            "methodVersion": 7 if "proximity" in variant else 6,
            "result": {
                "ascii": f"candidate-{index}\n",
                "rows": 1,
                "columns": 24,
                "processedPng": artifact_data_url,
                "renderedPng": artifact_data_url,
                "crop": [0, 0, 32, 24],
                "options": {"columns": 24, "profile": "person"},
            },
            "fillMaskPng": artifact_data_url,
            "surface": {"case_mechanical_gate": True},
        }
        for index, variant in enumerate(variants)
    ]
    comparison = {
        "comparisonKind": "surface-texture-v6",
        "generatorVersion": "deepaa-beam-v0.2",
        "recipeVersion": "surface-texture-proximity-v6",
        "candidates": candidates,
    }

    response = asyncio.run(
        app_module.save_surface_texture_proximity_preference(
            source=upload(app_module.decode_png_data_url(source_data_url)),
            comparison_json=json.dumps(comparison),
            selected_variant="surface-texture-proximity-v6",
            none_usable=False,
            source_origin="test-generated",
            rights_status="test-only",
        )
    )

    record = response["record"]
    assert record["schema_version"] == 6
    assert record["selected_variant"] == "surface-texture-proximity-v6"
    assert (
        app_module.list_surface_texture_proximity_preferences()["records"][0][
            "record_id"
        ]
        == record["record_id"]
    )
