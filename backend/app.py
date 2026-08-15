from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from training.yaruyomi import encode_unencodable_as_numeric_references

from .cumulative_generator import (
    P6RG1_RECIPE_VERSION,
    RECIPE_VERSION as SURFACE_MODEL_RECIPE_VERSION,
    generate_cumulative,
    regenerate_cumulative,
    regenerate_p6rg1,
)
from .correction_pairs import (
    MAX_SOURCE_BYTES,
    CorrectionPairStore,
    CorrectionPairSubmission,
    decode_png_data_url,
)
from .contracts import ConversionOptions, TextTransportRequest
from .deepaa import convert_deepaa
from .draft_preferences import (
    DraftCandidateSubmission,
    DraftPreferenceStore,
    DraftPreferenceSubmission,
)
from .information_recovery import generate_information_recovery_comparison
from .review_v2 import ReviewV2Store, ReviewV2Submission
from .surface_decisions import analyze_correction_pair_surfaces
from .surface_fill import generate_surface_fill_comparison
from .surface_fill_preferences import (
    DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT,
    DEFAULT_SURFACE_RECOVERY_PREFERENCE_ROOT,
    DEFAULT_SURFACE_TEXTURE_PROXIMITY_PREFERENCE_ROOT,
    DEFAULT_SURFACE_TEXTURE_PREFERENCE_ROOT,
    SurfaceFillCandidateSubmission,
    SurfaceFillPreferenceStore,
    SurfaceFillPreferenceSubmission,
)
from .surface_recovery import generate_surface_recovery_comparison
from .surface_texture import (
    generate_surface_texture_comparison,
    generate_surface_texture_proximity_comparison,
    load_texture_catalog,
    resolve_texture_baseline_options,
)


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
FONT_FILE = ROOT / "assets" / "fonts" / "Saitamaar.ttf"
REVIEW_STORE = ReviewV2Store()
CORRECTION_PAIR_STORE = CorrectionPairStore()
DRAFT_PREFERENCE_STORE = DraftPreferenceStore()
SURFACE_FILL_PREFERENCE_STORE = SurfaceFillPreferenceStore()
SURFACE_RECOVERY_PREFERENCE_STORE = SurfaceFillPreferenceStore(
    DEFAULT_SURFACE_RECOVERY_PREFERENCE_ROOT,
    comparison_kind="surface-recovery-v3",
    schema_version=3,
    require_common_processed=False,
)
INFORMATION_RECOVERY_PREFERENCE_STORE = SurfaceFillPreferenceStore(
    DEFAULT_INFORMATION_RECOVERY_PREFERENCE_ROOT,
    comparison_kind="information-recovery-v4",
    schema_version=4,
    require_common_processed=True,
)
SURFACE_TEXTURE_PREFERENCE_STORE = SurfaceFillPreferenceStore(
    DEFAULT_SURFACE_TEXTURE_PREFERENCE_ROOT,
    comparison_kind="surface-texture-v5",
    schema_version=5,
    require_common_processed=True,
    candidate_count=2,
    allow_no_preference=True,
)
SURFACE_TEXTURE_PROXIMITY_PREFERENCE_STORE = SurfaceFillPreferenceStore(
    DEFAULT_SURFACE_TEXTURE_PROXIMITY_PREFERENCE_ROOT,
    comparison_kind="surface-texture-v6",
    schema_version=6,
    require_common_processed=True,
    candidate_count=2,
    allow_no_preference=True,
)

app = FastAPI(title="Yaruo AA Studio", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/font/saitamaar.ttf", include_in_schema=False)
def saitamaar_font() -> FileResponse:
    if not FONT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="assets/fonts/Saitamaar.ttf was not found",
        )
    return FileResponse(FONT_FILE, media_type="font/ttf")


@app.post("/api/convert")
async def convert_image(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
    cumulative: bool = Form(True),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")

    try:
        options = ConversionOptions(
            columns=columns,
            detail=detail,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_component=min_component,
            abstraction=abstraction,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            max_rows=max_rows,
            profile=profile,
        ).normalized()
        if cumulative:
            generated = generate_cumulative(payload, options)
            result = generated.result
            pipeline = generated.pipeline
        else:
            result = convert_deepaa(payload, options)
            pipeline = {
                "generator_version": "deepaa-beam-v0.2",
                "recipe_version": "legacy-v0.2",
                "enabled": False,
                "applied": False,
                "stages": [],
            }
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "ascii": result.ascii_text,
        "rows": result.rows,
        "columns": result.columns,
        "processedPng": result.processed_png,
        "renderedPng": result.rendered_png,
        "crop": result.crop,
        "options": asdict(options),
        "pipeline": pipeline,
    }


@app.post("/api/surface-fill-comparison")
async def compare_surface_fill(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        options = ConversionOptions(
            columns=columns,
            detail=detail,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_component=min_component,
            abstraction=abstraction,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            max_rows=max_rows,
            profile=profile,
        ).normalized()
        return generate_surface_fill_comparison(payload, options)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/surface-recovery-comparison")
async def compare_surface_recovery(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        options = ConversionOptions(
            columns=columns,
            detail=detail,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_component=min_component,
            abstraction=abstraction,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            max_rows=max_rows,
            profile=profile,
        ).normalized()
        return generate_surface_recovery_comparison(payload, options)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/information-recovery-comparison")
async def compare_information_recovery(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        options = ConversionOptions(
            columns=columns,
            detail=detail,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_component=min_component,
            abstraction=abstraction,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            max_rows=max_rows,
            profile=profile,
        ).normalized()
        return generate_information_recovery_comparison(
            payload,
            options,
            SURFACE_RECOVERY_PREFERENCE_STORE,
        )
    except (OSError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/surface-texture-comparison")
async def compare_surface_texture(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        options = ConversionOptions(
            columns=columns,
            detail=detail,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            min_component=min_component,
            abstraction=abstraction,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            max_rows=max_rows,
            profile=profile,
        ).normalized()
        palette, adjustments, catalog_hash, config_hash = load_texture_catalog()
        return generate_surface_texture_comparison(
            payload,
            options,
            SURFACE_FILL_PREFERENCE_STORE,
            INFORMATION_RECOVERY_PREFERENCE_STORE,
            palette=palette,
            edge_adjustments=adjustments,
            catalog_sha256=catalog_hash,
            audit_config_sha256=config_hash,
        )
    except (OSError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/surface-texture-proximity-comparison")
async def compare_surface_texture_proximity(
    image: UploadFile = File(...),
    columns: int = Form(72),
    detail: int = Form(72),
    threshold_low: int = Form(45),
    threshold_high: int = Form(135),
    min_component: int = Form(8),
    abstraction: int = Form(35),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_width: float = Form(1.0),
    crop_height: float = Form(1.0),
    max_rows: int = Form(64),
    profile: str = Form("auto"),
) -> dict[str, object]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    payload = await image.read()
    if len(payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        options = resolve_texture_baseline_options(
            payload,
            SURFACE_FILL_PREFERENCE_STORE,
            INFORMATION_RECOVERY_PREFERENCE_STORE,
        )
        palette, adjustments, catalog_hash, config_hash = load_texture_catalog()
        return generate_surface_texture_proximity_comparison(
            payload,
            options,
            SURFACE_FILL_PREFERENCE_STORE,
            INFORMATION_RECOVERY_PREFERENCE_STORE,
            palette=palette,
            edge_adjustments=adjustments,
            catalog_sha256=catalog_hash,
            audit_config_sha256=config_hash,
        )
    except (OSError, ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/correction-pairs")
async def save_correction_pair(
    source: UploadFile = File(...),
    options_json: str = Form(...),
    crop_json: str = Form(...),
    draft_text: str = Form(...),
    corrected_text: str = Form(...),
    processed_png: str = Form(...),
    draft_rendered_png: str = Form(...),
    generation_json: str = Form("{}"),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        option_values = json.loads(options_json)
        crop_values = json.loads(crop_json)
        generation_values = json.loads(generation_json)
        if not isinstance(option_values, dict):
            raise ValueError("options_json must contain a JSON object")
        if not isinstance(generation_values, dict):
            raise ValueError("generation_json must contain a JSON object")
        if not (
            isinstance(crop_values, list)
            and len(crop_values) == 4
            and all(type(value) is int for value in crop_values)
        ):
            raise ValueError("crop_json must contain four integer coordinates")
        submission = CorrectionPairSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            options=ConversionOptions(**option_values),
            crop=tuple(crop_values),
            draft_text=draft_text,
            corrected_text=corrected_text,
            processed_png=decode_png_data_url(processed_png),
            draft_rendered_png=decode_png_data_url(draft_rendered_png),
            generation=generation_values,
        )
        return {"record": CORRECTION_PAIR_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/draft-preferences")
async def save_draft_preference(
    source: UploadFile = File(...),
    candidates_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        candidate_values = json.loads(candidates_json)
        if not isinstance(candidate_values, list):
            raise ValueError("candidates_json must contain a JSON array")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each candidate must be a JSON object")
            crop_values = value.get("crop")
            option_values = value.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be a JSON object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                DraftCandidateSubmission(
                    variant_id=str(value.get("variant_id", "")),
                    display_label=str(value.get("display_label", "")),
                    method_version=int(value.get("method_version", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(value.get("ascii", "")),
                    rows=int(value.get("rows", 0)),
                    columns=int(value.get("columns", 0)),
                    processed_png=decode_png_data_url(str(value.get("processed_png", ""))),
                    rendered_png=decode_png_data_url(str(value.get("rendered_png", ""))),
                )
            )
        submission = DraftPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": DRAFT_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/draft-preferences")
def list_draft_preferences() -> dict[str, object]:
    try:
        return {"records": DRAFT_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/surface-fill-preferences")
async def save_surface_fill_preference(
    source: UploadFile = File(...),
    comparison_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison_json must contain a JSON object")
        candidate_values = comparison.get("candidates")
        if not isinstance(candidate_values, list):
            raise ValueError("surface-fill comparison must contain candidates")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each surface-fill candidate must be an object")
            result = value.get("result")
            surface = value.get("surface")
            if not isinstance(result, dict) or not isinstance(surface, dict):
                raise ValueError("candidate result and surface metadata are required")
            crop_values = result.get("crop")
            option_values = result.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be an object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                SurfaceFillCandidateSubmission(
                    variant_id=str(value.get("variantId", "")),
                    display_label=str(value.get("displayLabel", "")),
                    method_version=int(value.get("methodVersion", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(result.get("ascii", "")),
                    rows=int(result.get("rows", 0)),
                    columns=int(result.get("columns", 0)),
                    processed_png=decode_png_data_url(
                        str(result.get("processedPng", ""))
                    ),
                    rendered_png=decode_png_data_url(
                        str(result.get("renderedPng", ""))
                    ),
                    fill_mask_png=decode_png_data_url(
                        str(value.get("fillMaskPng", ""))
                    ),
                    surface=surface,
                )
            )
        submission = SurfaceFillPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            comparison_kind=str(comparison.get("comparisonKind", "")),
            generator_version=str(comparison.get("generatorVersion", "")),
            recipe_version=str(comparison.get("recipeVersion", "")),
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": SURFACE_FILL_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/surface-fill-preferences")
def list_surface_fill_preferences() -> dict[str, object]:
    try:
        return {"records": SURFACE_FILL_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/surface-recovery-preferences")
async def save_surface_recovery_preference(
    source: UploadFile = File(...),
    comparison_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison_json must contain a JSON object")
        candidate_values = comparison.get("candidates")
        if not isinstance(candidate_values, list):
            raise ValueError("surface recovery comparison must contain candidates")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each surface recovery candidate must be an object")
            result = value.get("result")
            surface = value.get("surface")
            if not isinstance(result, dict) or not isinstance(surface, dict):
                raise ValueError("candidate result and surface metadata are required")
            crop_values = result.get("crop")
            option_values = result.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be an object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                SurfaceFillCandidateSubmission(
                    variant_id=str(value.get("variantId", "")),
                    display_label=str(value.get("displayLabel", "")),
                    method_version=int(value.get("methodVersion", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(result.get("ascii", "")),
                    rows=int(result.get("rows", 0)),
                    columns=int(result.get("columns", 0)),
                    processed_png=decode_png_data_url(
                        str(result.get("processedPng", ""))
                    ),
                    rendered_png=decode_png_data_url(
                        str(result.get("renderedPng", ""))
                    ),
                    fill_mask_png=decode_png_data_url(
                        str(value.get("fillMaskPng", ""))
                    ),
                    surface=surface,
                )
            )
        submission = SurfaceFillPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            comparison_kind=str(comparison.get("comparisonKind", "")),
            generator_version=str(comparison.get("generatorVersion", "")),
            recipe_version=str(comparison.get("recipeVersion", "")),
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": SURFACE_RECOVERY_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/surface-recovery-preferences")
def list_surface_recovery_preferences() -> dict[str, object]:
    try:
        return {"records": SURFACE_RECOVERY_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/information-recovery-preferences")
async def save_information_recovery_preference(
    source: UploadFile = File(...),
    comparison_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison_json must contain a JSON object")
        candidate_values = comparison.get("candidates")
        if not isinstance(candidate_values, list):
            raise ValueError("information recovery comparison must contain candidates")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each information recovery candidate must be an object")
            result = value.get("result")
            surface = value.get("surface")
            if not isinstance(result, dict) or not isinstance(surface, dict):
                raise ValueError("candidate result and recovery metadata are required")
            crop_values = result.get("crop")
            option_values = result.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be an object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                SurfaceFillCandidateSubmission(
                    variant_id=str(value.get("variantId", "")),
                    display_label=str(value.get("displayLabel", "")),
                    method_version=int(value.get("methodVersion", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(result.get("ascii", "")),
                    rows=int(result.get("rows", 0)),
                    columns=int(result.get("columns", 0)),
                    processed_png=decode_png_data_url(
                        str(result.get("processedPng", ""))
                    ),
                    rendered_png=decode_png_data_url(
                        str(result.get("renderedPng", ""))
                    ),
                    fill_mask_png=decode_png_data_url(
                        str(value.get("fillMaskPng", ""))
                    ),
                    surface=surface,
                )
            )
        submission = SurfaceFillPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            comparison_kind=str(comparison.get("comparisonKind", "")),
            generator_version=str(comparison.get("generatorVersion", "")),
            recipe_version=str(comparison.get("recipeVersion", "")),
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": INFORMATION_RECOVERY_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/information-recovery-preferences")
def list_information_recovery_preferences() -> dict[str, object]:
    try:
        return {"records": INFORMATION_RECOVERY_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/surface-texture-preferences")
async def save_surface_texture_preference(
    source: UploadFile = File(...),
    comparison_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison_json must contain a JSON object")
        candidate_values = comparison.get("candidates")
        if not isinstance(candidate_values, list):
            raise ValueError("surface texture comparison must contain candidates")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each surface texture candidate must be an object")
            result = value.get("result")
            surface = value.get("surface")
            if not isinstance(result, dict) or not isinstance(surface, dict):
                raise ValueError("candidate result and texture metadata are required")
            crop_values = result.get("crop")
            option_values = result.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be an object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                SurfaceFillCandidateSubmission(
                    variant_id=str(value.get("variantId", "")),
                    display_label=str(value.get("displayLabel", "")),
                    method_version=int(value.get("methodVersion", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(result.get("ascii", "")),
                    rows=int(result.get("rows", 0)),
                    columns=int(result.get("columns", 0)),
                    processed_png=decode_png_data_url(
                        str(result.get("processedPng", ""))
                    ),
                    rendered_png=decode_png_data_url(
                        str(result.get("renderedPng", ""))
                    ),
                    fill_mask_png=decode_png_data_url(
                        str(value.get("fillMaskPng", ""))
                    ),
                    surface=surface,
                )
            )
        submission = SurfaceFillPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            comparison_kind=str(comparison.get("comparisonKind", "")),
            generator_version=str(comparison.get("generatorVersion", "")),
            recipe_version=str(comparison.get("recipeVersion", "")),
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": SURFACE_TEXTURE_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/surface-texture-preferences")
def list_surface_texture_preferences() -> dict[str, object]:
    try:
        return {"records": SURFACE_TEXTURE_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/surface-texture-proximity-preferences")
async def save_surface_texture_proximity_preference(
    source: UploadFile = File(...),
    comparison_json: str = Form(...),
    selected_variant: str = Form(""),
    none_usable: bool = Form(False),
    source_origin: str = Form("local-user-provided"),
    rights_status: str = Form(
        "user-provided; local use only; redistribution not granted"
    ),
) -> dict[str, object]:
    if source.content_type and not source.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    source_payload = await source.read()
    if len(source_payload) > MAX_SOURCE_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 20 MB or smaller.")
    try:
        comparison = json.loads(comparison_json)
        if not isinstance(comparison, dict):
            raise ValueError("comparison_json must contain a JSON object")
        candidate_values = comparison.get("candidates")
        if not isinstance(candidate_values, list):
            raise ValueError("surface texture comparison must contain candidates")
        candidates = []
        for value in candidate_values:
            if not isinstance(value, dict):
                raise ValueError("each surface texture candidate must be an object")
            result = value.get("result")
            surface = value.get("surface")
            if not isinstance(result, dict) or not isinstance(surface, dict):
                raise ValueError("candidate result and texture metadata are required")
            crop_values = result.get("crop")
            option_values = result.get("options")
            if not isinstance(option_values, dict):
                raise ValueError("candidate options must be an object")
            if not (
                isinstance(crop_values, list)
                and len(crop_values) == 4
                and all(type(item) is int for item in crop_values)
            ):
                raise ValueError("candidate crop must contain four integers")
            candidates.append(
                SurfaceFillCandidateSubmission(
                    variant_id=str(value.get("variantId", "")),
                    display_label=str(value.get("displayLabel", "")),
                    method_version=int(value.get("methodVersion", 0)),
                    options=ConversionOptions(**option_values),
                    crop=tuple(crop_values),
                    ascii_text=str(result.get("ascii", "")),
                    rows=int(result.get("rows", 0)),
                    columns=int(result.get("columns", 0)),
                    processed_png=decode_png_data_url(
                        str(result.get("processedPng", ""))
                    ),
                    rendered_png=decode_png_data_url(
                        str(result.get("renderedPng", ""))
                    ),
                    fill_mask_png=decode_png_data_url(
                        str(value.get("fillMaskPng", ""))
                    ),
                    surface=surface,
                )
            )
        submission = SurfaceFillPreferenceSubmission(
            source_bytes=source_payload,
            source_filename=source.filename or "source-image",
            source_media_type=source.content_type or "application/octet-stream",
            source_origin=source_origin,
            rights_status=rights_status,
            comparison_kind=str(comparison.get("comparisonKind", "")),
            generator_version=str(comparison.get("generatorVersion", "")),
            recipe_version=str(comparison.get("recipeVersion", "")),
            candidates=tuple(candidates),
            selected_variant=selected_variant or None,
            none_usable=none_usable,
        )
        return {"record": SURFACE_TEXTURE_PROXIMITY_PREFERENCE_STORE.save(submission)}
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/surface-texture-proximity-preferences")
def list_surface_texture_proximity_preferences() -> dict[str, object]:
    try:
        return {"records": SURFACE_TEXTURE_PROXIMITY_PREFERENCE_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/correction-pairs")
def list_correction_pairs() -> dict[str, object]:
    try:
        return {"records": CORRECTION_PAIR_STORE.list()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/correction-pairs/{record_id}")
def get_correction_pair(record_id: str) -> dict[str, object]:
    try:
        return CORRECTION_PAIR_STORE.get(record_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get(
    "/api/correction-pairs/{record_id}/assets/{asset_name}",
    include_in_schema=False,
)
def correction_pair_asset(record_id: str, asset_name: str) -> FileResponse:
    try:
        path, media_type = CORRECTION_PAIR_STORE.asset(record_id, asset_name)
        return FileResponse(path, media_type=media_type)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/correction-pairs/{record_id}/extensions")
def update_correction_pair_extensions(
    record_id: str,
    updates: dict[str, object],
) -> dict[str, object]:
    try:
        return {"extensions": CORRECTION_PAIR_STORE.update_extensions(record_id, updates)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/correction-pairs/{record_id}/verify")
def verify_correction_pair(record_id: str) -> dict[str, object]:
    try:
        record = CORRECTION_PAIR_STORE.get(record_id)
        generation = record["manifest"]["conversion"].get("generation", {})
        recipe_version = generation.get("recipe_version")
        if recipe_version == SURFACE_MODEL_RECIPE_VERSION:
            converter = regenerate_cumulative
        elif recipe_version == P6RG1_RECIPE_VERSION:
            converter = regenerate_p6rg1
        else:
            converter = convert_deepaa
        return CORRECTION_PAIR_STORE.verify_regeneration(record_id, converter)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/correction-pairs/{record_id}/surface-analysis")
def analyze_correction_pair_surface_data(record_id: str) -> dict[str, object]:
    try:
        return analyze_correction_pair_surfaces(CORRECTION_PAIR_STORE, record_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/text/cp932-ncr")
def serialize_cp932_text(request: TextTransportRequest) -> dict[str, str]:
    return {
        "text": encode_unencodable_as_numeric_references(request.text),
        "encoding": "cp932-ncr",
    }


@app.get("/api/review/status")
def review_status() -> dict[str, object]:
    try:
        return REVIEW_STORE.status()
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/review/next")
def review_next(
    category: str | None = None,
    sensitive: str = "all",
    skip_entry_id: int | None = None,
) -> dict[str, object]:
    try:
        candidate = REVIEW_STORE.next_candidate(
            category=category,
            sensitive=sensitive,
            skip_entry_id=skip_entry_id,
        )
    except (FileNotFoundError, ValueError, sqlite3.Error, zipfile.BadZipFile) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if candidate is None:
        raise HTTPException(status_code=404, detail="No unreviewed AA matches this filter.")
    return candidate


@app.get("/api/review/entries/{entry_id}")
def review_entry(entry_id: int) -> dict[str, object]:
    try:
        candidate = REVIEW_STORE.get_entry(entry_id)
    except (FileNotFoundError, ValueError, sqlite3.Error, zipfile.BadZipFile) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"AA entry {entry_id} was not found.")
    return candidate


@app.put("/api/review/entries/{entry_id}")
def save_review(entry_id: int, submission: ReviewV2Submission) -> dict[str, object]:
    try:
        candidate = REVIEW_STORE.save_review(entry_id, submission)
        return {"candidate": candidate, "status": REVIEW_STORE.status()}
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (FileNotFoundError, ValueError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
