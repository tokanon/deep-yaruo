from __future__ import annotations

import numpy as np

import backend.cumulative_generator as cumulative
from backend.contracts import ConversionOptions, ConversionResult
from backend.deepaa_surface import DenseDecodeResult
from backend.deepaa_surface import decode_dense_row
from backend.image_io import image_to_data_url


def _result(text: str = "....\n") -> ConversionResult:
    image = image_to_data_url(np.full((18, 32), 255, dtype=np.uint8))
    return ConversionResult(
        ascii_text=text,
        rows=1,
        columns=24,
        processed_png=image,
        rendered_png=image,
        crop=(0, 0, 32, 18),
    )


def test_cumulative_pipeline_keeps_progress_across_later_stages(monkeypatch) -> None:
    surface = _result("surface\n")
    information = _result("information\n")
    textured = _result("texture\n")
    mask = np.zeros((18, 32), dtype=np.uint8)
    calls: list[str] = []

    def surface_stage(*_):
        calls.append("surface")
        return surface, mask, {"stage": "surface", "applied": True}

    def information_stage(*_):
        calls.append("information")
        return information, mask, {"stage": "information", "applied": True}

    def texture_stage(*_):
        calls.append("texture")
        return textured, {"stage": "texture", "applied": True}

    monkeypatch.setattr(cumulative, "_select_surface_base", surface_stage)
    monkeypatch.setattr(cumulative, "_apply_information_stage", information_stage)
    monkeypatch.setattr(cumulative, "_apply_texture_stage", texture_stage)

    generated = cumulative.generate_p6rg1(
        b"source",
        ConversionOptions(columns=24, profile="background"),
    )

    assert calls == ["surface", "information", "texture"]
    assert generated.result.ascii_text == "texture\n"
    assert generated.pipeline["recipe_version"] == "p6r-cumulative-g1-v1"


def test_line_art_profile_stays_on_the_tone_free_path(monkeypatch) -> None:
    raw = _result("lineart\n")
    monkeypatch.setattr(cumulative, "convert_deepaa", lambda *_: raw)
    monkeypatch.setattr(
        cumulative,
        "_select_surface_base",
        lambda *_: (_ for _ in ()).throw(AssertionError("surface path must not run")),
    )

    generated = cumulative.generate_cumulative(
        b"source",
        ConversionOptions(columns=24, profile="lineart"),
    )

    assert generated.result.ascii_text == "lineart\n"
    assert generated.pipeline["applied"] is False
    assert generated.pipeline["reason"] == "line-art-profile-has-no-tone-surface"


def test_color_profile_uses_surface_model_as_provisional_default(monkeypatch) -> None:
    processed = np.full((18, 192), 255, dtype=np.uint8)
    line = np.ones((1, 18, 192), dtype=np.float32)
    surface = np.zeros((3, 18, 192), dtype=np.float32)
    monkeypatch.setattr(
        cumulative,
        "_prepare_surface_model_inputs",
        lambda *_: (
            processed,
            line,
            surface,
            (0, 0, 192, 18),
            {"stage": "inputs", "applied": True},
        ),
    )
    monkeypatch.setattr(
        cumulative,
        "decode_surface_text",
        lambda *_: DenseDecodeResult(
            text="." * 48 + "\n",
            rows=1,
            target_width=192,
            row_widths=(192,),
            start_counts=(48,),
            all_whitespace_rows=(),
            model_sha256="model",
            vocabulary_sha256="vocabulary",
            checkpoint_sha256="checkpoint",
        ),
    )

    generated = cumulative.generate_cumulative(
        b"source", ConversionOptions(columns=24, profile="background")
    )

    assert generated.pipeline["recipe_version"] == cumulative.RECIPE_VERSION
    assert generated.pipeline["fallback_used"] is False
    assert generated.pipeline["stages"][1]["uses_target_start_count"] is False
    assert generated.pipeline["stages"][1]["model_sha256"] == "model"


def test_surface_model_whitespace_failure_falls_back_to_complete_p6rg1(monkeypatch) -> None:
    processed = np.full((18, 192), 255, dtype=np.uint8)
    line = np.zeros((1, 18, 192), dtype=np.float32)
    surface = np.zeros((3, 18, 192), dtype=np.float32)
    fallback = cumulative.CumulativeGeneration(
        result=_result("fallback\n"),
        pipeline={"recipe_version": cumulative.P6RG1_RECIPE_VERSION},
    )
    monkeypatch.setattr(
        cumulative,
        "_prepare_surface_model_inputs",
        lambda *_: (processed, line, surface, (0, 0, 192, 18), {}),
    )
    monkeypatch.setattr(
        cumulative,
        "decode_surface_text",
        lambda *_: DenseDecodeResult(
            text=" " * 48 + "\n",
            rows=1,
            target_width=192,
            row_widths=(192,),
            start_counts=(48,),
            all_whitespace_rows=(0,),
            model_sha256="model",
            vocabulary_sha256="vocabulary",
            checkpoint_sha256="checkpoint",
        ),
    )
    monkeypatch.setattr(cumulative, "generate_p6rg1", lambda *_: fallback)

    generated = cumulative.generate_cumulative(
        b"source", ConversionOptions(columns=24, profile="background")
    )

    assert generated.result.ascii_text == "fallback\n"
    assert generated.pipeline["fallback_used"] is True
    assert "all-whitespace" in generated.pipeline["fallback_reason"]


def test_dense_decoder_uses_learned_start_boundaries_without_target_count() -> None:
    characters = ("a", "b")
    advances = np.asarray([2, 4], dtype=np.int16)
    groups = {2: np.asarray([0]), 4: np.asarray([1])}
    character_logits = np.asarray(
        [[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]],
        dtype=np.float32,
    )
    # Starts at 0 and 2 are strongly preferred, so the exact-width path is aa.
    start_logits = np.asarray([5.0, -5.0, 5.0, -5.0], dtype=np.float32)

    decoded = decode_dense_row(
        character_logits,
        start_logits,
        characters=characters,
        advances=advances,
        advance_groups=groups,
        target_width=4,
        top_k=1,
    )

    assert decoded == ("aa", (0, 2))
