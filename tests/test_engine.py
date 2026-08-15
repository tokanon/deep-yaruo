from __future__ import annotations

import cv2
import numpy as np

from backend.contracts import ConversionOptions
from backend.deepaa import (
    _abstract_linework,
    _candidate_indices,
    _local_shape_cost,
    _sequence_repeat_cost,
    _simplify_background_lineart,
    _structure_mismatch_cost,
    generate_text,
    generate_text_beam,
    generate_text_viterbi,
    generate_text_viterbi_pruned,
    line_match_cost,
    load_assets,
    preprocess_for_deepaa,
)
from backend.image_io import decode_image
from backend.rendering import find_font, render_glyph_mask


def test_bundled_saitamaar_font_is_used() -> None:
    assert find_font().name == "Saitamaar.ttf"


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_glyphs_use_half_and_full_width() -> None:
    assert render_glyph_mask("/", 16).shape[1] == 8
    assert render_glyph_mask("／", 16).shape[1] == 16


def test_decode_image_reads_png() -> None:
    source = np.full((64, 64), 255, dtype=np.uint8)
    cv2.circle(source, (32, 32), 22, 0, 2)
    payload = encode_png(source)
    assert decode_image(payload).shape == (64, 64)


def test_deepaa_assets_and_inference() -> None:
    assets = load_assets()
    assert len(assets.characters) == 411
    assert len(assets.frequencies) == 411
    assert assets.session.get_inputs()[0].shape == ["batch", 64, 64, 1]
    line_image = np.full((18, 64), 255, dtype=np.uint8)
    text, predictions = generate_text(line_image)
    assert text.endswith("\n")
    assert len(predictions) == 1
    assert all(char in assets.glyphs for char in predictions[0])
    beam_text, beam_predictions = generate_text_beam(
        line_image,
        vertical_offset=0,
        beam_width=2,
        top_k=2,
    )
    assert beam_text.endswith("\n")
    assert len(beam_predictions) == 1
    viterbi_text, viterbi_predictions = generate_text_viterbi(
        line_image,
        vertical_offset=0,
        top_k=2,
    )
    assert viterbi_text.endswith("\n")
    assert len(viterbi_predictions) == 1
    rendered_width = sum(
        assets.glyphs[char].shape[1] for char in viterbi_predictions[0]
    )
    assert rendered_width >= line_image.shape[1]
    pruned_text, pruned_predictions = generate_text_viterbi_pruned(
        line_image,
        vertical_offset=0,
        baseline=predictions,
        top_k=2,
    )
    assert pruned_text.endswith("\n")
    assert len(pruned_predictions) == 1


def test_prior_corrected_candidates_include_raw_and_rare_favorites() -> None:
    assets = load_assets()
    scores = np.full(len(assets.characters), 1e-8, dtype=np.float32)
    common_index = int(np.argmax(assets.frequencies))
    rare_index = int(np.argmin(assets.frequencies))
    scores[common_index] = 0.51
    scores[rare_index] = 0.49
    candidates = _candidate_indices(scores, assets, top_k=1)
    assert common_index in candidates
    assert rare_index in candidates
    raw_candidates = _candidate_indices(
        scores,
        assets,
        top_k=1,
        diversity_strength=0.0,
    )
    assert raw_candidates.tolist() == [common_index]


def test_shape_cost_discourages_excess_ink_but_tone_repetition_is_allowed() -> None:
    target = np.zeros((16, 8), dtype=bool)
    target[:, 3] = True
    matching = target.copy()
    filled = np.ones_like(target)
    assert _local_shape_cost(target, matching) < _local_shape_cost(target, filled)
    assert _sequence_repeat_cost(list("||||||||")) > _sequence_repeat_cost(
        list("|/|/|/|/")
    )
    assert _sequence_repeat_cost(list("::::::::")) == 0.0


def test_structure_cost_prefers_matching_direction_and_boundaries() -> None:
    target = np.zeros((16, 8), dtype=bool)
    target[8, :] = True
    matching = target.copy()
    vertical = np.zeros_like(target)
    vertical[:, 3] = True
    assert _structure_mismatch_cost(target, matching) == 0
    assert _structure_mismatch_cost(target, vertical) > 0
    assert _local_shape_cost(
        target,
        matching,
        structure_strength=1.0,
    ) < _local_shape_cost(
        target,
        vertical,
        structure_strength=1.0,
    )
    narrow = np.zeros((16, 1), dtype=bool)
    narrow[:, 0] = True
    assert _structure_mismatch_cost(narrow, narrow) == 0


def test_abstraction_merges_parallel_detail_but_preserves_a_single_line() -> None:
    dense = np.zeros((90, 96), dtype=np.uint8)
    for x in range(3, 16, 2):
        cv2.line(dense, (x, 10), (x, 80), 255, 1)
    options = ConversionOptions(abstraction=60, profile="lineart").normalized()
    abstracted = _abstract_linework(dense, options)
    assert np.count_nonzero(abstracted) < np.count_nonzero(dense) // 2
    assert cv2.connectedComponents((abstracted > 0).astype(np.uint8))[0] - 1 == 1

    single = np.zeros_like(dense)
    cv2.line(single, (10, 10), (10, 80), 255, 1)
    preserved = _abstract_linework(single, options)
    assert np.count_nonzero(preserved) >= 60
    assert np.flatnonzero(np.any(preserved > 0, axis=0)).tolist() == [10]


def test_background_lineart_simplification_keeps_long_structure() -> None:
    ink = np.zeros((120, 240), dtype=np.uint8)
    cv2.line(ink, (10, 20), (230, 20), 255, 1)
    for y in range(50, 105, 4):
        for x in range(20, 220, 6):
            cv2.line(ink, (x, y), (x + 2, y + 1), 255, 1)
    options = ConversionOptions(
        abstraction=35,
        min_component=4,
        profile="background_lineart",
    ).normalized()
    simplified = _simplify_background_lineart(ink, options)
    assert np.count_nonzero(simplified) < np.count_nonzero(ink)
    assert np.count_nonzero(simplified[17:24]) >= 180


def test_deepaa_preprocess_uses_18_pixel_line_pitch() -> None:
    source = np.full((120, 80), 255, dtype=np.uint8)
    cv2.line(source, (10, 100), (70, 10), 0, 2)
    line_image, _ = preprocess_for_deepaa(
        source,
        ConversionOptions(columns=32, max_rows=30, min_component=1),
    )
    assert line_image.shape[1] == 32 * 8
    assert line_image.shape[0] % 18 == 0
    assert np.count_nonzero(line_image < 255) > 0


def test_deepaa_profiles_produce_distinct_line_images() -> None:
    source = np.full((180, 240), 230, dtype=np.uint8)
    cv2.rectangle(source, (20, 30), (220, 160), 40, 2)
    cv2.line(source, (20, 160), (220, 30), 80, 1)
    cv2.circle(source, (120, 90), 18, 100, 1)
    person, _ = preprocess_for_deepaa(
        source,
        ConversionOptions(columns=32, max_rows=30, min_component=1, profile="person"),
    )
    background, _ = preprocess_for_deepaa(
        source,
        ConversionOptions(columns=32, max_rows=30, min_component=1, profile="background"),
    )
    assert person.shape == background.shape
    assert not np.array_equal(person, background)


def test_lineart_profile_keeps_single_dark_strokes() -> None:
    source = np.full((180, 240), 255, dtype=np.uint8)
    cv2.line(source, (120, 10), (120, 170), 0, 1)
    lineart, _ = preprocess_for_deepaa(
        source,
        ConversionOptions(
            columns=32,
            max_rows=30,
            min_component=1,
            detail=72,
            profile="lineart",
        ),
    )
    middle = lineart[:, lineart.shape[1] // 2 - 2 : lineart.shape[1] // 2 + 3]
    ink_columns = np.flatnonzero(np.any(middle < 128, axis=0))
    assert 1 <= len(ink_columns) <= 2


def test_line_match_cost_prefers_aligned_ink() -> None:
    target = np.full((36, 64), 255, dtype=np.uint8)
    cv2.line(target, (8, 8), (55, 28), 0, 1)
    shifted = np.roll(target, 5, axis=1)
    assert line_match_cost(target, target) == 0
    assert line_match_cost(target, shifted) > 0
