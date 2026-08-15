from __future__ import annotations

import numpy as np

from backend.deepaa import HALF_WIDTH, LINE_PITCH
from training.audit_information_loss_p6r4i import analyze_information_loss


def canvas(columns: int = 4) -> np.ndarray:
    return np.full((LINE_PITCH, columns * HALF_WIDTH), 255, dtype=np.uint8)


def test_information_loss_audit_separates_pipeline_causes() -> None:
    source = canvas()
    # Cell 0: source edge lost before processed.
    source[:, HALF_WIDTH // 2 : HALF_WIDTH] = 20
    # Cell 1: flat dark tone never selected as a surface.
    source[:, HALF_WIDTH : 2 * HALF_WIDTH] = 50
    # Cell 2: flat dark tone selected, but no glyph was placed.
    source[:, 2 * HALF_WIDTH : 3 * HALF_WIDTH] = 55
    # Cell 3: processed evidence exists, but rendering is blank.
    source[:, 3 * HALF_WIDTH :] = 60
    processed = canvas()
    processed[:, 3 * HALF_WIDTH + 2] = 0
    rendered = canvas()
    fill_mask = np.zeros_like(source)
    fill_mask[:, 2 * HALF_WIDTH : 3 * HALF_WIDTH] = 255

    report, missing, categories = analyze_information_loss(
        source,
        crop=(0, 0, source.shape[1], source.shape[0]),
        processed=processed,
        rendered=rendered,
        fill_mask=fill_mask,
    )

    assert missing.tolist() == [[True, True, True, True]]
    assert categories.tolist() == [[1, 2, 3, 4]]
    assert report["blank_informative_cells"] == 4
    assert sum(report["cause_counts"].values()) == 4


def test_information_loss_audit_does_not_flag_rendered_cells() -> None:
    source = canvas(columns=1)
    source[:, :4] = 20
    processed = canvas(columns=1)
    rendered = canvas(columns=1)
    rendered[:, 1:4] = 0
    fill_mask = np.zeros_like(source)

    report, missing, categories = analyze_information_loss(
        source,
        crop=(0, 0, source.shape[1], source.shape[0]),
        processed=processed,
        rendered=rendered,
        fill_mask=fill_mask,
    )

    assert not missing.any()
    assert not categories.any()
    assert report["blank_informative_cells"] == 0


def test_information_loss_audit_requires_deepaa_grid() -> None:
    source = np.full((17, 8), 255, dtype=np.uint8)
    with np.testing.assert_raises_regex(ValueError, "8x18"):
        analyze_information_loss(
            source,
            crop=(0, 0, 8, 17),
            processed=source,
            rendered=source,
            fill_mask=np.zeros_like(source),
        )
