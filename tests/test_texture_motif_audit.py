from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backend.rendering import glyph_advance
from training.audit_texture_motifs_p6r4t import (
    MotifAuditConfig,
    _best_baselines,
    _motif_density,
    _owner_graph,
    audit_motif_catalog,
    extract_periodic_runs,
)


def test_periodic_motifs_allow_wide_patterns_but_exclude_wide_glyphs_and_rules() -> None:
    text = "`:" * 6 + "\n" + "二" * 12 + "\n" + "-" * 12

    runs = extract_periodic_runs(text)

    assert [run.motif for run in runs] == [":`"]
    assert runs[0].x_end - runs[0].x_start == (
        glyph_advance("`") + glyph_advance(":")
    ) * 6
    width, _ = _motif_density(":`")
    assert width == glyph_advance(":") + glyph_advance("`")
    assert width != 8


def test_short_periodic_motif_requires_vertical_support() -> None:
    isolated = extract_periodic_runs(".:.:.:")
    supported = extract_periodic_runs(".:.:.:\n:.:.:.")

    assert isolated == []
    assert len(supported) == 2
    assert {run.motif for run in supported} == {".:"}
    assert all(run.vertically_supported for run in supported)


def _snapshot(root: Path, texts: list[str]) -> None:
    records = []
    for index, text in enumerate(texts, start=1):
        relative = Path("aa") / "train" / f"{index:07d}.txt"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        records.append(
            {
                "entry_id": index,
                "decision": "accept",
                "text": relative.as_posix(),
            }
        )
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    (root / "records.jsonl").write_bytes(payload)
    (root / "snapshot.json").write_text(
        json.dumps(
            {
                "records": "records.jsonl",
                "records_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_catalog_reaggregates_all_snapshot_works_and_hashes_config(tmp_path: Path) -> None:
    _snapshot(tmp_path, ["`:" * 6, "`:" * 6])
    first_config = MotifAuditConfig(minimum_run_count=2, minimum_work_count=2)
    second_config = MotifAuditConfig(
        minimum_run_count=3,
        minimum_work_count=2,
    )

    first = audit_motif_catalog(tmp_path, config=first_config)
    second = audit_motif_catalog(tmp_path, config=second_config)

    motif = next(item for item in first["motifs"] if item["motif"] == ":`")
    assert first["accepted_entry_count"] == 2
    assert motif["run_count"] == 2
    assert motif["work_count"] == 2
    assert motif["supported"] is True
    assert second["supported_motif_count"] == 0
    assert first["catalog_sha256"] != second["catalog_sha256"]


def test_catalog_records_observed_left_and_right_edge_adjustments(tmp_path: Path) -> None:
    _snapshot(tmp_path, ["i" + ":" * 12 + "l", "i" + ":" * 12 + "l"])
    config = MotifAuditConfig(minimum_run_count=2, minimum_work_count=2)

    catalog = audit_motif_catalog(tmp_path, config=config)

    assert any(
        item["motif"] == ":" and item["character"] == "i"
        for item in catalog["edge_adjustments"]["left"]
    )
    assert any(
        item["motif"] == ":" and item["character"] == "l"
        for item in catalog["edge_adjustments"]["right"]
    )


def test_owner_graph_keeps_only_stable_proposal_ids() -> None:
    labels = np.array(
        [
            [1, 1, 2, 2, 0, 0],
            [1, 1, 2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    active = np.ones(labels.shape, dtype=bool)

    owner, edges = _owner_graph(active, labels)

    assert set(np.unique(owner)) == {0, 1, 2}
    assert edges == {(1, 2)}


def test_best_baselines_filters_earlier_recipe_records(monkeypatch) -> None:
    usable = {
        "recipe_version": "surface-fill-v2",
        "none_usable": False,
        "source": {"filename": "kept.png", "artifact": {"sha256": "kept"}},
    }
    obsolete = {
        "recipe_version": "surface-fill-v1",
        "none_usable": True,
        "source": {"filename": "old.png", "artifact": {"sha256": "old"}},
    }
    selected_v2 = [
        {
            **usable,
            "source": {
                "filename": f"kept-{index:02d}.png",
                "artifact": {"sha256": f"kept-{index:02d}"},
            },
        }
        for index in range(15)
    ]

    class Store:
        def __init__(self, root, **kwargs):
            self.root = root

        def list(self):
            return [obsolete, *selected_v2] if self.root.name == "v2" else []

    monkeypatch.setattr(
        "training.audit_texture_motifs_p6r4t.SurfaceFillPreferenceStore",
        Store,
    )
    baselines = _best_baselines(Path("v2"), Path("v4"))

    assert len(baselines) == 15
    assert all(record["recipe_version"] == "surface-fill-v2" for _, record in baselines)
