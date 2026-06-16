from __future__ import annotations

import numpy as np

from pyrrho.metrics import ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID

import importlib.util
from pathlib import Path


def _load_gap_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_moe_posthoc_quorum_gaps.py"
    spec = importlib.util.spec_from_file_location("analyze_moe_posthoc_quorum_gaps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gap = _load_gap_module()


def test_compare_prediction_sets_counts_target_and_candidate_wins() -> None:
    labels = np.asarray([TRUSTWORTHY_ID, TRUSTWORTHY_ID, DISPUTED_ID, ABSTAIN_ID])
    target = np.asarray([TRUSTWORTHY_ID, TRUSTWORTHY_ID, DISPUTED_ID, TRUSTWORTHY_ID])
    candidate = np.asarray([ABSTAIN_ID, TRUSTWORTHY_ID, TRUSTWORTHY_ID, ABSTAIN_ID])
    row_ids = ["a", "b", "c", "d"]
    metadata = {
        "a": {"route": "science", "taxonomy": "multi_source_corroboration", "query": "qa"},
        "b": {"route": "law", "taxonomy": "direct_answer", "query": "qb"},
        "c": {"route": "law", "taxonomy": "factual_contradiction", "query": "qc"},
        "d": {"route": "general", "taxonomy": "evidence_absent", "query": "qd"},
    }

    out = gap.compare_prediction_sets(
        name="candidate",
        target_predictions=target,
        candidate_predictions=candidate,
        labels=labels,
        row_ids=row_ids,
        metadata_by_id=metadata,
        top_k=3,
    )

    assert out["counts"]["both_correct"] == 1
    assert out["counts"]["target_wins"] == 2
    assert out["counts"]["candidate_wins"] == 1
    assert out["counts"]["candidate_extra_false_trustworthy"] == 1
    assert out["top"]["target_win_taxonomies"][0] == {
        "name": "multi_source_corroboration",
        "count": 1,
    }
