from __future__ import annotations

import numpy as np

from pyrrho.metrics import ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID
from pyrrho.moe.posthoc_thresholds import (
    guarded_predictions_at_threshold,
    non_trustworthy_fallback_from_probs,
    select_threshold_row,
    sweep_verifier_thresholds,
)


def test_guarded_predictions_at_threshold_demotes_only_low_score_trustworthy() -> None:
    preds = np.asarray([TRUSTWORTHY_ID, TRUSTWORTHY_ID, DISPUTED_ID])
    scores = np.asarray([0.9, 0.2, 0.1], dtype=np.float32)
    fallback = np.asarray([ABSTAIN_ID, DISPUTED_ID, ABSTAIN_ID])

    guarded = guarded_predictions_at_threshold(
        base_predictions=preds,
        accept_scores=scores,
        non_trustworthy_fallback=fallback,
        threshold=0.5,
    )

    assert guarded.tolist() == [TRUSTWORTHY_ID, DISPUTED_ID, DISPUTED_ID]


def test_non_trustworthy_fallback_uses_stronger_abstain_or_disputed_probability() -> None:
    probs = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    assert non_trustworthy_fallback_from_probs(probs).tolist() == [ABSTAIN_ID, DISPUTED_ID]


def test_threshold_selection_prefers_accuracy_within_target_ft() -> None:
    labels = np.asarray([TRUSTWORTHY_ID, DISPUTED_ID, DISPUTED_ID])
    base = np.asarray([TRUSTWORTHY_ID, TRUSTWORTHY_ID, DISPUTED_ID])
    scores = np.asarray([0.8, 0.4, 0.2], dtype=np.float32)
    fallback = np.asarray([ABSTAIN_ID, DISPUTED_ID, ABSTAIN_ID])
    rows = sweep_verifier_thresholds(
        labels=labels,
        base_predictions=base,
        accept_scores=scores,
        non_trustworthy_fallback=fallback,
        thresholds=[0.0, 0.5, 0.9],
    )

    selected = select_threshold_row(rows, target_ft=0.0)

    assert selected["threshold"] == 0.5
    assert selected["accuracy"] == 1.0
