from __future__ import annotations

import numpy as np
import pytest

from pyrrho.metrics import ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID
from pyrrho.moe.posthoc_policies import (
    build_default_policy_outputs,
    majority_vote,
    trustworthy_quorum,
)


def test_majority_vote_uses_safety_tie_for_three_way_ties() -> None:
    preds = np.asarray(
        [
            [TRUSTWORTHY_ID, TRUSTWORTHY_ID, ABSTAIN_ID],
            [DISPUTED_ID, TRUSTWORTHY_ID, DISPUTED_ID],
            [ABSTAIN_ID, DISPUTED_ID, DISPUTED_ID],
        ]
    )

    out = majority_vote(preds)

    assert out.predictions.tolist() == [ABSTAIN_ID, TRUSTWORTHY_ID, DISPUTED_ID]


def test_trustworthy_quorum_demotes_single_trustworthy_vote_to_best_non_t() -> None:
    preds = np.asarray(
        [
            [TRUSTWORTHY_ID, TRUSTWORTHY_ID],
            [DISPUTED_ID, ABSTAIN_ID],
            [ABSTAIN_ID, DISPUTED_ID],
        ]
    )
    mean_probs = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    out = trustworthy_quorum(preds, quorum=2, mean_probs=mean_probs)

    assert out.predictions.tolist() == [DISPUTED_ID, ABSTAIN_ID]


def test_default_policy_outputs_include_unanimous_for_three_seeds() -> None:
    preds = np.asarray(
        [
            [TRUSTWORTHY_ID, TRUSTWORTHY_ID],
            [TRUSTWORTHY_ID, ABSTAIN_ID],
            [TRUSTWORTHY_ID, DISPUTED_ID],
        ]
    )
    probs = np.full((3, 2, 3), 1 / 3, dtype=np.float32)

    outputs = build_default_policy_outputs(seed_predictions=preds, seed_probabilities=probs)

    assert [output.name for output in outputs] == [
        "majority_guarded_safety_tie",
        "trustworthy_quorum_2_of_3",
        "trustworthy_unanimous",
    ]
    assert outputs[2].predictions.tolist() == [TRUSTWORTHY_ID, ABSTAIN_ID]


def test_trustworthy_quorum_validates_quorum() -> None:
    preds = np.asarray([[TRUSTWORTHY_ID]])

    with pytest.raises(ValueError, match="quorum"):
        trustworthy_quorum(preds, quorum=2)
