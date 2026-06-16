"""Policy helpers for combining packaged post-hoc verifier predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrrho.metrics import ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID

NON_TRUSTWORTHY_IDS = (ABSTAIN_ID, DISPUTED_ID)
SAFETY_TIE_ORDER = (ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID)


@dataclass(frozen=True)
class PosthocPolicyOutput:
    name: str
    predictions: np.ndarray


def _validate_seed_predictions(seed_predictions: np.ndarray) -> np.ndarray:
    preds = np.asarray(seed_predictions, dtype=np.int64)
    if preds.ndim != 2:
        raise ValueError(f"seed_predictions must be 2D [seeds, rows], got {preds.shape}")
    if preds.shape[0] < 1:
        raise ValueError("at least one seed is required")
    return preds


def _validate_mean_probs(mean_probs: np.ndarray | None, rows: int) -> np.ndarray | None:
    if mean_probs is None:
        return None
    probs = np.asarray(mean_probs, dtype=np.float32)
    if probs.shape != (rows, 3):
        raise ValueError(f"mean_probs must have shape ({rows}, 3), got {probs.shape}")
    return probs


def _safety_tie_choice(candidates: set[int]) -> int:
    for label_id in SAFETY_TIE_ORDER:
        if label_id in candidates:
            return label_id
    raise ValueError(f"no valid candidates in {candidates}")


def _non_trustworthy_choice(labels: list[int], row_probs: np.ndarray | None = None) -> int:
    counts = {label_id: labels.count(label_id) for label_id in NON_TRUSTWORTHY_IDS}
    max_count = max(counts.values())
    candidates = {label_id for label_id, count in counts.items() if count == max_count}
    if len(candidates) == 1:
        return next(iter(candidates))
    if row_probs is not None:
        return int(max(candidates, key=lambda label_id: float(row_probs[label_id])))
    return _safety_tie_choice(candidates)


def majority_vote(
    seed_predictions: np.ndarray,
    *,
    mean_probs: np.ndarray | None = None,
    name: str = "majority_guarded_safety_tie",
) -> PosthocPolicyOutput:
    """Vote across seed predictions, using safety order for 3-way ties."""

    preds = _validate_seed_predictions(seed_predictions)
    probs = _validate_mean_probs(mean_probs, preds.shape[1])
    out = np.zeros(preds.shape[1], dtype=np.int64)
    for row_idx in range(preds.shape[1]):
        labels = [int(value) for value in preds[:, row_idx]]
        counts = {label_id: labels.count(label_id) for label_id in range(3)}
        max_count = max(counts.values())
        candidates = {label_id for label_id, count in counts.items() if count == max_count}
        if len(candidates) == 1:
            out[row_idx] = next(iter(candidates))
        elif probs is not None and TRUSTWORTHY_ID not in candidates:
            out[row_idx] = _non_trustworthy_choice(labels, probs[row_idx])
        else:
            out[row_idx] = _safety_tie_choice(candidates)
    return PosthocPolicyOutput(name=name, predictions=out)


def trustworthy_quorum(
    seed_predictions: np.ndarray,
    *,
    quorum: int,
    mean_probs: np.ndarray | None = None,
    name: str | None = None,
) -> PosthocPolicyOutput:
    """Predict TRUSTWORTHY only when enough seeds agree; otherwise choose non-T."""

    preds = _validate_seed_predictions(seed_predictions)
    if quorum < 1 or quorum > preds.shape[0]:
        raise ValueError(f"quorum must be between 1 and seed count {preds.shape[0]}, got {quorum}")
    probs = _validate_mean_probs(mean_probs, preds.shape[1])
    out = np.zeros(preds.shape[1], dtype=np.int64)
    for row_idx in range(preds.shape[1]):
        labels = [int(value) for value in preds[:, row_idx]]
        trustworthy_votes = labels.count(TRUSTWORTHY_ID)
        if trustworthy_votes >= quorum:
            out[row_idx] = TRUSTWORTHY_ID
        else:
            out[row_idx] = _non_trustworthy_choice(
                labels,
                probs[row_idx] if probs is not None else None,
            )
    return PosthocPolicyOutput(
        name=name or f"trustworthy_quorum_{quorum}",
        predictions=out,
    )


def build_default_policy_outputs(
    *,
    seed_predictions: np.ndarray,
    seed_probabilities: np.ndarray,
) -> list[PosthocPolicyOutput]:
    """Default policies to compare for 3-seed verifier packages."""

    preds = _validate_seed_predictions(seed_predictions)
    probs = np.asarray(seed_probabilities, dtype=np.float32)
    if probs.shape != (preds.shape[0], preds.shape[1], 3):
        raise ValueError(
            "seed_probabilities must have shape [seeds, rows, 3], "
            f"got {probs.shape}"
        )
    mean_probs = probs.mean(axis=0)
    outputs = [
        majority_vote(preds, mean_probs=mean_probs),
    ]
    if preds.shape[0] >= 2:
        outputs.append(
            trustworthy_quorum(
                preds,
                quorum=2,
                mean_probs=mean_probs,
                name="trustworthy_quorum_2_of_3",
            )
        )
    if preds.shape[0] >= 3:
        outputs.append(
            trustworthy_quorum(
                preds,
                quorum=preds.shape[0],
                mean_probs=mean_probs,
                name="trustworthy_unanimous",
            )
        )
    return outputs
