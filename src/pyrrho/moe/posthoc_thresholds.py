"""Threshold sweeps for post-hoc verifier accept scores."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pyrrho.data import ID2LABEL
from pyrrho.metrics import TRUSTWORTHY_ID, compute_classification_metrics


def non_trustworthy_fallback_from_probs(governance_probabilities: np.ndarray) -> np.ndarray:
    """Return the stronger ABSTAIN/DISPUTED fallback for each row."""

    probs = np.asarray(governance_probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"governance_probabilities must be [rows, classes], got {probs.shape}")
    return probs[:, :TRUSTWORTHY_ID].argmax(axis=1).astype(np.int64)


def guarded_predictions_at_threshold(
    *,
    base_predictions: np.ndarray,
    accept_scores: np.ndarray,
    non_trustworthy_fallback: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Demote candidate TRUSTWORTHY predictions below the verifier threshold."""

    preds = np.asarray(base_predictions, dtype=np.int64).copy()
    scores = np.asarray(accept_scores, dtype=np.float32)
    fallback = np.asarray(non_trustworthy_fallback, dtype=np.int64)
    if preds.shape != scores.shape or preds.shape != fallback.shape:
        raise ValueError(
            "base_predictions, accept_scores, and non_trustworthy_fallback must align; "
            f"got {preds.shape}, {scores.shape}, {fallback.shape}"
        )
    reject = (preds == TRUSTWORTHY_ID) & (scores < float(threshold))
    preds[reject] = fallback[reject]
    return preds


def _pred_counts(predictions: np.ndarray) -> dict[str, int]:
    return {
        ID2LABEL[label_id]: int((predictions == label_id).sum())
        for label_id in sorted(ID2LABEL)
    }


def metric_row(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    metrics = compute_classification_metrics(predictions, labels)
    return {
        "accuracy": float(metrics["accuracy"]),
        "false_trustworthy_rate": float(metrics["false_trustworthy_rate"]),
        "trustworthy_recall": float(metrics["recall_trustworthy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "pred_counts": _pred_counts(predictions),
    }


def sweep_verifier_thresholds(
    *,
    labels: np.ndarray,
    base_predictions: np.ndarray,
    accept_scores: np.ndarray,
    non_trustworthy_fallback: np.ndarray,
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    rows = []
    labels_arr = np.asarray(labels, dtype=np.int64)
    base_arr = np.asarray(base_predictions, dtype=np.int64)
    scores = np.asarray(accept_scores, dtype=np.float32)
    fallback = np.asarray(non_trustworthy_fallback, dtype=np.int64)
    for threshold in thresholds:
        preds = guarded_predictions_at_threshold(
            base_predictions=base_arr,
            accept_scores=scores,
            non_trustworthy_fallback=fallback,
            threshold=float(threshold),
        )
        rejected = (base_arr == TRUSTWORTHY_ID) & (scores < float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                **metric_row(preds, labels_arr),
                "rejected_candidate_trustworthy": int(rejected.sum()),
            }
        )
    return rows


def select_threshold_row(
    rows: list[dict[str, Any]],
    *,
    target_ft: float,
    min_accuracy: float | None = None,
) -> dict[str, Any]:
    """Select the highest-accuracy threshold satisfying eval constraints."""

    passing = [
        row
        for row in rows
        if float(row["false_trustworthy_rate"]) <= float(target_ft)
        and (min_accuracy is None or float(row["accuracy"]) >= float(min_accuracy))
    ]
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                float(row["accuracy"]),
                float(row["trustworthy_recall"]),
                -float(row["false_trustworthy_rate"]),
            ),
        )
        reason = "target_ft_and_accuracy_floor" if min_accuracy is not None else "target_ft"
    else:
        ft_passing = [
            row
            for row in rows
            if float(row["false_trustworthy_rate"]) <= float(target_ft)
        ]
        if ft_passing:
            selected = max(
                ft_passing,
                key=lambda row: (
                    float(row["accuracy"]),
                    float(row["trustworthy_recall"]),
                ),
            )
            reason = "target_ft_only"
        else:
            selected = min(
                rows,
                key=lambda row: (
                    float(row["false_trustworthy_rate"]),
                    -float(row["accuracy"]),
                    -float(row["trustworthy_recall"]),
                ),
            )
            reason = "min_false_trustworthy"
    return {**selected, "selection_reason": reason}
