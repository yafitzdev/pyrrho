"""Eval metrics for pyrrho governance models.

Hard-coded to the 3-class label space (ABSTAIN=0, DISPUTED=1, TRUSTWORTHY=2).
The release pass/fail gates compare against the fitz-sage v0.11 baseline:
    overall accuracy >= 78.7%
    false-trustworthy rate <= 5.7%
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


ABSTAIN_ID = 0
DISPUTED_ID = 1
TRUSTWORTHY_ID = 2

BASELINE_OVERALL = 0.787
BASELINE_FALSE_TRUSTWORTHY = 0.057
TIER0_SANITY_GATE = 0.95


def _as_preds(predictions: np.ndarray) -> np.ndarray:
    """Convert logits → predicted class ids. Pass-through if already class ids."""
    arr = np.asarray(predictions)
    if arr.ndim == 2:
        return arr.argmax(axis=1)
    return arr


def compute_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Core metric set: accuracy, macro F1, per-class P/R/F1, false-trustworthy rate."""
    preds = _as_preds(predictions)
    labels = np.asarray(labels)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[ABSTAIN_ID, DISPUTED_ID, TRUSTWORTHY_ID], zero_division=0
    )

    non_trustworthy = labels != TRUSTWORTHY_ID
    if non_trustworthy.sum() > 0:
        false_trustworthy = float((preds[non_trustworthy] == TRUSTWORTHY_ID).mean())
    else:
        false_trustworthy = 0.0

    # Custom selection metric: accuracy with a hard penalty for excursions above
    # the baseline FT rate. Rewards accuracy when FT is under the safety gate,
    # and punishes 3× per point of excess once FT > 5.7%. Designed for
    # metric_for_best_model so checkpoint selection respects the safety axis.
    ft_excess = max(0.0, false_trustworthy - BASELINE_FALSE_TRUSTWORTHY)
    ft_penalized_accuracy = float(accuracy) - 3.0 * ft_excess

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision_abstain": float(precision[0]),
        "precision_disputed": float(precision[1]),
        "precision_trustworthy": float(precision[2]),
        "recall_abstain": float(recall[0]),
        "recall_disputed": float(recall[1]),
        "recall_trustworthy": float(recall[2]),
        "f1_abstain": float(f1[0]),
        "f1_disputed": float(f1[1]),
        "f1_trustworthy": float(f1[2]),
        "false_trustworthy_rate": false_trustworthy,
        "ft_penalized_accuracy": ft_penalized_accuracy,
    }


def compute_metrics(eval_pred) -> dict[str, float]:
    """`transformers.Trainer`-compatible callback. `eval_pred` is `(logits, labels)`.

    If logits has 4 columns, both predictions and labels are collapsed to the 3-class
    space (HEDGED + DIRECT -> TRUSTWORTHY) before metric computation, so all metrics
    remain directly comparable to the fitz-sage 3-class baseline.
    """
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.shape[-1] == 4:
        preds_4 = logits.argmax(axis=-1)
        preds = np.where(preds_4 >= 2, TRUSTWORTHY_ID, preds_4)
        labels = np.where(labels >= 2, TRUSTWORTHY_ID, labels)
        return compute_classification_metrics(preds, labels)
    return compute_classification_metrics(logits, labels)


def breakdown_by(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-group accuracy + false-trustworthy. `groups` must match the prediction order."""
    preds = _as_preds(predictions)
    labels = np.asarray(labels)
    groups_arr = np.asarray(groups)

    out: dict[str, dict[str, float]] = {}
    for g in sorted(set(groups_arr.tolist())):
        mask = groups_arr == g
        if not mask.any():
            continue
        sub_preds = preds[mask]
        sub_labels = labels[mask]
        sub_non_t = sub_labels != TRUSTWORTHY_ID
        out[g] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(sub_labels, sub_preds)),
            "false_trustworthy_rate": float(
                (sub_preds[sub_non_t] == TRUSTWORTHY_ID).mean()
            )
            if sub_non_t.any()
            else 0.0,
        }
    return out


def check_release_gates(
    eval_metrics: dict[str, float], tier0_metrics: dict[str, float] | None = None
) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Release gates from HANDOFF / METHODOLOGY.

    Tier0 is kept as a diagnostic but is no longer a release gate.
    """
    gates = [
        (
            f"overall accuracy >= {BASELINE_OVERALL:.1%}",
            eval_metrics["accuracy"] >= BASELINE_OVERALL,
            f"got {eval_metrics['accuracy']:.1%}",
        ),
        (
            f"false_trustworthy <= {BASELINE_FALSE_TRUSTWORTHY:.1%}",
            eval_metrics["false_trustworthy_rate"] <= BASELINE_FALSE_TRUSTWORTHY,
            f"got {eval_metrics['false_trustworthy_rate']:.1%}",
        ),
    ]
    all_passed = all(passed for _, passed, _ in gates)
    return all_passed, gates


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def gated_predictions(
    logits: np.ndarray, threshold_t: float, num_classes: int = 3
) -> np.ndarray:
    """Argmax predictions collapsed to 3-class, with TRUSTWORTHY-prob threshold.

    Accepts either 3-class or 4-class logits. For 4-class, P(T) = P(HEDGED) + P(DIRECT).
    When the collapsed argmax is TRUSTWORTHY but P(T) < threshold_t, falls back to
    argmax over {ABSTAIN, DISPUTED}. Returned predictions are always 3-class ids.
    """
    logits = np.asarray(logits)
    probs = _softmax(logits)

    if num_classes == 3:
        argmax_3 = logits.argmax(axis=-1)
        p_t = probs[:, TRUSTWORTHY_ID]
    elif num_classes == 4:
        argmax_4 = logits.argmax(axis=-1)
        argmax_3 = np.where(argmax_4 >= 2, TRUSTWORTHY_ID, argmax_4)
        p_t = probs[:, 2] + probs[:, 3]
    else:
        raise ValueError(f"num_classes must be 3 or 4, got {num_classes}")

    fallback_mask = (argmax_3 == TRUSTWORTHY_ID) & (p_t < threshold_t)
    if fallback_mask.any():
        non_t = logits[fallback_mask][:, [ABSTAIN_ID, DISPUTED_ID]]
        runner_up = np.where(non_t[:, 0] >= non_t[:, 1], ABSTAIN_ID, DISPUTED_ID)
        argmax_3[fallback_mask] = runner_up
    return argmax_3


def sweep_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    grid_size: int = 66,
    num_classes: int = 3,
) -> list[dict[str, float]]:
    """Sweep τ ∈ [0.34, 0.99] on P(T). Return per-τ 3-class classification metrics.

    `labels` should be 3-class ids (collapse 4-class labels upstream if needed).
    """
    thresholds = np.linspace(0.34, 0.99, grid_size)
    labels = np.asarray(labels)
    out = []
    for t in thresholds:
        preds = gated_predictions(logits, float(t), num_classes=num_classes)
        m = compute_classification_metrics(preds, labels)
        m["threshold"] = float(t)
        out.append(m)
    return out


def find_optimal_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    target_ft: float = BASELINE_FALSE_TRUSTWORTHY,
    grid_size: int = 66,
    aux_logits: np.ndarray | None = None,
    aux_labels: np.ndarray | None = None,
    aux_min_accuracy: float = TIER0_SANITY_GATE,
    num_classes: int = 3,
) -> dict[str, float]:
    """Find τ that hits FT <= target on (logits, labels) with max accuracy.

    If `aux_logits/aux_labels` are given (typically tier0_sanity), additionally
    require that τ keeps aux_accuracy >= aux_min_accuracy. Falls back gracefully
    if no τ satisfies all constraints.
    """
    candidates = sweep_thresholds(logits, labels, grid_size, num_classes=num_classes)

    if aux_logits is not None and aux_labels is not None:
        aux_sweep = sweep_thresholds(aux_logits, aux_labels, grid_size, num_classes=num_classes)
        for c, a in zip(candidates, aux_sweep):
            c["aux_accuracy"] = a["accuracy"]
            c["aux_false_trustworthy_rate"] = a["false_trustworthy_rate"]

    primary_passing = [c for c in candidates if c["false_trustworthy_rate"] <= target_ft]

    if aux_logits is not None:
        joint_passing = [
            c for c in primary_passing if c.get("aux_accuracy", 0.0) >= aux_min_accuracy
        ]
        if joint_passing:
            best = max(joint_passing, key=lambda c: c["accuracy"])
            best["target_met"] = True
            best["aux_target_met"] = True
            return best

    if primary_passing:
        best = max(primary_passing, key=lambda c: c["accuracy"])
        best["target_met"] = True
        best["aux_target_met"] = False
        return best

    best = min(candidates, key=lambda c: c["false_trustworthy_rate"])
    best["target_met"] = False
    best["aux_target_met"] = False
    return best


def format_metrics_table(metrics: dict[str, float]) -> str:
    """Human-readable single-block report for stdout."""
    lines = [
        f"  accuracy             : {metrics['accuracy']:.4f}",
        f"  macro_f1             : {metrics['macro_f1']:.4f}",
        f"  false_trustworthy    : {metrics['false_trustworthy_rate']:.4f}",
        "",
        "  per-class            precision    recall      f1",
        f"  ABSTAIN              {metrics['precision_abstain']:.4f}      "
        f"{metrics['recall_abstain']:.4f}      {metrics['f1_abstain']:.4f}",
        f"  DISPUTED             {metrics['precision_disputed']:.4f}      "
        f"{metrics['recall_disputed']:.4f}      {metrics['f1_disputed']:.4f}",
        f"  TRUSTWORTHY          {metrics['precision_trustworthy']:.4f}      "
        f"{metrics['recall_trustworthy']:.4f}      {metrics['f1_trustworthy']:.4f}",
    ]
    return "\n".join(lines)
