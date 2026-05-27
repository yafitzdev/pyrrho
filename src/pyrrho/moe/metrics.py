"""Evaluation helpers for pyrrho-MoE prototypes."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pyrrho.metrics import (
    BASELINE_FALSE_TRUSTWORTHY,
    compute_classification_metrics,
    find_optimal_threshold,
)


def route_accuracy(route_preds: np.ndarray, route_labels: np.ndarray) -> float:
    return float((route_preds == route_labels).mean()) if len(route_labels) else 0.0


def taxonomy_accuracy(taxonomy_preds: np.ndarray, taxonomy_labels: np.ndarray) -> float:
    return float((taxonomy_preds == taxonomy_labels).mean()) if len(taxonomy_labels) else 0.0


def expert_traffic(route_preds: np.ndarray, route_labels: np.ndarray) -> dict[str, dict[str, int]]:
    pred_counts = Counter(int(v) for v in route_preds.tolist())
    gold_counts = Counter(int(v) for v in route_labels.tolist())
    keys = sorted(set(pred_counts) | set(gold_counts))
    return {
        str(k): {
            "predicted": int(pred_counts.get(k, 0)),
            "gold": int(gold_counts.get(k, 0)),
        }
        for k in keys
    }


def moe_eval_metrics(
    *,
    governance_logits: np.ndarray,
    labels: np.ndarray,
    route_logits: np.ndarray,
    route_labels: np.ndarray,
    taxonomy_logits: np.ndarray,
    taxonomy_labels: np.ndarray,
    calibration_grid_size: int = 66,
    target_ft: float = BASELINE_FALSE_TRUSTWORTHY,
) -> dict:
    governance_preds = governance_logits.argmax(axis=-1)
    route_preds = route_logits.argmax(axis=-1)
    taxonomy_preds = taxonomy_logits.argmax(axis=-1)
    return {
        "governance": compute_classification_metrics(governance_preds, labels),
        "governance_calibrated": find_optimal_threshold(
            governance_logits,
            labels,
            target_ft=target_ft,
            grid_size=calibration_grid_size,
            num_classes=3,
        ),
        "route_accuracy": route_accuracy(route_preds, route_labels),
        "taxonomy_accuracy": taxonomy_accuracy(taxonomy_preds, taxonomy_labels),
        "expert_traffic": expert_traffic(route_preds, route_labels),
    }
