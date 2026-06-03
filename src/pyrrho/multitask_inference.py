"""Runtime inference helpers for pyrrho multitask encoder packages."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer

from pyrrho.data import build_encoder_text, build_query_contract_text
from pyrrho.multitask import PyrrhoMultiTaskModernBert


def softmax(logits: Iterable[float]) -> list[float]:
    values = [float(value) for value in logits]
    if not values:
        return []
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def prediction_entropy(probabilities: Iterable[float]) -> float:
    return float(-sum(prob * math.log(prob) for prob in probabilities if prob > 0.0))


def class_prediction(
    logits: Iterable[float],
    id2label: dict[int, str],
    *,
    trustworthy_threshold: float | None = None,
) -> dict[str, Any]:
    """Return label/probability metadata for one classifier head."""
    logits_list = [float(value) for value in logits]
    probabilities = softmax(logits_list)
    if not probabilities:
        raise ValueError("cannot classify an empty logits vector")

    ranked_ids = sorted(range(len(probabilities)), key=lambda idx: probabilities[idx], reverse=True)
    raw_id = ranked_ids[0]
    final_id = raw_id
    used_threshold_fallback = False
    threshold = None if trustworthy_threshold is None else float(trustworthy_threshold)

    if (
        threshold is not None
        and id2label.get(raw_id) == "TRUSTWORTHY"
        and probabilities[raw_id] < threshold
    ):
        fallback_ids = [
            idx for idx in range(len(probabilities)) if id2label.get(idx) != "TRUSTWORTHY"
        ]
        final_id = max(fallback_ids, key=lambda idx: probabilities[idx])
        used_threshold_fallback = True

    runner_up_id = ranked_ids[1] if len(ranked_ids) > 1 else raw_id
    final_probability = probabilities[final_id]
    runner_up_probability = probabilities[runner_up_id]
    label_probabilities = {
        id2label[int(idx)]: float(probabilities[idx]) for idx in range(len(probabilities))
    }
    return {
        "raw_label": id2label[int(raw_id)],
        "final_label": id2label[int(final_id)],
        "used_threshold_fallback": used_threshold_fallback,
        "threshold": threshold,
        "confidence": float(final_probability),
        "probabilities": label_probabilities,
        "runner_up_label": id2label[int(runner_up_id)],
        "runner_up_probability": float(runner_up_probability),
        "margin_to_runner_up": float(final_probability - runner_up_probability),
        "entropy": prediction_entropy(probabilities),
    }


class PyrrhoMultiTaskPredictor:
    """Small runtime wrapper around a packaged `PyrrhoMultiTaskModernBert`."""

    def __init__(
        self,
        model: PyrrhoMultiTaskModernBert,
        tokenizer: Any,
        *,
        trustworthy_threshold: float | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.trustworthy_threshold = trustworthy_threshold
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        trustworthy_threshold: float | None = None,
        device: str | torch.device = "cpu",
    ) -> PyrrhoMultiTaskPredictor:
        model_path = Path(model_dir)
        if trustworthy_threshold is None:
            manifest_path = model_path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw_threshold = manifest.get("release", {}).get("trustworthy_threshold")
                trustworthy_threshold = float(raw_threshold) if raw_threshold is not None else None
        model = PyrrhoMultiTaskModernBert.from_pretrained(model_path, map_location="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(
            model,
            tokenizer,
            trustworthy_threshold=trustworthy_threshold,
            device=device,
        )

    @torch.no_grad()
    def predict(
        self,
        query: str,
        contexts: Iterable[str],
        *,
        max_seq_length: int = 4096,
        max_query_length: int = 256,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        context_list = [str(context) for context in contexts]
        full_text = build_encoder_text(query, context_list)
        query_text = build_query_contract_text(query)
        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        query_only = self.tokenizer(
            query_text,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        )
        outputs = self.model(
            input_ids=full["input_ids"].to(self.device),
            attention_mask=full["attention_mask"].to(self.device),
            query_input_ids=query_only["input_ids"].to(self.device),
            query_attention_mask=query_only["attention_mask"].to(self.device),
        )
        cfg = self.model.pyrrho_config
        scalars = outputs["scalar_preds"][0].detach().cpu().tolist()
        return {
            "schema_version": "pyrrho_multitask_prediction_v1",
            "num_contexts": len(context_list),
            "governance": class_prediction(
                outputs["governance_logits"][0].detach().cpu().tolist(),
                cfg.id2label,
                trustworthy_threshold=self.trustworthy_threshold,
            ),
            "query_contract": class_prediction(
                outputs["query_contract_logits"][0].detach().cpu().tolist(),
                cfg.query_contract_id2label,
            ),
            "route": class_prediction(
                outputs["route_logits"][0].detach().cpu().tolist(),
                cfg.route_id2label,
            ),
            "taxonomy": class_prediction(
                outputs["taxonomy_logits"][0].detach().cpu().tolist(),
                cfg.taxonomy_id2label,
            ),
            "scalars": {
                field: float(value) for field, value in zip(cfg.scalar_fields, scalars, strict=True)
            },
            "timing_ms": float((time.perf_counter() - started) * 1000.0),
        }
