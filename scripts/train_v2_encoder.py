"""Train a pyrrho v2-alpha multi-head encoder on fitz-gov-v2 rows.

The alpha head is a single 18-logit vector:
- evidence_verdict: 3-way CE
- failure_mode: 5-way CE
- retrieval_intents: 4 binary BCE logits
- evidence_kinds: 6 binary BCE logits

For dual-mode training, each source row becomes:
- [PYRRHO_PRE] query-only input, training retrieval/evidence-kind heads only
- [PYRRHO_POST] query-plus-sources input, training all heads

Example:
    python scripts/train_v2_encoder.py --config configs/encoder/modernbert_base_v2_alpha.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from pyrrho.manifest import write_manifest
from pyrrho.training import set_all_seeds
from pyrrho.v2 import (
    EVIDENCE_KIND_KEYS,
    EVIDENCE_VERDICTS,
    FAILURE_MODES,
    NUM_V2_LABELS,
    PYRRHO_POST_TAG,
    PYRRHO_PRE_TAG,
    RETRIEVAL_INTENT_KEYS,
    V2_FULL_LABEL_MASK,
    V2_PRE_LABEL_MASK,
    v2_label_names,
)

VERDICT_SLICE = slice(0, 3)
FAILURE_SLICE = slice(3, 8)
RETRIEVAL_SLICE = slice(8, 12)
EVIDENCE_KIND_SLICE = slice(12, 18)
VALID_INPUT_MODES = {"post", "dual"}


class V2AlphaTrainer(Trainer):
    def __init__(
        self,
        loss_weights: dict[str, float] | None = None,
        categorical_label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_weights = loss_weights or {
            "evidence_verdict": 1.0,
            "failure_mode": 1.0,
            "retrieval_intents": 1.0,
            "evidence_kinds": 1.0,
        }
        self.categorical_label_smoothing = categorical_label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(torch.float32)
        label_mask = inputs.pop("label_mask", None)
        if label_mask is None:
            label_mask = torch.ones_like(labels)
        else:
            label_mask = label_mask.to(torch.float32)
        outputs = model(**inputs)
        logits = outputs.logits

        verdict_labels = labels[:, VERDICT_SLICE].argmax(dim=-1)
        failure_labels = labels[:, FAILURE_SLICE].argmax(dim=-1)
        retrieval_labels = labels[:, RETRIEVAL_SLICE]
        evidence_kind_labels = labels[:, EVIDENCE_KIND_SLICE]

        verdict_loss = torch_functional.cross_entropy(
            logits[:, VERDICT_SLICE],
            verdict_labels,
            label_smoothing=self.categorical_label_smoothing,
            reduction="none",
        )
        failure_loss = torch_functional.cross_entropy(
            logits[:, FAILURE_SLICE],
            failure_labels,
            label_smoothing=self.categorical_label_smoothing,
            reduction="none",
        )
        retrieval_loss = torch_functional.binary_cross_entropy_with_logits(
            logits[:, RETRIEVAL_SLICE], retrieval_labels, reduction="none"
        )
        evidence_loss = torch_functional.binary_cross_entropy_with_logits(
            logits[:, EVIDENCE_KIND_SLICE], evidence_kind_labels, reduction="none"
        )

        loss_verdict = _masked_mean(verdict_loss, label_mask[:, VERDICT_SLICE].amax(dim=-1))
        loss_failure = _masked_mean(failure_loss, label_mask[:, FAILURE_SLICE].amax(dim=-1))
        loss_retrieval = _masked_mean(retrieval_loss, label_mask[:, RETRIEVAL_SLICE])
        loss_evidence = _masked_mean(evidence_loss, label_mask[:, EVIDENCE_KIND_SLICE])

        loss = (
            self.loss_weights.get("evidence_verdict", 1.0) * loss_verdict
            + self.loss_weights.get("failure_mode", 1.0) * loss_failure
            + self.loss_weights.get("retrieval_intents", 1.0) * loss_retrieval
            + self.loss_weights.get("evidence_kinds", 1.0) * loss_evidence
        )
        return (loss, outputs) if return_outputs else loss


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a loss tensor over active labels/examples only."""
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    numerator = (loss * mask).sum()
    denominator = mask.sum().clamp_min(1.0)
    return numerator / denominator


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_v2_processed(data_dir: Path) -> DatasetDict:
    hf_dir = data_dir / "hf_dataset"
    if not hf_dir.exists():
        raise FileNotFoundError(f"missing processed v2 dataset: {hf_dir}")
    ds = load_from_disk(str(hf_dir))
    missing = {"train", "eval"} - set(ds.keys())
    if missing:
        raise ValueError(f"dataset missing splits: {sorted(missing)}")
    return ds


def _with_mode_tag(text: str, tag: str) -> str:
    text = str(text or "").strip()
    if text.startswith("[PYRRHO_PRE]") or text.startswith("[PYRRHO_POST]"):
        return text
    return f"{tag}\n{text}"


def _mode_dataset(
    ds: Dataset, *, source_column: str, mode: str, tag: str, label_mask: tuple[float, ...]
) -> Dataset:
    if source_column not in ds.column_names:
        raise ValueError(f"dataset is missing required column for {mode} mode: {source_column}")

    mask = list(label_mask)

    def _map(batch):
        return {
            "training_text": [_with_mode_tag(text, tag) for text in batch[source_column]],
            "label_mask": [mask for _ in batch[source_column]],
            "pyrrho_mode": [mode for _ in batch[source_column]],
        }

    return ds.map(_map, batched=True)


def build_mode_datasets(
    ds: DatasetDict, *, input_mode: str, seed: int
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Build train/eval datasets for post-only or dual pre/post training."""
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(
            f"invalid input_mode={input_mode!r}; expected one of {sorted(VALID_INPUT_MODES)}"
        )

    train_post = _mode_dataset(
        ds["train"],
        source_column="text",
        mode="post",
        tag=PYRRHO_POST_TAG,
        label_mask=V2_FULL_LABEL_MASK,
    )
    eval_post = _mode_dataset(
        ds["eval"],
        source_column="text",
        mode="post",
        tag=PYRRHO_POST_TAG,
        label_mask=V2_FULL_LABEL_MASK,
    )
    if input_mode == "post":
        return train_post, eval_post, None

    train_pre = _mode_dataset(
        ds["train"],
        source_column="query_only_text",
        mode="pre",
        tag=PYRRHO_PRE_TAG,
        label_mask=V2_PRE_LABEL_MASK,
    )
    eval_pre = _mode_dataset(
        ds["eval"],
        source_column="query_only_text",
        mode="pre",
        tag=PYRRHO_PRE_TAG,
        label_mask=V2_PRE_LABEL_MASK,
    )
    train = concatenate_datasets([train_post, train_pre]).shuffle(seed=seed)
    return train, eval_post, eval_pre


def tokenize_dataset(ds, tokenizer, max_length: int, text_column: str):
    def _tok(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)

    keep = {"input_ids", "attention_mask", "labels", "label_mask"}
    drop = [column for column in ds.column_names if column not in keep]
    return ds.map(_tok, batched=True, remove_columns=drop)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def categorical_metrics(
    logits: np.ndarray, labels: np.ndarray, names: tuple[str, ...], prefix: str
) -> dict[str, float]:
    y_true = labels.argmax(axis=-1)
    y_pred = logits.argmax(axis=-1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(names))),
        zero_division=0,
    )
    out: dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for idx, name in enumerate(names):
        out[f"{prefix}_{name}_precision"] = float(precision[idx])
        out[f"{prefix}_{name}_recall"] = float(recall[idx])
        out[f"{prefix}_{name}_f1"] = float(f1[idx])
        out[f"{prefix}_{name}_support"] = float(support[idx])
    return out


def multilabel_metrics(
    logits: np.ndarray, labels: np.ndarray, names: tuple[str, ...], prefix: str
) -> dict[str, float]:
    probs = sigmoid(logits)
    preds = probs >= 0.5
    truth = labels >= 0.5
    f1_per_label = []
    precision_per_label = []
    recall_per_label = []
    out: dict[str, float] = {
        f"{prefix}_exact_match": float((preds == truth).all(axis=1).mean()),
        f"{prefix}_hamming_accuracy": float((preds == truth).mean()),
    }
    for idx, name in enumerate(names):
        p, r, f, _ = precision_recall_fscore_support(
            truth[:, idx].astype(int),
            preds[:, idx].astype(int),
            labels=[0, 1],
            zero_division=0,
        )
        precision_per_label.append(float(p[1]))
        recall_per_label.append(float(r[1]))
        f1_per_label.append(float(f[1]))
        out[f"{prefix}_{name}_precision"] = float(p[1])
        out[f"{prefix}_{name}_recall"] = float(r[1])
        out[f"{prefix}_{name}_f1"] = float(f[1])
        out[f"{prefix}_{name}_positive_rate"] = float(truth[:, idx].mean())
    out[f"{prefix}_macro_precision"] = float(np.mean(precision_per_label))
    out[f"{prefix}_macro_recall"] = float(np.mean(recall_per_label))
    out[f"{prefix}_macro_f1"] = float(np.mean(f1_per_label))
    return out


def compute_v2_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    out: dict[str, float] = {}
    out.update(
        categorical_metrics(
            logits[:, VERDICT_SLICE], labels[:, VERDICT_SLICE], EVIDENCE_VERDICTS, "verdict"
        )
    )
    out.update(
        categorical_metrics(
            logits[:, FAILURE_SLICE], labels[:, FAILURE_SLICE], FAILURE_MODES, "failure"
        )
    )
    out.update(
        multilabel_metrics(
            logits[:, RETRIEVAL_SLICE],
            labels[:, RETRIEVAL_SLICE],
            RETRIEVAL_INTENT_KEYS,
            "retrieval",
        )
    )
    out.update(
        multilabel_metrics(
            logits[:, EVIDENCE_KIND_SLICE],
            labels[:, EVIDENCE_KIND_SLICE],
            EVIDENCE_KIND_KEYS,
            "evidence_kind",
        )
    )

    verdict_true = labels[:, VERDICT_SLICE].argmax(axis=-1)
    verdict_pred = logits[:, VERDICT_SLICE].argmax(axis=-1)
    sufficient_id = EVIDENCE_VERDICTS.index("SUFFICIENT")
    non_sufficient = verdict_true != sufficient_id
    out["false_sufficient_rate"] = (
        float((verdict_pred[non_sufficient] == sufficient_id).mean())
        if non_sufficient.any()
        else 0.0
    )
    out["overall_score"] = float(
        np.mean(
            [
                out["verdict_macro_f1"],
                out["failure_macro_f1"],
                out["retrieval_macro_f1"],
                out["evidence_kind_macro_f1"],
            ]
        )
    )
    return out


def print_summary(metrics: dict[str, float]) -> None:
    fields = [
        "eval_overall_score",
        "eval_verdict_accuracy",
        "eval_verdict_macro_f1",
        "eval_false_sufficient_rate",
        "eval_failure_accuracy",
        "eval_failure_macro_f1",
        "eval_retrieval_exact_match",
        "eval_retrieval_macro_f1",
        "eval_evidence_kind_exact_match",
        "eval_evidence_kind_macro_f1",
    ]
    for field in fields:
        if field in metrics:
            print(f"  {field}: {metrics[field]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/v2_alpha"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_start = time.time()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42))
    set_all_seeds(seed)

    print(f"Config            : {args.config}")
    print(f"Data dir          : {args.data_dir}")
    print(f"Seed              : {seed}")

    ds = load_v2_processed(args.data_dir)
    print("\nDataset")
    print(f"  source train: {len(ds['train']):,}")
    print(f"  source eval : {len(ds['eval']):,}")

    base_model = cfg["model"]["base_model"]
    max_length = int(cfg["data"].get("max_seq_length", 2048))
    input_mode = cfg["data"].get("input_mode", "post")
    if input_mode not in VALID_INPUT_MODES:
        raise ValueError(
            f"invalid data.input_mode={input_mode!r}; expected one of {sorted(VALID_INPUT_MODES)}"
        )
    text_column = "training_text"
    output_dir = Path(args.output_dir or cfg["training"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nModel")
    print(f"  base_model     : {base_model}")
    print(f"  num_labels     : {NUM_V2_LABELS}")
    print(f"  input_mode     : {input_mode}")
    print(f"  text_column    : {text_column}")
    print(f"  max_seq_length : {max_length}")
    print(f"  output_dir     : {output_dir}")

    train_raw, eval_post_raw, eval_pre_raw = build_mode_datasets(
        ds, input_mode=input_mode, seed=seed
    )
    print("\nTraining view")
    print(f"  train examples : {len(train_raw):,}")
    print(f"  post eval      : {len(eval_post_raw):,}")
    if eval_pre_raw is not None:
        print(f"  pre eval       : {len(eval_pre_raw):,}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_V2_LABELS,
        id2label={i: name for i, name in enumerate(v2_label_names())},
        label2id={name: i for i, name in enumerate(v2_label_names())},
        problem_type="multi_label_classification",
    )

    print("\nTokenizing")
    train_ds = tokenize_dataset(train_raw, tokenizer, max_length, text_column)
    eval_ds = tokenize_dataset(eval_post_raw, tokenizer, max_length, text_column)
    eval_pre_ds = (
        tokenize_dataset(eval_pre_raw, tokenizer, max_length, text_column)
        if eval_pre_raw is not None
        else None
    )

    report_to = cfg["training"].get("report_to", "none")
    if args.no_wandb or os.environ.get("WANDB_DISABLED", "").lower() == "true":
        report_to = "none"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["training"].get("num_train_epochs", 3)),
        per_device_train_batch_size=int(cfg["training"].get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(cfg["training"].get("per_device_eval_batch_size", 32)),
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", False)),
        learning_rate=float(cfg["training"].get("learning_rate", 5e-5)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.1)),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "cosine"),
        bf16=bool(cfg["training"].get("bf16", True)),
        optim=cfg["training"].get("optim", "adamw_torch_fused"),
        eval_strategy=cfg["training"].get("eval_strategy", "epoch"),
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        save_total_limit=int(cfg["training"].get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg["training"].get("load_best_model_at_end", True)),
        metric_for_best_model=cfg["training"].get("metric_for_best_model", "overall_score"),
        greater_is_better=bool(cfg["training"].get("greater_is_better", True)),
        logging_steps=int(cfg["training"].get("logging_steps", 20)),
        report_to=report_to,
        run_name=cfg["training"].get("run_name", args.config.stem),
        seed=seed,
        dataloader_pin_memory=True,
    )

    callbacks = []
    patience = cfg["training"].get("early_stopping_patience")
    if patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = V2AlphaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_v2_metrics,
        loss_weights=cfg["training"].get("loss_weights"),
        categorical_label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        callbacks=callbacks,
    )

    if args.dry_run:
        print("\nDRY RUN — Trainer built. Exiting before training.")
        return 0

    print("\n=== TRAINING ===")
    train_result = trainer.train()

    print("\n=== FINAL EVAL ===")
    eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
    print_summary(eval_metrics)

    pre_eval_metrics: dict[str, float] = {}
    if eval_pre_ds is not None:
        print("\n=== PRE-RETRIEVAL EVAL ===")
        pre_prediction = trainer.predict(eval_pre_ds, metric_key_prefix="pre_eval")
        pre_eval_metrics = {
            k: float(v)
            for k, v in pre_prediction.metrics.items()
            if isinstance(v, (int, float))
            and (
                k.startswith("pre_eval_retrieval_")
                or k.startswith("pre_eval_evidence_kind_")
                or k
                in {
                    "pre_eval_loss",
                    "pre_eval_runtime",
                    "pre_eval_samples_per_second",
                    "pre_eval_steps_per_second",
                }
            )
        }
        print(
            "  pre_eval_retrieval_exact_match: "
            f"{pre_eval_metrics.get('pre_eval_retrieval_exact_match', 0.0):.4f}"
        )
        print(
            "  pre_eval_retrieval_macro_f1: "
            f"{pre_eval_metrics.get('pre_eval_retrieval_macro_f1', 0.0):.4f}"
        )
        print(
            "  pre_eval_evidence_kind_exact_match: "
            f"{pre_eval_metrics.get('pre_eval_evidence_kind_exact_match', 0.0):.4f}"
        )
        print(
            "  pre_eval_evidence_kind_macro_f1: "
            f"{pre_eval_metrics.get('pre_eval_evidence_kind_macro_f1', 0.0):.4f}"
        )

    metrics = {
        "train": {
            "global_step": int(train_result.global_step),
            "training_loss": float(train_result.training_loss),
        },
        "eval": {k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))},
        "pre_eval": pre_eval_metrics,
        "config_path": str(args.config),
        "base_model": base_model,
        "seed": seed,
        "input_mode": input_mode,
        "label_names": v2_label_names(),
    }
    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote metrics -> {metrics_path}")

    manifest_path = write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=seed,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=Path("C:/Users/yanfi/PycharmProjects/fitz-gov-modern_generator"),
        start_time=run_start,
        extra={
            "script": "scripts/train_v2_encoder.py",
            "dataset": str(args.data_dir.resolve()),
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
            "pre_eval_size": len(eval_pre_ds) if eval_pre_ds is not None else 0,
            "input_mode": input_mode,
            "mode_tags": {
                "pre": PYRRHO_PRE_TAG,
                "post": PYRRHO_POST_TAG,
            },
            "pre_label_mask": list(V2_PRE_LABEL_MASK),
            "base_model": base_model,
            "num_labels": NUM_V2_LABELS,
            "heads": {
                "evidence_verdict": list(EVIDENCE_VERDICTS),
                "failure_mode": list(FAILURE_MODES),
                "retrieval_intents": list(RETRIEVAL_INTENT_KEYS),
                "evidence_kinds": list(EVIDENCE_KIND_KEYS),
            },
        },
    )
    print(f"Wrote manifest -> {manifest_path}")

    final_path = output_dir / "best_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved best model -> {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
