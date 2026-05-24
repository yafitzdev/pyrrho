"""train_encoder.py — Fine-tune a HuggingFace encoder on the prepared fitz-gov splits.

Default target: release #1 (`pyrrho-nano-g1`). Driven by a YAML config so
the same script works for ModernBERT-base / DeBERTa-v3-base / DeBERTa-v3-large with
no code change.

Run from project root:
    python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml
    python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows (default cp1252 crashes on Greek/arrow characters).
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class WeightedLossTrainer(Trainer):
    """Trainer with weighted cross-entropy + optional label smoothing.

    Set class_weights=None to skip weighting. Set label_smoothing=0.0 to skip smoothing.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = (
            self.class_weights.to(logits.device, dtype=logits.dtype)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        return (loss, outputs) if return_outputs else loss

from pyrrho.data import ID2LABEL, LABEL2ID, load_processed
import numpy as np
import time
from pyrrho.manifest import write_manifest
from pyrrho.metrics import (
    check_release_gates,
    compute_classification_metrics,
    compute_metrics,
    find_optimal_threshold,
    format_metrics_table,
    gated_predictions,
    sweep_thresholds,
)
from pyrrho.training import set_all_seeds, tokenize_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Path to encoder YAML config")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory holding hf_dataset/ from prepare_data.py (default: data/processed)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override training.output_dir from the config",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb regardless of the config setting (useful for first run)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up everything but do not actually call Trainer.train()",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.seed from the config (used for multi-seed validation)",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def main() -> int:
    args = parse_args()
    run_start = time.time()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    print(f"Config            : {args.config}")
    print(f"Data dir          : {args.data_dir}")

    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42))
    set_all_seeds(seed)
    print(f"Seed              : {seed}")

    print("\nLoading dataset...")
    ds = load_processed(args.data_dir)
    print(f"  train         : {len(ds['train'])}")
    print(f"  eval          : {len(ds['eval'])}")
    if "test" in ds:
        print(f"  test          : {len(ds['test'])}")
    has_tier0 = "tier0_sanity" in ds and len(ds["tier0_sanity"]) > 0
    if has_tier0:
        print(f"  tier0_sanity  : {len(ds['tier0_sanity'])}")

    base_model = cfg["model"]["base_model"]
    max_length = int(cfg["data"]["max_seq_length"])
    num_labels = int(cfg["model"].get("num_labels", 3))
    id2label_cfg = cfg["model"].get("id2label")
    label2id_cfg = cfg["model"].get("label2id")
    id2label = {int(k): v for k, v in id2label_cfg.items()} if id2label_cfg else ID2LABEL
    label2id = dict(label2id_cfg) if label2id_cfg else LABEL2ID
    print(f"\nBase model        : {base_model}")
    print(f"num_labels        : {num_labels}")
    print(f"max_seq_length    : {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("\nTokenizing splits...")
    train_ds = tokenize_dataset(ds["train"], tokenizer, max_length)
    eval_ds = tokenize_dataset(ds["eval"], tokenizer, max_length)
    test_ds = tokenize_dataset(ds["test"], tokenizer, max_length) if "test" in ds else None
    tier0_ds = tokenize_dataset(ds["tier0_sanity"], tokenizer, max_length) if has_tier0 else None

    output_dir = args.output_dir or Path(cfg["training"]["output_dir"])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput dir        : {output_dir}")

    report_to = cfg["training"].get("report_to", "none")
    if args.no_wandb or os.environ.get("WANDB_DISABLED", "").lower() == "true":
        report_to = "none"

    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", False)),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "linear"),
        bf16=bool(cfg["training"].get("bf16", True)),
        optim=cfg["training"].get("optim", "adamw_torch"),
        eval_strategy=cfg["training"].get("eval_strategy", "epoch"),
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        save_total_limit=int(cfg["training"].get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg["training"].get("load_best_model_at_end", True)),
        metric_for_best_model=cfg["training"].get("metric_for_best_model", "macro_f1"),
        greater_is_better=bool(cfg["training"].get("greater_is_better", True)),
        logging_steps=int(cfg["training"].get("logging_steps", 20)),
        report_to=report_to,
        run_name=cfg["training"].get("run_name", args.config.stem),
        seed=seed,
        dataloader_pin_memory=True,
    )

    class_weights_cfg = cfg["training"].get("class_weights")
    class_weights = (
        torch.tensor(class_weights_cfg, dtype=torch.float)
        if class_weights_cfg
        else None
    )
    if class_weights is not None:
        print(f"Class weights      : {class_weights.tolist()} (ABSTAIN, DISPUTED, TRUSTWORTHY)")
    else:
        print("Class weights      : disabled (uniform 1.0)")

    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    print(f"Label smoothing    : {label_smoothing}")

    early_stopping_patience = cfg["training"].get("early_stopping_patience")
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience))
        )
        print(f"Early stopping     : patience={early_stopping_patience} on {targs.metric_for_best_model}")

    trainer = WeightedLossTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        callbacks=callbacks,
    )

    if args.dry_run:
        print("\nDRY RUN — Trainer built. Exiting before .train().")
        return 0

    print("\n=== TRAINING ===")
    trainer.train()

    def collapse_labels(labels_arr):
        """Collapse 4-class labels to 3-class so metrics stay comparable to baseline."""
        labels_arr = np.asarray(labels_arr)
        if num_labels == 4:
            return np.where(labels_arr >= 2, 2, labels_arr)
        return labels_arr

    def collapsed_argmax(logits_arr):
        logits_arr = np.asarray(logits_arr)
        preds = logits_arr.argmax(axis=-1)
        if num_labels == 4:
            preds = np.where(preds >= 2, 2, preds)
        return preds

    print("\n=== VALIDATION EVAL (checkpoint/threshold split, uncalibrated argmax, 3-class space) ===")
    eval_pred = trainer.predict(eval_ds)
    eval_labels_3 = collapse_labels(eval_pred.label_ids)
    eval_preds_3 = collapsed_argmax(eval_pred.predictions)
    eval_metrics = compute_classification_metrics(eval_preds_3, eval_labels_3)
    print(format_metrics_table(eval_metrics))

    test_pred = None
    test_labels_3 = None
    test_metrics = None
    if test_ds is not None:
        print("\n=== HELD-OUT TEST (uncalibrated argmax, 3-class space) ===")
        test_pred = trainer.predict(test_ds)
        test_labels_3 = collapse_labels(test_pred.label_ids)
        test_preds_3 = collapsed_argmax(test_pred.predictions)
        test_metrics = compute_classification_metrics(test_preds_3, test_labels_3)
        print(format_metrics_table(test_metrics))

    tier0_pred = None
    tier0_labels_3 = None
    tier0_metrics = None
    if tier0_ds is not None:
        print("\n=== TIER0 SANITY (uncalibrated argmax, diagnostic only) ===")
        tier0_pred = trainer.predict(tier0_ds)
        tier0_labels_3 = collapse_labels(tier0_pred.label_ids)
        tier0_preds_3 = collapsed_argmax(tier0_pred.predictions)
        tier0_metrics = compute_classification_metrics(tier0_preds_3, tier0_labels_3)
        print(format_metrics_table(tier0_metrics))

    print("\n=== THRESHOLD CALIBRATION (sweep τ on validation eval) ===")
    best_thr = find_optimal_threshold(
        eval_pred.predictions,
        eval_labels_3,
        num_classes=num_labels,
    )
    tau = best_thr["threshold"]
    print(f"  selected τ          : {tau:.4f}")
    print(f"  eval FT target met  : {best_thr['target_met']}")
    if tier0_ds is not None:
        print("  tier0 acc target met: diagnostic only (not a release gate)")

    full_sweep = sweep_thresholds(eval_pred.predictions, eval_labels_3, num_classes=num_labels)
    if tier0_pred is not None and tier0_labels_3 is not None:
        tier0_sweep = sweep_thresholds(tier0_pred.predictions, tier0_labels_3, num_classes=num_labels)
        print("\n  Sweep table (eval_acc / eval_FT / tier0_acc / tier0_FT):")
        print(f"  {'tau':>6s}  {'eval_acc':>9s}  {'eval_FT':>8s}  {'t0_acc':>7s}  {'t0_FT':>7s}")
        for s, t0 in zip(full_sweep, tier0_sweep):
            marker = " <--" if abs(s["threshold"] - tau) < 1e-6 else ""
            print(
                f"  {s['threshold']:6.3f}  {s['accuracy']:9.4f}  "
                f"{s['false_trustworthy_rate']:8.4f}  "
                f"{t0['accuracy']:7.4f}  {t0['false_trustworthy_rate']:7.4f}{marker}"
            )
    else:
        print("\n  Sweep table (eval_acc / eval_FT):")
        print(f"  {'tau':>6s}  {'eval_acc':>9s}  {'eval_FT':>8s}")
        for s in full_sweep:
            marker = " <--" if abs(s["threshold"] - tau) < 1e-6 else ""
            print(
                f"  {s['threshold']:6.3f}  {s['accuracy']:9.4f}  "
                f"{s['false_trustworthy_rate']:8.4f}{marker}"
            )

    print(f"\n  calibrated eval:")
    print(format_metrics_table(best_thr))

    tier0_calibrated = None
    if tier0_pred is not None and tier0_labels_3 is not None:
        print("\n=== TIER0 SANITY (calibrated with τ) ===")
        tier0_calibrated_preds = gated_predictions(tier0_pred.predictions, tau, num_classes=num_labels)
        tier0_calibrated = compute_classification_metrics(tier0_calibrated_preds, tier0_labels_3)
        print(format_metrics_table(tier0_calibrated))

    test_calibrated = None
    if test_pred is not None and test_labels_3 is not None:
        print("\n=== HELD-OUT TEST (calibrated with validation-selected τ) ===")
        test_calibrated_preds = gated_predictions(test_pred.predictions, tau, num_classes=num_labels)
        test_calibrated = compute_classification_metrics(test_calibrated_preds, test_labels_3)
        print(format_metrics_table(test_calibrated))

    all_metrics = {
        "eval_uncalibrated": eval_metrics,
        "eval_calibrated": {k: v for k, v in best_thr.items() if not k.startswith("_")},
        "threshold": tau,
        "target_met_on_eval": best_thr["target_met"],
        "target_met_on_test": bool(
            test_calibrated
            and test_calibrated["accuracy"] >= 0.787
            and test_calibrated["false_trustworthy_rate"] <= 0.057
        ),
        "config_path": str(args.config),
        "base_model": base_model,
        "seed": seed,
    }
    if test_metrics is not None and test_calibrated is not None:
        all_metrics["test_uncalibrated"] = test_metrics
        all_metrics["test_calibrated"] = test_calibrated
    if tier0_metrics is not None and tier0_calibrated is not None:
        all_metrics["tier0_uncalibrated"] = tier0_metrics
        all_metrics["tier0_calibrated"] = tier0_calibrated
    metrics_path = output_dir / "final_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nWrote final metrics -> {metrics_path}")

    # Reproducibility manifest — git/pip/hw/seed/timing, paired with final_metrics.json.
    fitz_gov_path = Path("../fitz-gov").resolve()
    manifest_path = write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=seed,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=fitz_gov_path if fitz_gov_path.exists() else None,
        start_time=run_start,
        extra={
            "script": "train_encoder.py",
            "base_model": base_model,
            "num_labels": num_labels,
            "threshold_selected": tau,
            "target_met_on_eval": bool(best_thr["target_met"]),
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
            "test_size": len(test_ds) if test_ds is not None else 0,
            "tier0_size": len(tier0_ds) if tier0_ds is not None else 0,
        },
    )
    print(f"Wrote manifest      -> {manifest_path}")

    gate_metrics = test_calibrated if test_calibrated is not None else best_thr
    gate_split = "held-out test" if test_calibrated is not None else "eval"
    passed, gates = check_release_gates(gate_metrics, tier0_calibrated)
    print("\n=== RELEASE GATES ===")
    print(f"  split: {gate_split}")
    for name, ok, detail in gates:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}  ({detail})")

    if not passed:
        print("\nDO NOT SHIP — at least one gate failed. Debug before pushing to HF.")
        return 2

    final_path = output_dir / "best_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAll gates passed. Best model saved -> {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
