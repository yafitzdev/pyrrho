"""eval.py — 5-fold cross-validation on tier1, mean/std reporting.

Matches fitz-sage's published protocol so numbers are directly comparable to
its 78.7%/86.5/86.1/70.0/5.7% baseline. Each fold retrains a fresh model on
80% of tier1 and evaluates on the held-out 20%, stratified by (label, difficulty).

Run from project root:
    python scripts/eval.py --config configs/encoder/modernbert_base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from pyrrho.data import ID2LABEL, LABEL2ID, load_processed
from pyrrho.metrics import (
    breakdown_by,
    compute_classification_metrics,
    compute_metrics,
    format_metrics_table,
)
from pyrrho.training import set_all_seeds, tokenize_dataset


METRIC_KEYS_TO_AGGREGATE = (
    "accuracy",
    "macro_f1",
    "precision_abstain",
    "precision_disputed",
    "precision_trustworthy",
    "recall_abstain",
    "recall_disputed",
    "recall_trustworthy",
    "f1_abstain",
    "f1_disputed",
    "f1_trustworthy",
    "false_trustworthy_rate",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cv_results"))
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def combined_tier1(data_dir: Path) -> Dataset:
    """Combine the 80/20 splits back into the full tier1_core for k-fold."""
    ds = load_processed(data_dir)
    return concatenate_datasets([ds["train"], ds["eval"]])


def train_one_fold(
    cfg: dict,
    train_split: Dataset,
    eval_split: Dataset,
    fold_idx: int,
    output_root: Path,
    report_to: str,
) -> dict[str, float]:
    base_model = cfg["model"]["base_model"]
    max_length = int(cfg["data"]["max_seq_length"])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_tok = tokenize_dataset(train_split, tokenizer, max_length)
    eval_tok = tokenize_dataset(eval_split, tokenizer, max_length)

    fold_dir = output_root / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=float(cfg["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "linear"),
        bf16=bool(cfg["training"].get("bf16", True)),
        optim=cfg["training"].get("optim", "adamw_torch"),
        eval_strategy=cfg["training"].get("eval_strategy", "epoch"),
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=int(cfg["training"].get("logging_steps", 50)),
        report_to=report_to,
        run_name=f"{cfg['training'].get('run_name', 'cv')}-fold{fold_idx}",
        seed=int(cfg["training"].get("seed", 42)) + fold_idx,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds = trainer.predict(eval_tok)
    metrics = compute_classification_metrics(preds.predictions, preds.label_ids)

    diffs = eval_split["difficulty"]
    domains = eval_split["domain"]
    metrics["_by_difficulty"] = breakdown_by(preds.predictions, preds.label_ids, diffs)
    metrics["_by_domain"] = breakdown_by(preds.predictions, preds.label_ids, domains)
    return metrics


def aggregate(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS_TO_AGGREGATE:
        values = [f[key] for f in per_fold]
        agg[key] = {
            "mean": float(mean(values)),
            "std": float(stdev(values)) if len(values) > 1 else 0.0,
            "values": [float(v) for v in values],
        }
    return agg


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_all_seeds(seed)

    print(f"Config            : {args.config}")
    print(f"Folds             : {args.folds}")
    print(f"Output dir        : {args.output_dir.resolve()}")

    full = combined_tier1(args.data_dir)
    n = len(full)
    print(f"\nFull tier1 size   : {n}")

    labels = np.array(full["label_id"])
    difficulties = np.array(full["difficulty"])
    strata = np.array([f"{l}|{d}" for l, d in zip(labels, difficulties)])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)

    report_to = "none" if args.no_wandb else cfg["training"].get("report_to", "none")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_fold: list[dict[str, float]] = []
    for fold_idx, (train_idx, eval_idx) in enumerate(skf.split(np.zeros(n), strata)):
        print(f"\n=== FOLD {fold_idx + 1}/{args.folds} "
              f"(train={len(train_idx)}, eval={len(eval_idx)}) ===")
        train_split = full.select(train_idx.tolist())
        eval_split = full.select(eval_idx.tolist())
        m = train_one_fold(cfg, train_split, eval_split, fold_idx, args.output_dir, report_to)
        per_fold.append(m)
        print(format_metrics_table({k: v for k, v in m.items() if not k.startswith("_")}))

    print("\n\n=== AGGREGATED (5-fold mean ± std) ===")
    agg = aggregate(per_fold)
    for key, stats in agg.items():
        print(f"  {key:30s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    out_path = args.output_dir / "cv_summary.json"
    payload = {
        "config_path": str(args.config),
        "base_model": cfg["model"]["base_model"],
        "n_folds": args.folds,
        "n_total": n,
        "aggregate": agg,
        "per_fold": [{k: v for k, v in f.items() if not k.startswith("_")} for f in per_fold],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote CV summary → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
