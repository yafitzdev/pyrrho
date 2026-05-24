"""eval_report.py — Load a trained checkpoint and emit a full evaluation report.

Goes beyond train_encoder.py's end-of-run summary by producing per-breakdown
metrics across the canonical V7 axes: difficulty, expert, taxonomy pattern,
and taxonomy cell.

Used for:
- Diagnosing where a model wins vs sklearn baseline
- Comparing two checkpoints (feed into compare_runs.py)
- Writing the model card (per-breakdown tables in the eval section)

Run from project root:
    python scripts/eval_report.py
    python scripts/eval_report.py --checkpoint outputs/modernbert_base_v1/checkpoint-730
    python scripts/eval_report.py --checkpoint <path> --output reports/v1_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, LABEL2ID, load_processed
from pyrrho.metrics import (
    breakdown_by,
    compute_classification_metrics,
    find_optimal_threshold,
    gated_predictions,
)


BREAKDOWN_FIELDS = (
    "difficulty",
    "expert",
    "taxonomy_pattern",
    "taxonomy_cell_id",
)


def latest_checkpoint(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* under {output_dir}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Specific checkpoint path. Default: latest under --search-dir.",
    )
    p.add_argument(
        "--search-dir",
        type=Path,
        default=Path("outputs/modernbert_base_v1"),
        help="Where to look for checkpoint-* dirs if --checkpoint not given.",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report. Default: <checkpoint_parent>/eval_report.json",
    )
    p.add_argument(
        "--min-bucket-size",
        type=int,
        default=10,
        help="Buckets with fewer than N cases are flagged as low-confidence in the report.",
    )
    return p.parse_args()


def infer_logits(model, tokenizer, dataset: Dataset, max_length: int, device: str) -> np.ndarray:
    """Run the model on every case in `dataset` and return raw logits (N, num_labels)."""
    model.eval()
    out = []
    with torch.no_grad():
        for case in dataset:
            enc = tokenizer(case["text"], truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits[0].float().cpu().numpy()
            out.append(logits)
    return np.stack(out, axis=0)


def format_breakdown(name: str, table: dict[str, dict[str, float]], min_n: int) -> str:
    if not table:
        return f"  [{name}] (no data)"
    lines = [f"  [{name}]"]
    lines.append(f"    {'bucket':<35s} {'n':>4s}  {'acc':>7s}  {'FT':>7s}")
    for bucket, stats in sorted(table.items(), key=lambda kv: kv[1]["n"], reverse=True):
        n = stats["n"]
        flag = " (small)" if n < min_n else ""
        lines.append(
            f"    {bucket:<35s} {n:>4d}  {stats['accuracy']:>7.4f}  "
            f"{stats['false_trustworthy_rate']:>7.4f}{flag}"
        )
    return "\n".join(lines)


def confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, int]]:
    """3x3 confusion: gold -> pred -> count."""
    out: dict[str, dict[str, int]] = {ID2LABEL[g]: {ID2LABEL[p]: 0 for p in range(3)} for g in range(3)}
    for g, p in zip(labels.tolist(), preds.tolist()):
        out[ID2LABEL[g]][ID2LABEL[p]] += 1
    return out


def report_split(
    name: str,
    dataset: Dataset,
    logits: np.ndarray,
    threshold: float,
    min_bucket: int,
) -> dict:
    """Produce uncalibrated + calibrated metrics with every breakdown for one split."""
    labels = np.array(dataset["label_id"])

    uncal_preds = logits.argmax(axis=-1)
    cal_preds = gated_predictions(logits, threshold, num_classes=logits.shape[-1])

    uncal_metrics = compute_classification_metrics(uncal_preds, labels)
    cal_metrics = compute_classification_metrics(cal_preds, labels)

    breakdowns_uncal = {}
    breakdowns_cal = {}
    for field in BREAKDOWN_FIELDS:
        if field not in dataset.column_names:
            continue
        groups = dataset[field]
        breakdowns_uncal[field] = breakdown_by(uncal_preds, labels, groups)
        breakdowns_cal[field] = breakdown_by(cal_preds, labels, groups)

    print(f"\n{'=' * 90}")
    print(f"[{name}]  n={len(dataset)}  threshold tau={threshold:.3f}")
    print(f"{'=' * 90}")
    for label_block, m in (("uncalibrated (argmax)", uncal_metrics), ("calibrated  (tau-gated)", cal_metrics)):
        print(f"\n  {label_block}")
        print(f"    accuracy             : {m['accuracy']:.4f}")
        print(f"    macro_f1             : {m['macro_f1']:.4f}")
        print(f"    false_trustworthy    : {m['false_trustworthy_rate']:.4f}")
        print(f"    {'class':<14s} {'precision':>10s}  {'recall':>8s}  {'f1':>8s}")
        for lbl in ("abstain", "disputed", "trustworthy"):
            print(
                f"    {lbl:<14s} {m[f'precision_{lbl}']:>10.4f}  "
                f"{m[f'recall_{lbl}']:>8.4f}  {m[f'f1_{lbl}']:>8.4f}"
            )

    print(f"\n  Confusion (gold -> pred), calibrated:")
    cm = confusion_matrix(cal_preds, labels)
    print(f"    {'':>13s} " + "  ".join(f"{l:>11s}" for l in ID2LABEL.values()))
    for g in ID2LABEL.values():
        row = [f"{cm[g][p]:>11d}" for p in ID2LABEL.values()]
        print(f"    {g:>13s} " + "  ".join(row))

    for field, table in breakdowns_cal.items():
        print()
        print(format_breakdown(f"calibrated, by {field}", table, min_bucket))

    return {
        "name": name,
        "n": len(dataset),
        "threshold": float(threshold),
        "uncalibrated": uncal_metrics,
        "calibrated": cal_metrics,
        "breakdowns_uncalibrated": breakdowns_uncal,
        "breakdowns_calibrated": breakdowns_cal,
        "confusion_calibrated": cm,
    }


def main() -> int:
    args = parse_args()

    ckpt = args.checkpoint or latest_checkpoint(args.search_dir)
    print(f"Checkpoint   : {ckpt}")

    ds = load_processed(args.data_dir)
    test_msg = f"  test={len(ds['test'])}" if "test" in ds else ""
    tier0_msg = f"  tier0={len(ds['tier0_sanity'])}" if "tier0_sanity" in ds else ""
    print(
        f"Splits       : train={len(ds['train'])}  eval={len(ds['eval'])}"
        f"{test_msg}{tier0_msg}"
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    num_classes = model.config.num_labels
    print(f"Device       : {device}")
    print(f"num_labels   : {num_classes}")

    print("\nRunning inference on eval split...")
    eval_logits = infer_logits(model, tokenizer, ds["eval"], args.max_seq_length, device)
    test_logits = None
    if "test" in ds:
        print("Running inference on test split...")
        test_logits = infer_logits(model, tokenizer, ds["test"], args.max_seq_length, device)
    tier0_logits = None
    if "tier0_sanity" in ds:
        print("Running inference on tier0_sanity split...")
        tier0_logits = infer_logits(model, tokenizer, ds["tier0_sanity"], args.max_seq_length, device)

    eval_labels = np.array(ds["eval"]["label_id"])

    # Find τ that hits FT target on eval/validation (same logic as train_encoder.py).
    best_thr = find_optimal_threshold(
        eval_logits,
        eval_labels,
        num_classes=num_classes,
    )
    tau = float(best_thr["threshold"])
    print(f"\nSelected tau : {tau:.3f}  (target_met={best_thr['target_met']})")

    eval_report = report_split("EVAL  (validation)", ds["eval"], eval_logits, tau, args.min_bucket_size)
    test_report = None
    if test_logits is not None:
        test_report = report_split("TEST  (held-out)", ds["test"], test_logits, tau, args.min_bucket_size)
    tier0_report = None
    if tier0_logits is not None:
        tier0_report = report_split("TIER0 (sanity, held-out)", ds["tier0_sanity"], tier0_logits, tau, args.min_bucket_size)

    payload = {
        "checkpoint": str(ckpt),
        "device": device,
        "num_labels": num_classes,
        "threshold": tau,
        "threshold_target_met": bool(best_thr["target_met"]),
        "eval": eval_report,
        "test": test_report,
        "tier0_sanity": tier0_report,
    }

    out_path = args.output or (ckpt.parent / "eval_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
