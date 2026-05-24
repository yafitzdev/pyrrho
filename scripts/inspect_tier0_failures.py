"""inspect_tier0_failures.py — Load a trained encoder checkpoint, predict on tier0_sanity,
print every misclassified case with full context.

Used to diagnose why tier0 accuracy gates fail. Default checkpoint is the
auto-selected best from outputs/modernbert_base_v1/.

Run from project root:
    python scripts/inspect_tier0_failures.py
    python scripts/inspect_tier0_failures.py --checkpoint outputs/modernbert_base_v1/checkpoint-584
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from pyrrho.data import ID2LABEL, ID2LABEL_4CLASS, load_processed
from pyrrho.metrics import _softmax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/modernbert_base_v1/checkpoint-584"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-context-chars", type=int, default=400)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"Loading model from {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    num_labels = model.config.num_labels
    id2label_full = ID2LABEL_4CLASS if num_labels == 4 else ID2LABEL
    print(f"Model has {num_labels} output classes")

    ds = load_processed(args.data_dir)
    tier0 = ds["tier0_sanity"]
    print(f"Loaded {len(tier0)} tier0 cases\n")

    # Predict
    preds = []
    probs = []
    with torch.no_grad():
        for ex in tier0:
            enc = tokenizer(
                ex["text"], truncation=True, max_length=4096, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits.cpu().numpy()[0]
            p = _softmax(logits)
            preds.append(int(np.argmax(logits)))
            probs.append(p)

    preds = np.array(preds)
    probs = np.array(probs)
    labels = np.array(tier0["label_id"])

    def collapse(arr):
        return np.where(arr >= 2, 2, arr) if num_labels == 4 else arr

    preds_3 = collapse(preds)
    labels_3 = collapse(labels)

    print(f"Strict ({num_labels}-class) accuracy: "
          f"{int((preds == labels).sum())}/{len(labels)} "
          f"= {(preds == labels).mean():.1%}")
    print(f"3-class collapsed accuracy:        "
          f"{int((preds_3 == labels_3).sum())}/{len(labels)} "
          f"= {(preds_3 == labels_3).mean():.1%}\n")

    print(f"Per-class breakdown ({num_labels}-class strict):")
    for cls_id, cls_name in sorted(id2label_full.items()):
        mask = labels == cls_id
        n = int(mask.sum())
        if n == 0:
            continue
        n_ok = int((preds[mask] == cls_id).sum())
        confusion = Counter(int(p) for p in preds[mask])
        confusion_str = ", ".join(
            f"{id2label_full[k]}={v}" for k, v in sorted(confusion.items())
        )
        print(f"  {cls_name:22s} {n_ok}/{n} ({n_ok / n:.1%})  preds: {confusion_str}")
    print()

    print("Per-class breakdown (3-class collapsed, gate-equivalent):")
    for cls_id in (0, 1, 2):
        cls_name = ID2LABEL[cls_id]
        mask = labels_3 == cls_id
        n = int(mask.sum())
        if n == 0:
            continue
        n_ok = int((preds_3[mask] == cls_id).sum())
        confusion = Counter(int(p) for p in preds_3[mask])
        confusion_str = ", ".join(f"{ID2LABEL[k]}={v}" for k, v in sorted(confusion.items()))
        print(f"  {cls_name:14s} {n_ok}/{n} ({n_ok / n:.1%})  preds: {confusion_str}")
    print()

    # Print every 3-class-collapsed misclassification (matches the gate)
    print("=" * 100)
    print(f"FAILURES (3-class collapsed, true -> predicted)")
    print("=" * 100)
    fail_idx = np.where(preds_3 != labels_3)[0]
    for idx in fail_idx:
        ex = tier0[int(idx)]
        true_lbl_full = id2label_full[int(labels[idx])]
        pred_lbl_full = id2label_full[int(preds[idx])]
        prob_parts = ", ".join(
            f"{id2label_full[i]}={probs[idx, i]:.3f}" for i in range(num_labels)
        )
        print(f"\n[case_id={ex['id']}] {true_lbl_full} -> {pred_lbl_full}  P({prob_parts})")
        print(f"  category    : {ex.get('category', '')}")
        print(f"  pattern     : {ex.get('taxonomy_pattern', '')}")
        print(f"  expert      : {ex.get('expert', '')}")
        print(f"  cell        : {ex.get('taxonomy_cell_id', '')}")
        print(f"  query       : {ex.get('query', '')}")
        for i, ctx in enumerate(ex.get("contexts", []), start=1):
            ctx_str = str(ctx).strip()
            if len(ctx_str) > args.max_context_chars:
                ctx_str = ctx_str[: args.max_context_chars] + " […truncated]"
            print(f"  context [{i}]: {ctx_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
