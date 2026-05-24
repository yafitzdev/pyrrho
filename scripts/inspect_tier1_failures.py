"""inspect_tier1_failures.py — Load a checkpoint, predict on the tier1 eval split,
aggregate failure patterns + show representative misclassified cases.

Tier1 has ~584 cases vs tier0's 60, so we *aggregate* before showing details.
For each canonical grouping axis (difficulty, expert, taxonomy_pattern,
taxonomy_cell_id), shows top-N failing buckets ranked by error count. Also
flags "high-confidence wrong" cases (model was very sure but wrong) — the
most concerning failure mode.

Used to diagnose what the model is *actually* bad at on the publishable
benchmark, not the noisy 60-case tier0 diagnostic.

Run from project root:
    python scripts/inspect_tier1_failures.py
    python scripts/inspect_tier1_failures.py --checkpoint outputs/modernbert_base_v1/checkpoint-730
    python scripts/inspect_tier1_failures.py --samples-per-bucket 3 --high-conf-threshold 0.85
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from pyrrho.data import ID2LABEL, ID2LABEL_4CLASS, load_processed
from pyrrho.metrics import _softmax


GROUPING_AXES = ("difficulty", "expert", "taxonomy_pattern", "taxonomy_cell_id")


def latest_checkpoint(search_dirs: list[Path]) -> Path | None:
    for base in search_dirs:
        if not base.exists():
            continue
        candidates = sorted(
            base.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if candidates:
            return candidates[-1]
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-context-chars", type=int, default=300)
    p.add_argument(
        "--samples-per-bucket",
        type=int,
        default=2,
        help="Number of failing cases to show per top bucket (default: 2)",
    )
    p.add_argument(
        "--top-buckets",
        type=int,
        default=8,
        help="Show top N failing buckets per axis (default: 8)",
    )
    p.add_argument(
        "--high-conf-threshold",
        type=float,
        default=0.80,
        help="Cases predicted with max-prob >= this are 'high-confidence wrong' (default: 0.80)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt = args.checkpoint or latest_checkpoint([
        Path("outputs/modernbert_base_v1"),
        Path("outputs/multi_seed/seed_42"),
        Path("outputs/multi_seed/seed_1337"),
        Path("outputs/multi_seed/seed_7"),
    ])
    if ckpt is None:
        print("No checkpoint found.", file=sys.stderr)
        return 1

    print(f"Loading model from {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    num_labels = model.config.num_labels
    id2label_full = ID2LABEL_4CLASS if num_labels == 4 else ID2LABEL
    print(f"Model has {num_labels} output classes")

    ds = load_processed(args.data_dir)
    eval_ds = ds["eval"]
    print(f"Loaded {len(eval_ds)} tier1 eval cases\n")

    preds = []
    probs = []
    with torch.no_grad():
        for ex in eval_ds:
            enc = tokenizer(
                ex["text"], truncation=True, max_length=4096, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits.cpu().numpy()[0]
            preds.append(int(np.argmax(logits)))
            probs.append(_softmax(logits))

    preds = np.array(preds)
    probs = np.array(probs)
    labels = np.array(eval_ds["label_id"])

    def collapse(arr):
        return np.where(arr >= 2, 2, arr) if num_labels == 4 else arr

    preds_3 = collapse(preds)
    labels_3 = collapse(labels)

    n = len(labels)
    n_correct_3 = int((preds_3 == labels_3).sum())
    n_wrong_3 = n - n_correct_3
    print(f"3-class accuracy: {n_correct_3}/{n} = {n_correct_3 / n:.3%}")
    print(f"Total failures  : {n_wrong_3}")
    fail_idx = np.where(preds_3 != labels_3)[0]
    fail_set = set(int(i) for i in fail_idx)
    print()

    # Confusion matrix (3-class)
    print("Confusion matrix (gold -> pred, count):")
    confusion = Counter(((int(g), int(p)) for g, p in zip(labels_3, preds_3)))
    print(f"  {'':>14s} " + "  ".join(f"{ID2LABEL[c]:>13s}" for c in range(3)))
    for g in range(3):
        row = [f"{confusion.get((g, p), 0):>13d}" for p in range(3)]
        print(f"  {ID2LABEL[g]:>14s} " + "  ".join(row))
    print()

    # Per-class recall
    print("Per-class recall (3-class collapsed):")
    for cls in range(3):
        mask = labels_3 == cls
        if not mask.any():
            continue
        rec = (preds_3[mask] == cls).mean()
        n_cls = int(mask.sum())
        n_ok = int((preds_3[mask] == cls).sum())
        print(f"  {ID2LABEL[cls]:>14s}: {n_ok}/{n_cls} = {rec:.3%}")
    print()

    # High-confidence wrong cases — most concerning
    high_conf_wrong = []
    for i in fail_idx:
        max_p = float(probs[i].max())
        if max_p >= args.high_conf_threshold:
            high_conf_wrong.append((int(i), max_p))
    high_conf_wrong.sort(key=lambda t: -t[1])
    print(f"High-confidence wrong (max prob >= {args.high_conf_threshold}): {len(high_conf_wrong)} of {n_wrong_3} failures")
    print()

    # Aggregate failures by each grouping axis
    for axis in GROUPING_AXES:
        if axis not in eval_ds.column_names:
            continue
        # Count: per bucket, total cases + failures + error rate
        bucket_total: dict[str, int] = Counter()
        bucket_fail: dict[str, int] = Counter()
        bucket_fail_indices: dict[str, list[int]] = defaultdict(list)
        values = eval_ds[axis]
        for i, v in enumerate(values):
            bucket_total[v] += 1
            if i in fail_set:
                bucket_fail[v] += 1
                bucket_fail_indices[v].append(i)

        # Rank by absolute failure count (most-impactful buckets to fix)
        ranked = sorted(bucket_fail.items(), key=lambda kv: -kv[1])
        ranked = [(b, c) for b, c in ranked if c > 0][: args.top_buckets]

        print(f"=== Top failing buckets by `{axis}` (ranked by error count) ===")
        print(f"  {'bucket':<42s} {'errs':>5s} {'total':>6s} {'err_rate':>9s}")
        for bucket, errs in ranked:
            total = bucket_total[bucket]
            rate = errs / total if total else 0.0
            print(f"  {bucket:<42s} {errs:>5d} {total:>6d}  {rate:>8.1%}")
        print()

    # Show sample failures per top taxonomy pattern.
    if "taxonomy_pattern" in eval_ds.column_names:
        print("=" * 100)
        print(f"SAMPLE FAILURES — top taxonomy patterns ({args.samples_per_bucket} per bucket)")
        print("=" * 100)
        values = eval_ds["taxonomy_pattern"]
        bucket_fail_indices: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(values):
            if i in fail_set:
                bucket_fail_indices[v].append(i)
        ranked = sorted(
            bucket_fail_indices.items(), key=lambda kv: -len(kv[1])
        )[: args.top_buckets]

        for bucket, indices in ranked:
            print(f"\n--- taxonomy_pattern: {bucket} ({len(indices)} failures) ---")
            for idx in indices[: args.samples_per_bucket]:
                ex = eval_ds[idx]
                true_lbl = id2label_full[int(labels[idx])]
                pred_lbl = id2label_full[int(preds[idx])]
                probs_str = ", ".join(
                    f"{id2label_full[i]}={probs[idx, i]:.3f}" for i in range(num_labels)
                )
                print(
                    f"  [id={ex['id']}]  {true_lbl} -> {pred_lbl}  ({probs_str})"
                )
                print(
                    f"    expert: {ex.get('expert', '')}  cell: {ex.get('taxonomy_cell_id', '')}  "
                    f"difficulty: {ex.get('difficulty', '')}"
                )
                print(f"    query : {ex.get('query', '')}")
                for i_ctx, ctx in enumerate(ex.get("contexts", []) or [], 1):
                    ctx_str = str(ctx).strip()
                    if len(ctx_str) > args.max_context_chars:
                        ctx_str = ctx_str[: args.max_context_chars] + " […truncated]"
                    print(f"    ctx[{i_ctx}]: {ctx_str}")

    # Show high-confidence wrong (most concerning)
    if high_conf_wrong:
        print()
        print("=" * 100)
        print(f"HIGH-CONFIDENCE WRONG (model was sure, but missed) — top {min(10, len(high_conf_wrong))}")
        print("=" * 100)
        for idx, max_p in high_conf_wrong[:10]:
            ex = eval_ds[idx]
            true_lbl = id2label_full[int(labels[idx])]
            pred_lbl = id2label_full[int(preds[idx])]
            probs_str = ", ".join(
                f"{id2label_full[i]}={probs[idx, i]:.3f}" for i in range(num_labels)
            )
            print(f"\n[id={ex['id']}]  {true_lbl} -> {pred_lbl}  (max_p={max_p:.3f})  ({probs_str})")
            print(
                f"  pattern: {ex.get('taxonomy_pattern', '')}  expert: {ex.get('expert', '')}  "
                f"cell: {ex.get('taxonomy_cell_id', '')}"
            )
            print(f"  query : {ex.get('query', '')}")
            for i_ctx, ctx in enumerate(ex.get("contexts", []) or [], 1):
                ctx_str = str(ctx).strip()
                if len(ctx_str) > args.max_context_chars:
                    ctx_str = ctx_str[: args.max_context_chars] + " […truncated]"
                print(f"  ctx[{i_ctx}]: {ctx_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
