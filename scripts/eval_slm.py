"""eval_slm.py - Run the decode-based eval pass on a saved SLM LoRA adapter.

Mirrors the eval portion of scripts/train_slm.py but without the training side,
so we can recover metrics for a checkpoint whose training completed but whose
in-script decode-eval hung or was interrupted. Writes final_metrics.json to
the adapter dir's parent (or wherever --output-dir says).

Run from project root:
    python scripts/eval_slm.py \\
        --config configs/slm/qwen3.5_0.8b_qlora_v1.1.yaml \\
        --adapter outputs/multi_seed_slm_g1_1/seed_1337/final \\
        --seed 1337
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 + unbuffered output. Block-buffered stdout under `>` redirection
# was what hid the decode-eval progress in the original train_slm.py run; this
# script writes ints incrementally so we get visible progress.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse helpers from train_slm.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_slm import (  # noqa: E402
    LABEL_NAMES,
    build_bnb_config,
    decode_eval,
    parse_failures,
    to_messages_dataset,
)

from pyrrho.data import LABEL2ID, load_processed  # noqa: E402
from pyrrho.metrics import (  # noqa: E402
    check_release_gates,
    compute_classification_metrics,
    format_metrics_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True, help="SLM YAML config (for inference settings + base_model id)")
    p.add_argument("--adapter", type=Path, required=True, help="Saved adapter dir (with adapter_model.safetensors)")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=None, help="Where to write final_metrics.json (default: adapter dir's parent)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=None, help="Decode batch size (default: from config inference.eval_batch_size or training.per_device_eval_batch_size)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1
    if not args.adapter.exists():
        print(f"ERROR: adapter dir not found: {args.adapter}", file=sys.stderr)
        return 1

    with args.config.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    output_dir = Path(args.output_dir) if args.output_dir else args.adapter.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = cfg["model"]["base_model"]
    max_new_tokens = int(cfg["inference"].get("max_new_tokens", 16))
    fallback_label = cfg["inference"].get("fallback_label", "ABSTAIN")
    decode_bs = int(args.batch_size or cfg.get("inference", {}).get("eval_batch_size", cfg["training"].get("per_device_eval_batch_size", 8)))

    run_start = time.time()
    print(f"Config        : {args.config}")
    print(f"Adapter       : {args.adapter}")
    print(f"Output dir    : {output_dir}")
    print(f"Base model    : {base_model_id}")
    print(f"Seed          : {args.seed}")
    print(f"Decode batch  : {decode_bs}")

    # Load processed data
    print("\nLoading processed dataset...")
    ds = load_processed(args.data_dir)
    print(f"  eval          : {len(ds['eval'])}")
    print(f"  tier0_sanity  : {len(ds['tier0_sanity'])}")
    eval_messages_prompt_only = to_messages_dataset(ds["eval"], include_label=False)
    tier0_messages_prompt_only = to_messages_dataset(ds["tier0_sanity"], include_label=False)

    # Load base + adapter
    print(f"\nLoading 4-bit base + LoRA adapter...")
    bnb = build_bnb_config(cfg)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
        attn_implementation=cfg["model"].get("attn_implementation"),
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    print(f"  model class: {type(model).__name__}")
    print(f"  device     : {next(model.parameters()).device}")

    # Decode-based eval
    eval_pred_ids, eval_raw = decode_eval(
        model,
        tokenizer,
        eval_messages_prompt_only,
        batch_size=decode_bs,
        max_new_tokens=max_new_tokens,
        fallback_label=fallback_label,
        desc="eval split",
    )
    eval_label_ids = [int(r["label_id"]) for r in eval_messages_prompt_only]
    eval_metrics = compute_classification_metrics(
        np.array(eval_pred_ids), np.array(eval_label_ids)
    )
    eval_metrics["decode_health"] = parse_failures(
        eval_pred_ids, eval_label_ids, eval_raw, fallback_label
    )
    print("\nEval metrics (decode-based):")
    print(format_metrics_table(eval_metrics))

    tier0_pred_ids, tier0_raw = decode_eval(
        model,
        tokenizer,
        tier0_messages_prompt_only,
        batch_size=decode_bs,
        max_new_tokens=max_new_tokens,
        fallback_label=fallback_label,
        desc="tier0_sanity",
    )
    tier0_label_ids = [int(r["label_id"]) for r in tier0_messages_prompt_only]
    tier0_metrics = compute_classification_metrics(
        np.array(tier0_pred_ids), np.array(tier0_label_ids)
    )
    tier0_metrics["decode_health"] = parse_failures(
        tier0_pred_ids, tier0_label_ids, tier0_raw, fallback_label
    )
    print("\nTier0 sanity metrics:")
    print(format_metrics_table(tier0_metrics))

    all_passed, gate_results = check_release_gates(eval_metrics, tier0_metrics)
    print("\nRelease gates:")
    for name, passed, detail in gate_results:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name:50s}  {detail}")
    print(f"\nAll gates passed: {all_passed}")

    # Decode sample for inspection
    sample = []
    for i in range(min(30, len(eval_raw))):
        sample.append({
            "id": eval_messages_prompt_only[i].get("id"),
            "label_true": LABEL_NAMES[eval_label_ids[i]],
            "label_pred": LABEL_NAMES[eval_pred_ids[i]],
            "raw_output": eval_raw[i],
        })
    with (output_dir / "eval_decode_sample.json").open("w", encoding="utf-8") as fh:
        json.dump(sample, fh, indent=2, ensure_ascii=False)

    final_metrics = {
        "eval": eval_metrics,
        "tier0_sanity": tier0_metrics,
        "training": {
            "note": "eval-only run on a pre-saved adapter; see seed dir for original training timing",
        },
        "config_path": str(args.config),
        "seed": int(args.seed),
        "smoke": False,
        "eval_only": True,
        "adapter_path": str(args.adapter.resolve()),
    }
    out_path = output_dir / "final_metrics.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2)
    print(f"\nWrote {out_path}")
    total_secs = time.time() - run_start
    print(f"Total wall-clock: {total_secs/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
