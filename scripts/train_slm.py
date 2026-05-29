"""train_slm.py - QLoRA fine-tune a generative SLM on the prepared fitz-gov splits.

Default target: pyrrho-small-g1 (Qwen3.5-0.8B). Driven by a YAML config so the
same script works for other Qwen / Gemma / Phi / LFM bases with no code change.

Pipeline:
  1. Load base model in 4-bit (bnb nf4 + bf16 compute), apply LoRA adapters.
  2. Transform the processed splits into HuggingFace `messages` format
     (system + user + assistant=label) and SFT with TRL's SFTTrainer using
     assistant_only_loss=True so loss only fires on label tokens.
  3. After training, run a decode-based eval pass on eval + tier0_sanity:
     greedy-generate up to N tokens, parse a label out of the decoded text,
     compute the standard classification metrics + release gates.
  4. Save LoRA adapter + final_metrics.json + manifest.json next to the
     checkpoint dir.

Run from project root:
    python scripts/train_slm.py --config configs/slm/qwen3.5_0.8b_qlora.yaml
    python scripts/train_slm.py --config <path> --seed 1337
    python scripts/train_slm.py --config <path> --smoke    # 50 train / 20 eval, 1 epoch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

# Force UTF-8 stdio + Python interop before importing transformers/trl
# (TRL 1.4 reads chat-template .jinja files with default encoding; cp1252 crashes).
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from pyrrho.data import LABEL2ID, SLM_SYSTEM_PROMPT, load_processed
from pyrrho.manifest import write_manifest
from pyrrho.metrics import (
    check_release_gates,
    compute_classification_metrics,
    format_metrics_table,
)


LABEL_NAMES = ("ABSTAIN", "DISPUTED", "TRUSTWORTHY")


class WeightedLossSFTTrainer(SFTTrainer):
    """SFTTrainer with per-example class-weighted CE + label smoothing.

    For SFT on a constrained 3-class output, transplants the encoder's anti-FT
    recipe (class_weights=[2.3, 2.3, 1.0] + label_smoothing=0.15) onto the
    token-level CE loss. Per-example weighting works by detecting which class
    each row's true assistant content represents (scanning the unmasked labels
    for the unique start token of ABSTAIN / DISPUTED / TRUSTWORTHY) and
    multiplying that row's averaged loss by the corresponding class weight.

    Reduces the SLM's false-trustworthy rate by directly penalizing the model
    when it gets a true ABSTAIN or DISPUTED case wrong (whereas plain SFT
    treats all wrong predictions equally and ends up biased toward
    TRUSTWORTHY because of class imbalance).
    """

    # First (unique) token of each label string under the Qwen3.5 tokenizer.
    # Probed once at init from the tokenizer to avoid relying on hard-coded ids.
    def __init__(
        self,
        *args,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cw = class_weights or [1.0, 1.0, 1.0]
        if len(cw) != 3:
            raise ValueError(f"class_weights must be a 3-element vector, got {cw!r}")
        self._class_weights = torch.tensor(cw, dtype=torch.float32)
        self._label_smoothing = float(label_smoothing)

        tok = self.processing_class
        self._label_start_ids: dict[int, int] = {}
        for cls_id, name in enumerate(LABEL_NAMES):
            ids = tok.encode(name, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Tokenizer returned no ids for {name!r}")
            self._label_start_ids[ids[0]] = cls_id

        if self.is_world_process_zero():
            print(
                f"[weighted-loss] class_weights={cw}  label_smoothing={self._label_smoothing}  "
                f"label_start_ids={self._label_start_ids}"
            )

    def _infer_class_ids(self, labels: torch.Tensor) -> torch.Tensor:
        """Return [B] tensor of class ids (0/1/2), defaulting to TRUSTWORTHY (weight 1.0) if not found."""
        B = labels.size(0)
        out = torch.full((B,), LABEL2ID["TRUSTWORTHY"], dtype=torch.long, device=labels.device)
        valid_mask = labels != -100
        for i in range(B):
            row = labels[i][valid_mask[i]]
            for tok_id, cls_id in self._label_start_ids.items():
                if (row == tok_id).any():
                    out[i] = cls_id
                    break
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Standard causal LM shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T_shift, V = shift_logits.shape
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
            label_smoothing=self._label_smoothing,
        ).view(B, T_shift)

        mask = (shift_labels != -100).to(per_token_loss.dtype)
        per_example_loss = (per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

        class_ids = self._infer_class_ids(shift_labels)
        weights = self._class_weights.to(per_example_loss.device, dtype=per_example_loss.dtype)[class_ids]
        loss = (per_example_loss * weights).sum() / weights.sum().clamp_min(1e-6)

        return (loss, outputs) if return_outputs else loss


# ---------- CLI ----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True, help="Path to SLM YAML config")
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
        "--seed",
        type=int,
        default=None,
        help="Override training.seed from the config (for multi-seed validation)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb regardless of the config setting",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 50 train cases, 20 eval cases, 1 epoch — verifies the pipeline",
    )
    p.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip the decode-based eval pass after training (e.g., to inspect a checkpoint manually)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Set up everything but skip trainer.train() and final eval",
    )
    return p.parse_args()


# ---------- Config + seeding ---------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ---------- Dataset transform --------------------------------------------------


def _build_user_content(query: str, contexts: Iterable[str]) -> str:
    numbered = "\n".join(
        f"[{i}] {str(c).strip()}" for i, c in enumerate(contexts or [], start=1)
    )
    return f"Question: {(query or '').strip()}\n\nSources:\n{numbered}"


def to_messages_dataset(ds: Dataset, include_label: bool) -> Dataset:
    """Transform a (query, contexts, label) dataset into TRL's `messages` format.

    Training rows include the assistant turn; eval rows do not (we generate it).
    Keeps `label_id` and a stable `id` so we can score generations downstream.
    """
    def _row(row):
        msgs = [
            {"role": "system", "content": SLM_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_content(row["query"], row["contexts"])},
        ]
        if include_label:
            msgs.append({"role": "assistant", "content": row["label"]})
        return {
            "messages": msgs,
            "label_id": int(row["label_id"]),
            "id": row.get("id"),
        }

    keep = {"messages", "label_id", "id"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.map(_row, remove_columns=drop)


# ---------- Model load ---------------------------------------------------------


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg.get("quantization", {})
    compute_dtype = getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=bool(q.get("load_in_4bit", True)),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(q.get("bnb_4bit_use_double_quant", True)),
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    l = cfg["lora"]
    task_str = l.get("task_type", "CAUSAL_LM")
    return LoraConfig(
        r=int(l.get("r", 16)),
        lora_alpha=int(l.get("alpha", 32)),
        lora_dropout=float(l.get("dropout", 0.05)),
        bias=l.get("bias", "none"),
        task_type=getattr(TaskType, task_str),
        target_modules=list(l["target_modules"]),
    )


# ---------- Decode-based eval --------------------------------------------------


def parse_label_from_text(text: str, fallback: str = "ABSTAIN") -> str:
    """Extract the first ABSTAIN/DISPUTED/TRUSTWORTHY token from a decoded string.

    Qwen3.5's chat template wraps the assistant content in `<think>...</think>`.
    We strip the think block first and search after it; if there's no think
    block, scan the whole string.
    """
    if not text:
        return fallback
    # Strip the empty think block the template injects.
    payload = text
    if "</think>" in payload:
        payload = payload.rsplit("</think>", 1)[1]
    payload = payload.strip()
    upper = payload.upper()
    for lab in LABEL_NAMES:
        if lab in upper:
            return lab
    # Final fallback — also scan the original text in case our strip ate the label.
    upper_full = text.upper()
    for lab in LABEL_NAMES:
        if lab in upper_full:
            return lab
    return fallback


def decode_eval(
    model,
    tokenizer,
    ds_messages: Dataset,
    *,
    batch_size: int,
    max_new_tokens: int,
    fallback_label: str,
    desc: str,
) -> tuple[list[int], list[str]]:
    """Greedy-decode each eval row, parse a label, return (pred_ids, raw_outputs)."""
    print(f"\n[decode-eval] {desc} on {len(ds_messages)} cases (batch={batch_size}, max_new_tokens={max_new_tokens})...")
    model.eval()

    # Render prompts in advance (no assistant turn, with generation prompt).
    prompts: list[str] = []
    for row in ds_messages:
        prompts.append(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
        )

    pred_ids: list[int] = []
    raw_outputs: list[str] = []
    t0 = time.time()

    # Process in mini-batches with left-padding so the new tokens are right-aligned.
    pad_side_orig = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
                enc = {k: v.to(model.device) for k, v in enc.items()}
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )
                # Strip the input portion from each row of `out` and decode just the new tokens.
                input_len = enc["input_ids"].shape[1]
                for i in range(out.shape[0]):
                    new_ids = out[i, input_len:].tolist()
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    raw_outputs.append(text)
                    label = parse_label_from_text(text, fallback=fallback_label)
                    pred_ids.append(LABEL2ID[label])
                done = start + len(batch_prompts)
                if done % max(batch_size * 4, 32) == 0 or done == len(prompts):
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (len(prompts) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(prompts)}  ({rate:.1f}/s, ETA {eta:.0f}s)")
    finally:
        tokenizer.padding_side = pad_side_orig

    elapsed = time.time() - t0
    print(f"[decode-eval] done in {elapsed:.1f}s ({len(prompts)/max(elapsed,1e-6):.2f} cases/s)")
    return pred_ids, raw_outputs


def parse_failures(
    pred_ids: list[int],
    label_ids: list[int],
    raw_outputs: list[str],
    fallback_label: str,
) -> dict:
    """Sanity stats on the decode pass: how many fell to fallback, etc."""
    fallback_id = LABEL2ID[fallback_label]
    n_fallback = 0
    for pred, raw in zip(pred_ids, raw_outputs):
        upper = raw.upper()
        if not any(lab in upper for lab in LABEL_NAMES):
            n_fallback += 1
    return {
        "n_fallback_to_default": int(n_fallback),
        "fallback_label": fallback_label,
        "fallback_label_id": int(fallback_id),
        "fraction_fallback": round(n_fallback / max(len(pred_ids), 1), 4),
    }


# ---------- Main ---------------------------------------------------------------


def main() -> int:
    args = parse_args()
    run_start = time.time()

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42))
    set_all_seeds(seed)

    # ---------- Env knobs ----------
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Reduce bitsandbytes warning spam.
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

    output_dir = Path(
        args.output_dir if args.output_dir else cfg["training"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = cfg["model"]["base_model"]
    max_length = int(cfg["data"].get("max_length", 4096))
    max_new_tokens = int(cfg["inference"].get("max_new_tokens", 16))
    fallback_label = cfg["inference"].get("fallback_label", "ABSTAIN")

    print(f"Config        : {args.config}")
    print(f"Output dir    : {output_dir}")
    print(f"Seed          : {seed}")
    print(f"Base model    : {base_model}")
    print(f"max_length    : {max_length}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"Smoke test    : {args.smoke}")

    # ---------- Data ----------
    print("\nLoading processed dataset...")
    ds = load_processed(args.data_dir)
    print(f"  train         : {len(ds['train'])}")
    print(f"  eval          : {len(ds['eval'])}")
    print(f"  tier0_sanity  : {len(ds['tier0_sanity'])}")

    train_ds = ds["train"]
    eval_ds = ds["eval"]
    tier0_ds = ds["tier0_sanity"]

    if args.smoke:
        train_ds = train_ds.shuffle(seed=seed).select(range(min(50, len(train_ds))))
        eval_ds = eval_ds.shuffle(seed=seed).select(range(min(20, len(eval_ds))))
        tier0_ds = tier0_ds.select(range(min(10, len(tier0_ds))))
        print(f"  -> smoke override: train={len(train_ds)} eval={len(eval_ds)} tier0={len(tier0_ds)}")

    train_messages = to_messages_dataset(train_ds, include_label=True)
    eval_messages_with_label = to_messages_dataset(eval_ds, include_label=True)
    eval_messages_prompt_only = to_messages_dataset(eval_ds, include_label=False)
    tier0_messages_prompt_only = to_messages_dataset(tier0_ds, include_label=False)

    # ---------- Tokenizer + base model ----------
    print(f"\nLoading tokenizer + 4-bit base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(cfg)
    model_load_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    attn_impl = cfg["model"].get("attn_implementation")
    if attn_impl:
        model_load_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)
    print(f"  model class: {type(model).__name__}")
    print(f"  dtype     : {next(model.parameters()).dtype}")

    # k-bit prep enables grad on the input embeddings + ensures the right autograd graph.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", True))
    )

    # ---------- LoRA ----------
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={lora_config.r} alpha={lora_config.lora_alpha} dropout={lora_config.lora_dropout}")
    print(f"  trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M ({100*n_trainable/n_total:.3f}%)")

    # ---------- SFTConfig ----------
    t = cfg["training"]
    num_epochs = 1 if args.smoke else int(t.get("num_train_epochs", 3))
    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 4)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        warmup_ratio=float(t.get("warmup_ratio", 0.05)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        bf16=bool(t.get("bf16", True)),
        optim=t.get("optim", "paged_adamw_8bit"),
        neftune_noise_alpha=t.get("neftune_noise_alpha"),
        eval_strategy=t.get("eval_strategy", "epoch"),
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=int(t.get("save_total_limit", 2)),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", True)),
        metric_for_best_model=t.get("metric_for_best_model", "loss"),
        greater_is_better=bool(t.get("greater_is_better", False)),
        logging_steps=int(t.get("logging_steps", 20)),
        report_to=t.get("report_to", []) if not args.no_wandb else "none",
        run_name=t.get("run_name", "pyrrho-small-g1"),
        seed=seed,
        max_length=max_length,
        packing=bool(cfg["data"].get("pack_examples", False)),
        assistant_only_loss=bool(t.get("assistant_only_loss", True)),
        chat_template_path=cfg["data"].get("chat_template_path"),
        # For training metric_for_best_model purposes, eval_loss is the only
        # cheap signal available without running generation per epoch. We
        # override metric_for_best_model to "eval_loss" / greater=False below
        # if the user kept ft_penalized_accuracy in the YAML (which we cannot
        # compute without decoding).
    )

    # Force eval_loss as the selection metric — we don't have decode-time
    # metrics during training. Other settings from YAML are respected.
    sft_cfg.metric_for_best_model = "eval_loss"
    sft_cfg.greater_is_better = False

    class_weights = t.get("class_weights")
    label_smoothing = float(t.get("label_smoothing", 0.0))
    use_weighted = bool(class_weights) or label_smoothing > 0.0
    if use_weighted:
        print(
            f"\nUsing WeightedLossSFTTrainer: class_weights={class_weights or 'uniform'}  "
            f"label_smoothing={label_smoothing}"
        )
        trainer = WeightedLossSFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_messages,
            eval_dataset=eval_messages_with_label,
            processing_class=tokenizer,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_messages,
            eval_dataset=eval_messages_with_label,
            processing_class=tokenizer,
        )

    if args.dry_run:
        print("\n--dry-run set, skipping trainer.train() and final eval.")
        return 0

    # ---------- Train ----------
    print("\nStarting training...")
    train_start = time.time()
    train_result = trainer.train()
    train_secs = time.time() - train_start
    print(f"\nTraining done in {train_secs/60:.1f} min")
    print(f"  global step : {train_result.global_step}")
    print(f"  train loss  : {train_result.training_loss:.4f}")

    # Save LoRA adapter + tokenizer to the output dir (in addition to checkpoints).
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  adapter saved to {final_dir}")

    # ---------- Final decode-based eval ----------
    eval_metrics: dict = {}
    tier0_metrics: dict = {}
    if not args.skip_final_eval:
        # Restore train-time eval batch size for decode (smaller — generation memory).
        decode_bs = max(1, int(cfg.get("inference", {}).get("eval_batch_size", t.get("per_device_eval_batch_size", 8))))

        eval_pred_ids, eval_raw = decode_eval(
            trainer.model,
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
            trainer.model,
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

        # Release gates (informational — the tier0 gate is dropped per HANDOFF
        # but we still report it for diagnostics).
        all_passed, gate_results = check_release_gates(eval_metrics, tier0_metrics)
        print("\nRelease gates:")
        for name, passed, detail in gate_results:
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {name:50s}  {detail}")
        print(f"\nAll gates passed: {all_passed}")

        # Dump the raw decoded outputs for the first 30 eval cases for
        # post-hoc inspection (e.g., multi_source_convergence behaviour).
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

    # ---------- Write final_metrics.json + manifest ----------
    final_metrics = {
        "eval": eval_metrics,
        "tier0_sanity": tier0_metrics,
        "training": {
            "global_step": int(train_result.global_step),
            "training_loss": float(train_result.training_loss),
            "duration_seconds": round(train_secs, 2),
        },
        "config_path": str(args.config),
        "seed": seed,
        "smoke": bool(args.smoke),
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, indent=2)
    print(f"\nWrote {output_dir / 'final_metrics.json'}")

    fitz_gov_repo = Path("../fitz-gov").resolve()
    write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=seed,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=fitz_gov_repo if fitz_gov_repo.exists() else None,
        start_time=run_start,
        extra={
            "script": "scripts/train_slm.py",
            "base_model": base_model,
            "smoke": bool(args.smoke),
        },
    )

    total_secs = time.time() - run_start
    print(f"\nTotal wall-clock: {total_secs/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
