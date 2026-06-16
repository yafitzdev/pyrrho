"""Run Qwen-seeded generative pyrrho-MoE adapter inference.

The emitted `selected_output` JSON is the authoritative governance output.
`raw_generation` is retained as audit/debug text because free generation can be
less safe than calibrated label-score selection.

Example:
    python scripts/infer_moe_qwen_sft.py \
      --input data/moe_v8/test.jsonl \
      --max-samples 8 \
      --output outputs/moe/qwen_generative_mvp_inference_smoke.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.moe.data import MoEVocab


LABELS = ("ABSTAIN", "DISPUTED", "TRUSTWORTHY")
DEFAULT_ADAPTER_PATH = Path(
    "outputs/moe/qwen_generative_sft_label_json_mild_weighted_512ctx_4096x512_tau047/final_adapter"
)


def load_training_module() -> Any:
    path = Path(__file__).with_name("train_moe_qwen_sft.py")
    spec = importlib.util.spec_from_file_location("pyrrho_qwen_sft_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import training helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows with query/contexts")
    parser.add_argument("--output", type=Path, default=None, help="Default: stdout JSONL")
    parser.add_argument("--seed-pack", type=Path, default=Path("outputs/moe/upcycling/qwen_alpha_seed_pack"))
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--metadata", type=Path, default=Path("data/moe_v8/metadata.json"))
    parser.add_argument("--threshold", type=float, default=0.50, help="TRUSTWORTHY label-score threshold")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=104)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb-4bit", "bnb-8bit"),
        default="none",
        help="Optional bitsandbytes quantized load mode.",
    )
    parser.add_argument("--bnb-4bit-quant-type", choices=("nf4", "fp4"), default="nf4")
    parser.add_argument("--bnb-4bit-double-quant", dest="bnb_4bit_double_quant", action="store_true", default=True)
    parser.add_argument("--no-bnb-4bit-double-quant", dest="bnb_4bit_double_quant", action="store_false")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--skip-generation", action="store_true", help="Only score labels; do not emit raw generation")
    parser.add_argument("--target-mode", choices=("json", "label-only", "label-json"), default="label-json")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rows.append(json.loads(raw))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gold_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    if "label" not in row:
        return None
    return {
        "classification": row.get("label"),
        "route": row.get("route"),
        "taxonomy_pattern": row.get("taxonomy_pattern"),
    }


def make_batch(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    helpers: Any,
    target_mode: str,
    max_length: int,
    pad_token_id: int,
) -> dict[str, Any]:
    encoded_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt = helpers.build_prompt(row, target_mode=target_mode)
        input_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"][:max_length]
        encoded_rows.append(
            {
                "id": row.get("id"),
                "prompt": prompt,
                "input_ids": input_ids,
            }
        )
    max_len = max(len(row["input_ids"]) for row in encoded_rows)
    input_ids_batch = []
    attention_mask_batch = []
    for row in encoded_rows:
        pad = max_len - len(row["input_ids"])
        input_ids_batch.append([pad_token_id] * pad + row["input_ids"])
        attention_mask_batch.append([0] * pad + [1] * len(row["input_ids"]))
    return {
        "ids": [row["id"] for row in encoded_rows],
        "prompts": [row["prompt"] for row in encoded_rows],
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
    }


def load_model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed_pack=args.seed_pack,
        dtype=args.dtype,
        device_map=args.device_map,
        quantization=args.quantization,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_double_quant=args.bnb_4bit_double_quant,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        eval_only=True,
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_target_modules="",
    )


def main() -> int:
    args = parse_args()
    helpers = load_training_module()
    vocab = MoEVocab.from_metadata(args.metadata)
    rows = read_jsonl(args.input, args.max_samples)
    model, tokenizer = helpers.load_model_and_tokenizer(load_model_args(args))
    helpers.disable_router_aux_outputs(model)
    model.eval()
    device = helpers.model_input_device(model)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    label_token_ids = helpers.label_candidate_token_ids(tokenizer)

    predictions: list[dict[str, Any]] = []
    counts = {label: 0 for label in LABELS}
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="qwen-sft-infer", leave=False):
            raw_batch = rows[start : start + args.batch_size]
            batch = make_batch(
                raw_batch,
                tokenizer=tokenizer,
                helpers=helpers,
                target_mode=args.target_mode,
                max_length=args.max_length,
                pad_token_id=pad_token_id,
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_scores, label_probs = helpers.score_label_candidates(
                model,
                input_ids,
                attention_mask,
                label_token_ids=label_token_ids,
                pad_token_id=pad_token_id,
                length_normalization="mean",
            )
            label_preds = helpers.apply_trustworthy_score_threshold(
                label_probs,
                threshold=args.threshold,
            )
            raw_texts = [""] * len(raw_batch)
            if not args.skip_generation:
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                prompt_len = input_ids.shape[1]
                raw_texts = [
                    tokenizer.decode(output_ids[idx, prompt_len:], skip_special_tokens=True)
                    for idx in range(output_ids.shape[0])
                ]
            score_rows = label_scores.detach().cpu().tolist()
            prob_rows = label_probs.detach().cpu().tolist()
            pred_ids = label_preds.detach().cpu().tolist()
            for idx, row in enumerate(raw_batch):
                selected_label_id = int(pred_ids[idx])
                selected_label = LABELS[selected_label_id]
                counts[selected_label] += 1
                text = raw_texts[idx]
                parsed = helpers.parse_generation(
                    text,
                    route2id=vocab.route2id,
                    taxonomy2id=vocab.taxonomy_pattern2id,
                    fallback_label="ABSTAIN",
                )
                selected_output = helpers.selected_governance_output(
                    text,
                    parsed,
                    selected_label_id=selected_label_id,
                    label_source="label-score",
                )
                out = {
                    "id": row.get("id"),
                    "classification": selected_label,
                    "selected_output": selected_output,
                    "label_score": {
                        "classification": selected_label,
                        "classification_id": selected_label_id,
                        "scores": {
                            label: float(score_rows[idx][label_idx])
                            for label_idx, label in enumerate(LABELS)
                        },
                        "probabilities": {
                            label: float(prob_rows[idx][label_idx])
                            for label_idx, label in enumerate(LABELS)
                        },
                        "length_normalization": "mean",
                        "trustworthy_threshold": float(args.threshold),
                    },
                    "parsed_generation": None if args.skip_generation else parsed,
                    "raw_generation": None if args.skip_generation else text,
                    "gold": gold_payload(row),
                }
                predictions.append(out)

    if args.output is not None:
        write_jsonl(args.output, predictions)
        print(f"Wrote predictions: {args.output}")
    else:
        for row in predictions:
            print(json.dumps(row, ensure_ascii=False))
    print(
        "Summary          : "
        f"rows={len(predictions)} threshold={args.threshold:g} counts={counts}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
