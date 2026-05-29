"""Analyze dense-seed compatibility with pyrrho-MoE parameter budgets.

This is a lightweight decision tool for the upcycling path. It compares the
canonical 64k-tokenizer baseline against seed-aligned variants and reports
whether each stays inside the 4B/A0.4B budget windows.

Run from project root:
    python scripts/analyze_moe_seed_budget.py
    python scripts/analyze_moe_seed_budget.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.moe import PyrrhoMoEConfig


@dataclass(frozen=True)
class BudgetVariant:
    name: str
    rationale: str
    architecture: dict


VARIANTS: tuple[BudgetVariant, ...] = (
    BudgetVariant(
        name="canonical_64k_baseline",
        rationale="Original architecture-doc baseline; tokenizer-neutral, not directly seed-aligned.",
        architecture={},
    ),
    BudgetVariant(
        name="qwen_vocab_old_shape",
        rationale="Naively keep old 24-layer/16-expert/3840-FFN shape but accept Qwen's vocab and real 128-dim attention heads.",
        architecture={
            "vocab_size": 151_936,
            "kv_heads": 8,
            "head_dim": 128,
            "attention_qk_norms": True,
        },
    ),
    BudgetVariant(
        name="qwen_exact_trunk_ffn3072",
        rationale="Use Qwen layer count, KV heads, head dim, vocab, and FFN dim; preserves more seed shape but exceeds active budget.",
        architecture={
            "layers": 28,
            "dense_ffn_layers": 4,
            "moe_ffn_layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "head_dim": 128,
            "kv_heads": 8,
            "attention_qk_norms": True,
            "ffn_dim": 3072,
            "experts_per_moe_layer": 16,
            "vocab_size": 151_936,
        },
    ),
    BudgetVariant(
        name="qwen_alpha_invalid_24e_ffn2112",
        rationale="Former alpha before real-weight inspection; fails active budget once Qwen head_dim=128 is counted.",
        architecture={
            "layers": 28,
            "dense_ffn_layers": 4,
            "moe_ffn_layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "head_dim": 128,
            "kv_heads": 8,
            "attention_qk_norms": True,
            "ffn_dim": 2112,
            "experts_per_moe_layer": 24,
            "vocab_size": 151_936,
        },
    ),
    BudgetVariant(
        name="qwen_budget_alpha_48e_ffn1056",
        rationale="Repaired Qwen-tokenizer alpha: preserves Qwen hidden/layer/KV/head-dim/vocab shape and restores budget by doubling experts and halving the prior 2112 FFN width.",
        architecture={
            "layers": 28,
            "dense_ffn_layers": 4,
            "moe_ffn_layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "head_dim": 128,
            "kv_heads": 8,
            "attention_qk_norms": True,
            "ffn_dim": 1056,
            "experts_per_moe_layer": 48,
            "vocab_size": 151_936,
        },
    ),
    BudgetVariant(
        name="qwen_budget_alpha_48e_ffn1024",
        rationale="Lower-margin alternate with power-of-two expert FFN width; slightly less total capacity than the selected 1056 variant.",
        architecture={
            "layers": 28,
            "dense_ffn_layers": 4,
            "moe_ffn_layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "head_dim": 128,
            "kv_heads": 8,
            "attention_qk_norms": True,
            "ffn_dim": 1024,
            "experts_per_moe_layer": 48,
            "vocab_size": 151_936,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="Print full JSON")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return p.parse_args()


def analyze() -> dict:
    rows = []
    for variant in VARIANTS:
        cfg = PyrrhoMoEConfig.from_mapping(variant.architecture)
        report = cfg.budget_report()
        rows.append(
            {
                "name": variant.name,
                "rationale": variant.rationale,
                "architecture": asdict(cfg),
                "derived": report["derived"],
                "parameters": report["parameters"],
                "budget_checks": report["budget_checks"],
                "passes_all_budget_checks": all(report["budget_checks"].values()),
            }
        )
    return {
        "seed": {
            "model_id": "Qwen/Qwen3-0.6B-Base",
            "license": "apache-2.0",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151_936,
        },
        "recommendation": "qwen_budget_alpha_48e_ffn1056",
        "variants": rows,
    }


def fmt_b(n: int) -> str:
    return f"{n / 1_000_000_000:.3f}B"


def main() -> int:
    args = parse_args()
    report = analyze()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("Seed: Qwen/Qwen3-0.6B-Base (Apache-2.0)")
    print(f"Recommendation: {report['recommendation']}")
    print()
    print(
        f"{'variant':34s} {'total':>8s} {'active':>8s} {'active-ex':>9s} "
        f"{'experts':>7s} {'ffn':>5s} {'pass':>5s}"
    )
    for row in report["variants"]:
        p = row["parameters"]
        a = row["architecture"]
        print(
            f"{row['name']:34s} {fmt_b(p['total']):>8s} "
            f"{fmt_b(p['active_inclusive']):>8s} "
            f"{fmt_b(p['active_excluding_embedding']):>9s} "
            f"{a['experts_per_moe_layer']:>7d} {a['ffn_dim']:>5d} "
            f"{'yes' if row['passes_all_budget_checks'] else 'no':>5s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
