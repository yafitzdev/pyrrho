"""Count pyrrho-MoE total and active parameters from a YAML config.

Run from project root:
    python scripts/count_moe_params.py --config configs/moe/pyrrho_moe_g3_alpha.yaml
    python scripts/count_moe_params.py --config configs/moe/pyrrho_moe_g3_alpha.yaml --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yaml

from pyrrho.moe import PyrrhoMoEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g3_alpha.yaml"),
        help="MoE YAML config (default: configs/moe/pyrrho_moe_g3_alpha.yaml)",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    p.add_argument("--json", action="store_true", help="Print JSON instead of a text table")
    return p.parse_args()


def fmt(n: int) -> str:
    return f"{n:,} ({n / 1_000_000_000:.3f}B)"


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    cfg = PyrrhoMoEConfig.from_mapping(raw.get("architecture"))
    report = cfg.budget_report()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    counts = report["parameters"]
    derived = report["derived"]
    checks = report["budget_checks"]

    print(f"Config                         : {args.config}")
    print(f"Layers                         : {cfg.layers}")
    print(f"Hidden / heads / KV heads      : {cfg.hidden_size} / {cfg.attention_heads} / {cfg.kv_heads}")
    print(f"Head dim / KV dim              : {derived['head_dim']} / {derived['kv_dim']}")
    print(f"Dense FFN / MoE FFN layers     : {cfg.dense_ffn_layers} / {cfg.moe_ffn_layers}")
    print(f"Experts per MoE layer          : {cfg.experts_per_moe_layer}")
    shards_per_group = derived["physical_shards_per_group"]
    if shards_per_group is None:
        shard_text = "mixed"
    else:
        shard_text = str(shards_per_group)
    print(f"Semantic groups / shards/group : {derived['semantic_group_count']} / {shard_text}")
    if shards_per_group is None:
        shard_map = ", ".join(
            f"{key}:{value}"
            for key, value in derived["semantic_expert_shards"].items()
        )
        print(f"Semantic shard map             : {shard_map}")
    print()
    for key in (
        "embedding",
        "attention_blocks",
        "dense_ffns",
        "moe_expert_bank",
        "routers",
        "norms",
        "task_heads",
        "total",
        "active_selected_experts",
        "active_inclusive",
        "active_excluding_embedding",
    ):
        print(f"{key:32s}: {fmt(counts[key])}")
    print()
    for key, passed in checks.items():
        print(f"{key:40s}: {'PASS' if passed else 'FAIL'}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
