"""Audit whether a MoE config is eligible for the g4-real runtime-shape gate.

This does not prove llama.cpp/LM Studio loadability. It catches local design
choices that already violate the gate before we spend time exporting weights.

Example:
    python scripts/audit_moe_runtime_shape.py \
      --config configs/moe/pyrrho_moe_g4_real_stock_runtime.yaml \
      --output outputs/moe/g4_real_runtime_shape_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pyrrho.moe import PyrrhoMoEConfig

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g4_real_stock_runtime.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def audit(raw: dict[str, Any]) -> dict[str, Any]:
    arch_raw = raw.get("architecture") or {}
    cfg = PyrrhoMoEConfig.from_mapping(arch_raw)
    budget = cfg.budget_report()
    runtime_gate = raw.get("runtime_gate") or {}

    runtime_target_text = " ".join(
        str(runtime_gate.get(key) or "")
        for key in ("target", "status", "selected_carrier")
    ).lower()
    checks = {
        "budget_passes": all(budget["budget_checks"].values()),
        "top_k_is_one": cfg.top_k == 1,
        "no_dense_only_ffn_layers": cfg.dense_ffn_layers == 0,
        "all_ffn_layers_are_moe": cfg.moe_ffn_layers == cfg.layers,
        "no_mlp_only_layers_key": "mlp_only_layers" not in arch_raw,
        "no_qwen3moe_runtime_target": "qwen3moe" not in runtime_target_text,
        "stock_runtime_target_declared": bool(runtime_gate.get("target")),
    }

    blockers: list[str] = []
    if not checks["budget_passes"]:
        blockers.append("parameter budget does not pass 4B/A0.4B gates")
    if not checks["top_k_is_one"]:
        blockers.append("top_k must be 1 for the CPU active-parameter target")
    if not checks["no_dense_only_ffn_layers"]:
        blockers.append("dense-only FFN layers reintroduce a mixed dense/MoE loader risk")
    if not checks["all_ffn_layers_are_moe"]:
        blockers.append("not every FFN layer is MoE")
    if not checks["no_mlp_only_layers_key"]:
        blockers.append("mlp_only_layers is forbidden for g4-real")
    if not checks["no_qwen3moe_runtime_target"]:
        blockers.append("runtime target still mentions qwen3moe")
    if not checks["stock_runtime_target_declared"]:
        blockers.append("stock runtime target is not declared")

    next_required = list(runtime_gate.get("required_next") or [])

    return {
        "schema_version": "pyrrho_moe_runtime_shape_audit_v1",
        "name": raw.get("name") or "pyrrho-MoE-g4-real",
        "config_path": str(raw.get("_config_path", "")),
        "checks": checks,
        "blockers": blockers,
        "passes_local_shape_gate": not blockers,
        "budget": budget,
        "runtime_gate": runtime_gate,
        "next_required": next_required,
        "note": (
            "passes_local_shape_gate means the config no longer violates the "
            "known patched-runtime failure mode. It is not a GGUF/LM Studio "
            "load proof."
        ),
    }


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    raw["_config_path"] = str(args.config)
    report = audit(raw)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["passes_local_shape_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
