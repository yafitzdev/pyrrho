"""OLMoE g4-real SFT smoke wrapper.

This keeps the active g4-real path out of the old Qwen-named defaults while
reusing the existing pyrrho generative SFT harness.

Example:
    python scripts/train_moe_olmoe_sft.py --max-steps 1 --max-train-samples 2 --max-eval-samples 2
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULTS: dict[str, str] = {
    "--seed-pack": "outputs/moe/g4_real_stock_runtime_carrier/olmoe_g4_real_full_random_hf",
    "--output-dir": "outputs/moe/olmoe_g4_real_sft_smoke",
    "--data-dir": "data/moe_v8",
    "--target-mode": "label-json",
    "--eval-label-source": "label-score",
    "--sample-mode": "balanced-label",
    "--dtype": "float16",
    "--device-map": "auto",
    "--attn-implementation": "sdpa",
    "--lora-target-modules": "q_proj,k_proj,v_proj,o_proj",
    "--run-label": "olmoe-generative-sft",
}

FLAGS: set[str] = {
    "--eval-skip-generation",
    "--save-adapter",
}


def has_option(args: list[str], option: str) -> bool:
    return option in args or any(arg.startswith(f"{option}=") for arg in args)


def main() -> int:
    args = sys.argv[1:]
    resolved: list[str] = []
    for option, value in DEFAULTS.items():
        if not has_option(args, option):
            resolved.extend([option, value])
    for flag in FLAGS:
        if not has_option(args, flag):
            resolved.append(flag)
    resolved.extend(args)

    script = Path(__file__).with_name("train_moe_qwen_sft.py")
    sys.argv = [str(script), *resolved]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
