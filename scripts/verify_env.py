"""verify_env.py — Sanity-check the pyrrho training environment.

Prints library versions and verifies the RTX 5090 (Blackwell sm_120) is visible to PyTorch.
With `--bnb`, also does a 4-bit bitsandbytes smoke-test (downloads a small model — ~500 MB).

Run from project root:
    python scripts/verify_env.py
    python scripts/verify_env.py --bnb       # also smoke-test 4-bit load
"""

from __future__ import annotations

import argparse
import importlib
import sys


CORE_LIBS = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
    "sklearn",
    "numpy",
    "pandas",
    "yaml",
]

OPTIONAL_LIBS = [
    "peft",
    "trl",
    "bitsandbytes",
    "optimum",
    "onnxruntime",
    "wandb",
]


def get_version(modname: str) -> str | None:
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        return None
    return getattr(mod, "__version__", "unknown")


def check_torch() -> int:
    import torch

    print(f"\nPyTorch info:")
    print(f"  version              : {torch.__version__}")
    print(f"  cuda available       : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — install cu128 wheels per SETUP.md")
        return 1

    print(f"  cuda runtime         : {torch.version.cuda}")
    print(f"  device count         : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  device {i}             : {torch.cuda.get_device_name(i)}  (sm_{cap[0]}{cap[1]})")
        if cap[0] >= 12:
            print(f"    -> Blackwell architecture detected (sm_{cap[0]}{cap[1]})")

    return 0


def smoke_test_bnb_4bit() -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        print(f"ERROR: bnb smoke-test requires transformers + bitsandbytes — {e}")
        return 1

    test_model = "Qwen/Qwen3.5-0.8B"
    print(f"\n4-bit smoke test on {test_model} (may download ~1.5 GB):")
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        m = AutoModelForCausalLM.from_pretrained(
            test_model, quantization_config=bnb, device_map="auto"
        )
        # Confirm it landed on GPU
        device = next(m.parameters()).device
        print(f"  loaded OK on {device}")
        return 0
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  -> Try WSL2 (SETUP.md path B) or Unsloth (path C) if this keeps failing")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bnb", action="store_true", help="Also run a 4-bit bitsandbytes smoke-test")
    args = parser.parse_args()

    print("Core libraries:")
    missing_core = []
    for name in CORE_LIBS:
        v = get_version(name)
        if v is None:
            missing_core.append(name)
            print(f"  {name:20s}: MISSING")
        else:
            print(f"  {name:20s}: {v}")

    print("\nOptional libraries:")
    missing_opt = []
    for name in OPTIONAL_LIBS:
        v = get_version(name)
        if v is None:
            missing_opt.append(name)
            print(f"  {name:20s}: MISSING")
        else:
            print(f"  {name:20s}: {v}")

    if missing_core:
        print(f"\nERROR: missing core libraries: {missing_core}")
        print("Install with: uv pip install -e .")
        return 1

    rc = check_torch()
    if rc != 0:
        return rc

    if args.bnb:
        rc = smoke_test_bnb_4bit()
        if rc != 0:
            return rc

    print("\nEnvironment OK.")
    if missing_opt:
        print(f"Note: optional libraries not installed: {missing_opt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
