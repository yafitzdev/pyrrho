# Setup — RTX 5090 / Blackwell (Windows + WSL2 fallback)

State of the world as of May 2026. Update this file as the stack moves.

## Hardware target

- NVIDIA RTX 5090 (Blackwell architecture, sm_120, 32 GB VRAM)
- Driver 560+ required for sm_120 features
- CUDA Toolkit 12.8 minimum (12.9 preferred for newer features)

## Path A — Native Windows (preferred for IDE comfort)

```powershell
# 1. Verify GPU is visible
nvidia-smi

# 2. Create Python 3.12 venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.8 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Verify Blackwell support
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert 'RTX 5090' in torch.cuda.get_device_name(0)"

# 5. Install the rest
pip install -e ".[dev]"
```

### Known Windows pitfalls

| Library | Issue | Workaround |
|---|---|---|
| bitsandbytes | Windows wheels may lag Blackwell support | Try `pip install bitsandbytes>=0.45` first; if it fails to load, build from source or switch to WSL2 |
| flash-attn | Pre-built Windows wheels limited for sm_120 | Fall back to `attn_implementation="sdpa"` (default in transformers) — slightly slower but works |
| triton | Windows support partial | Unsloth's bundled triton is often the easiest path |
| Unsloth | Studio works on Windows, but some installs flaky | Use WSL2 if Unsloth Studio fails after one install attempt |

## Path B — WSL2 fallback (use if Path A blocks on bitsandbytes/flash-attn)

```bash
# In WSL2 Ubuntu 24.04
wsl --install -d Ubuntu-24.04   # from Windows PowerShell, if not already installed

# Inside WSL2
sudo apt update && sudo apt install -y python3.11 python3.11-venv build-essential
python3.11 -m venv .venv
source .venv/bin/activate

# CUDA in WSL2 uses host driver; just install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install rest
pip install -e ".[dev]"
```

## Path C — Unsloth-only (fastest training, opinionated stack)

```bash
pip install unsloth
```

Unsloth bundles its own bitsandbytes + flash-attn replacement. Tradeoff: less flexibility, but installs work first-try on Blackwell more often than vanilla TRL/PEFT stacks.

## Verifying the environment

```python
# scripts/verify_env.py — TODO: write this
import torch
import transformers
import peft
import trl
import bitsandbytes
import optimum
import onnxruntime

print(f"torch:        {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"transformers: {transformers.__version__}")
print(f"peft:         {peft.__version__}")
print(f"trl:          {trl.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")
print(f"optimum:      {optimum.__version__}")
print(f"onnxruntime:  {onnxruntime.__version__}")

# Smoke-test 4-bit load (this is where Windows often fails)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Instruct", quantization_config=bnb, device_map="auto")
print("4-bit load OK")
```

If the 4-bit smoke test fails on Windows: switch to WSL2 (Path B) or Unsloth (Path C). Don't fight it.

## Sources

- [PyTorch RTX 5090 support](https://github.com/pytorch/pytorch/issues/159207)
- [Unsloth Blackwell docs](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth)
- [bitsandbytes CUDA 12.9 issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1642)
- [SaladCloud RTX 5090 PyTorch guide](https://docs.salad.com/container-engine/tutorials/machine-learning/pytorch-rtx5090)
