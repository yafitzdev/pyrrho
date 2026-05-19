"""export_onnx.py — Export a fine-tuned encoder checkpoint to ONNX and INT8-quantized ONNX.

The INT8 variant is the artifact fitz-sage will run on CPU at ~30 ms.
The FP32 ONNX is a fallback for environments where INT8 quantization causes accuracy drift.

Run from project root:
    python scripts/export_onnx.py --checkpoint outputs/multi_seed/seed_42/checkpoint-730
    python scripts/export_onnx.py --checkpoint <path> --output models/pyrrho-nano-g1
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a transformers-compatible checkpoint dir (config.json + model.safetensors + tokenizer files)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/pyrrho-nano-g1"),
        help="Where to write the ONNX artifacts (default: models/pyrrho-nano-g1)",
    )
    p.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Only export FP32 ONNX, skip the INT8 quantization step.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output     : {args.output.resolve()}")

    # Lazy import — optimum + onnxruntime aren't free on import time
    print("\nImporting optimum + onnxruntime...")
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    # 1. Convert safetensors -> ONNX (FP32). Also copy the original safetensors
    # checkpoint into the release dir so transformers users can load via
    # AutoModelForSequenceClassification.from_pretrained without ONNX runtime.
    print("\n[1/3] Exporting FP32 ONNX...")
    t0 = time.time()
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(args.checkpoint),
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    ort_model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    safetensors_src = args.checkpoint / "model.safetensors"
    if safetensors_src.exists():
        shutil.copy2(safetensors_src, args.output / "model.safetensors")
        print(f"      Copied model.safetensors from checkpoint")
    print(f"      FP32 export done in {time.time() - t0:.1f}s")

    # The FP32 ONNX usually lands as `model.onnx`; record its size.
    fp32_path = args.output / "model.onnx"
    if fp32_path.exists():
        fp32_mb = fp32_path.stat().st_size / 1e6
        print(f"      model.onnx          : {fp32_mb:.1f} MB")

    if args.skip_quantization:
        print("\n--skip-quantization set, exiting after FP32 export.")
        return 0

    # 2. INT8 dynamic quantization — sufficient for ModernBERT classification
    print("\n[2/3] Running INT8 dynamic quantization...")
    t0 = time.time()
    quantizer = ORTQuantizer.from_pretrained(str(args.output))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(args.output), quantization_config=qconfig)
    print(f"      INT8 quantization done in {time.time() - t0:.1f}s")

    # The quantizer typically writes `model_quantized.onnx` next to `model.onnx`.
    int8_path = args.output / "model_quantized.onnx"
    if int8_path.exists():
        int8_mb = int8_path.stat().st_size / 1e6
        print(f"      model_quantized.onnx: {int8_mb:.1f} MB")

    # 3. Smoke-test the quantized model
    print("\n[3/3] Smoke-testing the INT8 ONNX...")
    t0 = time.time()
    ort_int8 = ORTModelForSequenceClassification.from_pretrained(
        str(args.output),
        file_name="model_quantized.onnx",
    )
    sample = "Question: What is the speed of light?\n\nSources:\n[1] NIST defines the speed of light as exactly 299,792,458 m/s."
    enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
    out = ort_int8(**enc)
    import numpy as np
    probs = np.exp(out.logits.numpy() - out.logits.numpy().max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    labels = ["ABSTAIN", "DISPUTED", "TRUSTWORTHY"]
    print(f"      Sample input  : speed of light, single source")
    print(f"      Predicted     : {labels[int(probs.argmax())]} "
          f"(probs: A={probs[0,0]:.3f}, D={probs[0,1]:.3f}, T={probs[0,2]:.3f})")
    print(f"      Smoke-test done in {time.time() - t0:.2f}s")

    print(f"\nDONE. Artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
