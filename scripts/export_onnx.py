"""export_onnx.py — Export a fine-tuned encoder checkpoint to ONNX and INT8-quantized ONNX.

The INT8 variant is the artifact fitz-sage will run on CPU at ~30 ms.
The FP32 ONNX is a fallback for environments where INT8 quantization causes accuracy drift.

Run from project root:
    python scripts/export_onnx.py --checkpoint outputs/multi_seed/seed_42/checkpoint-730
    python scripts/export_onnx.py --checkpoint <path> --output models/pyrrho-nano-g1
"""

from __future__ import annotations

import argparse
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

    # Lazy imports — torch/onnxruntime aren't free on import time. We use a
    # direct torch export instead of optimum because the local stack currently
    # pairs transformers 5.x with optimum 2.1, whose exporter imports symbols
    # removed from transformers internals.
    print("\nImporting torch + onnxruntime...")
    import importlib

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # 1. Convert safetensors -> ONNX (FP32). Also copy the original safetensors
    # checkpoint into the release dir so transformers users can load via
    # AutoModelForSequenceClassification.from_pretrained without ONNX runtime.
    print("\n[1/3] Exporting FP32 ONNX...")
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(str(args.checkpoint)).eval().to("cpu")
    model = model.to(dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output))

    class SequenceClassifierOnnxWrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask):
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = "Question: What is the speed of light?\n\nSources:\n[1] NIST defines the speed of light as exactly 299,792,458 m/s."
    enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
    wrapper = SequenceClassifierOnnxWrapper(model).eval()
    fp32_path = args.output / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (enc["input_ids"], enc["attention_mask"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
        )
    print(f"      FP32 export done in {time.time() - t0:.1f}s")

    # The FP32 ONNX usually lands as `model.onnx`; record its size.
    if fp32_path.exists():
        fp32_mb = fp32_path.stat().st_size / 1e6
        print(f"      model.onnx          : {fp32_mb:.1f} MB")

    if args.skip_quantization:
        print("\n--skip-quantization set, exiting after FP32 export.")
        return 0

    # 2. INT8 dynamic quantization — sufficient for ModernBERT classification
    print("\n[2/3] Running INT8 dynamic quantization...")
    t0 = time.time()
    int8_path = args.output / "model_quantized.onnx"
    # ONNX Runtime 1.26 shape inference trips on ModernBERT's exported classifier
    # head (`768` vs `3`) before quantization. The model itself runs cleanly, so
    # load it without the eager shape-inference pass and supply the default
    # tensor type required by the dynamic MatMul quantizer.
    ort_quantize = importlib.import_module("onnxruntime.quantization.quantize")
    ort_onnx_quantizer = importlib.import_module("onnxruntime.quantization.onnx_quantizer")

    def _load_no_shape_infer(path):
        return onnx.load(str(path), load_external_data=True)

    def _passthrough_model(model_proto):
        return model_proto

    ort_quantize.load_model_with_shape_infer = _load_no_shape_infer
    ort_quantize.save_and_reload_model_with_shape_infer = _passthrough_model
    ort_onnx_quantizer.save_and_reload_model_with_shape_infer = _passthrough_model
    ort_quantize.quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        use_external_data_format=True,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print(f"      INT8 quantization done in {time.time() - t0:.1f}s")

    # The quantizer typically writes `model_quantized.onnx` next to `model.onnx`.
    if int8_path.exists():
        int8_mb = int8_path.stat().st_size / 1e6
        print(f"      model_quantized.onnx: {int8_mb:.1f} MB")

    # 3. Smoke-test the quantized model
    print("\n[3/3] Smoke-testing the INT8 ONNX...")
    t0 = time.time()
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": enc["input_ids"].cpu().numpy(),
        "attention_mask": enc["attention_mask"].cpu().numpy(),
    }
    logits = sess.run(["logits"], ort_inputs)[0]
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    labels = ["ABSTAIN", "DISPUTED", "TRUSTWORTHY"]
    print("      Sample input  : speed of light, single source")
    print(f"      Predicted     : {labels[int(probs.argmax())]} "
          f"(probs: A={probs[0,0]:.3f}, D={probs[0,1]:.3f}, T={probs[0,2]:.3f})")
    print(f"      Smoke-test done in {time.time() - t0:.2f}s")

    print(f"\nDONE. Artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
