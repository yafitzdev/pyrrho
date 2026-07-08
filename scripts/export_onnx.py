"""export_onnx.py — Export a fine-tuned encoder checkpoint to ONNX and INT8-quantized ONNX.

The INT8 variant is the artifact fitz-sage will run on CPU at ~30 ms.
The FP32 ONNX is a fallback for environments where INT8 quantization causes accuracy drift.

Run from project root:
    python scripts/export_onnx.py --checkpoint <path> --output models/pyrrho-v2-nano-g1
"""

from __future__ import annotations

import argparse
import json
import os
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
        default=Path("models/pyrrho-v2-nano-g1"),
        help="Where to write the ONNX artifacts (default: models/pyrrho-v2-nano-g1)",
    )
    p.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Only export FP32 ONNX, skip the INT8 quantization step.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output     : {args.output.resolve()}")

    # Lazy import — optimum + onnxruntime aren't free on import time. Newer
    # Transformers releases have moved a few symbols Optimum still imports, so
    # this path falls back to a direct PyTorch export when Optimum is not usable.
    print("\nImporting optimum + onnxruntime...")
    try:
        import transformers.modeling_utils as transformers_modeling_utils
        import transformers.utils as transformers_utils
        from transformers.utils import hub as transformers_hub

        if not hasattr(transformers_utils, "is_offline_mode"):
            transformers_utils.is_offline_mode = transformers_hub.is_offline_mode
        if not hasattr(transformers_modeling_utils, "get_parameter_dtype"):
            transformers_modeling_utils.get_parameter_dtype = (
                transformers_modeling_utils.get_state_dict_dtype
            )

        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except Exception as exc:
        print(f"      Optimum export unavailable: {type(exc).__name__}: {exc}")
        return _export_with_torch(args)

    from transformers import PreTrainedTokenizerFast

    # 1. Convert safetensors -> ONNX (FP32). Also copy the original safetensors
    # checkpoint into the release dir so transformers users can load via
    # AutoModelForSequenceClassification.from_pretrained without ONNX runtime.
    print("\n[1/3] Exporting FP32 ONNX...")
    t0 = time.time()
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(args.checkpoint),
        export=True,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.checkpoint / "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
    )
    ort_model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    safetensors_src = args.checkpoint / "model.safetensors"
    if safetensors_src.exists():
        shutil.copy2(safetensors_src, args.output / "model.safetensors")
        print("      Copied model.safetensors from checkpoint")
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
    config_path = args.output / "config.json"
    if config_path.exists():
        id2label = json.loads(config_path.read_text(encoding="utf-8")).get("id2label", {})
    else:
        id2label = {}
    labels = [str(id2label.get(str(i), f"LABEL_{i}")) for i in range(probs.shape[-1])]
    top_index = int(probs.argmax())
    top_values = ", ".join(
        f"{labels[i]}={probs[0, i]:.3f}"
        for i in probs[0].argsort()[-3:][::-1]
    )
    print("      Sample input  : speed of light, single source")
    print(f"      Predicted     : {labels[top_index]} (top3: {top_values})")
    print(f"      Smoke-test done in {time.time() - t0:.2f}s")

    print(f"\nDONE. Artifacts in {args.output.resolve()}")
    return 0


def _export_with_torch(args: argparse.Namespace) -> int:
    """Export a sequence-classification checkpoint without Optimum."""
    print("\nFalling back to direct torch.onnx export...")
    args.output.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.checkpoint / "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
    )
    print("\n[1/4] Copying release files...")
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
    ):
        src = args.checkpoint / filename
        if src.exists():
            shutil.copy2(src, args.output / filename)

    print("\n[2/4] Exporting FP32 ONNX with torch.onnx...")
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(str(args.checkpoint))
    model.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            return self.wrapped(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = (
        "[PYRRHO_POST]\n"
        "Question: What is the speed of light?\n\n"
        "Sources:\n[1] NIST defines the speed of light as exactly 299,792,458 m/s."
    )
    encoded = tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    fp32_path = args.output / "model.onnx"
    for stale in (fp32_path, args.output / "model.onnx.data", args.output / "model_quantized.onnx"):
        if stale.exists():
            stale.unlink()
    wrapper = _Wrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
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
        external_data=False,
        dynamo=False,
    )
    fp32_mb = fp32_path.stat().st_size / 1e6
    print(f"      model.onnx          : {fp32_mb:.1f} MB")
    print(f"      FP32 export done in {time.time() - t0:.1f}s")

    if args.skip_quantization:
        print("\n--skip-quantization set, exiting after FP32 export.")
        return 0

    print("\n[3/4] Running INT8 dynamic quantization...")
    t0 = time.time()
    int8_path = args.output / "model_quantized.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        use_external_data_format=False,
    )
    int8_mb = int8_path.stat().st_size / 1e6
    print(f"      model_quantized.onnx: {int8_mb:.1f} MB")
    print(f"      INT8 quantization done in {time.time() - t0:.1f}s")

    (args.output / "ort_config.json").write_text(
        json.dumps(
            {
                "optimization": {"level": "ORT_ENABLE_ALL"},
                "quantization": {"format": "dynamic", "weight_type": "QInt8"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[4/4] Smoke-testing the ONNX package...")
    t0 = time.time()
    import onnxruntime as ort

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    inputs = tokenizer(sample, return_tensors="np", truncation=True, max_length=512)
    declared = [node.name for node in session.get_inputs()]
    feed = {name: inputs[name].astype(np.int64) for name in declared if name in inputs}
    logits = session.run(None, feed)[0]
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label", {})
    labels = [str(id2label.get(str(i), f"LABEL_{i}")) for i in range(probs.shape[-1])]
    top_index = int(probs.argmax())
    top_values = ", ".join(
        f"{labels[i]}={probs[0, i]:.3f}"
        for i in probs[0].argsort()[-3:][::-1]
    )
    print(f"      Predicted     : {labels[top_index]} (top3: {top_values})")
    print(f"      Smoke-test done in {time.time() - t0:.2f}s")

    print(f"\nDONE. Artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
