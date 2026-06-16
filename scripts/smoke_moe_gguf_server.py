"""Smoke-test a pyrrho-MoE GGUF with llama.cpp's local HTTP server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from pyrrho.metrics import compute_classification_metrics
from pyrrho.moe.data import MoEVocab


LABELS = ("ABSTAIN", "DISPUTED", "TRUSTWORTHY")
SCORE_DECISION_MODES = ("first-token-label-score", "sequence-label-score")
DEFAULT_SERVER = Path("C:/Users/yanfi/.unsloth/llama.cpp/build/bin/Release/llama-server.exe")


def load_training_module() -> Any:
    path = Path(__file__).with_name("train_moe_qwen_sft.py")
    spec = importlib.util.spec_from_file_location("pyrrho_qwen_sft_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import training helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/moe/gguf/pyrrho-MoE-g3-mvp-merged-Q4_K_M.gguf"))
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--input", type=Path, default=Path("outputs/moe/package_hardening/qwen_mvp_32_random_test_input.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("data/moe_v8/metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/moe/gguf/smoke_q4_eval"))
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-ram-mib", type=int, default=0, help="llama-server prompt cache size in MiB")
    parser.add_argument("--n-predict", type=int, default=104)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--decision-mode",
        choices=("raw-generation", *SCORE_DECISION_MODES),
        default="raw-generation",
        help="How to choose the reported classification.",
    )
    parser.add_argument("--label-threshold", type=float, default=0.58)
    parser.add_argument(
        "--label-score-length-normalization",
        choices=("sum", "mean"),
        default="mean",
        help="How to normalize multi-token label sequence scores.",
    )
    parser.add_argument("--n-probs", type=int, default=5000)
    parser.add_argument("--health-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--no-request-cache-prompt", action="store_true")
    parser.add_argument("--no-cpu-moe", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(url: str, proc: subprocess.Popen[Any], *, timeout: float, peak: dict[str, int | None]) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update_peak_rss(proc, peak)
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def update_peak_rss(proc: subprocess.Popen[Any], peak: dict[str, int | None]) -> None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return
    try:
        rss = int(psutil.Process(proc.pid).memory_info().rss)
    except psutil.Error:
        return
    current = peak.get("bytes")
    peak["bytes"] = rss if current is None else max(int(current), rss)


def selected_label(parsed: dict[str, Any]) -> str:
    label = str(parsed.get("classification") or "ABSTAIN").upper()
    return label if label in LABELS else "ABSTAIN"


def softmax(values: list[float]) -> list[float]:
    pivot = max(values)
    exps = [math.exp(value - pivot) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def parse_top_logprobs(response: dict[str, Any]) -> dict[int, float]:
    probs = response.get("completion_probabilities") or []
    if not probs:
        return {}
    top_logprobs = probs[0].get("top_logprobs") or []
    return {int(item["id"]): float(item["logprob"]) for item in top_logprobs}


def select_with_threshold(label_probs: list[float], threshold: float) -> int:
    selected = int(np.argmax(label_probs))
    if LABELS[selected] == "TRUSTWORTHY" and label_probs[selected] < threshold:
        selected = int(np.argmax(label_probs[:2]))
    return selected


def tokenize_label_candidates(base_url: str, *, timeout: float) -> dict[str, list[int]]:
    label_token_ids: dict[str, list[int]] = {}
    for label in LABELS:
        tokenized = post_json(
            f"{base_url}/tokenize",
            {"content": label, "add_special": False, "parse_special": True},
            timeout=timeout,
        )
        tokens = tokenized.get("tokens") or []
        if not tokens:
            raise RuntimeError(f"label {label} produced no GGUF token ids")
        label_token_ids[label] = [int(token_id) for token_id in tokens]
    return label_token_ids


def score_label_sequences(
    completion_url: str,
    prompt: str,
    label_token_ids: dict[str, list[int]],
    args: argparse.Namespace,
    proc: subprocess.Popen[Any],
    peak: dict[str, int | None],
) -> tuple[list[float], dict[str, list[float]], dict[str, list[dict[str, int]]]]:
    scores: list[float] = []
    token_scores: dict[str, list[float]] = {}
    missing: dict[str, list[dict[str, int]]] = {}
    for label in LABELS:
        label_scores: list[float] = []
        label_missing: list[dict[str, int]] = []
        candidate_ids = label_token_ids[label]
        for offset, token_id in enumerate(candidate_ids):
            prefix_ids = candidate_ids[:offset]
            prompt_payload: str | list[Any] = prompt if not prefix_ids else [prompt, *prefix_ids]
            response = post_json(
                completion_url,
                {
                    "prompt": prompt_payload,
                    "n_predict": 1,
                    "temperature": -1,
                    "n_probs": args.n_probs,
                    "return_tokens": True,
                    "cache_prompt": not args.no_request_cache_prompt,
                    "stop": ["</s>", "<|im_end|>"],
                },
                timeout=args.request_timeout_seconds,
            )
            update_peak_rss(proc, peak)
            top_logprobs = parse_top_logprobs(response)
            value = top_logprobs.get(token_id)
            if value is None:
                value = -1.0e9
                label_missing.append({"offset": int(offset), "token_id": int(token_id)})
            label_scores.append(float(value))
        label_score = sum(label_scores)
        if args.label_score_length_normalization == "mean":
            label_score /= max(1, len(label_scores))
        scores.append(float(label_score))
        token_scores[label] = label_scores
        missing[label] = label_missing
    return scores, token_scores, missing


def main() -> int:
    args = parse_args()
    helpers = load_training_module()
    vocab = MoEVocab.from_metadata(args.metadata)
    rows = read_jsonl(args.input, args.max_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output_dir / "llama_server_stdout.log"
    stderr_path = args.output_dir / "llama_server_stderr.log"
    predictions_path = args.output_dir / "predictions.jsonl"
    report_path = args.output_dir / "report.json"

    server_args = [
        str(args.llama_server),
        "--model",
        str(args.model.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--no-webui",
        "--cache-ram",
        str(args.cache_ram_mib),
    ]
    if not args.no_cpu_moe:
        server_args.append("--cpu-moe")

    peak: dict[str, int | None] = {"bytes": None}
    started = time.monotonic()
    predictions: list[dict[str, Any]] = []
    gold_ids: list[int] = []
    pred_ids: list[int] = []
    counts = {label: 0 for label in LABELS}

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(server_args, stdout=stdout, stderr=stderr)
        try:
            health_url = f"http://{args.host}:{args.port}/health"
            ready = wait_for_health(health_url, proc, timeout=args.health_timeout_seconds, peak=peak)
            if not ready:
                raise RuntimeError(f"llama-server did not become ready; returncode={proc.poll()}")

            base_url = f"http://{args.host}:{args.port}"
            completion_url = f"http://{args.host}:{args.port}/completion"
            label_token_ids: dict[str, list[int]] | None = None
            if args.decision_mode in SCORE_DECISION_MODES:
                label_token_ids = tokenize_label_candidates(
                    base_url,
                    timeout=args.request_timeout_seconds,
                )

            for idx, row in enumerate(rows):
                prompt = helpers.build_prompt(row, target_mode="label-json")
                if args.decision_mode == "raw-generation":
                    response_payload = {
                        "prompt": prompt,
                        "n_predict": args.n_predict,
                        "temperature": args.temperature,
                        "stop": ["</s>", "<|im_end|>"],
                    }
                else:
                    response_payload = {
                        "prompt": prompt,
                        "n_predict": 1,
                        "temperature": -1,
                        "n_probs": args.n_probs,
                        "return_tokens": True,
                        "cache_prompt": not args.no_request_cache_prompt,
                        "stop": ["</s>", "<|im_end|>"],
                    }
                label_score = None
                if args.decision_mode == "raw-generation":
                    response = post_json(completion_url, response_payload, timeout=args.request_timeout_seconds)
                    update_peak_rss(proc, peak)
                    text = str(response.get("content") or "")
                    parsed = helpers.parse_generation(
                        text,
                        route2id=vocab.route2id,
                        taxonomy2id=vocab.taxonomy_pattern2id,
                        fallback_label="ABSTAIN",
                    )
                    label = selected_label(parsed)
                elif args.decision_mode == "first-token-label-score":
                    assert label_token_ids is not None
                    response = post_json(completion_url, response_payload, timeout=args.request_timeout_seconds)
                    update_peak_rss(proc, peak)
                    text = str(response.get("content") or "")
                    top_logprobs = parse_top_logprobs(response)
                    scores = [
                        top_logprobs.get(label_token_ids[label][0], -1.0e9)
                        for label in LABELS
                    ]
                    missing = [
                        label
                        for label in LABELS
                        if label_token_ids[label][0] not in top_logprobs
                    ]
                    label_probs = softmax(scores)
                    label_id = select_with_threshold(label_probs, args.label_threshold)
                    label = LABELS[label_id]
                    parsed = {
                        "json_parsed": False,
                        "label_parsed": not missing,
                        "classification": label,
                        "classification_id": label_id,
                        "route": "",
                        "route_id": -1,
                        "taxonomy_pattern": "",
                        "taxonomy_pattern_id": -1,
                    }
                    label_score = {
                        "classification": label,
                        "classification_id": label_id,
                        "scores": {label_name: float(scores[i]) for i, label_name in enumerate(LABELS)},
                        "probabilities": {
                            label_name: float(label_probs[i])
                            for i, label_name in enumerate(LABELS)
                        },
                        "trustworthy_threshold": float(args.label_threshold),
                        "mode": args.decision_mode,
                        "n_probs": int(args.n_probs),
                        "missing_top_logprobs": missing,
                        "label_token_ids": label_token_ids,
                    }
                else:
                    assert label_token_ids is not None
                    text = ""
                    scores, token_scores, missing = score_label_sequences(
                        completion_url,
                        prompt,
                        label_token_ids,
                        args,
                        proc,
                        peak,
                    )
                    label_probs = softmax(scores)
                    label_id = select_with_threshold(label_probs, args.label_threshold)
                    label = LABELS[label_id]
                    has_missing = any(entries for entries in missing.values())
                    parsed = {
                        "json_parsed": False,
                        "label_parsed": not has_missing,
                        "classification": label,
                        "classification_id": label_id,
                        "route": "",
                        "route_id": -1,
                        "taxonomy_pattern": "",
                        "taxonomy_pattern_id": -1,
                    }
                    label_score = {
                        "classification": label,
                        "classification_id": label_id,
                        "scores": {label_name: float(scores[i]) for i, label_name in enumerate(LABELS)},
                        "probabilities": {
                            label_name: float(label_probs[i])
                            for i, label_name in enumerate(LABELS)
                        },
                        "token_scores": token_scores,
                        "trustworthy_threshold": float(args.label_threshold),
                        "length_normalization": args.label_score_length_normalization,
                        "mode": args.decision_mode,
                        "n_probs": int(args.n_probs),
                        "missing_top_logprobs": missing,
                        "label_token_ids": label_token_ids,
                    }
                counts[label] += 1
                predictions.append(
                    {
                        "index": idx,
                        "id": row.get("id"),
                        "classification": label,
                        "parsed": parsed,
                        "label_score": label_score,
                        "raw_generation": text,
                        "gold": {
                            "classification": row.get("label"),
                            "route": row.get("route"),
                            "taxonomy_pattern": row.get("taxonomy_pattern"),
                        },
                    }
                )
                if row.get("label") in LABELS:
                    pred_ids.append(LABELS.index(label))
                    gold_ids.append(LABELS.index(str(row["label"])))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)

    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "ok": True,
        "model": str(args.model),
        "input": str(args.input),
        "predictions": str(predictions_path),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "rows": len(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_bytes": peak["bytes"],
        "peak_rss_gib": None if peak["bytes"] is None else round(int(peak["bytes"]) / (1024**3), 3),
        "n_predict": args.n_predict,
        "cache_ram_mib": args.cache_ram_mib,
        "request_cache_prompt": not args.no_request_cache_prompt,
        "decision_mode": args.decision_mode,
        "label_threshold": args.label_threshold,
        "label_score_length_normalization": (
            args.label_score_length_normalization
            if args.decision_mode in SCORE_DECISION_MODES
            else None
        ),
        "n_probs": args.n_probs if args.decision_mode in SCORE_DECISION_MODES else None,
        "counts": counts,
        "label_parse_rate": (
            sum(1 for row in predictions if row["parsed"].get("label_parsed")) / len(predictions)
            if predictions
            else 0.0
        ),
        "json_parse_rate": None
        if args.decision_mode != "raw-generation"
        else (
            sum(1 for row in predictions if row["parsed"].get("json_parsed")) / len(predictions)
            if predictions
            else 0.0
        ),
    }
    if gold_ids:
        metrics = compute_classification_metrics(
            np.array(pred_ids, dtype=np.int64),
            np.array(gold_ids, dtype=np.int64),
        )
        report["classification"] = metrics
    write_jsonl(predictions_path, predictions)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
