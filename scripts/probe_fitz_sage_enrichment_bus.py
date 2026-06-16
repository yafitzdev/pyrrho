"""Probe whether a local LLM can drive fitz-sage's KRAG enrichment bus.

This script imports the real ``KragEnricher`` from a sibling fitz-sage checkout
and feeds it a small mixed code/document fixture. Success means the model can
return the JSON array shape fitz-sage expects, not just plausible prose.

Examples:
    python scripts/probe_fitz_sage_enrichment_bus.py --backend hf \
      --model-path outputs/external_baselines/Qwen3.5-0.8B-Base

    python scripts/probe_fitz_sage_enrichment_bus.py --backend endpoint \
      --endpoint-base-url http://localhost:1234/v1 --endpoint-model local-model
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FITZ_SAGE_ROOT = ROOT.parent / "fitz-sage"
DEFAULT_OUTPUT = ROOT / "outputs" / "enrichment_bus_probe" / "report.json"
DEFAULT_QWEN = ROOT / "outputs" / "external_baselines" / "Qwen3.5-0.8B-Base"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Mirror fitz-sage's local-provider cleanup for reasoning models."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        return text.split("</think>")[-1].lstrip() if "</think>" in text else text.split("<think>")[0].rstrip()
    return text


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RecordedCall:
    elapsed_s: float
    expected_items: int
    response_chars: int
    raw_response: str
    parsed_type: str
    parsed_count: int
    parse_ok: bool
    count_ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completion_tokens_per_s: float | None = None


@dataclass
class RecordingChat:
    calls: list[RecordedCall] = field(default_factory=list)

    def _record(
        self,
        messages: list[dict[str, Any]],
        raw_response: str,
        elapsed_s: float,
        usage: dict[str, Any] | None = None,
    ) -> None:
        from fitz_sage.core.json_utils import parse_llm_json

        user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        expected_items = len(re.findall(r"(?m)^Item \d+:", user))
        parsed = parse_llm_json(raw_response, as_array=True)
        parsed_count = len(parsed) if isinstance(parsed, list) else 0
        prompt_tokens = _safe_int((usage or {}).get("prompt_tokens"))
        completion_tokens = _safe_int((usage or {}).get("completion_tokens"))
        total_tokens = _safe_int((usage or {}).get("total_tokens"))
        self.calls.append(
            RecordedCall(
                elapsed_s=elapsed_s,
                expected_items=expected_items,
                response_chars=len(raw_response),
                raw_response=raw_response,
                parsed_type=type(parsed).__name__,
                parsed_count=parsed_count,
                parse_ok=isinstance(parsed, list) and bool(parsed),
                count_ok=isinstance(parsed, list) and parsed_count >= expected_items,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                completion_tokens_per_s=(
                    completion_tokens / elapsed_s if completion_tokens is not None and elapsed_s > 0 else None
                ),
            )
        )


class HFLocalChat(RecordingChat):
    def __init__(
        self,
        model_path: Path,
        *,
        device: str,
        device_map: str | None,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        chat_template: str,
        trust_remote_code: bool,
    ) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._chat_template = chat_template

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype != "none":
            model_kwargs["torch_dtype"] = dtype
        if device_map:
            model_kwargs["device_map"] = device_map

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if not device_map:
            actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            self._model.to(actual_device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> str:
        prompt = self._render_messages(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        first_device = next(self._model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}

        do_sample = self._temperature > 0.0
        start = time.perf_counter()
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=do_sample,
                temperature=self._temperature if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        elapsed_s = time.perf_counter() - start
        new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        response = strip_thinking(raw).strip()
        self._record(messages, response, elapsed_s)
        return response

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        use_template = self._chat_template == "always" or (
            self._chat_template == "auto" and bool(getattr(self._tokenizer, "chat_template", None))
        )
        if use_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                if self._chat_template == "always":
                    raise

        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)


class EndpointChat(RecordingChat):
    def __init__(
        self,
        *,
        base_url: str,
        model: str | None,
        api_key: str | None,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._model = model or self._discover_model(api_key, timeout_s)
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_s = timeout_s

    @property
    def model(self) -> str:
        return self._model

    def chat(self, messages: list[dict[str, Any]], **_: Any) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        start = time.perf_counter()
        data = self._post_json(f"{self._base_url}/chat/completions", payload, self._api_key, self._timeout_s)
        elapsed_s = time.perf_counter() - start
        response = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        response = strip_thinking(response).strip()
        self._record(messages, response, elapsed_s, usage=data.get("usage") or {})
        return response

    def _discover_model(self, api_key: str | None, timeout_s: float) -> str:
        data = self._get_json(f"{self._base_url}/models", api_key, timeout_s)
        models = data.get("data") or []
        if not models:
            raise RuntimeError(f"No models returned by {self._base_url}/models")
        model_id = models[0].get("id")
        if not model_id:
            raise RuntimeError(f"First model entry has no id: {models[0]}")
        return str(model_id)

    @staticmethod
    def _headers(api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _get_json(cls, url: str, api_key: str | None, timeout_s: float) -> dict[str, Any]:
        request = urllib.request.Request(url, headers=cls._headers(api_key), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Endpoint request failed: {url}: {exc}") from exc

    @classmethod
    def _post_json(
        cls,
        url: str,
        payload: dict[str, Any],
        api_key: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=cls._headers(api_key), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Endpoint request failed: {url}: {exc}") from exc


def build_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    symbols = [
        {
            "id": "sym_auth_validate",
            "name": "AuthService.validateToken",
            "kind": "function",
            "summary": (
                "Validates OAuth2 JWT_TOKEN for /api/v2/users. Rejects expired tokens "
                "after AuthService v2.3.1 rollout in 2024-03."
            ),
        },
        {
            "id": "sym_retry_budget",
            "name": "RetryBudget.MAX_RETRIES",
            "kind": "constant",
            "summary": (
                "Defines RETRY_LIMIT for ECU-17 acceptance tests. Test case TC-1001 "
                "changed from 3 to 5 retries on 2025-02-14 for firmware 1.8.4."
            ),
        },
        {
            "id": "sym_entity_graph",
            "name": "EvidenceGraph.link_claims",
            "kind": "method",
            "summary": (
                "Links KRAG chunks through EntityGraphStore, SQLite FTS5, and bm25() "
                "sparse matches for related-unit discovery."
            ),
        },
        {
            "id": "sym_json_parser",
            "name": "parse_llm_json",
            "kind": "function",
            "summary": (
                "Extracts JSON arrays from markdown ```json fences and prose before "
                "falling back to empty enrichment objects."
            ),
        },
    ]
    sections = [
        {
            "id": "sec_auth",
            "title": "API v2 Authentication",
            "summary": (
                "The AuthService module uses OAuth2 and JWT_TOKEN headers for "
                "/api/v2/users. PostgreSQL stores revoked-token state for v2.3.1."
            ),
            "content": "",
        },
        {
            "id": "sec_ecu",
            "title": "ECU Release Acceptance",
            "summary": (
                "ECU-17 release notes say TC-1001 failed on firmware 1.8.4 until "
                "the retry limit changed on 2025-02-14."
            ),
            "content": "",
        },
        {
            "id": "sec_graph",
            "title": "KRAG Entity Graph",
            "summary": (
                "EntityGraphStore records named entities from enriched sections so "
                "SQLite FTS5 and bm25() can retrieve related KRAG units."
            ),
            "content": "",
        },
        {
            "id": "sec_runtime",
            "title": "Q4 Runtime Notes",
            "summary": (
                "The GGUF Q4_K_M runtime was tested in LM Studio and llama.cpp, with "
                "a 7.40 GiB host-memory load for the model."
            ),
            "content": "",
        },
    ]
    expected_anchors = {
        "sym_auth_validate": ["AuthService", "OAuth2", "JWT_TOKEN", "/api/v2/users", "v2.3.1", "2024-03"],
        "sym_retry_budget": ["MAX_RETRIES", "RETRY_LIMIT", "ECU-17", "TC-1001", "2025-02-14", "1.8.4"],
        "sym_entity_graph": ["KRAG", "EntityGraphStore", "SQLite FTS5", "bm25"],
        "sym_json_parser": ["parse_llm_json", "JSON", "markdown", "fences"],
        "sec_auth": ["AuthService", "OAuth2", "JWT_TOKEN", "/api/v2/users", "PostgreSQL", "v2.3.1"],
        "sec_ecu": ["ECU-17", "TC-1001", "firmware 1.8.4", "2025-02-14"],
        "sec_graph": ["EntityGraphStore", "SQLite FTS5", "bm25", "KRAG"],
        "sec_runtime": ["GGUF", "Q4_K_M", "LM Studio", "llama.cpp", "7.40 GiB"],
    }
    return symbols, sections, expected_anchors


def summarize_items(items: list[dict[str, Any]], expected_anchors: dict[str, list[str]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_hits = 0
    total_expected = 0
    nonempty_keywords = 0
    nonempty_entities = 0
    temporal_count = 0
    shape_ok = 0

    for item in items:
        item_id = item["id"]
        keywords = item.get("keywords")
        entities = item.get("entities")
        temporal = (item.get("metadata") or {}).get("temporal")
        item_shape_ok = isinstance(keywords, list) and isinstance(entities, list)
        shape_ok += int(item_shape_ok)
        nonempty_keywords += int(isinstance(keywords, list) and bool(keywords))
        nonempty_entities += int(isinstance(entities, list) and bool(entities))
        temporal_count += int(isinstance(temporal, dict) and any(temporal.get(k) for k in ("dates", "versions", "refs")))

        haystack = json.dumps(
            {
                "keywords": keywords or [],
                "entities": entities or [],
                "temporal": temporal or {},
            },
            ensure_ascii=False,
        ).lower()
        expected = expected_anchors[item_id]
        hits = [anchor for anchor in expected if anchor.lower() in haystack]
        total_hits += len(hits)
        total_expected += len(expected)
        rows.append(
            {
                "id": item_id,
                "keywords": keywords or [],
                "entities": entities or [],
                "temporal": temporal or {},
                "expected_anchors": expected,
                "hit_anchors": hits,
                "anchor_recall": len(hits) / len(expected) if expected else 0.0,
            }
        )

    return {
        "items": rows,
        "item_count": len(items),
        "shape_ok_items": shape_ok,
        "nonempty_keywords_items": nonempty_keywords,
        "nonempty_entities_items": nonempty_entities,
        "temporal_items": temporal_count,
        "anchor_hits": total_hits,
        "anchor_total": total_expected,
        "anchor_recall": total_hits / total_expected if total_expected else 0.0,
    }


def import_fitz_sage(fitz_sage_root: Path) -> None:
    import sys

    if not fitz_sage_root.exists():
        raise FileNotFoundError(f"fitz-sage root not found: {fitz_sage_root}")
    sys.path.insert(0, str(fitz_sage_root))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import_fitz_sage(args.fitz_sage_root)
    from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher

    if args.backend == "hf":
        chat: RecordingChat = HFLocalChat(
            args.model_path,
            device=args.device,
            device_map=args.device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            chat_template=args.chat_template,
            trust_remote_code=args.trust_remote_code,
        )
        model_label = str(args.model_path)
    else:
        endpoint_chat = EndpointChat(
            base_url=args.endpoint_base_url,
            model=args.endpoint_model,
            api_key=args.endpoint_api_key,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_s=args.endpoint_timeout_s,
        )
        chat = endpoint_chat
        model_label = f"{args.endpoint_base_url}::{endpoint_chat.model}"

    symbols, sections, expected_anchors = build_fixture()
    enricher = KragEnricher(chat, batch_size=args.batch_size)

    start = time.perf_counter()
    enricher.enrich_symbols(symbols)
    enricher.enrich_sections(sections)
    total_elapsed_s = time.perf_counter() - start

    symbol_summary = summarize_items(symbols, expected_anchors)
    section_summary = summarize_items(sections, expected_anchors)
    combined_items = symbols + sections
    combined_summary = summarize_items(combined_items, expected_anchors)
    calls = [call.__dict__ for call in chat.calls]
    completion_tokens = [call["completion_tokens"] for call in calls if call.get("completion_tokens") is not None]
    timed_completion_s = [
        (call["completion_tokens"], call["elapsed_s"])
        for call in calls
        if call.get("completion_tokens") is not None and call["elapsed_s"] > 0
    ]
    total_completion_tokens = sum(completion_tokens) if completion_tokens else None
    total_generation_s = sum(elapsed for _, elapsed in timed_completion_s) if timed_completion_s else None
    aggregate_completion_tokens_per_s = (
        sum(tokens for tokens, _ in timed_completion_s) / total_generation_s if total_generation_s else None
    )
    all_calls_parse = bool(calls) and all(call["parse_ok"] and call["count_ok"] for call in calls)
    all_shapes_ok = combined_summary["shape_ok_items"] == combined_summary["item_count"]
    useful_keywords = combined_summary["nonempty_keywords_items"] >= max(1, combined_summary["item_count"] - 1)
    anchor_floor_ok = combined_summary["anchor_recall"] >= args.min_anchor_recall
    bus_usable = all_calls_parse and all_shapes_ok and useful_keywords and anchor_floor_ok

    return {
        "backend": args.backend,
        "model": model_label,
        "fitz_sage_root": str(args.fitz_sage_root),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "min_anchor_recall": args.min_anchor_recall,
        "total_elapsed_s": total_elapsed_s,
        "total_completion_tokens": total_completion_tokens,
        "total_generation_s": total_generation_s,
        "aggregate_completion_tokens_per_s": aggregate_completion_tokens_per_s,
        "fitz_sage_enrichment_bus_usable": bus_usable,
        "gates": {
            "all_calls_parse_and_count_ok": all_calls_parse,
            "all_items_have_expected_shape": all_shapes_ok,
            "mostly_nonempty_keywords": useful_keywords,
            "anchor_recall_floor_ok": anchor_floor_ok,
        },
        "calls": calls,
        "symbols": symbol_summary,
        "sections": section_summary,
        "combined": combined_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["hf", "endpoint"], default="hf")
    parser.add_argument("--fitz-sage-root", type=Path, default=DEFAULT_FITZ_SAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-anchor-recall", type=float, default=0.50)

    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument("--dtype", default="auto", help="torch_dtype for from_pretrained; use 'none' to omit.")
    parser.add_argument("--chat-template", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--endpoint-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--endpoint-model", default=None)
    parser.add_argument("--endpoint-api-key", default=None)
    parser.add_argument("--endpoint-timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_probe(args)

    if args.output == DEFAULT_OUTPUT and args.backend == "hf":
        args.output = (
            ROOT
            / "outputs"
            / "enrichment_bus_probe"
            / f"fitz_sage_enrichment_bus_{sanitize_name(args.model_path.name)}.json"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    combined = report["combined"]
    print(f"wrote: {args.output}")
    print(f"usable: {report['fitz_sage_enrichment_bus_usable']}")
    print(f"calls_parse_count_ok: {report['gates']['all_calls_parse_and_count_ok']}")
    if report.get("aggregate_completion_tokens_per_s") is not None:
        print(
            "completion_tok_s: "
            f"{report['aggregate_completion_tokens_per_s']:.2f} "
            f"({report['total_completion_tokens']} tokens / {report['total_generation_s']:.2f}s)"
        )
    print(
        "items: "
        f"{combined['item_count']} shape_ok={combined['shape_ok_items']} "
        f"kw_nonempty={combined['nonempty_keywords_items']} "
        f"entity_nonempty={combined['nonempty_entities_items']} "
        f"temporal={combined['temporal_items']}"
    )
    print(f"anchor_recall: {combined['anchor_recall']:.3f} ({combined['anchor_hits']}/{combined['anchor_total']})")


if __name__ == "__main__":
    main()
