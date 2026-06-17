"""Audit fitz-gov-sage subagent JSONL outputs before materialization/training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_WORKPACK_DIR = Path("data/fitz_gov_sage_v1_workpacks")
DEFAULT_INPUT_DIR = DEFAULT_WORKPACK_DIR / "subagent_outputs"
STAGES = {"query_planning", "evidence_governance"}
PLACEHOLDER_RE = re.compile(r"\[(?:entity|date|project|product|system|source|document)\]", re.I)
SYNTHETIC_MISSING_RE = re.compile(
    r"\b(?:missing item|missing evidence|missing placeholder|the pack lacks|does not include the decisive|does not contain the decisive)\b",
    re.I,
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            row["_audit_path"] = str(path)
            row["_audit_line"] = line_no
            yield row


def load_source_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in read_jsonl(path):
        queries[str(row["source_id"])] = str(row.get("query") or "")
    if not queries:
        raise ValueError(f"no source rows loaded from {path}")
    return queries


def text_fields(row: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key in ("query", "query_text", "text"):
        value = row.get(key)
        if isinstance(value, str):
            yield key, value
    contexts = row.get("contexts")
    if isinstance(contexts, list):
        for idx, value in enumerate(contexts):
            if isinstance(value, str):
                yield f"contexts[{idx}]", value
    pack = row.get("pack_metadata")
    if isinstance(pack, dict):
        for idx, item in enumerate(pack.get("items") or []):
            if isinstance(item, dict):
                for key in ("anchor", "why_present", "role"):
                    value = item.get(key)
                    if isinstance(value, str):
                        yield f"pack_metadata.items[{idx}].{key}", value


def evidence_leakage_fields(row: dict[str, Any]) -> Iterable[tuple[str, str]]:
    contexts = row.get("contexts")
    if isinstance(contexts, list):
        for idx, value in enumerate(contexts):
            if isinstance(value, str):
                yield f"contexts[{idx}]", value
    pack = row.get("pack_metadata")
    if isinstance(pack, dict):
        for idx, item in enumerate(pack.get("items") or []):
            if isinstance(item, dict):
                for key in ("anchor", "why_present"):
                    value = item.get(key)
                    if isinstance(value, str):
                        yield f"pack_metadata.items[{idx}].{key}", value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-dir", type=Path, default=DEFAULT_WORKPACK_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workpack_dir = args.workpack_dir.resolve()
    input_dir = args.input_dir.resolve()
    source_queries = load_source_queries(workpack_dir / "source_selection.jsonl")
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL outputs found in {input_dir}")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    violations: list[dict[str, Any]] = []
    rows_seen = 0
    for path in files:
        for row in read_jsonl(path):
            rows_seen += 1
            source_id = str(row.get("source_id") or "")
            stage = str(row.get("stage") or "")
            location = {"path": row["_audit_path"], "line": row["_audit_line"], "source_id": source_id, "stage": stage}
            if source_id not in source_queries:
                violations.append({**location, "kind": "unknown_source_id"})
                continue
            if stage not in STAGES:
                violations.append({**location, "kind": "invalid_stage"})
                continue
            if stage in grouped[source_id]:
                violations.append({**location, "kind": "duplicate_stage"})
            grouped[source_id][stage] = row
            if str(row.get("query") or "") != source_queries[source_id]:
                violations.append({**location, "kind": "query_changed"})
            contexts = row.get("contexts")
            if stage == "query_planning" and contexts not in ([], None):
                violations.append({**location, "kind": "query_planning_has_contexts"})
            if stage == "evidence_governance" and not contexts:
                violations.append({**location, "kind": "evidence_governance_missing_contexts"})
            pack = row.get("pack_metadata")
            if stage == "evidence_governance" and isinstance(pack, dict):
                items = pack.get("items") or []
                if isinstance(contexts, list) and len(items) != len(contexts):
                    violations.append(
                        {
                            **location,
                            "kind": "pack_item_context_count_mismatch",
                            "items": len(items),
                            "contexts": len(contexts),
                        }
                    )
                for item in items:
                    if isinstance(item, dict) and str(item.get("role") or "") == "missing_placeholder":
                        violations.append({**location, "kind": "missing_placeholder_role"})
            for field, value in text_fields(row):
                if PLACEHOLDER_RE.search(value):
                    violations.append({**location, "kind": "placeholder_token", "field": field})
                if SYNTHETIC_MISSING_RE.search(value):
                    violations.append({**location, "kind": "synthetic_missing_context", "field": field})
            if stage == "evidence_governance":
                for field, value in evidence_leakage_fields(row):
                    if source_id and source_id in value:
                        violations.append({**location, "kind": "source_id_leakage", "field": field})

    for source_id, stages in grouped.items():
        missing = sorted(STAGES - set(stages))
        if missing:
            violations.append({"source_id": source_id, "kind": "missing_stage", "missing": missing})

    report = {
        "input_dir": str(input_dir),
        "files": len(files),
        "rows_seen": rows_seen,
        "source_ids_seen": len(grouped),
        "violations": len(violations),
        "violation_examples": violations[:100],
    }
    output = args.output or (workpack_dir / "subagent_output_audit.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
