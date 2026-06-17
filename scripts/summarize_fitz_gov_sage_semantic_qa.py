"""Summarize fitz-gov-sage semantic QA reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            row["_qa_file"] = path.name
            row["_qa_line"] = line_no
            yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-dir", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repair-output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workpack_dir = args.workpack_dir.resolve()
    qa_dir = args.qa_dir.resolve()
    expected_ids = {
        str(row.get("source_id") or row.get("id"))
        for row in read_jsonl(workpack_dir / "source_selection.jsonl")
    }
    reports = sorted(qa_dir.glob("pack_*.semantic_qa.jsonl"))
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for path in reports:
        local_seen: set[str] = set()
        for row in read_jsonl(path):
            source_id = str(row.get("source_id") or "")
            location = {"file": row["_qa_file"], "line": row["_qa_line"], "source_id": source_id}
            if not source_id:
                errors.append({**location, "kind": "missing_source_id"})
                continue
            if source_id in local_seen:
                errors.append({**location, "kind": "duplicate_source_id_in_report"})
            if source_id in seen:
                errors.append({**location, "kind": "duplicate_source_id_across_reports"})
            if source_id not in expected_ids:
                errors.append({**location, "kind": "unknown_source_id"})
            local_seen.add(source_id)
            seen.add(source_id)
            if row.get("verdict") not in ("accept", "repair"):
                errors.append({**location, "kind": "invalid_verdict", "verdict": row.get("verdict")})
            rows.append(row)

    missing = sorted(expected_ids - seen)
    if missing:
        errors.append({"kind": "missing_qa_rows", "count": len(missing), "examples": missing[:20]})

    counts = Counter(str(row.get("verdict")) for row in rows)
    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    repairs: list[dict[str, Any]] = []
    for row in rows:
        by_file[str(row["_qa_file"])][str(row.get("verdict"))] += 1
        if row.get("verdict") == "repair":
            repairs.append(
                {
                    "source_id": row.get("source_id"),
                    "label": row.get("label"),
                    "issue_types": row.get("issue_types") or [],
                    "issues": row.get("issues") or [],
                    "repair_instruction": row.get("repair_instruction") or "",
                    "qa_file": row.get("_qa_file"),
                }
            )

    summary = {
        "workpack_dir": str(workpack_dir),
        "qa_dir": str(qa_dir),
        "report_files": len(reports),
        "expected_source_ids": len(expected_ids),
        "qa_rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "by_file": {key: dict(value) for key, value in sorted(by_file.items())},
        "repair_count": len(repairs),
        "errors": errors,
        "repairs": repairs,
    }
    output = args.output or (qa_dir / "semantic_qa_summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    repair_output = args.repair_output or (qa_dir / "repair_manifest.jsonl")
    with repair_output.open("w", encoding="utf-8") as fh:
        for row in repairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

