"""Audit fitz-gov-sage outputs preserve source labels and scalar targets."""

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


PRE_RETRIEVAL_LABEL_FIELDS = (
    "query_contract",
    "route",
    "retrieval_action",
    "gap_type",
    "answerability_shape",
    "retrieval_modality",
    "retrieval_obligation",
)
EVIDENCE_LABEL_FIELDS = (
    "label",
    "label_id",
    "query_contract",
    "route",
    "taxonomy_pattern",
    "retrieval_action",
    "gap_type",
    "answerability_shape",
    "retrieval_modality",
    "retrieval_obligation",
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
            row["_audit_file"] = str(path)
            row["_audit_line"] = line_no
            yield row


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        source_id = str(row.get("source_id") or row.get("id") or "")
        if not source_id:
            raise ValueError(f"{path}: source row missing source_id/id")
        sources[source_id] = row
    if not sources:
        raise ValueError(f"no source rows loaded from {path}")
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workpack_dir = args.workpack_dir.resolve()
    input_dir = args.input_dir.resolve()
    sources = load_sources(workpack_dir / "source_selection.jsonl")
    stage_counts = Counter()
    grouped: dict[str, set[str]] = defaultdict(set)
    violations: list[dict[str, Any]] = []

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL outputs found in {input_dir}")

    for path in files:
        for row in read_jsonl(path):
            source_id = str(row.get("source_id") or "")
            stage = str(row.get("stage") or "")
            location = {
                "path": row["_audit_file"],
                "line": row["_audit_line"],
                "source_id": source_id,
                "stage": stage,
            }
            if source_id not in sources:
                violations.append({**location, "kind": "unknown_source_id"})
                continue
            source = sources[source_id]
            source_labels = source.get("labels") or {}
            out_labels = row.get("labels") or {}
            stage_counts[stage] += 1
            grouped[source_id].add(stage)

            if stage == "query_planning":
                if out_labels.get("label_id") != -1:
                    violations.append({**location, "kind": "planning_label_id_not_masked", "value": out_labels.get("label_id")})
                if out_labels.get("taxonomy_pattern") is not None:
                    violations.append(
                        {
                            **location,
                            "kind": "planning_taxonomy_not_masked",
                            "value": out_labels.get("taxonomy_pattern"),
                        }
                    )
                if row.get("scalar_targets") not in ({}, None):
                    violations.append({**location, "kind": "planning_scalar_targets_not_empty"})
                for field in PRE_RETRIEVAL_LABEL_FIELDS:
                    if out_labels.get(field) != source_labels.get(field):
                        violations.append(
                            {
                                **location,
                                "kind": "planning_label_mismatch",
                                "field": field,
                                "expected": source_labels.get(field),
                                "observed": out_labels.get(field),
                            }
                        )
            elif stage == "evidence_governance":
                for field in EVIDENCE_LABEL_FIELDS:
                    if out_labels.get(field) != source_labels.get(field):
                        violations.append(
                            {
                                **location,
                                "kind": "evidence_label_mismatch",
                                "field": field,
                                "expected": source_labels.get(field),
                                "observed": out_labels.get(field),
                            }
                        )
                if row.get("scalar_targets") != source.get("scalar_targets"):
                    violations.append({**location, "kind": "evidence_scalar_targets_mismatch"})
            else:
                violations.append({**location, "kind": "unknown_stage"})

    for source_id in sorted(sources):
        stages = grouped.get(source_id, set())
        if stages != {"query_planning", "evidence_governance"}:
            violations.append({"source_id": source_id, "kind": "missing_or_extra_stages", "stages": sorted(stages)})

    report = {
        "workpack_dir": str(workpack_dir),
        "input_dir": str(input_dir),
        "files": len(files),
        "source_rows": len(sources),
        "stage_counts": dict(sorted(stage_counts.items())),
        "violations": len(violations),
        "violation_examples": violations[:100],
    }
    output = args.output or (workpack_dir / "label_preservation_audit.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

