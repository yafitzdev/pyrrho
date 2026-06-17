"""Audit whether fitz-gov-sage outputs are real retrieval-pack transforms."""

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
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def normalize_contexts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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


def load_evidence_outputs(input_dir: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("stage") != "evidence_governance":
                continue
            source_id = str(row.get("source_id") or "")
            if not source_id:
                raise ValueError(f"{path}: evidence row missing source_id")
            if source_id in grouped:
                raise ValueError(f"{source_id}: duplicate evidence_governance output")
            grouped[source_id] = row
    if not grouped:
        raise ValueError(f"no evidence_governance rows found in {input_dir}")
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-exact-source-context-fraction", type=float, default=0.10)
    parser.add_argument("--min-changed-context-set-fraction", type=float, default=0.80)
    parser.add_argument("--min-mean-context-count", type=float, default=4.0)
    parser.add_argument("--max-short-pack-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workpack_dir = args.workpack_dir.resolve()
    input_dir = args.input_dir.resolve()
    sources = load_sources(workpack_dir / "source_selection.jsonl")
    evidence_rows = load_evidence_outputs(input_dir)

    missing_outputs = sorted(set(sources) - set(evidence_rows))
    unknown_outputs = sorted(set(evidence_rows) - set(sources))
    stats = Counter()
    context_counts = Counter()
    pack_shapes = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_id, source in sources.items():
        row = evidence_rows.get(source_id)
        if row is None:
            continue
        source_contexts = normalize_contexts(source.get("contexts"))
        output_contexts = normalize_contexts(row.get("contexts"))
        source_set = set(source_contexts)
        output_set = set(output_contexts)
        pack = row.get("pack_metadata") if isinstance(row.get("pack_metadata"), dict) else {}
        pack_shape = str(pack.get("pack_shape") or "<missing>")

        context_counts[len(output_contexts)] += 1
        pack_shapes[pack_shape] += 1
        stats["evidence_rows"] += 1
        if output_contexts == source_contexts:
            stats["exact_source_context_list_rows"] += 1
            if len(examples["exact_source_context_list"]) < 5:
                examples["exact_source_context_list"].append(
                    {"source_id": source_id, "query": source.get("query"), "contexts": len(output_contexts)}
                )
        if output_set != source_set:
            stats["changed_context_set_rows"] += 1
        if output_set - source_set:
            stats["rows_with_added_contexts"] += 1
        if source_set - output_set:
            stats["rows_with_removed_source_contexts"] += 1
        if len(output_contexts) >= 4:
            stats["retrieval_pack_context_count_rows"] += 1
        if pack_shape == "short_pack_1_3":
            stats["short_pack_rows"] += 1
        items = pack.get("items") or []
        if len(items) != len(output_contexts):
            stats["pack_item_context_count_mismatch"] += 1
            if len(examples["pack_item_context_count_mismatch"]) < 5:
                examples["pack_item_context_count_mismatch"].append(
                    {"source_id": source_id, "items": len(items), "contexts": len(output_contexts)}
                )

    evidence_count = max(int(stats["evidence_rows"]), 1)
    mean_context_count = sum(count * n for count, n in context_counts.items()) / evidence_count
    exact_fraction = stats["exact_source_context_list_rows"] / evidence_count
    changed_fraction = stats["changed_context_set_rows"] / evidence_count
    short_fraction = stats["short_pack_rows"] / evidence_count

    violations: list[dict[str, Any]] = []
    if missing_outputs:
        violations.append({"kind": "missing_evidence_outputs", "count": len(missing_outputs), "examples": missing_outputs[:10]})
    if unknown_outputs:
        violations.append({"kind": "unknown_evidence_outputs", "count": len(unknown_outputs), "examples": unknown_outputs[:10]})
    if stats["pack_item_context_count_mismatch"]:
        violations.append(
            {
                "kind": "pack_item_context_count_mismatch",
                "count": int(stats["pack_item_context_count_mismatch"]),
                "examples": examples["pack_item_context_count_mismatch"],
            }
        )
    if exact_fraction > args.max_exact_source_context_fraction:
        violations.append(
            {
                "kind": "too_many_exact_source_context_lists",
                "observed": exact_fraction,
                "max_allowed": args.max_exact_source_context_fraction,
            }
        )
    if changed_fraction < args.min_changed_context_set_fraction:
        violations.append(
            {
                "kind": "too_few_changed_context_sets",
                "observed": changed_fraction,
                "min_required": args.min_changed_context_set_fraction,
            }
        )
    if mean_context_count < args.min_mean_context_count:
        violations.append(
            {
                "kind": "mean_context_count_too_low",
                "observed": mean_context_count,
                "min_required": args.min_mean_context_count,
            }
        )
    if short_fraction > args.max_short_pack_fraction:
        violations.append(
            {
                "kind": "too_many_short_packs",
                "observed": short_fraction,
                "max_allowed": args.max_short_pack_fraction,
            }
        )

    report = {
        "workpack_dir": str(workpack_dir),
        "input_dir": str(input_dir),
        "source_rows": len(sources),
        "evidence_rows": int(stats["evidence_rows"]),
        "context_count_distribution": dict(sorted(context_counts.items())),
        "pack_shape_counts": dict(sorted(pack_shapes.items())),
        "mean_context_count": mean_context_count,
        "exact_source_context_list_rows": int(stats["exact_source_context_list_rows"]),
        "exact_source_context_list_fraction": exact_fraction,
        "changed_context_set_rows": int(stats["changed_context_set_rows"]),
        "changed_context_set_fraction": changed_fraction,
        "rows_with_added_contexts": int(stats["rows_with_added_contexts"]),
        "rows_with_removed_source_contexts": int(stats["rows_with_removed_source_contexts"]),
        "retrieval_pack_context_count_rows": int(stats["retrieval_pack_context_count_rows"]),
        "short_pack_rows": int(stats["short_pack_rows"]),
        "short_pack_fraction": short_fraction,
        "violations": len(violations),
        "violation_examples": violations[:100],
        "examples": examples,
    }
    output = args.output or (workpack_dir / "sage_shape_audit.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

