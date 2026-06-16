"""Validate subagent outputs and materialize fitz-gov-sage training JSONL.

Subagents write stage rows, but the source V10 row remains the authority for
labels. This script uses subagent output for evidence-pack shape and source
V10 metadata for labels/ids, then writes trainer-ready split files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.data import build_encoder_text, build_query_contract_text
from pyrrho.manifest import write_manifest


DEFAULT_WORKPACK_DIR = Path("data/fitz_gov_sage_v1_workpacks")
DEFAULT_SOURCE_DATA_DIR = Path("data/multitask_g5_1_v10_repaired")
DEFAULT_OUTPUT_DIR = Path("data/fitz_gov_sage_v1")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_sage_g1_v10_packview.yaml")
STAGES = ("query_planning", "evidence_governance")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_id(mapping: dict[str, int], value: Any, *, field: str, source_id: str) -> int:
    if value in (None, "", "<masked>"):
        return -1
    value = str(value)
    if value not in mapping:
        raise ValueError(f"{source_id}: unknown {field} label {value!r}")
    return int(mapping[value])


def load_source_selection(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"source selection is empty: {path}")
    return {str(row["source_id"]): row for row in rows}


def load_stage_outputs(input_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no subagent output JSONL files found in {input_dir}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in files:
        for row in read_jsonl(path):
            source_id = str(row.get("source_id") or "")
            stage = str(row.get("stage") or "")
            if not source_id:
                raise ValueError(f"{path}: row missing source_id")
            if stage not in STAGES:
                raise ValueError(f"{path}: {source_id}: invalid stage {stage!r}")
            if stage in grouped[source_id]:
                raise ValueError(f"{source_id}: duplicate stage {stage!r}")
            grouped[source_id][stage] = row
    return grouped


def ensure_query(source: dict[str, Any], stage_row: dict[str, Any]) -> str:
    query = str(source.get("query") or "").strip()
    output_query = str(stage_row.get("query") or "").strip()
    if output_query and output_query != query:
        raise ValueError(f"{source['source_id']}: query changed in {stage_row.get('stage')}")
    if not query:
        raise ValueError(f"{source['source_id']}: empty source query")
    return query


def contexts_from_stage(source: dict[str, Any], stage_row: dict[str, Any]) -> list[str]:
    raw_contexts = stage_row.get("contexts")
    if raw_contexts is None:
        raw_contexts = source.get("contexts") or []
    if not isinstance(raw_contexts, list):
        raise ValueError(f"{source['source_id']}: contexts must be a list")
    contexts = [str(item).strip() for item in raw_contexts if str(item).strip()]
    if stage_row.get("stage") == "evidence_governance" and not contexts:
        raise ValueError(f"{source['source_id']}: evidence_governance row has no contexts")
    return contexts


def materialize_stage(
    *,
    source: dict[str, Any],
    stage_row: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    labels = source["labels"]
    source_id = str(source["source_id"])
    stage = str(stage_row["stage"])
    query = ensure_query(source, stage_row)
    contexts = [] if stage == "query_planning" else contexts_from_stage(source, stage_row)

    row = {
        "id": f"{source_id}::sage::{stage}",
        "source_id": source_id,
        "split": source["source_split"],
        "stage": stage,
        "version": "fitz-gov-sage-v1",
        "dataset_version": "sage_v1",
        "source_dataset_version": source.get("source_dataset_version"),
        "query": query,
        "query_rewritten": source.get("query_rewritten"),
        "query_text": build_query_contract_text(query),
        "contexts": contexts,
        "context_features": source.get("context_features") or [],
        "text": build_encoder_text(query, contexts),
        "route": labels.get("route"),
        "route_id": label_id(metadata["route2id"], labels.get("route"), field="route", source_id=source_id),
        "query_contract": labels.get("query_contract"),
        "query_contract_id": label_id(
            metadata["query_contract2id"],
            labels.get("query_contract"),
            field="query_contract",
            source_id=source_id,
        ),
        "retrieval_action": labels.get("retrieval_action"),
        "retrieval_action_id": label_id(
            metadata["retrieval_action2id"],
            labels.get("retrieval_action"),
            field="retrieval_action",
            source_id=source_id,
        ),
        "gap_type": labels.get("gap_type"),
        "gap_type_id": label_id(metadata["gap_type2id"], labels.get("gap_type"), field="gap_type", source_id=source_id),
        "answerability_shape": labels.get("answerability_shape"),
        "answerability_shape_id": label_id(
            metadata["answerability_shape2id"],
            labels.get("answerability_shape"),
            field="answerability_shape",
            source_id=source_id,
        ),
        "retrieval_modality": labels.get("retrieval_modality"),
        "retrieval_modality_id": label_id(
            metadata["retrieval_modality2id"],
            labels.get("retrieval_modality"),
            field="retrieval_modality",
            source_id=source_id,
        ),
        "retrieval_obligation": labels.get("retrieval_obligation"),
        "retrieval_obligation_id": label_id(
            metadata["retrieval_obligation2id"],
            labels.get("retrieval_obligation"),
            field="retrieval_obligation",
            source_id=source_id,
        ),
        "alpha_source": "fitz_gov_sage_v1_from_v10_clean_source",
        "sage_metadata": stage_row.get("sage_metadata") or {},
        "pack_metadata": stage_row.get("pack_metadata") or {},
    }

    if stage == "query_planning":
        row.update(
            {
                "label": None,
                "label_id": -1,
                "taxonomy_pattern": None,
                "taxonomy_pattern_id": -1,
                "scalar_targets": {},
            }
        )
    else:
        row.update(
            {
                "label": labels.get("label"),
                "label_id": int(labels.get("label_id", -1)),
                "taxonomy_pattern": labels.get("taxonomy_pattern"),
                "taxonomy_pattern_id": label_id(
                    metadata["taxonomy_pattern2id"],
                    labels.get("taxonomy_pattern"),
                    field="taxonomy_pattern",
                    source_id=source_id,
                ),
                "scalar_targets": source.get("scalar_targets") or {},
            }
        )
    return row


def summarize(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    return {
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "stage_counts": dict(Counter(str(row.get("stage")) for row in all_rows)),
        "label_counts": dict(Counter(str(row.get("label")) for row in all_rows if row.get("label"))),
        "source_rows": len({str(row.get("source_id")) for row in all_rows}),
        "rows": len(all_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workpack-dir", type=Path, default=DEFAULT_WORKPACK_DIR)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_WORKPACK_DIR / "subagent_outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    workpack_dir = args.workpack_dir.resolve()
    source_data_dir = args.source_data_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    source_selection = load_source_selection(workpack_dir / "source_selection.jsonl")
    source_metadata = read_json(source_data_dir / "metadata.json")
    stage_outputs = load_stage_outputs(input_dir)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": [], "test": []}
    missing_sources: list[str] = []
    for source_id, source in source_selection.items():
        stage_group = stage_outputs.get(source_id)
        if not stage_group:
            missing_sources.append(source_id)
            if args.allow_partial:
                continue
            raise ValueError(f"{source_id}: no subagent output rows")
        missing_stages = sorted(set(STAGES) - set(stage_group))
        if missing_stages:
            if args.allow_partial:
                continue
            raise ValueError(f"{source_id}: missing stages {missing_stages}")
        for stage in STAGES:
            row = materialize_stage(
                source=source,
                stage_row=stage_group[stage],
                metadata=source_metadata,
            )
            rows_by_split[str(source["source_split"])].append(row)

    total_rows = sum(len(rows) for rows in rows_by_split.values())
    if total_rows == 0:
        raise ValueError("no sage rows materialized")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    out_metadata = {
        "source": {
            "source_data_dir": str(source_data_dir),
            "workpack_dir": str(workpack_dir),
            "input_dir": str(input_dir),
            "fitz_gov_source": "v10.0.0_clean_g5_1_repaired_prep",
            "excluded_versions": ["v11", "v12", "v12.1", "v12.2"],
        },
        **summarize(rows_by_split),
        "missing_sources": len(missing_sources),
        "route2id": source_metadata["route2id"],
        "taxonomy_pattern2id": source_metadata["taxonomy_pattern2id"],
        "query_contract2id": source_metadata["query_contract2id"],
        "retrieval_action2id": source_metadata["retrieval_action2id"],
        "gap_type2id": source_metadata["gap_type2id"],
        "answerability_shape2id": source_metadata["answerability_shape2id"],
        "answerability_shape_collapse_map": source_metadata.get("answerability_shape_collapse_map", {}),
        "retrieval_modality2id": source_metadata["retrieval_modality2id"],
        "retrieval_obligation2id": source_metadata["retrieval_obligation2id"],
        "require_query_contract": True,
        "require_retrieval_control": True,
        "stage_aware": True,
        "scalar_fields": source_metadata["scalar_fields"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(out_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        output_dir,
        args.config,
        seed=0,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=Path.cwd().parent / "fitz-gov",
        start_time=start,
        extra={
            "script": "materialize_fitz_gov_sage_data.py",
            "source_data_dir": str(source_data_dir),
            "workpack_dir": str(workpack_dir),
            "input_dir": str(input_dir),
            "stage_aware": True,
        },
    )

    print(f"output      : {output_dir}")
    print(f"rows        : {total_rows:,}")
    print(f"splits      : {out_metadata['splits']}")
    print(f"stage counts: {out_metadata['stage_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
