"""Dataset utilities for pyrrho-MoE Stage 0 prototypes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def stable_token_id(token: str, vocab_size: int) -> int:
    """Stable token hash in `[1, vocab_size)`, reserving 0 for padding."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return 1 + (value % (vocab_size - 1))


def hash_tokenize(text: str, vocab_size: int, max_length: int) -> list[int]:
    """Cheap deterministic tokenizer for Stage 0 route/loss plumbing."""
    tokens = TOKEN_RE.findall((text or "").lower())
    ids = [stable_token_id(tok, vocab_size) for tok in tokens[:max_length]]
    return ids or [0]


def load_teacher_logits(path: Path) -> dict[str, list[float]]:
    """Load governance-teacher logits keyed by row id."""
    if not path.exists():
        raise FileNotFoundError(f"teacher logits sidecar not found: {path}")
    out: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            logits = row.get("logits") or row.get("teacher_logits")
            if not isinstance(logits, list) or len(logits) != 3:
                raise ValueError(f"invalid teacher logits row for id={row.get('id')!r}")
            out[str(row["id"])] = [float(v) for v in logits]
    return out


@dataclass(frozen=True)
class MoEVocab:
    route2id: dict[str, int]
    taxonomy_pattern2id: dict[str, int]
    scalar_fields: tuple[str, ...]

    @classmethod
    def from_metadata(cls, path: Path) -> MoEVocab:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            route2id={str(k): int(v) for k, v in raw["route2id"].items()},
            taxonomy_pattern2id={
                str(k): int(v) for k, v in raw["taxonomy_pattern2id"].items()
            },
            scalar_fields=tuple(str(v) for v in raw.get("scalar_fields", [])),
        )


class MoEJsonlDataset(Dataset):
    """JSONL dataset written by `scripts/prepare_moe_data.py`."""

    def __init__(
        self,
        path: Path,
        *,
        vocab: MoEVocab,
        token_vocab_size: int,
        max_length: int,
        max_query_length: int = 96,
        max_sources: int = 8,
        max_source_length: int = 192,
        teacher_logits_path: Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.vocab = vocab
        self.token_vocab_size = int(token_vocab_size)
        self.max_length = int(max_length)
        self.max_query_length = int(max_query_length)
        self.max_sources = int(max_sources)
        self.max_source_length = int(max_source_length)
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                self.rows.append(json.loads(raw))
                if limit is not None and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise ValueError(f"no rows loaded from {self.path}")
        if teacher_logits_path is not None:
            teacher_logits = load_teacher_logits(teacher_logits_path)
            for row in self.rows:
                logits = teacher_logits.get(str(row["id"]))
                if logits is not None:
                    row["teacher_logits"] = logits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        scalar_targets = row.get("scalar_targets") or {}
        scalar_values = []
        scalar_mask = []
        for field in self.vocab.scalar_fields:
            value = scalar_targets.get(field)
            if isinstance(value, int | float):
                scalar_values.append(float(value))
                scalar_mask.append(1.0)
            else:
                scalar_values.append(0.0)
                scalar_mask.append(0.0)
        contexts = row.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = []
        source_input_ids = [
            hash_tokenize(str(context), self.token_vocab_size, self.max_source_length)
            for context in contexts[: self.max_sources]
        ]
        source_valid_mask = [1.0] * len(source_input_ids)
        while len(source_input_ids) < self.max_sources:
            source_input_ids.append([0])
            source_valid_mask.append(0.0)
        return {
            "id": row["id"],
            "input_ids": hash_tokenize(row["text"], self.token_vocab_size, self.max_length),
            "query_input_ids": hash_tokenize(
                str(row.get("query") or ""),
                self.token_vocab_size,
                self.max_query_length,
            ),
            "source_input_ids": source_input_ids,
            "source_valid_mask": source_valid_mask,
            "label_id": int(row["label_id"]),
            "route_id": int(row["route_id"]),
            "taxonomy_pattern_id": int(row["taxonomy_pattern_id"]),
            "scalar_targets": scalar_values,
            "scalar_mask": scalar_mask,
            "teacher_logits": row.get("teacher_logits"),
        }


def collate_moe_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in rows)
    max_query_len = max(len(row.get("query_input_ids", [0])) for row in rows)
    max_source_count = max(len(row.get("source_input_ids", [])) for row in rows)
    max_source_count = max(max_source_count, 1)
    max_source_len = 1
    for row in rows:
        for source_ids in row.get("source_input_ids", [])[:max_source_count]:
            max_source_len = max(max_source_len, len(source_ids))
    input_ids = []
    attention_mask = []
    query_input_ids = []
    query_attention_mask = []
    source_input_ids = []
    source_attention_mask = []
    source_valid_mask = []
    teacher_values = []
    teacher_masks = []
    for row in rows:
        ids = row["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)

        query_ids = row.get("query_input_ids", [0])
        query_pad = max_query_len - len(query_ids)
        query_input_ids.append(query_ids + [0] * query_pad)
        query_attention_mask.append([1] * len(query_ids) + [0] * query_pad)

        row_sources = list(row.get("source_input_ids", []))[:max_source_count]
        row_valid = list(row.get("source_valid_mask", []))[:max_source_count]
        while len(row_sources) < max_source_count:
            row_sources.append([0])
            row_valid.append(0.0)
        padded_sources = []
        padded_source_masks = []
        for source_ids in row_sources:
            source_pad = max_source_len - len(source_ids)
            padded_sources.append(source_ids + [0] * source_pad)
            padded_source_masks.append([1] * len(source_ids) + [0] * source_pad)
        source_input_ids.append(padded_sources)
        source_attention_mask.append(padded_source_masks)
        source_valid_mask.append(row_valid)

        teacher_logits = row.get("teacher_logits")
        if isinstance(teacher_logits, list) and len(teacher_logits) == 3:
            teacher_values.append([float(v) for v in teacher_logits])
            teacher_masks.append(1.0)
        else:
            teacher_values.append([0.0, 0.0, 0.0])
            teacher_masks.append(0.0)

    return {
        "ids": [row["id"] for row in rows],
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        "query_input_ids": torch.tensor(query_input_ids, dtype=torch.long),
        "query_attention_mask": torch.tensor(query_attention_mask, dtype=torch.float32),
        "source_input_ids": torch.tensor(source_input_ids, dtype=torch.long),
        "source_attention_mask": torch.tensor(source_attention_mask, dtype=torch.float32),
        "source_valid_mask": torch.tensor(source_valid_mask, dtype=torch.float32),
        "labels": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
        "route_ids": torch.tensor([row["route_id"] for row in rows], dtype=torch.long),
        "taxonomy_ids": torch.tensor(
            [row["taxonomy_pattern_id"] for row in rows], dtype=torch.long
        ),
        "scalar_targets": torch.tensor(
            [row["scalar_targets"] for row in rows], dtype=torch.float32
        ),
        "scalar_mask": torch.tensor([row["scalar_mask"] for row in rows], dtype=torch.float32),
        "teacher_logits": torch.tensor(teacher_values, dtype=torch.float32),
        "teacher_mask": torch.tensor(teacher_masks, dtype=torch.float32),
    }
