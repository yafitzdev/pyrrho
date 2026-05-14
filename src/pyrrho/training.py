"""Training utilities shared between train_encoder.py and eval.py."""

from __future__ import annotations

import random

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, set_seed


def set_all_seeds(seed: int) -> None:
    """Seed Python, numpy, torch (CPU + CUDA), and transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def tokenize_dataset(
    ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> Dataset:
    """Tokenize the `text` column, drop everything else, and rename `label_id` → `labels`."""
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    keep = {"input_ids", "attention_mask", "label_id"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.map(_tok, batched=True, remove_columns=drop).rename_column("label_id", "labels")
