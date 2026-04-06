from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from csrc_rag.utils import read_jsonl


@dataclass(frozen=True)
class SplitDataset:
    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime(1900, 1, 1)
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return datetime(1900, 1, 1)


def load_party_samples(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: (_parse_date(row.get("declare_date")), row.get("sample_id")))
    return rows


def time_split(rows: list[dict[str, Any]], train_ratio: float = 0.7, valid_ratio: float = 0.15) -> SplitDataset:
    n = len(rows)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return SplitDataset(train=rows[:train_end], valid=rows[train_end:valid_end], test=rows[valid_end:])


def build_label_vocab(rows: list[dict[str, Any]]) -> list[str]:
    labels = sorted({label for row in rows for label in row.get("labels", [])})
    return labels


def encode_multilabel(rows: list[dict[str, Any]], vocab: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(vocab)}
    target = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for label in row.get("labels", []):
            target[row_idx, label_to_idx[label]] = 1.0
    return target

