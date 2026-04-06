from __future__ import annotations


def recall_at_k(ranked_event_ids: list[str], gold_event_id: str, k: int) -> float:
    return 1.0 if gold_event_id in ranked_event_ids[:k] else 0.0


def reciprocal_rank(ranked_event_ids: list[str], gold_event_id: str) -> float:
    for idx, event_id in enumerate(ranked_event_ids, start=1):
        if event_id == gold_event_id:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_event_ids: list[str], gold_event_id: str, k: int) -> float:
    for idx, event_id in enumerate(ranked_event_ids[:k], start=1):
        if event_id == gold_event_id:
            return 1.0 / (idx.bit_length())
    return 0.0

