from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

