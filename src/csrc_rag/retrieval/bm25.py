from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from csrc_rag.retrieval.tokenizer import tokenize


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[tuple[str, str]] = []
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_len = 0.0

    def fit(self, documents: list[tuple[str, str]]) -> None:
        self.documents = documents
        total_length = 0
        for doc_id, text in documents:
            tokens = tokenize(text)
            term_freqs = Counter(tokens)
            self.doc_term_freqs[doc_id] = term_freqs
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            for token in term_freqs:
                self.doc_freqs[token] += 1
        self.avg_doc_len = total_length / max(len(documents), 1)

    def idf(self, token: str) -> float:
        doc_count = len(self.documents)
        freq = self.doc_freqs.get(token, 0)
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, allowed_doc_ids: set[str] | None = None) -> list[BM25Hit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[str, float] = {}
        for doc_id, _ in self.documents:
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = 0.0
            doc_len = self.doc_lengths.get(doc_id, 0)
            term_freqs = self.doc_term_freqs.get(doc_id, Counter())
            for token in query_tokens:
                tf = term_freqs.get(token, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_len, 1e-8))
                score += self.idf(token) * numerator / max(denominator, 1e-8)
            if score > 0:
                scores[doc_id] = score

        return [BM25Hit(doc_id=doc_id, score=score) for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

