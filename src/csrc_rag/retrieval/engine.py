from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from csrc_rag.orchestration.intents import IntentDecision, IntentSpec, load_registry, route_query
from csrc_rag.retrieval.bm25 import BM25Index
from csrc_rag.retrieval.dense import SentenceTransformerDenseEncoder, SvdTfidfDenseEncoder
from csrc_rag.retrieval.hybrid import reciprocal_rank_fusion
from csrc_rag.retrieval.query_builder import QueryPlan, build_query_plan
from csrc_rag.response.responder import build_responder
from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR
from csrc_rag.utils import read_json


@dataclass(frozen=True)
class EventResult:
    event_id: str
    title: str | None
    score: float
    declare_date: str | None
    promulgator: str | None
    punishment_types: list[str]
    snippets: list[str]
    laws: list[str]


@dataclass(frozen=True)
class SearchResponse:
    intent: str
    intent_confidence: float
    intent_method: str
    intent_scores: dict[str, float]
    response_backend: str
    response_model: str | None
    query_plan: dict[str, Any]
    answer: str
    events: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class RetrievalEngine:
    def __init__(
        self,
        chunk_path: Path | None = None,
        event_path: Path | None = None,
        retrieval_config_path: Path | None = None,
        intents_path: Path | None = None,
        retrieval_mode: str = "bm25",
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.chunk_path = chunk_path or PROCESSED_DIR / "event_chunks.jsonl"
        self.event_path = event_path or PROCESSED_DIR / "event_corpus.jsonl"
        retrieval_config = _read_json(retrieval_config_path or CONFIG_DIR / "retrieval.json")
        self.model_config = read_json(CONFIG_DIR / "models.json")
        self.registry = load_registry(intents_path)
        self.responder = build_responder()
        self.event_docs = {row["event_id"]: row for row in _load_jsonl(self.event_path)}
        self.chunks = _load_jsonl(self.chunk_path)
        self.chunk_by_id = {row["chunk_id"]: row for row in self.chunks}
        self.index = BM25Index(
            k1=retrieval_config["bm25"]["k1"],
            b=retrieval_config["bm25"]["b"],
        )
        self.index.fit([(row["chunk_id"], row["retrieval_text"]) for row in self.chunks])
        self.retrieval_config = retrieval_config
        self.dense_encoder = None
        if self.retrieval_mode in {"dense", "hybrid"}:
            dense_cfg = self.model_config["dense_retrieval"]
            backend = dense_cfg["backend"]
            if backend == "svd_tfidf":
                params = dense_cfg["svd_tfidf"]
                self.dense_encoder = SvdTfidfDenseEncoder(
                    max_features=params["max_features"],
                    ngram_range=tuple(params["ngram_range"]),
                    n_components=params["n_components"],
                )
            else:
                self.dense_encoder = SentenceTransformerDenseEncoder(dense_cfg["sentence_transformer_model"])
            self.dense_encoder.fit(
                [row["chunk_id"] for row in self.chunks],
                [row["retrieval_text"] for row in self.chunks],
            )

    def _allowed_doc_ids(self, query_plan: QueryPlan) -> set[str] | None:
        filters = query_plan.metadata_filters
        if not filters:
            return None
        allowed: set[str] = set()
        for chunk in self.chunks:
            year = chunk.get("year")
            listed = chunk.get("is_listed_company")
            promulgator = chunk.get("promulgator") or ""
            if "year" in filters and year != filters["year"]:
                continue
            if "is_listed_company" in filters and listed != filters["is_listed_company"]:
                continue
            if "regulator_hint" in filters and filters["regulator_hint"] not in promulgator:
                continue
            allowed.add(chunk["chunk_id"])
        return allowed

    def search(
        self,
        query: str,
        forced_intent: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> SearchResponse:
        if forced_intent:
            intent_decision = IntentDecision(
                spec=self.registry[forced_intent],
                confidence=1.0,
                method="forced",
                scores={forced_intent: 1.0},
            )
        else:
            intent_decision = route_query(query, self.registry)
        intent = intent_decision.spec
        query_plan = build_query_plan(query, intent)
        allowed = self._allowed_doc_ids(query_plan)
        hits = self._search_chunks(query, allowed_doc_ids=allowed)

        grouped_scores: dict[str, float] = defaultdict(float)
        grouped_snippets: dict[str, list[str]] = defaultdict(list)
        for hit in hits[:50]:
            chunk = self.chunk_by_id[hit.doc_id]
            event_id = chunk["event_id"]
            grouped_scores[event_id] = max(grouped_scores[event_id], hit.score)
            if len(grouped_snippets[event_id]) < 3:
                grouped_snippets[event_id].append(chunk["chunk_text"][:220])

        ranked_events: list[EventResult] = []
        for event_id, score in sorted(grouped_scores.items(), key=lambda item: item[1], reverse=True)[: intent.top_k]:
            event = self.event_docs[event_id]
            ranked_events.append(
                EventResult(
                    event_id=event_id,
                    title=event.get("title"),
                    score=round(score, 4),
                    declare_date=event.get("declare_date"),
                    promulgator=event.get("promulgator"),
                    punishment_types=event.get("punishment_types", []),
                    snippets=grouped_snippets[event_id],
                    laws=[event.get("law")] if event.get("law") else [],
                )
            )

        response_output = self.responder.generate(
            query=query,
            intent=intent,
            ranked_events=ranked_events,
            history=history,
        )
        return SearchResponse(
            intent=intent.name,
            intent_confidence=intent_decision.confidence,
            intent_method=intent_decision.method,
            intent_scores=intent_decision.scores,
            response_backend=response_output.backend,
            response_model=response_output.model_name,
            query_plan=asdict(query_plan),
            answer=response_output.text,
            events=[asdict(event) for event in ranked_events],
        )

    def _search_chunks(self, query: str, allowed_doc_ids: set[str] | None) -> list:
        candidate_pool = self.model_config["hybrid_retrieval"]["candidate_pool"]
        if self.retrieval_mode == "bm25":
            return self.index.score(query, allowed_doc_ids=allowed_doc_ids)
        if self.retrieval_mode == "dense":
            return self.dense_encoder.search(query, top_k=candidate_pool, allowed_doc_ids=allowed_doc_ids)
        if self.retrieval_mode == "hybrid":
            bm25_hits = self.index.score(query, allowed_doc_ids=allowed_doc_ids)[:candidate_pool]
            dense_hits = self.dense_encoder.search(query, top_k=candidate_pool, allowed_doc_ids=allowed_doc_ids)
            fused = reciprocal_rank_fusion(
                [
                    [(hit.doc_id, hit.score) for hit in bm25_hits],
                    [(hit.doc_id, hit.score) for hit in dense_hits],
                ],
                top_k=candidate_pool,
                rrf_k=self.model_config["hybrid_retrieval"]["rrf_k"],
            )
            return [type("HybridHit", (), {"doc_id": doc_id, "score": score}) for doc_id, score in fused]
        raise ValueError(f"Unsupported retrieval_mode: {self.retrieval_mode}")
