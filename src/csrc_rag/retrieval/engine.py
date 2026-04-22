from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from csrc_rag.orchestration.intent_model import load_intent_classifier
from csrc_rag.orchestration.intents import IntentDecision, IntentSpec, load_registry, route_query
from csrc_rag.orchestration.topic_guard import is_out_of_scope
from csrc_rag.retrieval.bm25 import BM25Index
from csrc_rag.retrieval.dense import (
    BgeZhDenseEncoder,
    NumpyEmbeddingIndex,
    SentenceTransformerDenseEncoder,
    SvdTfidfDenseEncoder,
)
from csrc_rag.retrieval.hybrid import reciprocal_rank_fusion
from csrc_rag.retrieval.query_builder import QueryPlan, build_query_plan
from csrc_rag.retrieval.reranker import RerankConfig, Reranker, RrfCandidate
from csrc_rag.response.responder import build_responder
from csrc_rag.settings import CONFIG_DIR, PROCESSED_DIR, PROJECT_ROOT
from csrc_rag.utils import read_json


# ---------------------------------------------------------------------------
# Planner v2 early-exit labels (see docs/strategies/01-reject-strategy.md §4)
# ---------------------------------------------------------------------------
#
# The v2 classifier predicts 7 classes; 3 of them (greeting / chitchat /
# out_of_scope) must bypass retrieval and respond from a fixed template, per
# the L1 fallback tier of the reject strategy.  The legacy ``route_query`` +
# ``intents.json`` only know the 4 "productive" labels; rather than expanding
# that registry we intercept the v2-only labels here and short-circuit the
# pipeline with canned responses from ``prompts/planner/fallback_responses.md``.
_PLANNER_V2_REJECT_LABELS: frozenset[str] = frozenset(
    {"greeting", "chitchat", "out_of_scope"}
)

_PLANNER_V2_FALLBACK_MESSAGES: dict[str, str] = {
    "greeting": (
        "你好！我是证监会违规处罚案例智能问答助手。\n"
        "我可以帮你完成四件事：\n"
        "1. 案例检索 — 类似「2023 年内幕交易被罚的案例有哪些？」\n"
        "2. 法规依据 — 类似「信息披露违规通常违反哪条法规？」\n"
        "3. 处罚推荐 — 类似「上市公司虚假陈述一般怎么罚？」\n"
        "4. 趋势分析 — 类似「近五年操纵市场案件的处罚趋势？」"
    ),
    "chitchat": (
        "这个问题有点超出我的专长 😊。\n"
        "我专注于 证监会处罚案例 的检索、法规依据、处罚分析与趋势统计。\n"
        "要不要试试：「近两年信披违规案例有哪些？」或「证券法第 197 条适用什么情形？」"
    ),
    "out_of_scope": (
        "抱歉，该问题不在本系统覆盖范围内。\n"
        "数据来源：仅限中国证监会公开处罚案例（证券 / 基金 / 期货 / 上市公司）。\n"
        "不涵盖：股价预测、个股推荐、编程问题、娱乐内容、医疗 / 法律咨询。"
    ),
}


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
        rerank_enabled: bool | None = None,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.chunk_path = chunk_path or PROCESSED_DIR / "event_chunks.jsonl"
        self.event_path = event_path or PROCESSED_DIR / "event_corpus.jsonl"
        retrieval_config = _read_json(retrieval_config_path or CONFIG_DIR / "retrieval.json")
        self.model_config = read_json(CONFIG_DIR / "models.json")
        self.registry = load_registry(intents_path)
        self.responder = build_responder()
        # Pre-load the v2 Planner classifier so the main thread is primed and
        # the early-exit branch below can peek at the prediction without
        # re-reading the pickle on every request.
        self._planner = load_intent_classifier()
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
            self.dense_encoder = self._build_dense_encoder()
            self.dense_encoder.fit(
                [row["chunk_id"] for row in self.chunks],
                [row["retrieval_text"] for row in self.chunks],
            )

        # Reranker: optional L3-tail cross-encoder. Lazy-loaded on first use.
        reranker_cfg = self.model_config.get("reranker", {}) or {}
        default_enabled = bool(reranker_cfg.get("enabled", False))
        self.rerank_enabled = default_enabled if rerank_enabled is None else rerank_enabled
        self._reranker: Reranker | None = None
        self._reranker_cfg_dict: dict[str, Any] = reranker_cfg

    # ------------------------------------------------------------------
    # Dense backend factory
    # ------------------------------------------------------------------

    def _build_dense_encoder(self):
        """Build the configured dense encoder.

        Supports:
          * ``active_backend = "bge_small_zh"`` — pre-built bge npy index.
          * legacy ``backend = "prebuilt"``   — all-MiniLM-L6-v2 npy index.
          * legacy ``backend = "svd_tfidf"``  — on-the-fly TF-IDF + SVD.
          * fallback: sentence_transformer_model on-the-fly encoding.
        """
        dense_cfg = self.model_config["dense_retrieval"]
        active_backend = dense_cfg.get("active_backend")

        if active_backend == "bge_small_zh":
            params = dense_cfg["bge_small_zh"]
            npy_path = PROJECT_ROOT / params["npy_path"]
            order_path = PROJECT_ROOT / params["order_path"]
            cache_folder_raw = params.get("model_cache_folder")
            cache_folder = (
                str(PROJECT_ROOT / cache_folder_raw) if cache_folder_raw else None
            )
            return BgeZhDenseEncoder(
                npy_path=npy_path,
                order_path=order_path,
                model_name=params.get("model_name", "BAAI/bge-small-zh-v1.5"),
                model_cache_folder=cache_folder,
                query_instruction=params.get(
                    "query_instruction",
                    "为这个句子生成表示以用于检索相关文章：",
                ),
                max_seq_length=int(params.get("max_seq_length", 512)),
            )

        if active_backend == "svd_tfidf":
            params = dense_cfg["svd_tfidf"]
            return SvdTfidfDenseEncoder(
                max_features=params["max_features"],
                ngram_range=tuple(params["ngram_range"]),
                n_components=params["n_components"],
            )

        # Legacy path for backward compatibility.
        backend = dense_cfg.get("backend", "prebuilt")
        if backend == "prebuilt":
            params = dense_cfg["prebuilt"]
            npy_path = PROJECT_ROOT / params["npy_path"]
            order_path = PROJECT_ROOT / params["order_path"]
            return NumpyEmbeddingIndex(
                npy_path=npy_path,
                order_path=order_path,
                query_model=params.get(
                    "query_model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )
        if backend == "svd_tfidf":
            params = dense_cfg["svd_tfidf"]
            return SvdTfidfDenseEncoder(
                max_features=params["max_features"],
                ngram_range=tuple(params["ngram_range"]),
                n_components=params["n_components"],
            )
        return SentenceTransformerDenseEncoder(dense_cfg["sentence_transformer_model"])

    # ------------------------------------------------------------------
    # Reranker (lazy load)
    # ------------------------------------------------------------------

    def _get_reranker(self) -> Reranker:
        if self._reranker is not None:
            return self._reranker
        cfg_dict = dict(self._reranker_cfg_dict or {})
        # RerankConfig does not know about enabled / cache_folder; strip them.
        cfg_dict.pop("enabled", None)
        cache_folder = cfg_dict.pop("model_cache_folder", None)
        if cache_folder:
            import os
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / cache_folder))
            os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / cache_folder))

        config = RerankConfig(**cfg_dict) if cfg_dict else RerankConfig()

        class _ChunkLookup:
            def __init__(self, table: dict[str, dict[str, Any]]) -> None:
                self._table = table

            def get(self, chunk_id: str) -> dict[str, Any] | None:
                return self._table.get(chunk_id)

        reranker = Reranker(config=config, chunk_lookup=_ChunkLookup(self.chunk_by_id))
        reranker.load()
        self._reranker = reranker
        return reranker

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

    def _planner_v2_early_exit(self, query: str) -> SearchResponse | None:
        """Short-circuit greeting / chitchat / out_of_scope predictions.

        The legacy 4-label ``route_query`` cannot represent these classes, so
        we consult the pre-loaded v2 classifier once at the top of the
        pipeline. If it predicts one of the reject-class labels we build a
        template ``SearchResponse`` directly and skip retrieval + responder
        entirely — this matches the L1 tier of the reject strategy.

        Returns ``None`` when the planner is unavailable or predicts a
        productive label (in which case the usual pipeline continues).
        """
        if self._planner is None:
            return None
        prediction = self._planner.predict(query)
        if prediction.name not in _PLANNER_V2_REJECT_LABELS:
            return None

        message = _PLANNER_V2_FALLBACK_MESSAGES[prediction.name]
        return SearchResponse(
            intent=prediction.name,
            intent_confidence=float(prediction.confidence),
            intent_method=prediction.method,
            intent_scores=dict(prediction.scores),
            response_backend="planner_v2_fallback",
            response_model=None,
            query_plan={"retrieval_unit": "-", "top_k": 0, "metadata_filters": {}},
            answer=message,
            events=[],
        )

    def search(
        self,
        query: str,
        forced_intent: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> SearchResponse:
        # ── 0. Planner v2 early exit (greeting / chitchat / out_of_scope) ────
        if not forced_intent:
            early = self._planner_v2_early_exit(query)
            if early is not None:
                return early

        # ── 0b. Topic guard ──────────────────────────────────────────────────
        out_of_scope, guard_reason = is_out_of_scope(query)
        if out_of_scope:
            return SearchResponse(
                intent="out_of_scope",
                intent_confidence=1.0,
                intent_method="topic_guard",
                intent_scores={},
                response_backend="topic_guard",
                response_model=None,
                query_plan={"retrieval_unit": "-", "top_k": 0, "metadata_filters": {}},
                answer=guard_reason or _PLANNER_V2_FALLBACK_MESSAGES["out_of_scope"],
                events=[],
            )

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

        # Optional cross-encoder rerank (L3 tail).
        rerank_event_order: list[str] | None = None
        if self.rerank_enabled and hits:
            try:
                reranker = self._get_reranker()
                candidates = [
                    RrfCandidate(
                        chunk_id=hit.doc_id,
                        rrf_score=float(getattr(hit, "score", 0.0)),
                        rank_before=i,
                    )
                    for i, hit in enumerate(hits, start=1)
                ]
                reranked = reranker.rerank(
                    query=query,
                    candidates=candidates,
                    intent=intent.name,
                    top_k=max(intent.top_k, 10),
                )
                if reranked:
                    rerank_event_order = [r.event_id for r in reranked]
            except Exception as exc:  # pragma: no cover - defensive
                # Rerank failures should degrade gracefully to RRF order.
                import logging
                logging.getLogger(__name__).warning(
                    "Reranker failed (%s); falling back to hybrid order.",
                    exc,
                )

        grouped_scores: dict[str, float] = defaultdict(float)
        grouped_snippets: dict[str, list[str]] = defaultdict(list)
        for hit in hits[:50]:
            chunk = self.chunk_by_id[hit.doc_id]
            event_id = chunk["event_id"]
            grouped_scores[event_id] = max(grouped_scores[event_id], hit.score)
            if len(grouped_snippets[event_id]) < 3:
                grouped_snippets[event_id].append(chunk["chunk_text"][:220])

        if rerank_event_order:
            # Re-order by the cross-encoder's event-level ranking; any events
            # missing from the rerank output keep their hybrid order as tail.
            seen: set[str] = set()
            ordered_events: list[tuple[str, float]] = []
            for rank_pos, event_id in enumerate(rerank_event_order, start=1):
                if event_id not in grouped_scores or event_id in seen:
                    continue
                # Fake a monotonically-decreasing score so the downstream
                # formatter can still sort by score if needed.
                ordered_events.append((event_id, 1.0 - rank_pos * 1e-3))
                seen.add(event_id)
            # Append anything grouped but not reranked in original hybrid order.
            for event_id, score in sorted(
                grouped_scores.items(), key=lambda it: it[1], reverse=True
            ):
                if event_id in seen:
                    continue
                ordered_events.append((event_id, score))
                seen.add(event_id)
        else:
            ordered_events = sorted(
                grouped_scores.items(), key=lambda item: item[1], reverse=True
            )

        ranked_events: list[EventResult] = []
        for event_id, score in ordered_events[: intent.top_k]:
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
