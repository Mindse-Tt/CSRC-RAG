"""Cross-encoder reranker for the retrieval layer (L3 tail).

Strategy doc: docs/strategies/06-reranking-strategy.md

This module is a thin wrapper around a HuggingFace cross-encoder (default:
``BAAI/bge-reranker-v2-m3``). It takes the output of RRF fusion, scores each
(query, chunk) pair, applies business-level boosts (authority of the
promulgating body, severity of the punishment), and returns event-level
ranked candidates for the evidence-assembly layer.

The concrete model loading is kept lazy so that importing this module does
not require torch / transformers at dev time (tests can mock ``Reranker``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RrfCandidate:
    """A post-RRF candidate entering the reranker."""

    chunk_id: str
    rrf_score: float
    rank_before: int  # 1-based rank in the RRF output


@dataclass(frozen=True)
class RerankedCandidate:
    """Event-level output schema contracted with the evidence-assembly layer.

    See docs/strategies/06-reranking-strategy.md §2 for field semantics.
    """

    event_id: str
    rerank_score: float  # sigmoid(CE_logit) * auth_boost * severity_boost
    raw_score: float     # sigmoid(CE_logit), un-boosted
    auth_boost: float
    severity_boost: float
    rank_before: int
    rank_after: int
    top_chunk_id: str
    snippet: str


@dataclass(frozen=True)
class RerankConfig:
    """Runtime configuration for the reranker.

    Loaded from ``config/rerank.json`` (see strategy doc §10).
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    fallback_model_name: str = "BAAI/bge-reranker-base"
    max_length: int = 512
    batch_size: int = 32
    candidate_pool_max: int = 60
    final_top_k_events: int = 6
    device: str = "auto"  # "cpu" / "cuda" / "auto"
    use_fp16: bool = True
    use_onnx_int8: bool = False
    enable_auth_boost: bool = True
    enable_severity_boost: bool = True
    auth_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "证监会": 1.25,
            "证监局": 1.15,
            "交易所": 1.10,
            "协会": 1.05,
        }
    )
    severity_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "市场禁入": 1.20,
            "刑事移送": 1.20,
            "吊销": 1.20,
            "没收": 1.15,
            "罚款": 1.05,
        }
    )

    @classmethod
    def from_file(cls, path: Path) -> "RerankConfig":
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Protocols (for testability)
# ---------------------------------------------------------------------------


class ChunkLookup(Protocol):
    """Minimal view of the chunk corpus needed by the reranker."""

    def get(self, chunk_id: str) -> dict[str, Any] | None: ...


class CrossEncoderBackend(Protocol):
    """Duck-typed backend. Real impl wraps sentence_transformers.CrossEncoder."""

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        ...


# ---------------------------------------------------------------------------
# Default HuggingFace backend (lazy)
# ---------------------------------------------------------------------------


class SentenceTransformerCrossEncoder:
    """Thin adapter over ``sentence_transformers.CrossEncoder``.

    Import of sentence_transformers is deferred to ``__init__`` so the module
    can be imported in environments without torch (e.g. unit tests that mock
    the backend).
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        device: str = "auto",
        use_fp16: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        resolved_device = self._resolve_device(device)
        logger.info("Loading cross-encoder %s on %s", model_name, resolved_device)
        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=resolved_device,
        )
        if use_fp16 and resolved_device.startswith("cuda"):
            # sentence-transformers does not expose a direct fp16 switch; we
            # convert the underlying transformer parameters if available.
            try:
                self._model.model.half()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                logger.warning("FP16 conversion failed; continuing in FP32.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "cpu"

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    """Cross-encoder reranker with business-aware boosting.

    Usage::

        reranker = Reranker(config, chunk_lookup)
        reranker.load()  # lazy; skip in tests with mocked backend
        results = reranker.rerank(
            query="2023 年上市公司财务造假案",
            candidates=[RrfCandidate("E123-c01", 0.021, 1), ...],
            intent="case_retrieval",
        )
    """

    def __init__(
        self,
        config: RerankConfig,
        chunk_lookup: ChunkLookup,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        self.config = config
        self.chunk_lookup = chunk_lookup
        self._backend: CrossEncoderBackend | None = backend

    # -- lifecycle --------------------------------------------------------

    def load(self) -> None:
        """Load the default backend. Idempotent."""
        if self._backend is not None:
            return
        try:
            self._backend = SentenceTransformerCrossEncoder(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device,
                use_fp16=self.config.use_fp16,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Primary reranker %s failed to load (%s); falling back to %s",
                self.config.model_name,
                exc,
                self.config.fallback_model_name,
            )
            self._backend = SentenceTransformerCrossEncoder(
                model_name=self.config.fallback_model_name,
                max_length=self.config.max_length,
                device=self.config.device,
                use_fp16=self.config.use_fp16,
            )

    # -- public API -------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[RrfCandidate],
        intent: str = "case_retrieval",
        top_k: int | None = None,
    ) -> list[RerankedCandidate]:
        """Rerank chunk-level candidates and return event-level results.

        Args:
            query: user query (already rewritten by L2).
            candidates: output of ``reciprocal_rank_fusion``.
            intent: drives which business boosts are applied.
            top_k: override ``final_top_k_events``.

        Returns:
            event-level ranked list, deduplicated by ``event_id``.
        """
        if not candidates:
            return []
        if self._backend is None:
            raise RuntimeError("Reranker.load() must be called before rerank().")

        # 1. Truncate candidate pool.
        pool = list(candidates)[: self.config.candidate_pool_max]

        # 2. Build (query, chunk_text) pairs.
        pairs: list[tuple[str, str]] = []
        chunks: list[dict[str, Any]] = []
        for cand in pool:
            chunk = self.chunk_lookup.get(cand.chunk_id)
            if chunk is None:
                logger.debug("chunk %s not found in lookup", cand.chunk_id)
                continue
            text = chunk.get("retrieval_text") or chunk.get("chunk_text") or ""
            pairs.append((query, text))
            chunks.append(chunk)

        if not pairs:
            return []

        # 3. Cross-encoder scoring (batched).
        t0 = time.perf_counter()
        logits = self._backend.predict(pairs, batch_size=self.config.batch_size)
        logger.debug(
            "cross-encoder scored %d pairs in %.1f ms",
            len(pairs),
            (time.perf_counter() - t0) * 1000.0,
        )

        # 4. Apply business boosts and aggregate to event level.
        best_per_event: dict[str, RerankedCandidate] = {}
        for cand, chunk, logit in zip(pool[: len(pairs)], chunks, logits):
            raw_score = _sigmoid(logit)
            auth_boost = self._auth_boost(chunk)
            severity_boost = self._severity_boost(chunk, intent)
            final_score = raw_score * auth_boost * severity_boost

            event_id = chunk.get("event_id") or cand.chunk_id.split("-")[0]
            snippet = (chunk.get("chunk_text") or "")[:220]

            entry = RerankedCandidate(
                event_id=event_id,
                rerank_score=round(final_score, 6),
                raw_score=round(raw_score, 6),
                auth_boost=auth_boost,
                severity_boost=severity_boost,
                rank_before=cand.rank_before,
                rank_after=0,  # filled after sort
                top_chunk_id=cand.chunk_id,
                snippet=snippet,
            )

            prev = best_per_event.get(event_id)
            if prev is None or entry.rerank_score > prev.rerank_score:
                best_per_event[event_id] = entry

        # 5. Sort and assign final rank.
        ordered = sorted(
            best_per_event.values(),
            key=lambda e: e.rerank_score,
            reverse=True,
        )
        k = top_k or self.config.final_top_k_events
        finalized: list[RerankedCandidate] = []
        for i, entry in enumerate(ordered[:k], start=1):
            finalized.append(
                RerankedCandidate(
                    event_id=entry.event_id,
                    rerank_score=entry.rerank_score,
                    raw_score=entry.raw_score,
                    auth_boost=entry.auth_boost,
                    severity_boost=entry.severity_boost,
                    rank_before=entry.rank_before,
                    rank_after=i,
                    top_chunk_id=entry.top_chunk_id,
                    snippet=entry.snippet,
                )
            )
        return finalized

    # -- boosting helpers -------------------------------------------------

    def _auth_boost(self, chunk: dict[str, Any]) -> float:
        if not self.config.enable_auth_boost:
            return 1.0
        promulgator = str(chunk.get("promulgator") or "")
        best = 1.0
        for keyword, boost in self.config.auth_boosts.items():
            if keyword in promulgator and boost > best:
                best = boost
        return best

    def _severity_boost(self, chunk: dict[str, Any], intent: str) -> float:
        if not self.config.enable_severity_boost:
            return 1.0
        if intent != "sanction_recommendation":
            return 1.0
        # PunishmentMeasure must NOT be used as a generation-time feature,
        # but it is allowed as a ranking signal (see strategy doc §5.2).
        measure = str(chunk.get("punishment_measure") or "")
        types = " ".join(chunk.get("punishment_types") or [])
        haystack = f"{measure} {types}"
        best = 1.0
        for keyword, boost in self.config.severity_boosts.items():
            if keyword in haystack and boost > best:
                best = boost
        return best


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_reranker(
    config_path: Path | None,
    chunk_lookup: ChunkLookup,
    backend: CrossEncoderBackend | None = None,
) -> Reranker:
    """Build a :class:`Reranker` from an optional JSON config file.

    Note: does **not** call :meth:`Reranker.load`; call it explicitly in the
    engine wiring so test code can inject a mocked backend.
    """
    config = RerankConfig.from_file(config_path) if config_path else RerankConfig()
    return Reranker(config=config, chunk_lookup=chunk_lookup, backend=backend)


__all__ = [
    "RrfCandidate",
    "RerankedCandidate",
    "RerankConfig",
    "ChunkLookup",
    "CrossEncoderBackend",
    "SentenceTransformerCrossEncoder",
    "Reranker",
    "build_reranker",
]
