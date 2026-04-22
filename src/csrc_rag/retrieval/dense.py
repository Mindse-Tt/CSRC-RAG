from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class DenseHit:
    doc_id: str
    score: float


class DenseEncoder(ABC):
    @abstractmethod
    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        raise NotImplementedError


class SvdTfidfDenseEncoder(DenseEncoder):
    def __init__(self, max_features: int = 30000, ngram_range: tuple[int, int] = (1, 2), n_components: int = 256) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.n_components = n_components
        self.svd: TruncatedSVD | None = None
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.n_components, max(2, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        doc_vectors = self.svd.fit_transform(tfidf)
        self.doc_ids = list(doc_ids)
        self.doc_vectors = self._normalize(doc_vectors)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("Dense encoder is not fitted.")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        return self._normalize(vectors)

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("Dense encoder is not fitted.")
        query_vec = self.encode_queries([query])[0]
        scores = self.doc_vectors @ query_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class NumpyEmbeddingIndex(DenseEncoder):
    """Load pre-built L2-normalised float32 embeddings from a .npy file.

    The .npy row order must match the chunk_id list in chunk_id_order.json,
    which is produced by scripts/rebuild_from_hybrid_pkg.py.

    encode_query uses all-MiniLM-L6-v2 (via sentence-transformers) at search
    time so queries are embedded in the same 384-dim space as the corpus.
    """

    def __init__(
        self,
        npy_path: str | Path,
        order_path: str | Path,
        query_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.npy_path = Path(npy_path)
        self.order_path = Path(order_path)
        self.query_model = query_model
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for NumpyEmbeddingIndex query encoding."
            ) from exc
        self._encoder = SentenceTransformer(self.query_model)

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        """Load embeddings from disk; doc_ids and texts are ignored (pre-built)."""
        with self.order_path.open(encoding="utf-8") as fh:
            self.doc_ids = json.load(fh)
        self.doc_vectors = np.load(str(self.npy_path)).astype(np.float32)
        if len(self.doc_ids) != self.doc_vectors.shape[0]:
            raise ValueError(
                f"NumpyEmbeddingIndex: doc_ids length {len(self.doc_ids)} != "
                f"embeddings rows {self.doc_vectors.shape[0]}"
            )

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("NumpyEmbeddingIndex is not fitted.")
        self._ensure_encoder()
        q_vec = np.asarray(
            self._encoder.encode([query], normalize_embeddings=True)[0],  # type: ignore[union-attr]
            dtype=np.float32,
        )
        scores = self.doc_vectors @ q_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0.0:
                break
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked


class SentenceTransformerDenseEncoder(DenseEncoder):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers backend is unavailable. Install sentence-transformers, transformers, and torch first."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.doc_ids: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> None:
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        self.doc_ids = list(doc_ids)
        self.doc_vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, top_k: int, allowed_doc_ids: set[str] | None = None) -> list[DenseHit]:
        if self.doc_vectors is None:
            raise RuntimeError("Dense encoder is not fitted.")
        query_vec = np.asarray(self.model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
        scores = self.doc_vectors @ query_vec
        ranked: list[DenseHit] = []
        for idx in np.argsort(scores)[::-1]:
            doc_id = self.doc_ids[idx]
            if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                continue
            score = float(scores[idx])
            if score <= 0:
                continue
            ranked.append(DenseHit(doc_id=doc_id, score=score))
            if len(ranked) >= top_k:
                break
        return ranked

