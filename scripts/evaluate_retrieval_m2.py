"""M2 retrieval ablation: compare 4 retrieval backends on the sanity set.

Computes Recall@5 / MRR / nDCG@10 for:

    1. BM25-only
    2. Dense-only (bge-small-zh-v1.5)
    3. Hybrid (BM25 + Dense + RRF)
    4. Hybrid + Reranker (bge-reranker-v2-m3)

The sanity set is the same one used by ``scripts/evaluate_retrieval_sanity.py``:
for each event we use ``event.activity`` as the query and the same ``event_id``
as the single gold label. This is an auto-bootstrap evaluation (the source
event always "knows" its own evidence chunks), so absolute numbers are
optimistic; what matters is the *relative* improvement between backends.

Usage
-----
    python scripts/evaluate_retrieval_m2.py \
        --limit 200 \
        --output docs/reports/m2_retrieval_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Route HuggingFace through the mirror and point cache to artifacts/models.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_CACHE = str(PROJECT_ROOT / "artifacts" / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _CACHE)
os.environ.setdefault("HF_HOME", _CACHE)
os.environ.setdefault("HF_HUB_CACHE", _CACHE)

from csrc_rag.evaluation.retrieval_metrics import (  # noqa: E402
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from csrc_rag.response.responder import TemplateResponder  # noqa: E402
from csrc_rag.retrieval.engine import RetrievalEngine  # noqa: E402
from csrc_rag.settings import PROCESSED_DIR  # noqa: E402

LOGGER = logging.getLogger("evaluate_retrieval_m2")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_query(event: dict) -> str | None:
    activity = event.get("activity")
    if not activity:
        return None
    return activity[:160]


def evaluate(
    engine: RetrievalEngine,
    events: list[dict],
    *,
    label: str,
) -> dict:
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    tested = 0
    t0 = time.perf_counter()
    for i, event in enumerate(events):
        query = build_query(event)
        if not query:
            continue
        try:
            response = engine.search(query, forced_intent="case_retrieval")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[%s] query %d failed: %s", label, i, exc)
            continue
        ranked_event_ids = [item["event_id"] for item in response.events]
        gold = event["event_id"]
        recalls.append(recall_at_k(ranked_event_ids, gold, 5))
        mrrs.append(reciprocal_rank(ranked_event_ids, gold))
        ndcgs.append(ndcg_at_k(ranked_event_ids, gold, 10))
        tested += 1

    elapsed = time.perf_counter() - t0
    result = {
        "label": label,
        "tested_queries": tested,
        "Recall@5": round(mean(recalls), 4) if recalls else 0.0,
        "MRR": round(mean(mrrs), 4) if mrrs else 0.0,
        "nDCG@10": round(mean(ndcgs), 4) if ndcgs else 0.0,
        "latency_total_s": round(elapsed, 2),
        "latency_per_query_ms": round(elapsed * 1000.0 / max(tested, 1), 1),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 retrieval ablation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reports" / "m2_retrieval_eval.json",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["bm25", "dense", "hybrid", "rerank"],
        help="Skip specific conditions (useful for iterative debugging).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    events = load_jsonl(PROCESSED_DIR / "event_corpus.jsonl")[: args.limit]
    LOGGER.info("Loaded %d events for evaluation", len(events))

    results: list[dict] = []

    # 1. BM25-only
    if "bm25" not in args.skip:
        LOGGER.info("### 1/4 BM25-only")
        engine = RetrievalEngine(retrieval_mode="bm25", rerank_enabled=False)
        engine.responder = TemplateResponder()
        results.append(evaluate(engine, events, label="bm25"))
        LOGGER.info("BM25 result: %s", results[-1])
        del engine

    # 2. Dense-only (bge-small-zh)
    if "dense" not in args.skip:
        LOGGER.info("### 2/4 Dense-only (bge-small-zh)")
        engine = RetrievalEngine(retrieval_mode="dense", rerank_enabled=False)
        engine.responder = TemplateResponder()
        results.append(evaluate(engine, events, label="dense_bge"))
        LOGGER.info("Dense result: %s", results[-1])
        del engine

    # 3. Hybrid (BM25 + Dense + RRF)
    hybrid_engine = None
    if "hybrid" not in args.skip or "rerank" not in args.skip:
        hybrid_engine = RetrievalEngine(
            retrieval_mode="hybrid", rerank_enabled=False
        )
        hybrid_engine.responder = TemplateResponder()

    if "hybrid" not in args.skip and hybrid_engine is not None:
        LOGGER.info("### 3/4 Hybrid (RRF)")
        results.append(evaluate(hybrid_engine, events, label="hybrid_rrf"))
        LOGGER.info("Hybrid result: %s", results[-1])

    # 4. Hybrid + Reranker
    if "rerank" not in args.skip and hybrid_engine is not None:
        LOGGER.info("### 4/4 Hybrid + Reranker (bge-reranker-v2-m3)")
        hybrid_engine.rerank_enabled = True
        # Pre-load the reranker once.
        hybrid_engine._get_reranker()
        results.append(evaluate(hybrid_engine, events, label="hybrid_rerank"))
        LOGGER.info("Rerank result: %s", results[-1])

    output = {
        "config": {
            "limit": args.limit,
            "dataset": "sanity_self_bootstrap (activity → event_id)",
            "skip": args.skip,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", args.output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
