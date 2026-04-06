from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from csrc_rag.orchestration.intent_model import load_intent_classifier
from csrc_rag.settings import CONFIG_DIR


@dataclass(frozen=True)
class IntentSpec:
    name: str
    description: str
    retrieval_unit: str
    top_k: int
    response_sections: list[str]


@dataclass(frozen=True)
class IntentDecision:
    spec: IntentSpec
    confidence: float
    method: str
    scores: dict[str, float]


def load_registry(path: str | Path | None = None) -> dict[str, IntentSpec]:
    config_path = Path(path) if path else CONFIG_DIR / "intents.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        name: IntentSpec(
            name=name,
            description=config["description"],
            retrieval_unit=config["retrieval_unit"],
            top_k=config["top_k"],
            response_sections=config["response_sections"],
        )
        for name, config in payload.items()
    }


@lru_cache(maxsize=1)
def _load_router():
    return load_intent_classifier()


def _heuristic_route(query: str, registry: dict[str, IntentSpec]) -> IntentDecision:
    if any(token in query for token in ["处罚", "建议", "推荐", "罚款", "市场禁入"]):
        spec = registry["sanction_recommendation"]
        return IntentDecision(spec=spec, confidence=0.92, method="heuristic", scores={spec.name: 0.92})
    if any(token in query for token in ["法条", "法规", "依据", "违反"]):
        spec = registry["law_grounding"]
        return IntentDecision(spec=spec, confidence=0.9, method="heuristic", scores={spec.name: 0.9})
    if any(token in query for token in ["趋势", "统计", "分布", "近年", "变化"]):
        spec = registry["trend_analysis"]
        return IntentDecision(spec=spec, confidence=0.88, method="heuristic", scores={spec.name: 0.88})
    spec = registry["case_retrieval"]
    return IntentDecision(spec=spec, confidence=0.75, method="heuristic", scores={spec.name: 0.75})


def route_query(query: str, registry: dict[str, IntentSpec]) -> IntentDecision:
    router = _load_router()
    if router is not None:
        prediction = router.predict(query)
        if prediction.name in registry:
            return IntentDecision(
                spec=registry[prediction.name],
                confidence=prediction.confidence,
                method=prediction.method,
                scores=prediction.scores,
            )
    return _heuristic_route(query, registry)
