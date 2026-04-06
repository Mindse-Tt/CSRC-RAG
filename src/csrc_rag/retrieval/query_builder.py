from __future__ import annotations

import re
from dataclasses import dataclass

from csrc_rag.orchestration.intents import IntentSpec


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    retrieval_unit: str
    top_k: int
    metadata_filters: dict[str, str]
    query_text: str


YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def build_query_plan(query: str, intent: IntentSpec) -> QueryPlan:
    filters: dict[str, str] = {}
    year_match = YEAR_PATTERN.search(query)
    if year_match:
        filters["year"] = year_match.group(0)
    if "上市公司" in query:
        filters["is_listed_company"] = "1"
    if "证监会" in query:
        filters["regulator_hint"] = "证监会"
    return QueryPlan(
        intent=intent.name,
        retrieval_unit=intent.retrieval_unit,
        top_k=intent.top_k,
        metadata_filters=filters,
        query_text=query,
    )

