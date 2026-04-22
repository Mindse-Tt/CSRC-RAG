"""Prompt templates for the four supported intents.

StrategyAgent-G · L5 Generation layer.

This module is intentionally self-contained: it only exposes pure functions that
build a string prompt from a (query, intent, evidence) tuple. The actual LLM
call lives in `responder.py`; the L7 validation lives in `validators.py`.

Design notes
------------
* `SYSTEM_PROMPT` is shared across all intents. It encodes the "hard rules"
  (evidence-only, must-cite EventID, must-cite legal articles, fallback phrase).
* `case_retrieval` and `trend_analysis` use **zero-shot** prompts because their
  output structure is simple and the examples would waste tokens.
* `law_grounding` and `sanction_recommendation` use **four-shot** prompts
  because the citation format and disclaimer wording must be demonstrated.
* Evidence block format is fixed so that `validators.py` can parse `[EventID=...]`
  deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# System prompt (shared by every intent)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是证监会违规处罚案例的分析助手。严格遵循以下铁律：
1. 【仅凭证据】只能使用 <EVIDENCE> 块中的事实，禁止编造案例、法条、金额、日期。
2. 【必须引证】
   - 每提及一个具体案例，必须以 [EventID=xxxx] 格式标注（xxxx 为证据中的 event_id 原值）。
   - 每引用一条法规条款，必须以 [法条：《xx法》第xx条] 格式标注。
3. 【证据不足处理】当 <EVIDENCE> 不足以回答用户问题时，直接输出："证据不足，建议缩小查询范围"。
4. 【格式】使用简体中文；不输出 markdown 表情符号；每段不超过 4 行。
5. 【边界】不给出正式法律意见；处罚推荐仅供参考，必须附免责声明。
"""

# ---------------------------------------------------------------------------
# Evidence block formatting
# ---------------------------------------------------------------------------

MAX_CTX_TOKENS_DEFAULT = 1800
PER_EV_MAX_CHARS = 320
SNIPPET_CAP = 180


@dataclass(frozen=True)
class EvidenceCard:
    event_id: str
    title: str
    declare_date: str
    promulgator: str
    punishment_types: list[str]
    laws: list[str]
    snippet: str
    score: float = 0.0

    @classmethod
    def from_hit(cls, hit: Any) -> "EvidenceCard":
        return cls(
            event_id=str(getattr(hit, "event_id", "") or ""),
            title=getattr(hit, "title", "") or "",
            declare_date=getattr(hit, "declare_date", "") or "",
            promulgator=getattr(hit, "promulgator", "") or "",
            punishment_types=list(getattr(hit, "punishment_types", []) or []),
            laws=list(getattr(hit, "laws", []) or []),
            snippet=(getattr(hit, "snippets", []) or [""])[0],
            score=float(getattr(hit, "score", 0.0) or 0.0),
        )


def format_evidence_line(ev: EvidenceCard, *, compact: bool = False) -> str:
    """Render a single evidence row. Deterministic format — parsers depend on it."""
    pt = "、".join(ev.punishment_types[:3]) or "未记录"
    laws = "；".join(l for l in ev.laws[:2] if l) or "未提取"
    snippet = (ev.snippet or "").strip().replace("\n", " ")
    if compact:
        snippet = snippet[:60]
    else:
        snippet = snippet[:SNIPPET_CAP]
    title = ev.title or ev.event_id
    return (
        f"[EventID={ev.event_id}] 标题：{title} | 时间：{ev.declare_date or '未知'} | "
        f"机构：{ev.promulgator or '未知'} | 处罚类型：{pt} | 法规：{laws} | 摘要：{snippet}"
    )


def _token_est(text: str) -> int:
    """Cheap token estimate for Chinese-dominant text (1 token ~ 1.6 chars)."""
    return max(1, int(len(text) / 1.6))


def assemble_evidence_block(
    events: Iterable[Any],
    *,
    k: int = 6,
    max_ctx_tokens: int = MAX_CTX_TOKENS_DEFAULT,
    compact: bool = False,
) -> tuple[str, list[str]]:
    """Build the <EVIDENCE> block with tail-drop truncation.

    Returns (block_text, event_id_list_kept).
    """
    cards = [EvidenceCard.from_hit(e) for e in list(events)[:k]]
    lines = [format_evidence_line(c, compact=compact) for c in cards]

    # Tail-drop loop: never drop the top-1; shrink/drop low-rank ones until fit.
    while lines and _token_est("\n".join(lines)) > max_ctx_tokens:
        if len(lines) > 3:
            lines.pop()  # drop lowest-scoring tail
            cards.pop()
        else:
            # Shrink the last remaining one by switching to compact mode.
            cards[-1] = cards[-1]
            lines[-1] = format_evidence_line(cards[-1], compact=True)
            if _token_est("\n".join(lines)) > max_ctx_tokens and len(lines) > 1:
                lines.pop()
                cards.pop()
            else:
                break

    kept_ids = [c.event_id for c in cards if c.event_id]
    return "\n".join(lines), kept_ids


# ---------------------------------------------------------------------------
# History handling (multi-turn)
# ---------------------------------------------------------------------------

HISTORY_INTENTS = {"case_retrieval", "trend_analysis"}
HISTORY_MAX_TURNS = 2  # user+assistant pairs
HISTORY_MAX_CHARS = 80


def format_history(history: list[dict[str, str]] | None, intent_name: str) -> str:
    if not history or intent_name not in HISTORY_INTENTS:
        return ""
    recent = history[-(HISTORY_MAX_TURNS * 2):]
    rows = []
    for turn in recent:
        role = "用户" if turn.get("role") == "user" else "助手"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        rows.append(f"{role}：{content[:HISTORY_MAX_CHARS]}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Few-shot example banks
# ---------------------------------------------------------------------------

LAW_GROUNDING_EXAMPLES = """示例1:
Q: 内幕交易一般违反哪条法规？
Evidence: [EventID=E010] 法规：《证券法》第五十三条
A: 内幕交易行为主要违反 [法条：《证券法》第五十三条]，代表案例 [EventID=E010]。

示例2:
Q: 信息披露违规的处罚依据？
Evidence: [EventID=E020] 法规：《证券法》第一百九十七条
A: 信息披露违规的常见依据是 [法条：《证券法》第一百九十七条]，代表 [EventID=E020]。

示例3:
Q: 操纵市场引用什么法条？
Evidence: [EventID=E033] 法规：《证券法》第五十五条
A: 操纵证券市场行为违反 [法条：《证券法》第五十五条]，参见 [EventID=E033]。

示例4:
Q: 证据不匹配的示例
Evidence: （无匹配法条）
A: 证据不足，建议缩小查询范围。
"""

SANCTION_EXAMPLES = """示例1:
Q: 上市公司虚假陈述一般怎么罚？
Evidence: 5 个相似案例，处罚方式分布：警告+罚款 3、没收违法所得 1、市场禁入 1
A: 推荐处罚方式：
1) 警告+罚款（占 60%），代表 [EventID=E101]、[EventID=E102]。
2) 没收违法所得（占 20%），代表 [EventID=E103]。
法规依据：[法条：《证券法》第一百九十七条]。
⚠ 仅基于历史案例统计，不构成正式执法意见。

示例2:
Q: 操纵证券市场通常如何处罚？
Evidence: 处罚方式分布：没收违法所得 4、罚款 3、市场禁入 2
A: 推荐：
1) 没收违法所得+罚款（占 70%），代表 [EventID=E201]、[EventID=E202]。
2) 市场禁入（占 20%），代表 [EventID=E203]。
法规依据：[法条：《证券法》第一百九十二条]。
⚠ 仅供参考。

示例3:
Q: 从业人员违规买卖股票如何处理？
Evidence: 警告+罚款 5 例
A: 主要处罚：警告+罚款（占 100%），代表 [EventID=E301]、[EventID=E302]。
法规依据：[法条：《证券法》第五十四条]。
⚠ 仅供参考。

示例4:
Q: 证据不足的示例
Evidence: 仅 1 条模糊案例
A: 证据不足，建议缩小查询范围。
"""

# ---------------------------------------------------------------------------
# Intent-specific prompt builders
# ---------------------------------------------------------------------------

def _wrap_sections(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def build_case_retrieval_prompt(
    query: str, evidence_block: str, history_block: str = ""
) -> str:
    task = (
        "<TASK>\n任务：案例检索。用户在问“类似的违规案例”，请从证据中筛选最相关的 3–5 条案例，"
        "按相似度降序列出。"
    )
    hist = f"<HISTORY>\n{history_block}" if history_block else ""
    user = f"<USER_QUERY>\n{query}"
    ev = f"<EVIDENCE>\n{evidence_block}"
    fmt = (
        "<OUTPUT_FORMAT>\n"
        "结论：一句话总结。\n"
        "相关案例：\n"
        "1. [EventID=xxx] 标题 · 时间 · 机构 · 一句话事由\n"
        "2. [EventID=xxx] ...\n"
        "补充说明（可选，≤ 2 句）。"
    )
    return _wrap_sections(SYSTEM_PROMPT, task, hist, ev, user, fmt)


def build_law_grounding_prompt(
    query: str, evidence_block: str, history_block: str = ""
) -> str:
    task = "<TASK>\n任务：法规定位。从证据中抽取被引用的法条，并对应到案例。"
    examples = f"<EXAMPLES>\n{LAW_GROUNDING_EXAMPLES}"
    ev = f"<EVIDENCE>\n{evidence_block}"
    user = f"<USER_QUERY>\n{query}"
    fmt = (
        "<OUTPUT_FORMAT>\n"
        "1. 主要法条：[法条：《xx》第xx条] — 适用场景\n"
        "2. 代表案例：[EventID=...]（至多 3 条）\n"
        "3. 风险提示（1 句）"
    )
    return _wrap_sections(SYSTEM_PROMPT, task, examples, ev, user, fmt)


def build_sanction_recommendation_prompt(
    query: str, evidence_block: str, history_block: str = ""
) -> str:
    task = (
        "<TASK>\n任务：处罚方式推荐。基于历史相似案例统计，给出最可能的处罚方式及代表案例。"
        "不得作为正式执法依据，必须附免责声明。"
    )
    examples = f"<EXAMPLES>\n{SANCTION_EXAMPLES}"
    ev = f"<EVIDENCE>\n{evidence_block}"
    user = f"<USER_QUERY>\n{query}"
    fmt = (
        "<OUTPUT_FORMAT>\n"
        "推荐处罚方式（Top 3，带占比，需引 [EventID=...]）；法规依据（[法条：...]）；"
        "代表案例；免责声明（必含“仅供参考”或“不构成执法”字样）。"
    )
    return _wrap_sections(SYSTEM_PROMPT, task, examples, ev, user, fmt)


def build_trend_analysis_prompt(
    query: str, evidence_block: str, history_block: str = ""
) -> str:
    task = (
        "<TASK>\n任务：趋势分析。给出年度分布 / 机构分布 / 处罚类型分布，"
        "基于证据中已统计好的数字，禁止外推。"
    )
    hist = f"<HISTORY>\n{history_block}" if history_block else ""
    ev = f"<EVIDENCE>\n{evidence_block}"
    user = f"<USER_QUERY>\n{query}"
    fmt = (
        "<OUTPUT_FORMAT>\n"
        "总览一句话；年度趋势；主要处罚类型；峰值年份；代表案例 [EventID=...] × 2；"
        "统计口径声明。"
    )
    return _wrap_sections(SYSTEM_PROMPT, task, hist, ev, user, fmt)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BUILDERS = {
    "case_retrieval": build_case_retrieval_prompt,
    "law_grounding": build_law_grounding_prompt,
    "sanction_recommendation": build_sanction_recommendation_prompt,
    "trend_analysis": build_trend_analysis_prompt,
}


def build_prompt(
    intent_name: str,
    query: str,
    ranked_events: list[Any],
    history: list[dict[str, str]] | None = None,
    *,
    k: int | None = None,
    max_ctx_tokens: int = MAX_CTX_TOKENS_DEFAULT,
) -> tuple[str, list[str]]:
    """Build the full prompt string plus the list of kept EventIDs.

    Returns
    -------
    prompt_text : str
    kept_event_ids : list[str]
        The EventIDs that actually made it into the <EVIDENCE> block.
        `validators.py` uses this as the ground truth set for L7-1.
    """
    effective_k = k if k is not None else (6 if intent_name == "case_retrieval" else 4)
    evidence_block, kept_ids = assemble_evidence_block(
        ranked_events,
        k=effective_k,
        max_ctx_tokens=max_ctx_tokens,
        compact=(intent_name == "case_retrieval"),
    )
    history_block = format_history(history, intent_name)
    builder = _BUILDERS.get(intent_name, build_case_retrieval_prompt)
    prompt = builder(query=query, evidence_block=evidence_block, history_block=history_block)
    return prompt, kept_ids


__all__ = [
    "SYSTEM_PROMPT",
    "EvidenceCard",
    "assemble_evidence_block",
    "format_evidence_line",
    "format_history",
    "build_case_retrieval_prompt",
    "build_law_grounding_prompt",
    "build_sanction_recommendation_prompt",
    "build_trend_analysis_prompt",
    "build_prompt",
    "MAX_CTX_TOKENS_DEFAULT",
]
