from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csrc_rag.orchestration.intents import IntentSpec
from csrc_rag.settings import CONFIG_DIR
from csrc_rag.utils import read_json


@dataclass(frozen=True)
class ResponseOutput:
    text: str
    backend: str
    model_name: str | None = None


# ---------------------------------------------------------------------------
# Post-processing filter for LoRA outputs
# ---------------------------------------------------------------------------
# The LoRA-tuned Qwen-0.5B occasionally leaks training-data artefacts that
# should never appear in user-facing answers:
#   • Python-style variable names such as ``most_like_thing_1=``,
#     ``most_faq_affected_event_1=None``.
#   • Full-width semicolons between EventID tags, e.g.
#     ``[EventID=40109363；EventID=401737]`` (should be ``][`` separated).
#   • Dangling ``=None`` / ``None`` literals.
#
# These patterns are removed / normalised here so the end user never sees
# them. This is a conservative safety net, not a substitute for cleaner
# training data; see ``docs/reports/bad_cases.md`` for root-cause notes.

# Variable-name leaks: ``<name>_<int>=`` up to the next separator.
_VAR_LEAK_RE = re.compile(
    r"(?:most_like_[a-z_]*\d*|most_faq_[a-z_]*\d*|affected_event_\d+|"
    r"faq_result_\d+|faq_affected_event_\d+|like_thing_\d+)"
    r"\s*=\s*[^；;\n]*(?:；|;|\n|$)",
    flags=re.IGNORECASE,
)

# Bare "=None" or leading comma/semicolon followed by None
_NONE_LITERAL_RE = re.compile(r"(?:[;；,，]\s*)?=\s*None\b", flags=re.IGNORECASE)

# Two or more consecutive EventID tags glued with full-width / half-width
# semicolons: ``[EventID=xxx；EventID=yyy]`` → ``[EventID=xxx][EventID=yyy]``.
_EID_SEMICOLON_RE = re.compile(r"(\[EventID=\d+)[；;,，]\s*(EventID=\d+)")

# Leftover marker words that sometimes slip through the LoRA: a bare
# ``most_faq_result_1`` with no value following.
_BARE_MARKER_RE = re.compile(
    r"(?:most_like_[a-z_]*\d*|most_faq_[a-z_]*\d*|affected_event_\d+|"
    r"faq_result_\d+)\b",
    flags=re.IGNORECASE,
)

# Multiple spaces / stray whitespace cleanup
_MULTI_WS_RE = re.compile(r"[ \t]{2,}")


def _postprocess_answer(text: str) -> str:
    """Strip training-data leaks and normalise EventID separators.

    Applied to every local HF responder decode before it is returned to the
    user. Idempotent: running this on already-cleaned text is a no-op.
    """
    if not text:
        return text

    original = text

    # 1) Normalise ``[EventID=a；EventID=b]`` → ``[EventID=a][EventID=b]``.
    # Apply repeatedly until stable (in case three+ EIDs are chained).
    while True:
        new = _EID_SEMICOLON_RE.sub(r"\1][\2", text)
        if new == text:
            break
        text = new

    # 2) Drop ``most_like_xxx_1='...' ；`` style leaks.
    text = _VAR_LEAK_RE.sub("", text)

    # 3) Strip bare ``=None`` residues left behind.
    text = _NONE_LITERAL_RE.sub("", text)

    # 4) Remove any remaining bare marker words that survived step 2.
    text = _BARE_MARKER_RE.sub("", text)

    # 5) Collapse consecutive punctuation left by the removals.
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"\s*[；;]\s*(?=[。\n]|$)", "", text)
    text = _MULTI_WS_RE.sub(" ", text)

    cleaned = text.strip(" \t；;，,")

    # If post-processing ate everything, fall back to the raw text so the
    # caller can still use a downgrade path.
    return cleaned if cleaned else original


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def _fmt_amount(value: float) -> str:
    if value >= 1e8:
        return f"{value / 1e8:.2f}\u4ebf\u5143"
    if value >= 1e4:
        return f"{value / 1e4:.1f}\u4e07\u5143"
    return f"{value:.0f}\u5143"


class TemplateResponder:
    """Rich rule-based responder with per-intent answer strategies."""

    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        if not ranked_events:
            return ResponseOutput(
                text=(
                    "\u672a\u68c0\u7d22\u5230\u4e0e\u300c"
                    + query
                    + "\u300d\u9ad8\u5ea6\u76f8\u5173\u7684\u6848\u4f8b\uff0c\u8bf7\u5c1d\u8bd5\u7f29\u77ed\u63cf\u8ff0\u3001"
                    "\u6362\u4e00\u79cd\u8868\u8fbe\u65b9\u5f0f\uff0c\u6216\u63d0\u4f9b\u66f4\u591a\u8fdd\u89c4\u884c\u4e3a\u7ec6\u8282\u3002"
                ),
                backend="template",
            )

        name = intent.name
        if name == "case_retrieval":
            return self._case_retrieval(query, ranked_events)
        if name == "law_grounding":
            return self._law_grounding(query, ranked_events)
        if name == "sanction_recommendation":
            return self._sanction_recommendation(query, ranked_events)
        if name == "trend_analysis":
            return self._trend_analysis(query, ranked_events)
        return self._case_retrieval(query, ranked_events)

    # -- Case retrieval -------------------------------------------------------
    def _case_retrieval(self, query: str, events: list[Any]) -> ResponseOutput:
        lines = [
            "\u3010\u6848\u4f8b\u68c0\u7d22\u3011\u9488\u5bf9\u300c"
            + query
            + "\u300d\uff0c\u5171\u53ec\u56de "
            + str(len(events))
            + " \u6761\u76f8\u5173\u6848\u4f8b\uff0c\u4ee5\u4e0b\u4e3a\u6700\u76f8\u5173\u6848\u4f8b\uff1a",
            "",
        ]
        for i, ev in enumerate(events[:6], 1):
            title = ev.title or ev.event_id
            date = ev.declare_date or "\u672a\u77e5"
            org = ev.promulgator or "\u672a\u77e5"
            pt = "\u3001".join(ev.punishment_types[:3]) if ev.punishment_types else "\u672a\u8bb0\u5f55"
            snippet = ev.snippets[0][:120] if ev.snippets else ""
            lines.append(f"\u258c \u6848\u4f8b {i}  {title}")
            lines.append(f"  \u65f6\u95f4\uff1a{date}\u3000\u53d1\u5e03\u673a\u6784\uff1a{org}")
            lines.append(f"  \u5904\u7f5a\u65b9\u5f0f\uff1a{pt}")
            if snippet:
                lines.append(f"  \u6458\u8981\uff1a{snippet}\u2026")
            lines.append("")
        lines.append(
            "\u4ee5\u4e0a\u6848\u4f8b\u6309\u76f8\u4f3c\u5ea6\u7531\u9ad8\u5230\u4f4e\u6392\u5217\u3002"
            "\u53ef\u8fdb\u4e00\u6b65\u63d0\u95ee\u5177\u4f53\u6848\u4ef6\u8be6\u60c5\u3001\u6cd5\u89c4\u4f9d\u636e\u6216\u5904\u7f5a\u63a8\u8350\u3002"
        )
        return ResponseOutput(text="\n".join(lines), backend="template")

    # -- Law grounding --------------------------------------------------------
    def _law_grounding(self, query: str, events: list[Any]) -> ResponseOutput:
        law_counter: Counter = Counter()
        law_cases: dict[str, list[str]] = {}
        for ev in events:
            for law in ev.laws:
                if law:
                    law_counter[law] += 1
                    if law not in law_cases:
                        law_cases[law] = []
                    if len(law_cases[law]) < 3:
                        law_cases[law].append(ev.title or ev.event_id)

        lines = [
            "\u3010\u6cd5\u89c4\u4f9d\u636e\u3011\u9488\u5bf9\u300c"
            + query
            + "\u300d\uff0c\u4ee5\u4e0b\u6cd5\u89c4\u5728\u76f8\u4f3c\u6848\u4f8b\u4e2d\u88ab\u9ad8\u9891\u5f15\u7528\uff1a",
            "",
        ]
        if not law_counter:
            lines.append(
                "\u672a\u4ece\u53ec\u56de\u6848\u4f8b\u4e2d\u63d0\u53d6\u5230\u5177\u4f53\u6cd5\u89c4\u4fe1\u606f\uff0c"
                "\u5efa\u8bae\u7ed3\u5408\u6848\u4f8b\u539f\u6587\u67e5\u9605\u3002"
            )
        else:
            for rank, (law, cnt) in enumerate(law_counter.most_common(5), 1):
                cases_str = "\u3001".join(law_cases[law][:2])
                lines.append(f"{rank}. {law[:160]}")
                lines.append(f"   \u5f15\u7528\u6b21\u6570\uff1a{cnt}  \u4ee3\u8868\u6848\u4f8b\uff1a{cases_str}")
                lines.append("")
        lines.append(
            "\u672c\u56de\u7b54\u57fa\u4e8e\u53ec\u56de\u6848\u4f8b\u7684\u6cd5\u6761\u5f15\u7528\uff0c"
            "\u4e0d\u6784\u6210\u6b63\u5f0f\u6cd5\u5f8b\u610f\u89c1\u3002"
        )
        return ResponseOutput(text="\n".join(lines), backend="template")

    # -- Sanction recommendation ----------------------------------------------
    def _sanction_recommendation(self, query: str, events: list[Any]) -> ResponseOutput:
        pt_counter: Counter = Counter()
        pt_cases: dict[str, list[str]] = {}
        penalties: list[float] = []

        for ev in events:
            boost = 1 + round(float(ev.score), 4) * 2
            for pt in ev.punishment_types:
                if pt:
                    pt_counter[pt] += boost
                    if pt not in pt_cases:
                        pt_cases[pt] = []
                    if len(pt_cases[pt]) < 2:
                        pt_cases[pt].append(ev.title or ev.event_id)
            raw = getattr(ev, "sum_penalty", None)
            if raw is not None:
                try:
                    v = float(raw)
                    if v > 0:
                        penalties.append(v)
                except (TypeError, ValueError):
                    pass

        lines = [
            "\u3010\u5904\u7f5a\u63a8\u8350\u3011\u6839\u636e "
            + str(len(events))
            + " \u4e2a\u76f8\u4f3c\u6848\u4f8b\u5206\u6790\uff0c\u6700\u53ef\u80fd\u9002\u7528\u7684\u5904\u7f5a\u65b9\u5f0f\uff1a",
            "",
        ]
        top_pts = pt_counter.most_common(5)
        total_weight = sum(w for _, w in top_pts) or 1.0
        for rank, (pt, weight) in enumerate(top_pts, 1):
            pct = f"{weight / total_weight * 100:.1f}%"
            cases_str = "\u3001".join(pt_cases.get(pt, [])[:2])
            lines.append(f"{rank}. **{pt}**\u3000\uff08\u5360\u76f8\u4f3c\u6848\u4f8b\u52a0\u6743 {pct}\uff09")
            if cases_str:
                lines.append(f"   \u4ee3\u8868\u6848\u4f8b\uff1a{cases_str}")
            lines.append("")

        if penalties:
            mn = min(penalties)
            mx = max(penalties)
            med = statistics.median(penalties)
            lines.append(
                f"\u7f5a\u6b3e\u91d1\u989d\u53c2\u8003\uff08{len(penalties)} \u4e2a\u6709\u8bb0\u5f55\u6848\u4f8b\uff09\uff1a\n"
                f"  \u6700\u4f4e {_fmt_amount(mn)}  /  \u4e2d\u4f4d\u6570 {_fmt_amount(med)}  /  \u6700\u9ad8 {_fmt_amount(mx)}"
            )
            lines.append("")

        lines.append(
            "\u26a0 \u4e0a\u8ff0\u5efa\u8bae\u57fa\u4e8e\u76f8\u4f3c\u5386\u53f2\u6848\u4f8b\u7684\u7edf\u8ba1\u6a21\u5f0f\uff0c"
            "\u4e0d\u66ff\u4ee3\u76d1\u7ba1\u673a\u6784\u7684\u6b63\u5f0f\u6267\u6cd5\u5224\u65ad\u3002"
        )
        lines.append(
            "  \u5177\u4f53\u5904\u7f5a\u9700\u7ed3\u5408\u8fdd\u89c4\u60c5\u8282\u3001\u4e3b\u89c2\u6545\u610f\u7a0b\u5ea6\u53ca\u76f8\u5173\u6cd5\u5f8b\u8ba4\u5b9a\u3002"
        )
        return ResponseOutput(text="\n".join(lines), backend="template")

    # -- Trend analysis -------------------------------------------------------
    def _trend_analysis(self, query: str, events: list[Any]) -> ResponseOutput:
        year_counter: Counter = Counter()
        org_counter: Counter = Counter()
        pt_counter: Counter = Counter()

        for ev in events:
            if ev.declare_date and len(ev.declare_date) >= 4:
                year = ev.declare_date[:4]
                if year.isdigit():
                    year_counter[year] += 1
            if ev.promulgator:
                org_counter[ev.promulgator] += 1
            for pt in ev.punishment_types:
                if pt:
                    pt_counter[pt] += 1

        total = len(events)
        lines = [
            "\u3010\u8d8b\u52bf\u5206\u6790\u3011\u9488\u5bf9\u300c"
            + query
            + "\u300d\uff0c\u4ee5\u4e0b\u7edf\u8ba1\u57fa\u4e8e\u53ec\u56de\u7684 "
            + str(total)
            + " \u4e2a\u76f8\u5173\u6848\u4f8b\uff1a",
            "",
        ]

        if year_counter:
            sorted_years = sorted(year_counter.items())
            lines.append("\u258c \u5e74\u5ea6\u5206\u5e03\uff1a")
            for yr, cnt in sorted_years:
                bar = "\u2588" * min(cnt, 20)
                lines.append(f"  {yr}  {bar} {cnt}\u4ef6 ({_pct(cnt, total)})")
            lines.append("")
            peak_year, peak_cnt = year_counter.most_common(1)[0]
            lines.append(
                f"  \u5cf0\u503c\u5e74\u4efd\uff1a{peak_year}"
                f"\uff08{peak_cnt} \u4ef6\uff0c\u5360 {_pct(peak_cnt, total)}\uff09"
            )
            lines.append("")

        if pt_counter:
            lines.append("\u258c \u4e3b\u8981\u5904\u7f5a\u65b9\u5f0f\uff08Top 5\uff09\uff1a")
            for pt, cnt in pt_counter.most_common(5):
                lines.append(f"  - {pt}\uff1a{cnt} \u6b21 ({_pct(cnt, total)})")
            lines.append("")

        if org_counter:
            lines.append("\u258c \u4e3b\u8981\u53d1\u5e03\u673a\u6784\uff08Top 3\uff09\uff1a")
            for org, cnt in org_counter.most_common(3):
                lines.append(f"  - {org}\uff1a{cnt} \u4ef6")
            lines.append("")

        lines.append(
            "\u6ce8\uff1a\u7edf\u8ba1\u57fa\u4e8e\u5f53\u524d\u68c0\u7d22\u7ed3\u679c\uff0c"
            "\u4e0d\u4ee3\u8868\u5b8c\u6574\u5386\u53f2\u6570\u636e\u3002"
            "\u5982\u9700\u7cbe\u786e\u7edf\u8ba1\uff0c\u8bf7\u7f29\u5c0f\u67e5\u8be2\u8303\u56f4\u3002"
        )
        return ResponseOutput(text="\n".join(lines), backend="template")


class LocalHFResponder:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        device = (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _build_prompt(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None,
    ) -> str:
        _unk = "\u672a\u77e5"
        _none = "\u672a\u63d0\u53d6"
        evidence_lines: list[str] = []
        for index, event in enumerate(ranked_events[:4], start=1):
            pt_str = "\u3001".join(event.punishment_types) or _none
            law_str = "\uff1b".join(filter(None, event.laws[:2])) or _none
            snip_str = " ".join(event.snippets[:2]) or _none
            evidence_lines.append(
                "\n".join(
                    [
                        f"[\u6848\u4f8b{index}] \u6807\u9898\uff1a{event.title or event.event_id}",
                        f"\u65f6\u95f4\uff1a{event.declare_date or _unk}",
                        f"\u673a\u6784\uff1a{event.promulgator or _unk}",
                        f"\u5904\u7f5a\u65b9\u5f0f\uff1a{pt_str}",
                        f"\u6cd5\u89c4\uff1a{law_str}",
                        f"\u8bc1\u636e\u7247\u6bb5\uff1a{snip_str}",
                    ]
                )
            )
        history_lines = []
        for turn in (history or [])[-4:]:
            role = "\u7528\u6237" if turn.get("role") == "user" else "\u52a9\u624b"
            history_lines.append(f"{role}\uff1a{turn.get('content', '').strip()}")

        sections = [
            "\u4f60\u662f\u8bc1\u76d1\u4f1a\u5904\u7f5a\u6848\u4f8b\u667a\u80fd\u5206\u6790\u52a9\u624b\u3002",
            "\u4f60\u53ea\u80fd\u6839\u636e\u7ed9\u5b9a\u6848\u4f8b\u8bc1\u636e\u56de\u7b54\uff0c"
            "\u7981\u6b62\u7f16\u9020\u672a\u51fa\u73b0\u7684\u6cd5\u6761\u3001\u5904\u7f5a\u7ed3\u679c\u3001\u91d1\u989d\u6216\u4e8b\u5b9e\u3002",
            "\u5982\u679c\u8bc1\u636e\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u5199\u201c\u8bc1\u636e\u4e0d\u8db3\u201d\u3002",
            f"\u5f53\u524d\u4efb\u52a1\uff1a{intent.description}",
        ]
        if history_lines:
            sections.extend(["\u5bf9\u8bdd\u5386\u53f2\uff1a", "\n".join(history_lines)])
        sections.extend(
            [
                f"\u7528\u6237\u95ee\u9898\uff1a{query}",
                "\u68c0\u7d22\u8bc1\u636e\uff1a",
                "\n\n".join(evidence_lines),
                "\u8bf7\u8f93\u51fa\uff1a1. \u7ed3\u8bba\u6458\u8981 2. \u4f9d\u636e\u8981\u70b9 3. \u76f8\u4f3c\u6848\u4f8b\u63d0\u793a 4. \u98ce\u9669\u4e0e\u4e0d\u8db3\u3002",
                "\u8f93\u51fa\u8981\u6c42\uff1a",
                "- \u7ed3\u8bba\u6458\u8981\u5148\u76f4\u63a5\u56de\u7b54\u95ee\u9898\uff0c\u4e0d\u8981\u7a7a\u8bdd\u3002",
                "- \u5982\u679c\u662f\u5904\u7f5a\u63a8\u8350\uff0c\u660e\u786e\u5199\u51fa1\u52303\u4e2a\u6700\u53ef\u80fd\u7684\u5904\u7f5a\u65b9\u5f0f\u3002",
                "- \u4f9d\u636e\u8981\u70b9\u4f18\u5148\u5f15\u7528\u6cd5\u89c4\u540d\u79f0\u6216\u5904\u7f5a\u7c7b\u578b\uff0c\u4e0d\u8981\u9010\u5b57\u590d\u5236\u957f\u6bb5\u539f\u6587\u3002",
                "- \u76f8\u4f3c\u6848\u4f8b\u63d0\u793a\u53ea\u5199\u6848\u4f8b\u6807\u9898\u3001\u5e74\u4efd\u548c\u673a\u6784\uff0c\u4e0d\u8981\u8d34\u6574\u6bb5\u8bc1\u636e\u3002",
                "- \u6bcf\u4e00\u90e8\u5206\u5c3d\u91cf\u7b80\u6d01\uff0c\u907f\u514d\u8d85\u8fc74\u6761\u3002",
                # B2 fix: even when no single retrieval result perfectly matches
                # the query, the responder MUST still cite the most relevant
                # EventID(s) from the retrieved evidence. Empty "参考历史相似案例"
                # boilerplate without citation is forbidden.
                "- **\u5fc5\u987b\u5f15\u7528\u6848\u4f8b**\uff1a\u56de\u7b54\u4e2d\u81f3\u5c11\u51fa\u73b0\u4e00\u4e2a [EventID=xxx] \u683c\u5f0f\u7684\u5f15\u7528\uff0c"
                "\u4f18\u5148\u5f15\u7528\u8bc1\u636e\u4e2d\u6700\u76f8\u5173\u7684\u6848\u4f8b "
                "[EventID=" + (ranked_events[0].event_id if ranked_events else "xxx") + "]\u3002"
                "\u5373\u4f7f\u68c0\u7d22\u8fd4\u56de\u7684\u6848\u4f8b\u4e0e\u67e5\u8be2\u4e0d\u5b8c\u5168\u5339\u914d\uff0c"
                "\u4e5f\u8981\u5f15\u7528\u6700\u63a5\u8fd1\u7684\u6848\u4f8b\u5e76\u8bf4\u660e\u76f8\u4f3c\u4e4b\u5904\uff0c"
                "**\u4e0d\u5141\u8bb8\u7528\u201c\u53c2\u8003\u5386\u53f2\u76f8\u4f3c\u6848\u4f8b\u201d\u7b49\u7a7a\u8bdd\u66ff\u4ee3\u5177\u4f53\u5f15\u7528**\u3002",
                "- \u5982\u679c\u8bc1\u636e\u7a7a\u96c6\uff0c\u76f4\u63a5\u5199\u201c\u8bc1\u636e\u4e0d\u8db3\u201d\uff0c"
                "\u8fd9\u662f\u552f\u4e00\u53ef\u4ee5\u4e0d\u5f15\u7528 EventID \u7684\u60c5\u51b5\u3002",
                "\u56de\u7b54\u7528\u4e2d\u6587\u3002",
            ]
        )
        return "\n\n".join(sections)

    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        if not ranked_events:
            return ResponseOutput(
                text="\u672a\u68c0\u7d22\u5230\u4e0e\u300c" + query + "\u300d\u9ad8\u5ea6\u76f8\u5173\u7684\u6848\u4f8b\uff0c\u8bf7\u7f29\u77ed\u63cf\u8ff0\u6216\u6362\u4e00\u79cd\u95ee\u6cd5\u3002",
                backend="local_hf_empty",
                model_name=self.model_name,
            )
        self._ensure_model()

        import torch  # type: ignore

        prompt = self._build_prompt(query=query, intent=intent, ranked_events=ranked_events, history=history)
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None or self._device is None:
            raise RuntimeError("Local HF responder failed to initialize.")

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "\u4f60\u662f\u8bc1\u76d1\u4f1a\u5904\u7f5a\u6848\u4f8b\u667a\u80fd\u5206\u6790\u52a9\u624b\uff0c\u53ea\u80fd\u4f9d\u636e\u7ed9\u5b9a\u8bc1\u636e\u56de\u7b54\u3002"
                        "\u56de\u7b54\u5fc5\u987b\u5f15\u7528 [EventID=xxx] \u683c\u5f0f\u7684\u5177\u4f53\u6848\u4f8b\uff0c"
                        "\u4e0d\u5141\u8bb8\u7528\u201c\u53c2\u8003\u5386\u53f2\u76f8\u4f3c\u6848\u4f8b\u201d\u7b49\u7a7a\u8bdd\u66ff\u4ee3\u3002"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        answer = _postprocess_answer(answer)
        # B2 fallback: if the LoRA still produces boilerplate with no EventID
        # citation but the retriever did return evidence, force-append the
        # top-1 EventID so the answer is never a citation-less dead end.
        if ranked_events and "[EventID=" not in answer:
            top1_eid = ranked_events[0].event_id
            if top1_eid:
                answer = (
                    f"{answer.rstrip('。;；,，。 ')}"
                    f"；参考案例见 [EventID={top1_eid}]\u3002"
                )
        if not answer:
            answer = (
                "\u8bc1\u636e\u5df2\u53ec\u56de\uff0c\u4f46\u672c\u5730\u56de\u590d\u6a21\u578b\u672a\u751f\u6210\u6709\u6548\u6587\u672c\uff0c"
                "\u8bf7\u5148\u67e5\u770b\u4e0b\u65b9\u6848\u4f8b\u8bc1\u636e\u3002"
            )
        return ResponseOutput(text=answer, backend="local_hf", model_name=self.model_name)


class CompositeResponder:
    def __init__(self, primary: Any, fallback: TemplateResponder) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(
        self,
        query: str,
        intent: IntentSpec,
        ranked_events: list[Any],
        history: list[dict[str, str]] | None = None,
    ) -> ResponseOutput:
        try:
            return self.primary.generate(query=query, intent=intent, ranked_events=ranked_events, history=history)
        except Exception:
            return self.fallback.generate(query=query, intent=intent, ranked_events=ranked_events, history=history)


def build_responder(config_path: str | Path | None = None) -> CompositeResponder:
    config = read_json(config_path or CONFIG_DIR / "models.json")
    response_cfg = config.get("response_generation", {})
    fallback = TemplateResponder()
    backend = response_cfg.get("backend", "template")
    if backend == "local_hf":
        primary = LocalHFResponder(
            model_name=response_cfg["model_name"],
            max_new_tokens=int(response_cfg.get("max_new_tokens", 256)),
            temperature=float(response_cfg.get("temperature", 0.2)),
            top_p=float(response_cfg.get("top_p", 0.9)),
        )
        return CompositeResponder(primary=primary, fallback=fallback)
    return CompositeResponder(primary=fallback, fallback=fallback)
