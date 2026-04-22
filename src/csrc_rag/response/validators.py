"""L7 citation validation.

StrategyAgent-G · post-processing.

Runs on the raw text emitted by the LLM (or TemplateResponder) and checks:

    L7-1  Every [EventID=xxx] appearing in the answer must be in the evidence set.
    L7-2  At least one [EventID=] must appear (except for `trend_analysis`).
    L7-3  [法条：《xx》第xx条] must match the canonical regex.
    L7-4  The law name (xx) must be in a small allow-list of PRC securities laws.
    L7-5  Numbers / money / percentages must sit in a sentence that also cites
          an EventID or a law article; otherwise logged as "unsupported claim".
    L7-6  Total length ≤ 800 characters (auto-truncate).
    L7-7  Forbidden phrases such as "法院已判决" get replaced with hedging words.
    L7-8  sanction_recommendation output must contain a disclaimer keyword.

The module is pure / side-effect-free; it returns a `ValidationReport` plus a
possibly-rewritten text (for L7-6 / L7-7 / L7-8 auto-fixes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

EVENT_ID_RE = re.compile(r"\[EventID=([^\]\s]+?)\]")

# Chinese numerals + Arabic digits for article numbers, optional "之N" suffix.
LAW_CITE_RE = re.compile(
    r"\[法条：《([^》]{1,40})》第([0-9一二三四五六七八九十百零两]+)条(之[0-9一二三四五六七八九十]+)?\]"
)

NUMERIC_CLAIM_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|％|万元|亿元|元|人|年|次|起|件))"
)

FORBIDDEN_PATTERNS = [
    (re.compile(r"法院已判决"), "建议参考"),
    (re.compile(r"根据法律，?必须判处"), "历史案例中常见处罚为"),
    (re.compile(r"本案应判处"), "历史案例中类似情形通常处以"),
]

DISCLAIMER_KEYWORDS = ("仅供参考", "不构成执法", "不构成正式执法")

MAX_ANSWER_CHARS = 800

# Law name allow-list (L7-4). Extend as the corpus grows.
LAW_WHITELIST = {
    "证券法",
    "公司法",
    "证券投资基金法",
    "基金法",
    "期货和衍生品法",
    "期货法",
    "刑法",
    "反洗钱法",
    "商业银行法",
    "行政处罚法",
    "证券投资者保护法",
    "上市公司信息披露管理办法",
    "证券发行与承销管理办法",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    passed: bool = True
    missing_event_ids: list[str] = field(default_factory=list)
    invalid_laws: list[str] = field(default_factory=list)
    unknown_law_names: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    truncated: bool = False
    disclaimer_added: bool = False
    severity: str = "ok"   # ok | low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "missing_event_ids": self.missing_event_ids,
            "invalid_laws": self.invalid_laws,
            "unknown_law_names": self.unknown_law_names,
            "unsupported_claims": self.unsupported_claims,
            "forbidden_hits": self.forbidden_hits,
            "truncated": self.truncated,
            "disclaimer_added": self.disclaimer_added,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    text: str                       # possibly rewritten text (auto-fixes applied)
    cited_event_ids: list[str]
    cited_laws: list[str]
    report: ValidationReport


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_event_ids(text: str) -> list[str]:
    seen = set()
    out: list[str] = []
    for m in EVENT_ID_RE.finditer(text):
        eid = m.group(1).strip()
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def extract_law_citations(text: str) -> list[str]:
    """Return list of canonical `[法条：《xx》第xx条]` strings."""
    seen = set()
    out: list[str] = []
    for m in LAW_CITE_RE.finditer(text):
        full = m.group(0)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def extract_law_names(text: str) -> list[str]:
    return [m.group(1) for m in LAW_CITE_RE.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    # Split on Chinese & Latin sentence terminators, keep content.
    parts = re.split(r"(?<=[。！？!?;；\n])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def check_event_ids(
    text: str, allowed_ids: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Return (cited_valid, cited_missing)."""
    allowed = {str(a) for a in allowed_ids}
    cited = extract_event_ids(text)
    valid = [c for c in cited if c in allowed]
    missing = [c for c in cited if c not in allowed]
    return valid, missing


def check_law_format(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (valid_cites, invalid_cite_fragments, unknown_law_names).

    An "invalid_cite_fragment" is anything that looks like a broken legal
    citation (contains "《" and "法条" markers but doesn't match the regex).
    """
    valid = extract_law_citations(text)
    names = extract_law_names(text)
    unknown = [n for n in names if not any(w in n for w in LAW_WHITELIST)]
    # Heuristic: find broken "[法条：..." fragments that failed the strict regex
    broken = []
    for raw in re.findall(r"\[法条：[^\]]{0,60}\]", text):
        if raw not in valid:
            broken.append(raw)
    return valid, broken, unknown


def check_unsupported_claims(text: str) -> list[str]:
    """Sentences containing numbers but no [EventID=...] or [法条：...] citation."""
    unsupported: list[str] = []
    for sent in _split_sentences(text):
        if NUMERIC_CLAIM_RE.search(sent) and not (
            EVENT_ID_RE.search(sent) or LAW_CITE_RE.search(sent)
        ):
            unsupported.append(sent[:80])
    return unsupported


def apply_forbidden_rewrites(text: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    out = text
    for pat, repl in FORBIDDEN_PATTERNS:
        if pat.search(out):
            hits.append(pat.pattern)
            out = pat.sub(repl, out)
    return out, hits


def enforce_length(text: str, *, limit: int = MAX_ANSWER_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    # Truncate at the nearest sentence boundary before the limit.
    cut = text[:limit]
    tail = re.search(r"[。！？.!?]", cut[::-1])
    if tail:
        cut = cut[: limit - tail.start()]
    return cut.rstrip() + "……", True


def ensure_disclaimer(text: str) -> tuple[str, bool]:
    if any(k in text for k in DISCLAIMER_KEYWORDS):
        return text, False
    return text.rstrip() + "\n⚠ 以上内容仅基于历史案例统计，仅供参考，不构成正式执法意见。", True


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate(
    text: str,
    *,
    intent_name: str,
    evidence_event_ids: Iterable[str],
) -> ValidationResult:
    """Run all L7 rules. Returns possibly-rewritten text + structured report.

    The rewrite pipeline (in order):
        L7-7 forbidden phrase substitution
        L7-6 length enforcement
        L7-8 disclaimer for sanction_recommendation
    """
    report = ValidationReport()

    # L7-7: forbidden rewrites (always applied)
    text, forbidden_hits = apply_forbidden_rewrites(text)
    if forbidden_hits:
        report.forbidden_hits = forbidden_hits
        report.severity = _bump(report.severity, "high")

    # L7-1 / L7-2: EventID checks
    valid_ids, missing_ids = check_event_ids(text, evidence_event_ids)
    if missing_ids:
        report.missing_event_ids = missing_ids
        report.passed = False
        report.severity = _bump(report.severity, "critical")
    if not valid_ids and intent_name != "trend_analysis":
        report.passed = False
        report.severity = _bump(report.severity, "high")

    # L7-3 / L7-4: law citation format + whitelist
    valid_laws, invalid_laws, unknown_names = check_law_format(text)
    if invalid_laws:
        report.invalid_laws = invalid_laws
        report.passed = False
        report.severity = _bump(report.severity, "high")
    if unknown_names:
        report.unknown_law_names = unknown_names
        report.severity = _bump(report.severity, "medium")

    # L7-5: numeric claims without citation
    unsupported = check_unsupported_claims(text)
    if unsupported:
        report.unsupported_claims = unsupported
        report.severity = _bump(report.severity, "medium")

    # L7-6: length
    text, truncated = enforce_length(text)
    report.truncated = truncated

    # L7-8: disclaimer for sanction recommendation
    if intent_name == "sanction_recommendation":
        text, added = ensure_disclaimer(text)
        report.disclaimer_added = added

    return ValidationResult(
        text=text,
        cited_event_ids=valid_ids,
        cited_laws=valid_laws,
        report=report,
    )


# ---------------------------------------------------------------------------
# Confidence scoring (§6 of the strategy doc)
# ---------------------------------------------------------------------------

def compute_confidence(
    *,
    avg_top3_score: float,
    answer_text: str,
    n_evidence: int,
    report: ValidationReport,
) -> float:
    """Confidence ∈ [0, 1]."""
    # Citation coverage: #citations / #sentences (roughly).
    sentences = _split_sentences(answer_text)
    if not sentences:
        citation_coverage = 0.0
    else:
        cited_sents = sum(
            1 for s in sentences if EVENT_ID_RE.search(s) or LAW_CITE_RE.search(s)
        )
        citation_coverage = cited_sents / len(sentences)

    evidence_norm = min(n_evidence, 5) / 5.0

    total_claims = max(1, len(_split_sentences(answer_text)))
    unsupported_ratio = len(report.unsupported_claims) / total_claims

    score = (
        0.4 * max(0.0, min(1.0, avg_top3_score))
        + 0.3 * citation_coverage
        + 0.2 * evidence_norm
        + 0.1 * (1.0 - unsupported_ratio)
    )
    if report.missing_event_ids:
        score *= 0.5  # strong penalty for fabricated citations
    return max(0.0, min(1.0, round(score, 4)))


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = ["ok", "low", "medium", "high", "critical"]


def _bump(current: str, candidate: str) -> str:
    try:
        return _SEVERITY_ORDER[max(
            _SEVERITY_ORDER.index(current), _SEVERITY_ORDER.index(candidate)
        )]
    except ValueError:
        return candidate


__all__ = [
    "EVENT_ID_RE",
    "LAW_CITE_RE",
    "LAW_WHITELIST",
    "ValidationReport",
    "ValidationResult",
    "extract_event_ids",
    "extract_law_citations",
    "check_event_ids",
    "check_law_format",
    "check_unsupported_claims",
    "apply_forbidden_rewrites",
    "enforce_length",
    "ensure_disclaimer",
    "validate",
    "compute_confidence",
]
