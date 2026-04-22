"""Tokenizer for BM25 / sparse retrieval.

Supports two backends, switchable via ``configs/retrieval.json::tokenizer``:

* ``"jieba"``         -- jieba precise-mode word segmentation + stop-word
                         pruning + synonyms.json canonicals loaded as
                         user_dict so domain terms like ``"内幕交易"``,
                         ``"操纵市场"``, ``"证券法"`` survive as single
                         tokens.
* ``"regex_bigram"``  -- legacy fallback: ALNUM + law-name + CJK sliding
                         bigram. Kept for ablation studies so the M2
                         baseline can be reproduced.

The active backend is chosen by reading ``configs/retrieval.json`` once
per process; callers do not need to pass it in. Downstream indexes
(``BM25Index``) call :func:`tokenize` with a plain string.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared regex for the legacy (regex_bigram) backend
# ---------------------------------------------------------------------------

ALNUM_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
LAW_PATTERN = re.compile(r"《[^》]{1,40}》")


# ---------------------------------------------------------------------------
# Inline Chinese + English stop-word list (kept small and domain-aware).
#
# Notes
# -----
# * We intentionally keep domain terms like "公司" / "股份" OUT of the list
#   because they carry signal in our corpus (firm-name matching).
# * Punctuation and single-character helpers dominate; this is enough to
#   recover BM25 discrimination without dragging in a full HIT list.
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        # Chinese function words
        "的", "了", "和", "与", "及", "或", "而", "及其", "等", "是", "在", "于",
        "对", "对于", "以", "被", "把", "将", "从", "从而", "因", "因此",
        "所以", "并", "并且", "也", "还", "又", "就", "都", "不", "没", "没有",
        "有", "这", "那", "这个", "那个", "这些", "那些", "之", "其", "其中",
        "如", "如同", "即", "乃", "但", "但是", "然而", "则", "向", "到",
        "为", "给", "让", "使", "令", "着", "过", "了",
        "吧", "吗", "呢", "啊", "呀", "哦", "嗯", "哈", "哼",
        "已", "已经", "正在", "正", "尚", "曾",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
        "请问", "请", "帮", "帮我", "告诉", "说明", "介绍",
        "怎么", "如何", "什么", "哪些", "哪个", "多少", "几", "若干",
        "可", "可以", "能", "能够", "需要", "需", "应", "应当", "必须",
        "的话", "来", "去", "出", "上", "下", "里", "内", "外",
        # Punctuation often leaks through
        "。", "，", "、", "；", "：", "？", "！", "“", "”", "‘", "’",
        "（", "）", "【", "】", "《", "》", "—", "…", "·",
        ".", ",", ";", ":", "?", "!", "\"", "'",
        "(", ")", "[", "]", "-", "—", "…",
        # English function words (robustness)
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "at",
        "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its",
    }
)


def _project_root() -> Path:
    # src/csrc_rag/retrieval/tokenizer.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Backend state (lazy-loaded, per-process)
# ---------------------------------------------------------------------------

_BACKEND: str | None = None
_JIEBA_READY: bool = False
_JIEBA_MODULE = None


def _load_backend_from_config() -> str:
    """Read ``tokenizer`` field from configs/retrieval.json. Default 'regex_bigram'."""
    cfg_path = _project_root() / "configs" / "retrieval.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        LOGGER.warning("retrieval.json not found at %s; defaulting to regex_bigram", cfg_path)
        return "regex_bigram"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to parse retrieval.json (%s); defaulting to regex_bigram", exc)
        return "regex_bigram"
    backend = str(cfg.get("tokenizer", "regex_bigram")).lower()
    if backend not in {"jieba", "regex_bigram"}:
        LOGGER.warning("Unknown tokenizer=%s; defaulting to regex_bigram", backend)
        return "regex_bigram"
    return backend


def _canonical_terms_from_synonyms() -> list[str]:
    """Collect canonical terms from configs/synonyms.json for jieba user_dict.

    Both the categorised (``raw[category][canonical] = [aliases]``) and the
    legacy flat (``raw['synonyms'][canonical] = [aliases]``) layouts are
    supported, matching the tolerance already in slot_filler.
    """
    path = _project_root() / "configs" / "synonyms.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("tokenizer: failed to read synonyms.json: %s", exc)
        return []

    canonicals: set[str] = set()
    if isinstance(raw, dict):
        # Categorised layout
        for key, block in raw.items():
            if key.startswith("_") or not isinstance(block, dict):
                continue
            if key == "synonyms":
                # Flat layout lives under this key
                for canon, aliases in block.items():
                    canonicals.add(canon)
                    if isinstance(aliases, list):
                        for a in aliases:
                            if isinstance(a, str):
                                canonicals.add(a)
                continue
            # Categorised: e.g. raw["violation"]["内幕交易"] = [aliases]
            for canon, aliases in block.items():
                if not isinstance(canon, str):
                    continue
                canonicals.add(canon)
                if isinstance(aliases, list):
                    for a in aliases:
                        if isinstance(a, str):
                            canonicals.add(a)
        # violation_types list (legacy)
        vio_list = raw.get("violation_types")
        if isinstance(vio_list, list):
            for v in vio_list:
                if isinstance(v, str):
                    canonicals.add(v)
    return sorted(canonicals, key=len, reverse=True)


def _init_jieba() -> bool:
    """Initialise jieba + inject canonicals as user_dict. Returns False on failure."""
    global _JIEBA_READY, _JIEBA_MODULE
    if _JIEBA_READY:
        return True
    try:
        import jieba  # type: ignore
    except ImportError:
        LOGGER.warning(
            "jieba not installed; BM25 tokenizer will fall back to regex_bigram. "
            "Run `pip install jieba` to enable."
        )
        return False
    # Inject canonicals as user_dict entries so multi-char domain terms are
    # not over-segmented (e.g. "内幕交易", "信息披露违规").
    for term in _canonical_terms_from_synonyms():
        if len(term) >= 2 and len(term) <= 20:
            # weight=1000 is high enough to beat default frequencies; no POS tag.
            jieba.add_word(term, freq=1000)
    # Silence jieba's initial "Building prefix dict from the default dictionary" log.
    jieba.initialize()
    _JIEBA_MODULE = jieba
    _JIEBA_READY = True
    return True


def _get_backend() -> str:
    """Return the resolved backend, falling back if jieba is unavailable."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    backend = _load_backend_from_config()
    if backend == "jieba" and not _init_jieba():
        backend = "regex_bigram"
    _BACKEND = backend
    LOGGER.info("tokenizer backend resolved to: %s", backend)
    return backend


def reset_backend_cache() -> None:
    """Reset the resolved backend (test hook / ablation harness)."""
    global _BACKEND
    _BACKEND = None


# ---------------------------------------------------------------------------
# Legacy regex + bigram backend (kept byte-identical for ablation)
# ---------------------------------------------------------------------------


def _tokenize_regex_bigram(text: str) -> list[str]:
    normalized = text.lower().strip()
    tokens: list[str] = []

    for law_name in LAW_PATTERN.findall(normalized):
        tokens.append(law_name)

    for token in ALNUM_PATTERN.findall(normalized):
        tokens.append(token)

    for chunk in CJK_PATTERN.findall(normalized):
        compact = chunk.strip()
        if not compact:
            continue
        if len(compact) <= 2:
            tokens.append(compact)
            continue
        tokens.append(compact[:6])
        for idx in range(len(compact) - 1):
            tokens.append(compact[idx : idx + 2])

    return tokens


# ---------------------------------------------------------------------------
# Jieba backend
# ---------------------------------------------------------------------------


def _tokenize_jieba(text: str) -> list[str]:
    """jieba precise-mode + stop-word filter + law-name preservation.

    Preservation steps
    ------------------
    1. Pull out ``《...》`` spans first and emit them as single tokens --
       jieba would split these otherwise.
    2. Lower-case and feed the remainder through ``jieba.lcut`` precise
       mode (default).
    3. Drop pure whitespace, stop-words, and 1-char CJK pieces unless they
       are alnum (we keep 1-char English/number tokens like 'a' filtered
       out via stop-words too).
    """
    normalized = text.lower().strip()
    if not normalized:
        return []

    tokens: list[str] = []

    # Step 1: extract law names first so jieba does not split "《证券法》".
    masked = normalized
    for law_name in LAW_PATTERN.findall(normalized):
        tokens.append(law_name)
        # Replace in masked text with spaces so indices of downstream
        # segmentation are not disturbed.
        masked = masked.replace(law_name, " " * len(law_name), 1)

    # Step 2: jieba precise-mode segmentation on the masked string.
    assert _JIEBA_MODULE is not None  # backend init guarantees this
    for seg in _JIEBA_MODULE.lcut(masked, cut_all=False, HMM=True):
        tok = seg.strip()
        if not tok:
            continue
        if tok in _STOPWORDS:
            continue
        # Drop 1-char Chinese filler (keep 1-char alnum like 'A', '5' only
        # when it is clearly a code identifier; easier rule: drop any 1-char
        # Chinese or punctuation token not already whitelisted).
        if len(tok) == 1 and CJK_PATTERN.fullmatch(tok):
            continue
        # Remove pure punctuation that slipped past stop-words.
        if not re.search(r"[\w\u4e00-\u9fff]", tok):
            continue
        tokens.append(tok)

    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tokenize(text: str | None) -> list[str]:
    """Tokenise ``text`` using the backend configured in retrieval.json."""
    if not text:
        return []

    backend = _get_backend()
    if backend == "jieba":
        return _tokenize_jieba(text)
    return _tokenize_regex_bigram(text)
