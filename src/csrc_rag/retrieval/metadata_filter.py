"""Metadata hard-filter for L3a.

Consumes structured slots produced by the L2 slot_filler and produces the
set of chunk_ids that pass the hard filter, to be passed into the
BM25 / Dense encoders as ``allowed_doc_ids``.

Design notes
------------
- ``year`` / ``violation_type`` / ``org``  -> hard filter (equal / contains)
- ``company``                              -> soft filter (kept as boost hint)
- If the resulting ``allowed`` set is too small (< ``min_allowed_fallback``),
  we degrade to soft filter (return ``None`` => keep full corpus) to avoid
  empty recall.

The module is stateless apart from the pre-indexed metadata; the engine
should instantiate ``MetadataFilter`` once at startup.

Interface
---------
    mf = MetadataFilter.from_chunks(chunks)
    decision = mf.apply(
        slots={"year": "2024", "violation_type": "信息披露违规", "org": "证监会"},
        slot_confidence={"year": 0.95, "violation_type": 0.80, "org": 0.90},
    )
    # decision.allowed_doc_ids  -> set[str] | None
    # decision.applied_filters  -> dict[str, str]
    # decision.fallback          -> bool  (True if degraded to soft)
    # decision.boost_hints       -> dict[str, str]  (e.g. {"company": "中金公司"})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

LOGGER = logging.getLogger(__name__)


# Minimum number of chunks after hard-filter before we degrade to soft-filter.
DEFAULT_MIN_ALLOWED_FALLBACK = 20

# Minimum slot confidence to trust the slot for hard filtering.
DEFAULT_SLOT_CONFIDENCE_THRESHOLD = 0.6

# Slot name -> chunk field(s) mapping
HARD_FILTER_SLOTS = ("year", "violation_type", "org")
SOFT_FILTER_SLOTS = ("company",)


@dataclass(frozen=True)
class FilterDecision:
    allowed_doc_ids: set[str] | None
    applied_filters: dict[str, str]
    boost_hints: dict[str, str]
    fallback: bool
    diagnostics: dict[str, int]


@dataclass
class ChunkMetaRow:
    chunk_id: str
    year: str | None
    promulgator: str
    supervisor: str
    violation_types: tuple[str, ...]
    title: str


class MetadataFilter:
    """Apply structured slot filters on the chunk corpus."""

    def __init__(
        self,
        rows: list[ChunkMetaRow],
        *,
        min_allowed_fallback: int = DEFAULT_MIN_ALLOWED_FALLBACK,
        slot_confidence_threshold: float = DEFAULT_SLOT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._rows = rows
        self._min_allowed_fallback = min_allowed_fallback
        self._slot_confidence_threshold = slot_confidence_threshold

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Mapping[str, Any]],
        *,
        min_allowed_fallback: int = DEFAULT_MIN_ALLOWED_FALLBACK,
        slot_confidence_threshold: float = DEFAULT_SLOT_CONFIDENCE_THRESHOLD,
    ) -> "MetadataFilter":
        rows: list[ChunkMetaRow] = []
        for chunk in chunks:
            rows.append(
                ChunkMetaRow(
                    chunk_id=chunk["chunk_id"],
                    year=chunk.get("year"),
                    promulgator=chunk.get("promulgator") or "",
                    supervisor=chunk.get("supervisor") or "",
                    violation_types=tuple(chunk.get("violation_types") or []),
                    title=chunk.get("title") or "",
                )
            )
        return cls(
            rows,
            min_allowed_fallback=min_allowed_fallback,
            slot_confidence_threshold=slot_confidence_threshold,
        )

    # --------------------------------------------------------------- public
    def apply(
        self,
        slots: Mapping[str, Any] | None,
        *,
        slot_confidence: Mapping[str, float] | None = None,
    ) -> FilterDecision:
        """Apply hard/soft filters based on the slot dict.

        Parameters
        ----------
        slots: normalised slot values, e.g. {"year": "2024", "org": "证监会"}.
        slot_confidence: optional confidence per slot (0.0-1.0). Slots whose
            confidence is below ``slot_confidence_threshold`` are ignored for
            hard filtering but still used as soft boost hints.
        """
        slots = dict(slots or {})
        confidences = dict(slot_confidence or {})

        applied: dict[str, str] = {}
        boost_hints: dict[str, str] = {}
        trusted_hard_slots: dict[str, str] = {}

        for slot_name in HARD_FILTER_SLOTS:
            value = _normalise_slot_value(slots.get(slot_name))
            if value is None:
                continue
            conf = float(confidences.get(slot_name, 1.0))
            if conf < self._slot_confidence_threshold:
                # Low-confidence slot: skip hard filter, keep as soft hint.
                boost_hints[slot_name] = value
                continue
            trusted_hard_slots[slot_name] = value
            applied[slot_name] = value

        for slot_name in SOFT_FILTER_SLOTS:
            value = _normalise_slot_value(slots.get(slot_name))
            if value is None:
                continue
            boost_hints[slot_name] = value

        # No hard filters -> no restriction.
        if not trusted_hard_slots:
            return FilterDecision(
                allowed_doc_ids=None,
                applied_filters={},
                boost_hints=boost_hints,
                fallback=False,
                diagnostics={"allowed": len(self._rows), "total": len(self._rows)},
            )

        allowed = {
            row.chunk_id
            for row in self._rows
            if self._row_matches(row, trusted_hard_slots)
        }

        diagnostics = {
            "allowed": len(allowed),
            "total": len(self._rows),
        }

        if len(allowed) < self._min_allowed_fallback:
            LOGGER.info(
                "Metadata hard-filter returned %d < %d chunks; degrading to soft.",
                len(allowed),
                self._min_allowed_fallback,
            )
            # Promote hard slots to soft boost hints so downstream rerank can use them.
            for slot_name, value in trusted_hard_slots.items():
                boost_hints.setdefault(slot_name, value)
            return FilterDecision(
                allowed_doc_ids=None,
                applied_filters={},
                boost_hints=boost_hints,
                fallback=True,
                diagnostics=diagnostics,
            )

        return FilterDecision(
            allowed_doc_ids=allowed,
            applied_filters=applied,
            boost_hints=boost_hints,
            fallback=False,
            diagnostics=diagnostics,
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _row_matches(row: ChunkMetaRow, slots: Mapping[str, str]) -> bool:
        if "year" in slots and row.year != slots["year"]:
            return False
        if "violation_type" in slots:
            needle = slots["violation_type"]
            if not any(needle in vt for vt in row.violation_types):
                return False
        if "org" in slots:
            needle = slots["org"]
            if needle not in row.promulgator and needle not in row.supervisor:
                return False
        return True


def _normalise_slot_value(value: Any) -> str | None:
    """Strip / reject empty slot values. Returns None if unusable."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # Slot filler may return a list; take the first non-empty entry.
        for item in value:
            norm = _normalise_slot_value(item)
            if norm is not None:
                return norm
        return None
    text = str(value).strip()
    if not text:
        return None
    return text
