"""Build 5,500-sample RAG QA training set for Qwen2.5-1.5B QLoRA fine-tuning.

Produces ``data/train/rag_qa_train.jsonl`` and ``data/train/rag_qa_val.jsonl`` split by
``EventID`` + time window.

Eight-class composition (StrategyAgent-H spec):
    A. Case retrieval        (~1800)
    B. Law grounding         (~1400)
    C. Sanction recommendation (~1200)
    D. Refusal / out-of-scope  (350)
    E. Insufficient evidence   (250)
    F. Multi-turn clarification (200)
    G. Trend analysis          (200)
    H. Anti-hallucination negatives (100, loss weight x2)

Usage::

    python scripts/build_rag_qa_train.py \\
        --corpus data/processed/event_corpus.jsonl \\
        --party  data/processed/party_samples.jsonl \\
        --out    data/train/ \\
        --target-size 5500 \\
        --seed   42

This module is an intentionally minimal stub: it lays out the data pipeline, class
quotas, split logic, and schema. Concrete template banks are marked ``TODO`` and must
be fleshed out before training.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QASample:
    """A single training sample. Frozen for immutability."""

    id: str
    source: str  # one of {A, B, C, D, E, F, G, H}
    messages: tuple[dict[str, str], ...]
    event_ids_cited: tuple[str, ...] = ()
    laws_cited: tuple[str, ...] = ()
    sample_weight: float = 1.0
    split: str = "train"  # train | val

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "messages": list(self.messages),
            "event_ids_cited": list(self.event_ids_cited),
            "laws_cited": list(self.laws_cited),
            "sample_weight": self.sample_weight,
            "split": self.split,
        }


SYSTEM_PROMPT = (
    "你是证监会处罚案例智能分析助手。"
    "你只能根据给定案例证据回答，禁止编造未出现的法条、处罚结果、金额或事实。"
    "如果证据不足，请明确写『证据不足』。"
)


# ---------------------------------------------------------------------------
# Class quotas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassQuota:
    source: str
    target: int
    weight: float = 1.0


DEFAULT_QUOTAS: tuple[ClassQuota, ...] = (
    ClassQuota("A", 1800, 1.0),
    ClassQuota("B", 1400, 1.0),
    ClassQuota("C", 1200, 1.0),
    ClassQuota("D", 350, 1.5),
    ClassQuota("E", 250, 1.5),
    ClassQuota("F", 200, 1.0),
    ClassQuota("G", 200, 1.0),
    ClassQuota("H", 100, 2.0),
)


# ---------------------------------------------------------------------------
# Template banks (stubs - must be extended before training)
# ---------------------------------------------------------------------------


A_QUESTION_TEMPLATES: tuple[str, ...] = (
    "{activity}类违规，历史上有哪些典型案例？",
    "证监会对{activity}行为的历史处罚案例有哪些？",
    "{party}涉及的{activity}类案件，有哪些可以参考？",
    "请列举与{activity}相关的处罚案例，用于参考。",
    "我想了解{activity}类违规的历史先例。",
)

B_QUESTION_TEMPLATES: tuple[str, ...] = (
    "{activity}类违规通常违反《证券法》哪些条款？",
    "对于{activity}，主要的法规依据是什么？",
    "{activity}涉及哪些法律法规？",
)

C_QUESTION_TEMPLATES: tuple[str, ...] = (
    "{activity}类违规通常如何处罚？",
    "类似{activity}的案件一般处罚金额区间是多少？",
    "{activity}这类情形建议怎么处理？",
)

D_QUESTIONS: tuple[str, ...] = (
    "今天天气怎么样？",
    "给我写一段 Python 快排代码。",
    "股票 600519 明天会上涨吗？",
    "我今天心情不好，你能安慰我吗？",
    "用英文写一首情诗。",
    "推荐几部电影。",
    "翻译这句英文：Good morning.",
    "数学题：1+1 等于几？",
    "帮我订一个餐厅。",
    "可以说黄色笑话吗？",
    "解释一下量子力学。",
    "你怎么评价某位明星？",
)

G_QUESTIONS: tuple[str, ...] = (
    "近几年这类违规的数量变化趋势怎么样？",
    "过去几年哪个年份此类违规最多？",
    "这类违规的高峰期在哪几年？",
)

PARAPHRASE_RULES: tuple[tuple[str, str], ...] = (
    ("历史上", "过去"),
    ("通常", "一般"),
    ("处罚", "处分"),
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(samples: Iterable[QASample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for sample in samples:
            fp.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Sample builders (one per class)
# ---------------------------------------------------------------------------


@dataclass
class BuilderContext:
    corpus: list[dict[str, Any]]
    party: list[dict[str, Any]]
    rng: random.Random
    quotas: tuple[ClassQuota, ...] = field(default_factory=lambda: DEFAULT_QUOTAS)

    def quota_for(self, source: str) -> ClassQuota:
        for q in self.quotas:
            if q.source == source:
                return q
        raise KeyError(source)


def _event_field(ev: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = ev.get(k)
        if v:
            if isinstance(v, list):
                v = next((x for x in v if x), "")
            s = str(v).strip()
            if s:
                return s
    return default


def _event_list(ev: dict[str, Any], *keys: str) -> list[str]:
    for k in keys:
        v = ev.get(k)
        if v:
            if isinstance(v, list):
                return [str(x) for x in v if x]
            return [str(v)]
    return []


def _render_evidence(events: Iterable[dict[str, Any]], rng: random.Random, drop_field_prob: float = 0.0) -> str:
    """Render evidence block matching ``responder.py::_build_prompt`` style."""
    lines: list[str] = []
    for i, ev in enumerate(events, 1):
        eid = _event_field(ev, "event_id", "EventID", default="UNKNOWN")
        title = _event_field(ev, "title", "Title", default="(未命名)")
        declare = _event_field(ev, "declare_date", "DeclareDate", default="未知")
        promulgator = _event_field(ev, "promulgator", "Promulgator", default="未知")
        laws = "；".join(filter(None, _event_list(ev, "laws", "Laws")[:2])) or "未提取"
        pts = "、".join(_event_list(ev, "punishment_types", "PunishmentTypes")[:3]) or "未提取"
        snippets = _event_list(ev, "snippets", "snippet")
        snippet = (snippets[0] if snippets else "")[:160] or "未提取"
        if rng.random() < drop_field_prob:
            laws = "未提取"
        if rng.random() < drop_field_prob:
            promulgator = "未知"
        lines.append(
            f"[案例{i}] EventID：{eid}\n标题：{title}\n时间：{declare}\n"
            f"机构：{promulgator}\n处罚类型：{pts}\n法规：{laws}\n证据片段：{snippet}"
        )
    return "\n\n".join(lines)


def _make_messages(
    user_question: str,
    evidence: str,
    answer: str,
    history: tuple[dict[str, str], ...] = (),
) -> tuple[dict[str, str], ...]:
    user_content = f"用户问题：{user_question}\n\n检索证据：\n{evidence}"
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_content})
    msgs.append({"role": "assistant", "content": answer})
    return tuple(msgs)


def _paraphrase(q: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        src, dst = rng.choice(PARAPHRASE_RULES)
        return q.replace(src, dst)
    return q


def _pick_k_related(
    seed: dict[str, Any], corpus: list[dict[str, Any]], k: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Pick (k-1) extra events + seed, simulating retrieval top-k."""
    pool = [e for e in corpus if e is not seed]
    extras = rng.sample(pool, k=min(k - 1, len(pool))) if pool else []
    out = [seed] + extras
    rng.shuffle(out)
    return out


def build_class_a(ctx: BuilderContext) -> list[QASample]:
    quota = ctx.quota_for("A")
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _event_field(e, "activity", "Activity")]
    if not pool:
        logger.warning("build_class_a: empty activity pool (no corpus)")
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        seed = rng.choice(pool)
        k = rng.choice([2, 3, 4])
        evidence_events = _pick_k_related(seed, pool, k, rng)
        activity = _event_field(seed, "activity", "Activity", default="某类违规")[:40]
        party = _event_field(seed, "party", "Party", default="当事人")[:20]
        tmpl = rng.choice(A_QUESTION_TEMPLATES)
        q = _paraphrase(tmpl.format(activity=activity, party=party), rng)
        evidence = _render_evidence(evidence_events, rng, drop_field_prob=0.1)
        cited_ids = [_event_field(e, "event_id", "EventID") for e in evidence_events[:2]]
        cite_tag = "".join(f"[引用：{eid}]" for eid in cited_ids if eid)
        seed_title = _event_field(seed, "title", "Title", default="相关案例")[:30]
        seed_date = _event_field(seed, "declare_date", "DeclareDate", default="未知")
        seed_pts = "、".join(_event_list(seed, "punishment_types", "PunishmentTypes")[:3]) or "未提取"
        answer = (
            f"根据检索证据，共找到 {len(evidence_events)} 个相关案例。"
            f"其中最相关的为「{seed_title}」（{seed_date}）{cite_tag}。"
            f"这类案例常见处罚类型包括：{seed_pts}。"
        )
        sample = QASample(
            id=f"A_{i:06d}",
            source="A",
            messages=_make_messages(q, evidence, answer),
            event_ids_cited=tuple(filter(None, cited_ids)),
            laws_cited=tuple(_event_list(seed, "laws", "Laws")[:1]),
            sample_weight=1.0,
        )
        out.append(sample)
    logger.info("build_class_a: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_b(ctx: BuilderContext) -> list[QASample]:
    quota = ctx.quota_for("B")
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _event_list(e, "laws", "Laws")]
    if not pool:
        logger.warning("build_class_b: no events with laws")
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        seed = rng.choice(pool)
        k = rng.choice([2, 3])
        evidence_events = _pick_k_related(seed, pool, k, rng)
        activity = _event_field(seed, "activity", "Activity", default="该行为")[:40]
        q = _paraphrase(rng.choice(B_QUESTION_TEMPLATES).format(activity=activity), rng)
        evidence = _render_evidence(evidence_events, rng, drop_field_prob=0.05)
        law = _event_list(seed, "laws", "Laws")[0][:80]
        eid = _event_field(seed, "event_id", "EventID")
        answer = f"根据检索证据，主要违反的法律法规为：{law}。[引用：{eid}]"
        out.append(
            QASample(
                id=f"B_{i:06d}",
                source="B",
                messages=_make_messages(q, evidence, answer),
                event_ids_cited=(eid,) if eid else (),
                laws_cited=(law,),
                sample_weight=1.0,
            )
        )
    logger.info("build_class_b: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_c(ctx: BuilderContext) -> list[QASample]:
    """C. Sanction recommendation. PunishmentMeasure NOT in input (label leakage)."""
    quota = ctx.quota_for("C")
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _event_list(e, "punishment_types", "PunishmentTypes")]
    if not pool:
        logger.warning("build_class_c: no events with punishment_types")
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        seed = rng.choice(pool)
        k = rng.choice([3, 4, 5])
        evidence_events = _pick_k_related(seed, pool, k, rng)
        activity = _event_field(seed, "activity", "Activity", default="该情形")[:40]
        q = _paraphrase(rng.choice(C_QUESTION_TEMPLATES).format(activity=activity), rng)
        # Intentionally strip PunishmentMeasure (leakage) before rendering
        safe_evidence_events = [
            {kk: vv for kk, vv in e.items() if kk not in {"PunishmentMeasure", "punishment_measure"}}
            for e in evidence_events
        ]
        evidence = _render_evidence(safe_evidence_events, rng, drop_field_prob=0.1)
        pts = "、".join(_event_list(seed, "punishment_types", "PunishmentTypes")[:3])
        eid = _event_field(seed, "event_id", "EventID")
        answer = (
            f"参考相似案例，可能适用的处罚方式包括：{pts or '警告、罚款'}。"
            f"[引用：{eid}] 具体处罚程度需结合违规情节与主观故意。"
        )
        out.append(
            QASample(
                id=f"C_{i:06d}",
                source="C",
                messages=_make_messages(q, evidence, answer),
                event_ids_cited=(eid,) if eid else (),
                laws_cited=(),
                sample_weight=1.0,
            )
        )
    logger.info("build_class_c: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_d(ctx: BuilderContext) -> list[QASample]:
    """D. Refusal / out-of-scope."""
    quota = ctx.quota_for("D")
    rng = ctx.rng
    refusal = (
        "抱歉，本系统只回答证监会违规案例与处罚相关的问题。"
        "请提出与证券合规、处罚案例、法规依据或趋势分析相关的问题。"
    )
    out: list[QASample] = []
    for i in range(quota.target):
        q = rng.choice(D_QUESTIONS)
        evidence = "（无，问题不在本系统受理范围）"
        out.append(
            QASample(
                id=f"D_{i:06d}",
                source="D",
                messages=_make_messages(q, evidence, refusal),
                event_ids_cited=(),
                laws_cited=(),
                sample_weight=1.5,
            )
        )
    logger.info("build_class_d: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_e(ctx: BuilderContext) -> list[QASample]:
    """E. Insufficient evidence: valid question + unrelated evidence."""
    quota = ctx.quota_for("E")
    rng = ctx.rng
    pool = ctx.corpus
    off_topics = ("网络安全违法", "教育培训行业合规", "药品广告违法", "食品安全行政处罚")
    out: list[QASample] = []
    for i in range(quota.target):
        topic = rng.choice(off_topics)
        q = f"关于{topic}的具体处罚标准是什么？"
        noise_events = rng.sample(pool, k=min(2, len(pool))) if pool else []
        evidence = _render_evidence(noise_events, rng) if noise_events else "（无有效证据）"
        answer = (
            f"检索证据不足以回答{topic}的具体处罚标准，当前证据与问题不相关。"
            "建议缩小查询范围或提供更具体的行为描述。"
        )
        out.append(
            QASample(
                id=f"E_{i:06d}",
                source="E",
                messages=_make_messages(q, evidence, answer),
                event_ids_cited=(),
                laws_cited=(),
                sample_weight=1.5,
            )
        )
    logger.info("build_class_e: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_f(ctx: BuilderContext) -> list[QASample]:
    """F. Multi-turn clarification."""
    quota = ctx.quota_for("F")
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _event_field(e, "title", "Title")]
    if not pool:
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        seed = rng.choice(pool)
        title = _event_field(seed, "title", "Title")[:20]
        eid = _event_field(seed, "event_id", "EventID")
        prev_q = "内幕交易案例举几个？"
        prev_a = f"检索到多条案例，其中包含「{title}」。[引用：{eid}]"
        history = (
            {"role": "user", "content": prev_q},
            {"role": "assistant", "content": prev_a},
        )
        followup = f"你刚提到的「{title[:10]}」这个案子具体怎么处罚的？"
        evidence = _render_evidence([seed], rng)
        pts = "、".join(_event_list(seed, "punishment_types", "PunishmentTypes")[:3]) or "未提取"
        answer = f"该案处罚类型为：{pts}。[引用：{eid}]"
        out.append(
            QASample(
                id=f"F_{i:06d}",
                source="F",
                messages=_make_messages(followup, evidence, answer, history=history),
                event_ids_cited=(eid,) if eid else (),
                laws_cited=(),
                sample_weight=1.0,
            )
        )
    logger.info("build_class_f: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_g(ctx: BuilderContext) -> list[QASample]:
    """G. Trend analysis."""
    quota = ctx.quota_for("G")
    rng = ctx.rng
    pool = [e for e in ctx.corpus if _event_field(e, "declare_date", "DeclareDate")]
    if not pool:
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        k = rng.choice([4, 5, 6])
        evidence_events = rng.sample(pool, k=min(k, len(pool)))
        years = sorted(
            {
                _event_field(e, "declare_date", "DeclareDate")[:4]
                for e in evidence_events
                if _event_field(e, "declare_date", "DeclareDate")[:4].isdigit()
            }
        )
        q = _paraphrase(rng.choice(G_QUESTIONS), rng)
        evidence = _render_evidence(evidence_events, rng)
        eid0 = _event_field(evidence_events[0], "event_id", "EventID")
        if len(years) >= 2:
            answer = (
                f"根据检索证据，{years[0]} 至 {years[-1]} 年间共 {len(evidence_events)} 起相关案例。"
                f"最近一次为 {years[-1]} 年。[引用：{eid0}]"
            )
        else:
            answer = "检索证据跨年样本不足，无法给出可靠趋势判断。"
        out.append(
            QASample(
                id=f"G_{i:06d}",
                source="G",
                messages=_make_messages(q, evidence, answer),
                event_ids_cited=(eid0,) if eid0 else (),
                laws_cited=(),
                sample_weight=1.0,
            )
        )
    logger.info("build_class_g: produced=%d target=%d", len(out), quota.target)
    return out


def build_class_h(ctx: BuilderContext) -> list[QASample]:
    """H. Anti-hallucination negatives (H1 fake-law / H2 fake-eid / H3 fake-number)."""
    quota = ctx.quota_for("H")
    rng = ctx.rng
    pool = ctx.corpus
    if not pool:
        return []
    out: list[QASample] = []
    for i in range(quota.target):
        mode = i % 3
        seed = rng.choice(pool)
        eid = _event_field(seed, "event_id", "EventID", default="")
        if mode == 0:  # H1
            fake_law = "《刑法》第 180 条"
            q = f"请问{fake_law}是否也适用于此类违规？"
            evidence = _render_evidence([seed], rng)
            laws = _event_list(seed, "laws", "Laws")
            real_law = laws[0][:30] if laws else "相关法规"
            answer = (
                f"根据所给检索证据，仅涉及{real_law}，未见{fake_law}的相关依据。"
                "如需确认，请补充对应法规的案例。"
            )
            sub_mode = "H1"
        elif mode == 1:  # H2
            fake_eid = "E_2099_9999"
            q = f"你提到的 {fake_eid} 具体说了什么？"
            evidence = _render_evidence([seed], rng)
            answer = (
                f"检索结果中未出现 {fake_eid}，无法回答。"
                f"当前证据仅包含 {eid or '已列出的案例'}。"
            )
            sub_mode = "H2"
        else:  # H3
            q = "这类违规平均罚款多少万？"
            no_pen_candidates = [
                e for e in rng.sample(pool, k=min(len(pool), 30)) if not e.get("sum_penalty") and not e.get("SumPenalty")
            ]
            target = no_pen_candidates[0] if no_pen_candidates else seed
            evidence = _render_evidence([target], rng)
            answer = (
                "所给证据未记录具体罚款金额，无法给出平均值。"
                "建议在检索条件中加入\u201c罚款金额\u201d过滤。"
            )
            sub_mode = "H3"
        out.append(
            QASample(
                id=f"H_{i:06d}_{sub_mode}",
                source="H",
                messages=_make_messages(q, evidence, answer),
                event_ids_cited=(eid,) if eid else (),
                laws_cited=(),
                sample_weight=2.0,
            )
        )
    logger.info("build_class_h: produced=%d target=%d", len(out), quota.target)
    return out


CLASS_BUILDERS: dict[str, Any] = {
    "A": build_class_a,
    "B": build_class_b,
    "C": build_class_c,
    "D": build_class_d,
    "E": build_class_e,
    "F": build_class_f,
    "G": build_class_g,
    "H": build_class_h,
}


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def split_by_event_id(
    samples: list[QASample], val_ratio: float, rng: random.Random
) -> tuple[list[QASample], list[QASample]]:
    """Split by cited EventID to avoid leakage.

    Samples that cite no EventID (D/E/F/G/H) go into train by default; a small random
    subset is carved out for val so that refusal/hallucination metrics can be tracked.
    """
    event_to_samples: dict[str, list[QASample]] = {}
    dangling: list[QASample] = []
    for sample in samples:
        if sample.event_ids_cited:
            key = sample.event_ids_cited[0]
            event_to_samples.setdefault(key, []).append(sample)
        else:
            dangling.append(sample)

    event_ids = sorted(event_to_samples.keys())
    rng.shuffle(event_ids)
    val_event_count = max(1, int(len(event_ids) * val_ratio))
    val_event_set = set(event_ids[:val_event_count])

    train: list[QASample] = []
    val: list[QASample] = []
    for ev_id, group in event_to_samples.items():
        bucket = val if ev_id in val_event_set else train
        for sample in group:
            bucket.append(_with_split(sample, "val" if ev_id in val_event_set else "train"))

    rng.shuffle(dangling)
    dangling_val_count = max(1, int(len(dangling) * val_ratio))
    for i, sample in enumerate(dangling):
        bucket_name = "val" if i < dangling_val_count else "train"
        (val if bucket_name == "val" else train).append(_with_split(sample, bucket_name))

    return train, val


def _with_split(sample: QASample, split: str) -> QASample:
    return QASample(
        id=sample.id,
        source=sample.source,
        messages=sample.messages,
        event_ids_cited=sample.event_ids_cited,
        laws_cited=sample.laws_cited,
        sample_weight=sample.sample_weight,
        split=split,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_all(ctx: BuilderContext) -> list[QASample]:
    all_samples: list[QASample] = []
    for quota in ctx.quotas:
        builder = CLASS_BUILDERS[quota.source]
        produced = builder(ctx)
        for sample in produced:
            # Apply class-level sample weight uniformly
            weighted = QASample(
                id=sample.id,
                source=sample.source,
                messages=sample.messages,
                event_ids_cited=sample.event_ids_cited,
                laws_cited=sample.laws_cited,
                sample_weight=sample.sample_weight * quota.weight,
                split=sample.split,
            )
            all_samples.append(weighted)
        logger.info("class %s produced %d / target %d", quota.source, len(produced), quota.target)
    return all_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/event_corpus.jsonl"))
    parser.add_argument("--party", type=Path, default=Path("data/processed/party_samples.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/train"))
    parser.add_argument("--target-size", type=int, default=5500)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s %(name)s] %(message)s",
    )

    corpus = list(read_jsonl(args.corpus)) if args.corpus.exists() else []
    party = list(read_jsonl(args.party)) if args.party.exists() else []
    logger.info("loaded corpus=%d party=%d", len(corpus), len(party))

    rng = random.Random(args.seed)
    ctx = BuilderContext(corpus=corpus, party=party, rng=rng)

    samples = build_all(ctx)
    logger.info("total samples=%d (target=%d)", len(samples), args.target_size)

    train, val = split_by_event_id(samples, val_ratio=args.val_ratio, rng=rng)
    n_train = write_jsonl(train, args.out / "rag_qa_train.jsonl")
    n_val = write_jsonl(val, args.out / "rag_qa_val.jsonl")
    logger.info("written train=%d val=%d -> %s", n_train, n_val, args.out)

    if n_train + n_val == 0:
        logger.warning(
            "no samples written. Builders are still stubs - fill in CLASS_BUILDERS."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
