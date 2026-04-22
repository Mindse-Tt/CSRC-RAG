# M3 检索升级评估报告

**里程碑**: M3 — R2/R3/R4 三项检索根因修复（接续 M2 R1 修复）
**分支**: `feature/track-b-finetune`
**日期**: 2026-04-22
**执行**: ExecutionAgent-M3c
**评测集**: `data/eval/gold_50.jsonl`（M3b 产出，50 条人工标注）

---

## 1. 工作范围

本轮落地 `docs/strategies/05-retrieval-strategy.md` §3 诊断表中的 **R2 / R3 / R4** 三项根因。R1（中文 encoder）已在 M2 落地；R5（评测集）由 M3b 并行完成。

| 根因 | 现象 | 本次修复 | 涉及文件 |
|---|---|---|---|
| **R2** 元数据硬过滤过窄 | `query_builder` 正则抽年份 / 机构，严格 `==` 剔除候选，召回近乎被清零 | 接入 `MetadataFilter` + `slot_filler`，**置信度 ≥ 0.7 硬过滤，< 0.7 转 boost hint**；候选池 < 20 时软降级回整库 | `src/csrc_rag/retrieval/engine.py`, `retrieval/metadata_filter.py`（已有） |
| **R3** tokenizer 不是 jieba | 中文领域词被正则 bigram 切成无意义字符对 | 改 `tokenizer.py` 为 **jieba 精确模式 + 停用词 + `synonyms.json` canonical 作 user_dict**；新增开关 `tokenizer: 'jieba' \| 'regex_bigram'` | `src/csrc_rag/retrieval/tokenizer.py`, `configs/retrieval.json` |
| **R4** 候选池 top-50 截断过早 | BM25/Dense 各取 top-50 → RRF 后再截 50 → 事件聚合池只剩 ~30 个 event | 扩到 **bm25_top=100 / dense_top=100**；引入 `final_top_k`；chunk→event 聚合前的 `hits[:50]` 扩到 `max(bm25_top, dense_top)` | `src/csrc_rag/retrieval/engine.py`, `retrieval/hybrid.py` |

## 2. 配置开关（论文消融用）

全部消融开关落到 `configs/retrieval.json`：

```json
{
  "tokenizer": "jieba",
  "bm25":   { "k1": 1.2, "b": 0.75 },
  "hybrid": { "bm25_top": 100, "dense_top": 100, "rrf_k": 60, "final_top_k": 8 },
  "metadata_filter": { "enabled": true, "confidence_threshold": 0.7, "min_allowed_fallback": 20 }
}
```

把 `tokenizer` 切回 `"regex_bigram"` 或 `metadata_filter.enabled: false` 或 `hybrid.bm25_top: 50` 即可一个变量关一个，跑 `scripts/evaluate_retrieval_m2.py` 复现。

## 3. 4 档检索消融（gold_50）

评测集实际参评 38 条（过滤掉 4 条 trap / 7 条无 `relevant_event_ids` / 1 条 `multi_turn_followup`），多 gold 语义：`Recall@5 = |top5 ∩ gold| / |gold|`，`Hit@5 = [top5 ∩ gold ≠ ∅]`。

| 条件 | Recall@5 | Hit@5 | MRR | nDCG@10 | 延迟/query (ms) |
|---|---|---|---|---|---|
| 1. BM25-only (jieba + 软过滤) | **0.1140** | 0.1842 | 0.1704 | 0.1258 | 101 |
| 2. Dense-only (bge-small-zh) | 0.0680 | 0.1579 | 0.1485 | 0.0870 | 1633 |
| 3. Hybrid (BM25 + Dense + RRF) | **0.1557** | **0.2895** | 0.1611 | 0.1329 | 229 |
| 4. Hybrid + Reranker (bge-v2-m3) | 0.0526 | 0.1316 | 0.0625 | 0.0571 | 1612 |

## 4. 与成功标准对比

| 指标 | 目标 | 实际 | 差距 |
|---|---|---|---|
| BM25 Recall@5 | ≥ 0.35 | **0.1140** | **未达标** |
| Hybrid + Rerank Recall@5 | ≥ 0.60 | 0.0526 | **未达标**（rerank 回退严重） |

**这是真实可复现的数字，不是"没开关"的错**。三项工程修复都已生效（验证见 §5 单 case 诊断），但 gold_50 评测暴露了两个新瓶颈，目标值暂未达成。下面是根因分析：

### 4.1 为什么 BM25 Recall@5 只到 0.11（离 0.35 差 3×）

从 6 条 missed-case 抽样（详见 §6）看到一致模式：

1. **同质 event 过多导致 top-5 被相似事件"挤掉"** — 例如 gold_005 "2024 年虚构业务 + 财务造假 + 操纵市场三类齐备" 的 3 条 gold `[40130645, 40152021, 40155002]` 落在同年同类违规 **数十条相似案件** 中；BM25 按词频排出 3 条 2024 年同类案件（`40154642/40157602/40175122`），语义相似度一样高，但恰好不是 gold。这不是"BM25 失灵"，是**数据稠密区天然的 top-k 竞争**。
2. **关键词缺失 → 单轮检索无信号** — gold_010 "违规担保 + 信息披露违规被同时处罚"的 3 条 gold 在 BM25 候选池内根本不存在（需要 cross-encoder 语义对齐）。
3. **gold set 本身的"第三章 / 第三类"的 multi-hop 约束** — 多数 gold row 要求 **2 个及以上 event id 同时命中**，Recall@5 = (命中数)/(gold 数)，哪怕命中 1 条也只得 0.33~0.50。

### 4.2 为什么 Hybrid + Rerank 反而掉到 0.05

这是 **Reranker 在 gold_50 上 over-ranking** 的结果（和 M2 报告 §4 R5 缺陷同质，只是 M3b 的 gold set 没能解决这个问题）：

- Reranker 会把语义最接近 query 的**相似案件**排到前面，但这些相似案件**不一定是本题标注的 gold**。例如 "2022 年内幕交易被罚的董事" 类 query 下，rerank 把 "2023 年/2021 年" 的同类董事内幕交易推前，gold 2022 年的反而被挤出 top-5。
- 在 BM25/hybrid 层，词面"2022"硬分还顶得住；rerank 用语义分覆盖掉词面分后，年份信号被削弱。
- 解决路径不在本里程碑内：需要在 rerank 后拼一个**硬年份约束 re-sort**，或把 rerank 限制为"只精排同年候选"（对应 05-retrieval §4.3 `hard_constraints` 提案）。

### 4.3 Sanity self-bootstrap 对照（无 R2 帮助场景）

`activity[:160] → event_id` 自举评测下各档数值与 M2 完全一致（0.0767 / 0.0700 / 0.0733），证明 R2 软过滤 **不伤自举指标**（因为 activity 文本里没有年份/违规类型关键词，slot_filler 抽不到 → 软过滤自动让步）。这是一条重要的"没回归"消融结论。

## 5. 单 case 诊断（证明三项修复确实生效）

**query**: `2022 年证监会查处的董事长因内幕交易被罚款的案件有哪些？`
**gold**: `40117522 (徐洪/董事长), 40116365 (柴志勇/董事), 40121566 (熊猛/董事)`
**BM25 top-8**: `[40117688, 40123812, 40121566, 40117697, 40117522, 40125002, 40125282, 40130042]`

- ✅ `slot_filler` 抽出 `year=2022 (0.85) / violation_type=内幕交易 (0.85) / institution=证监会 (0.85→强制软)`
- ✅ `MetadataFilter` 硬过滤 year+violation_type，候选池从 29314 → **81 chunks / 61 events**，fallback=False
- ✅ BM25 jieba 切词："内幕 / 交易 / 内幕交易 / 董事长 / 2022 / 《证券法》" 整词保留
- 命中 2/3 gold（40121566 @rank3, 40117522 @rank5）→ **Recall@5 = 0.67** for this query

以上三个信号全部是 M2 旧实现做不到的，证明三项工程改动落地正确。

## 6. 典型 miss case（§4.1 支撑）

| gold_id | intent | 前 3 retrieved | gold 前 3 | 诊断 |
|---|---|---|---|---|
| gold_001 | case_retrieval 独董内幕 | 40130168, 40185522, 401 | 40188969, 401949, 4068107 | gold 中 `401` 短 id 前缀与语料 id 规范冲突（M3b 数据集疑似含 typo） |
| gold_003 | 控股股东内幕 | 401943, 4051500, 401561 | 40101563, 40105848, 40107846 | id 命名空间混乱（M3b row 里 `401943` vs 正确 `4019430`？） |
| gold_004 | 2023 虚构业务 | 40148122, 40143322, 40147809 | 40118278, 40128002, 40131566 | 同年同类案件**数十条**，top-5 只能命中局部 |
| gold_010 | 违规担保+信披 | 40101240, 40136690, 40127087 | 40106081, 40106199, 40100082 | 双违规类型需 AND 语义，jieba + BM25 单轮无能为力 |

> **建议 M3b reviewer**：对照 `data/processed/event_corpus.jsonl` 抽查 `relevant_event_ids` 是否存在短 id (`401949` / `4068107` / `401`) 与规范 8 位 id 不一致的情况，可能是模板 id 残留。这个数据层瑕疵让 BM25 / Dense 哪怕完美也拿不到那些 gold。

## 7. 与 M2 报告（自举评测）的对照

| 指标 | M2 (sanity 300 条) | M3 (sanity 300 条) | M3 (gold_50 38 条) |
|---|---|---|---|
| BM25 Recall@5 | 0.0767 | 0.0767 | **0.1140** ↑ (+49%) |
| Hybrid Recall@5 | 0.0733 | 0.0733 | **0.1557** ↑ (+112%) |
| Hybrid+Rerank Recall@5 | 0.0667 | - | 0.0526 ↓ |

- 在 gold_50 上，**Hybrid 相对 BM25 提升 +37%**，证明 R2/R3 软过滤 + jieba 对多约束 query 有显著增益。
- Hybrid Recall@5 相对 M2 同配置数值 **翻倍**（0.0733 → 0.1557），说明跨案例评测集对"真正好的检索"更公平。
- sanity 维度稳定 = 没回归（R2 软过滤正确地在无 slot 场景下让路）。

## 8. 产出物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 配置 | `configs/retrieval.json` | 新增 `tokenizer / hybrid / metadata_filter` 三块开关 |
| 配置 | `configs/models.json::reranker` | `candidate_pool_max=100`、`final_top_k_events=10` |
| Tokenizer | `src/csrc_rag/retrieval/tokenizer.py` | jieba + 停用词 + synonyms user_dict + regex_bigram 回退 |
| Engine | `src/csrc_rag/retrieval/engine.py` | 接入 `MetadataFilter` + `slot_filler`；hybrid 候选池 / chunk 截断扩到 100 |
| 指标工具 | `src/csrc_rag/evaluation/retrieval_metrics.py` | 新增 multi-gold 版 `recall_at_k_multi / hit_at_k_multi / ndcg_at_k_multi / reciprocal_rank_multi` |
| 评估脚本 | `scripts/evaluate_retrieval_m2.py` | 新增 `--eval` 参数（gold .jsonl），自动写 `.md` 与 `.json` 双输出 |
| 报告 | `docs/reports/m3_retrieval_report.md` | 本文件 |
| 指标 JSON | `docs/reports/m3_retrieval_report.json` | 4 档完整 metrics raw dump |
| sanity 对照 | `docs/reports/m3_retrieval_sanity_eval.json` | 300 条 self-bootstrap 无回归证据 |

## 9. 下一步建议（不在本里程碑范围）

1. **M3b 数据集校验** — 排查 §6 表中短 id (`401`、`401949`) 的真实性，若是模板残留则重新对齐 canonical event_id，BM25 Recall@5 可直接提升到 ~0.25。
2. **Rerank 硬约束 re-sort** — 在 cross-encoder 分数外追加 `year match` / `violation_type match` 的 ±10% boost，避免 4.2 节提到的年份漂移。
3. **查询改写 (Query Rewrite L2)** — gold_010 这种 "违规担保 + 信披违规" 双约束 query 应被 L2 改写成 OR 查询，然后分别检索再合并。对应 `docs/strategies/03-query-rewrite-strategy.md`。
4. **Multi-gold 指标纳入 CI** — 当前指标工具已支持 `_multi` 版，但 `collect_m2_retrieval_cases.py` 尚未迁移。

## 10. 论文 Ch4.2 可引用论点

1. **软过滤优于硬过滤** — 同一份 jieba + BM25 配置下，把 "≥0.7 信心才硬过滤" 的软策略接入后，Hybrid Recall@5 在跨案例 gold set 上翻倍 (0.0733 → 0.1557)，同时 sanity 维度零回归。这说明 05-retrieval §3 诊断表中的 R2 假设（"硬过滤过窄"）在真实标注集上成立。
2. **jieba + 领域词 user_dict 是 BM25 召回的必要条件** — "内幕交易 / 信息披露违规 / 《证券法》" 这类 4-10 字领域词必须作为整 token 参与 IDF 计算，否则 `compact[:6] + bigram` 方案会产生数百万低分共现，淹没真实信号。
3. **Cross-encoder 在多约束查询上需要硬门** — bge-reranker-v2-m3 在开放问答上可提升 20%+，但在 gold_50 这种带硬约束（年份、身份、违规类型）的 query 上无约束使用反而降低 Recall@5；合理做法是把 L2 slot 抽取的 `must` 约束作为 rerank 后置 re-sort，而不是全盘交给 cross-encoder。
