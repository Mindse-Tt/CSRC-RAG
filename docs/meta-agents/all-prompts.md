# 18 个 Meta-Agent 的 Prompt 结构归档

> 归档者：**ArchiveAgent-Meta**（Opus 4.7）
> 归档时间：2026-04-22
> 说明：以下 prompt 为"**精炼再创作版**"，忠实反映派令时的任务结构、输入输出要求与禁止事项，但**不宣称是逐字原文**。每个 agent 的 prompt 原文以 `docs/strategies/0x-*.md` 文件头部的 "Owner/目标/约束" 段落作为事实锚点可交叉校验。

---

## 0. 共享上下文（所有第 1 批 agent 派令前置）

派出 12 个策略 agent 前，Coordinator 先把 `docs/strategies/00-agent-team-shared-context.md` 作为**全员必读共识**推送。其中锁定：
- 项目一句话（赛道 B · 5 人 · 2-4 周）
- 5 个硬约束（LoRA 对比 / 幻觉缓解三组 / ≥5 组消融 / EventID+时间三切 / `PunishmentMeasure` 不进输入）
- 现有资产（`event_corpus.jsonl` 4,233 / `event_chunks.jsonl` 29,314 / `party_samples.jsonl` 14,740）
- 七层链路 L0→L7
- 禁止事项：不动别人文件、不 commit、不改评估口径与数据切分

---

## 1. 第一批 · 12 个策略 Agent（2026-04-22 并行派出）

### Agent A · 拒答策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~180 秒（单次并行输出 500–1500 字策略文档）
- **核心职责**：为 L1 意图识别 + L3/L5/L7 多级兜底设计拒答策略，拦住越界与幻觉。
- **Prompt 结构**（精炼版）：
  - 背景：接到共享上下文，知道本项目是证监会处罚 RAG；7 类意图已预设。
  - 任务清单：
    1. 定义 `greeting / chitchat / out_of_scope / case_retrieval / law_grounding / sanction_recommendation / trend_analysis` 7 类。
    2. 给出 4 级兜底（L1 预判 / L3 空召回 / L5 低置信度 / L7 引证失败）。
    3. 针对"保险/银行"近邻金融场景设计**软越界**话术。
    4. 产出 `configs/reject.yaml` 骨架 + ≥7 条话术模板。
  - 产出要求：500–1500 字 + 至少 1 张 mermaid 流程图 + JSON schema + 风险兜底。
  - 禁止：不动 `src/csrc_rag/orchestration/` 别人的现有文件；不 commit。
- **产出资产**：
  - 文档：`docs/strategies/01-reject-strategy.md`
  - Stub：`configs/reject.yaml` 雏形，`src/csrc_rag/orchestration/reject_policy.py` 签名
- **人工把关点（Coordinator）**：
  - 把"保险类全部拒答"改为"软越界 + 引导回证券"（裁决 C2，记录在 99-leader §8）。
  - 对外 7 类但允许 case_retrieval 内部分 3 子意图（裁决 C5）。

---

### Agent B · 数据处理策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~220 秒
- **核心职责**：从 14,740 行原始表产出两级索引 + 6 类衍生数据 + 三切 splits，严防 `PunishmentMeasure` 泄漏。
- **Prompt 结构**（精炼版）：
  - 背景：CNRDS 原表 14,740 行 × 24 列，已有 `event_corpus.jsonl` (4,233) / `event_chunks.jsonl` (29,314) / `party_samples.jsonl` (14,740)。
  - 任务清单：
    1. 设计 6 类衍生数据用途（检索语料 / 监督标签 / QA 构造 / 评估金标 / 反幻觉负样本 / 词表）。
    2. 按 **EventID + 时间** 双重切分（Train 1994-2021 / Val 2022-2023 / Test 2024-2025）。
    3. 质量检查器：断言 `PunishmentMeasure` 不流入 model input。
    4. 输出 JSON schema（每条 chunk 的字段名、类型、取值范围）。
  - 产出要求：风险登记（数据口径差异、类别不平衡）、兜底方案。
  - 禁止：不改评估口径；不改切分规则。
- **产出资产**：
  - 文档：`docs/strategies/02-data-strategy.md`
  - Stub：`scripts/split_by_event.py` 签名、`src/csrc_rag/data/quality_checker.py` 签名
- **人工把关点**：
  - 强制按 EventID+时间三切而非随机（裁决 C7）。
  - I 评估模块额外留 20% 随机切做 sanity check（裁决 C7 补充）。
  - 数据口径统一写"14,740 原始 → 4,233 事件 + 14,740 当事人"，"8,000" 废弃（开题报告对齐）。

---

### Agent C · Query 改写策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~160 秒
- **核心职责**：L2 层 Query 重写——共指消解 + 同义词扩展 + 槽位抽取（年份/机构/违规类型）。
- **Prompt 结构**（精炼版）：
  - 背景：用户提问口语化，直接检索 Recall 低；需要 rewrite 到更接近语料的表达。
  - 任务清单：
    1. 规则优先 + LLM fallback 的二层架构（贵的模型只在规则失败时用）。
    2. 同义词分强/弱两级：强同义词直接 OR；弱同义词权重 0.3。
    3. 槽位抽取 F1 目标：年份 ≥0.90 / 机构 ≥0.80 / 违规类型 ≥0.75。
    4. 依赖 Agent B 的 `configs/synonyms.json`（约 200 对）。
  - 产出要求：接口 schema、评估方法、至少 1 张 mermaid。
  - 禁止：不写实际 LLM 调用代码（留 stub 给 D7 合并）。
- **产出资产**：
  - 文档：`docs/strategies/03-query-rewrite-strategy.md`
  - Stub：`src/csrc_rag/orchestration/rewriter.py` 签名；附录 A ~60 条同义词样例
- **人工把关点**：
  - 同义词分级策略来自冲突裁决 C4（C 激进扩 vs E 精准扩）。
  - `configs/synonyms.json` 最终扩到 **930 短语**（M1c 完成），远超策略期 200 对下限。

---

### Agent D · Planner 训练策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~170 秒
- **核心职责**：L1 意图 7 分类器的训练流水线，Macro-F1 ≥ 0.92。
- **Prompt 结构**（精炼版）：
  - 背景：现有意图分类器太弱；需升级到 7 类 + 子意图双层标签。
  - 任务清单：
    1. 阶段 1 baseline：TF-IDF + LR（分钟级训完，作为论文消融对照）。
    2. 阶段 2 主路线：MiniLM-L12-v2 冷冻特征 + LR（论文主提交）。
    3. 数据规模：7 类 × 500 = **3,500 条**（30 种子 + LLM 扩增 470）。
    4. 验收：Macro-F1 ≥ 0.92 · P95 延迟 ≤ 5ms。
  - 产出要求：输出 `QueryPlan` schema（intent / sub_intent / confidence / slots）。
  - 禁止：不写前端展示；不和 responder 耦合。
- **产出资产**：
  - 文档：`docs/strategies/04-planner-training-strategy.md`
  - Stub：`scripts/train_intent_classifier_v2.py`（后由 M1b 落地）
- **人工把关点**：
  - case_retrieval 内部分 3 子意图（案例/相似/机构类），采纳裁决 C5。
  - 训练数据从 210 种子升到 3500（M1b 执行中）。

---

### Agent E · 检索策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~240 秒
- **核心职责**：L3 双路召回（BM25 + bge-small-zh）+ RRF 融合，Recall@5 从 0.09 拉到 ≥0.70。
- **Prompt 结构**（精炼版）：
  - 背景：现 `chunk_embeddings.npy` 是 MiniLM 英文向量，中文检索严重失配；硬过滤过度；tokenizer 非 jieba。
  - 任务清单：
    1. 识别 **5 个根因**并分别给修复方案（英文 embedding 换 bge-small-zh / 硬过滤降级为软加权 / tokenizer 切 jieba / RRF k=30/60 / 双层索引兼容 chunk 粒度）。
    2. 输出 `build_dense_index_bge.py` 骨架 + `metadata_filter.py` 软加权策略。
    3. 目标 Recall@5 ≥ 0.70、nDCG@10 ≥ 0.65。
  - 产出要求：3 组基线对比表模板（BM25 / Hybrid / +Rerank）。
- **产出资产**：
  - 文档：`docs/strategies/05-retrieval-strategy.md`
  - Stub：`scripts/build_dense_index_bge.py`、`src/csrc_rag/retrieval/metadata_filter.py`
- **人工把关点**：
  - chunk 长度从"150 字"改为 **300–500 字** + `top_k=5~8`（裁决 C1）。
  - Recall 目标口径澄清：05 的 0.70 是"修复后基线"，09 的 0.85 是"+Rerank 目标"（裁决 C13）。

---

### Agent F · 重排策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~150 秒
- **核心职责**：L3d 重排层——bge-reranker-v2-m3 + 业务加权（机构权威 × 严重性）。
- **Prompt 结构**（精炼版）：
  - 背景：60 候选 → 5–8 条事件；需要 cross-encoder 精排 + 业务先验加权。
  - 任务清单：
    1. 基础重排：bge-reranker-v2-m3（FP16 GPU / ONNX-INT8 CPU）。
    2. 业务加权：仅在 `sanction_recommendation` 意图开 severity boost；其他意图只用基础分。
    3. 目标 nDCG@10 ≥ 0.80，P95 延迟 ≤ 3s。
  - 产出要求：`configs/rerank.json` 超参 + 降级策略（reranker-base / 候选 top-20）。
- **产出资产**：
  - 文档：`docs/strategies/06-reranking-strategy.md`
  - Stub：`src/csrc_rag/retrieval/reranker.py`
- **人工把关点**：
  - 默认 `top_k=5`，消融时测 3/5/8（裁决 C6）。
  - severity boost 只对 sanction 意图开，防止污染其他意图（Coordinator 拍板）。

---

### Agent G · 回复策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~200 秒
- **核心职责**：L5 生成 + L7 引证校验 —— 四意图四模板 + 强证据 Prompt + 3 级兜底。
- **Prompt 结构**（精炼版）：
  - 背景：LoRA 后的 Qwen2.5-1.5B 生成答案；必须强制引证 EventID 白名单。
  - 任务清单：
    1. 四意图各写一套 Jinja2 模板（system + few-shot + user）。
    2. 强证据 System Prompt 三档：**宽 / 中 / 严**，用于 G0–G4 消融对照。
    3. L7 引证校验 8 条规则（EventID 白名单、法条正则、禁编造当事人等）。
    4. 3 级兜底链：生成失败 → 模板退化 → 拒答话术。
  - 产出要求：引证命中 ≥ 95%、幻觉率 ≤ 5%。
- **产出资产**：
  - 文档：`docs/strategies/07-response-strategy.md`
  - Stub：`configs/prompt_templates/*.j2`、`src/csrc_rag/response/responder.py` hook
- **人工把关点**：
  - Prompt 严格度作为消融维度保留，最终提交用 G3=LoRA+严格（裁决 C11）。
  - trend 意图先表格 + 柱状图，复杂图 v2（裁决 C8）。

---

### Agent H · 回复模型训练策略（QLoRA）
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~230 秒
- **核心职责**：Qwen2.5-1.5B + QLoRA 训练方案 + 三阶段 Ablation V1/V2/V3。
- **Prompt 结构**（精炼版）：
  - 背景：赛道 B 硬性要求 LoRA 前后对比；本机起初被标注为"无 GPU"→ Colab T4。
  - 任务清单：
    1. 超参：NF4 量化、double_quant、`r=16, α=32, dropout=0.05, lr=2e-4, 3 epoch, cosine`。
    2. 训练数据：5,500 条 8 类 QA（A 案例 1,800 / B 法条 1,200 / C 处罚 1,000 / D 趋势 400 / E 拒答 300 / F 证据不足 300 / G 多轮 200 / H 反幻觉 300）。
    3. 三阶段 Ablation：V1 (A+B+C=4000) / V2 (+D+E+F=5000) / V3 (+G+H=5500)。
    4. 验收：BERTScore-F1 +3pp、幻觉率 -5pp。
  - 产出要求：超参 JSON + Colab 一键脚本 + 抗模板过拟合 4 条技巧。
  - **原 prompt 指定 `compute_dtype=bfloat16, per_device_batch=4`**。
- **产出资产**：
  - 文档：`docs/strategies/08-response-training-strategy.md`
  - Stub：`scripts/train_qlora_qwen.py`、`configs/qlora_config.json`
- **人工把关点（关键）**：
  - ⚠️ **用户后来拍板本机是 RTX 2060 SUPER 8GB（Turing 架构，不支持 bf16）**。Coordinator 裁决 C12：**全链路改 fp16** + `per_device_batch=2, grad_accum=8`（有效 batch 仍 16）。
  - 该修正由 M1a 执行 bf16→fp16 patch，M1d 做 smoke test 验证。
  - Epoch=3 + val-loss 早停 patience=2（裁决 C9）。

---

### Agent I · 评估策略
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~210 秒
- **核心职责**：横切全员 —— 三级评估 + 7 组消融 + 50 条人工 blind 打分 + Fleiss' κ。
- **Prompt 结构**（精炼版）：
  - 背景：论文需定量数据填表；组员需统一评测 SDK。
  - 任务清单：
    1. 三级评估：L1 组件（Recall@k/MRR/nDCG）→ L2 生成 G0-G4（五组对照）→ L3 端到端 blind（5 人 Likert）。
    2. 7 组消融矩阵：BM25 / Dense / Hybrid / +Rerank / 改写 / metadata 过滤 / LoRA / Prompt 约束 / top-k。
    3. 50 条金标：4 类意图分层（case 15 / law 10 / sanction 10 / trend 10 / 边界 5）。
    4. 一致性：Fleiss' κ ≥ 0.6，<0.6 重标。
  - 产出要求：统一评测 SDK（基类 + `compute()` 接口），各 agent 截止 D7 对接。
- **产出资产**：
  - 文档：`docs/strategies/09-evaluation-strategy.md`
  - Stub：`scripts/evaluate_retrieval_sanity.py` / `evaluate_generation.py` / `evaluate_end_to_end.py`
- **人工把关点**：
  - 评测 SDK 接口集中 vs 代码分散（裁决 C10，I 定接口、各 agent 只实现 compute）。
  - 幻觉率自动化方案：L7 自动判 EventID 白名单 + 法条正则 + 人工 20% 抽检。

---

### Agent J · 模型选择
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~190 秒
- **核心职责**：6 模型家族对比矩阵 + 最终选型 + `configs/models.recommended.json`。
- **Prompt 结构**（精炼版）：
  - 背景：需要锁定 生成器 / Embedding / Reranker / 意图分类 / 辅线 MacBERT / API 对照 六类模型。
  - 任务清单：
    1. 对每类列 3–5 个候选，从中文能力 / 尺寸 / license / 推理开销 四维评分。
    2. 给出最终推荐 + 降级路径。
    3. 与 Agent H 串联：QLoRA 基座必须在候选内。
  - 产出要求：`configs/models.recommended.json`。
- **产出资产**：
  - 文档：`docs/strategies/10-model-selection.md`
  - 配置：`configs/models.recommended.json`
  - 最终选型：Qwen2.5-1.5B-Instruct / bge-small-zh-v1.5 / bge-reranker-v2-m3 / MiniLM-L12-v2+LR（+MacBERT 辅线） / DeepSeek-Chat（G4 API 对照）
- **人工把关点**：
  - 生成基座锁 1.5B（裁决 C3），3B/7B 作为 `experiments.alternate_generator` 可选。
  - 原 prompt 假设本机无 GPU → 后被用户拍板 RTX 2060S 覆盖。

---

### Agent K · 前后端工程
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~200 秒
- **核心职责**：把现 `run_demo_server.py + web/index.html` 升级为**七层可视化答辩 Demo**。
- **Prompt 结构**（精炼版）：
  - 背景：现前端只有三层 MVP；答辩需要展示 L0–L7 全链路 trace。
  - 任务清单：
    1. 后端：FastAPI + Pydantic v2，一次请求返 26 字段 trace；保留旧 http.server 作离线回退。
    2. 前端：HTML + ECharts 本地化（不走 CDN，防断网）；7 类意图配色 badge。
    3. **降级优先 > 正确性优先 > 速度优先**：12 层降级路径。
    4. 所有配置走环境变量 + `configs/*.json`。
  - 产出要求：API P95 < 3s (template) / < 8s (LoRA CPU)。
- **产出资产**：
  - 文档：`docs/strategies/11-engineering-strategy.md`
  - Stub：`src/csrc_rag/api/{schemas.py, server_v2.py}`、`web/index_v2.html`、4 幕 Demo 口播脚本
- **人工把关点**：
  - trend 图复杂度裁决 C8（先柱状图，折线 v2）。
  - Demo 现场断网风险 R5：预录 3 条离线录屏 + ECharts 本地化。

---

### Agent L · Leader（总统筹）
- **模型**：Opus 4.7
- **派出时间**：2026-04-22（与 11 策略 agent 同批）
- **运行时长**：~280 秒（最重，要读全部其他 agent 的预期产出）
- **核心职责**：**不写具体技术方案**，只做项目管理 + 依赖编排 + 风险识别 + 冲突裁决。
- **Prompt 结构**（精炼版）：
  - 背景：接 00-shared-context.md；预知 A–K 11 个 agent 会产出 11 份策略。
  - 任务清单：
    1. 项目目标一页纸 + 5 硬约束 + 4 交付物。
    2. 12-Agent 职责矩阵 + 依赖关系图（mermaid）。
    3. 5 人分工 D1-D7 任务清单。
    4. D0-D14 甘特 + 6 关里程碑 M1–M6 + 验收标准。
    5. 风险登记册（R1–R12，按 可能性 × 影响 排序）。
    6. 11 条冲突裁决（C1–C11）。
    7. 给 Coordinator 的"红/绿/黄"下一步清单。
  - 禁止：不写任何技术方案代码；不覆盖其他 agent 的策略细节。
- **产出资产**：
  - 文档：`docs/strategies/99-leader-orchestration.md`
- **人工把关点**：
  - 原文档假设"本机无 GPU"，由 Coordinator 在第 2 批合稿时修正为 RTX 2060S 8GB。
  - 原 Top-3 最紧急为 R1（Colab）/ R2（评测集）/ R11（文档跳票），全部在 D2 前消除。

---

## 2. 第二批 · SynthesisAgent（合稿）

### Agent 13 · SynthesisAgent
- **模型**：Opus 4.7
- **派出时间**：2026-04-22（12 份策略全部入库后立即派出）
- **运行时长**：~360 秒（需并发消化 13 份 md）
- **核心职责**：把 12 份策略合并成 2 份可执行文档，抓出冲突、对齐口径。
- **Prompt 结构**（精炼版）：
  - 背景：12 个 agent 并行产出存在口径不一致（bf16 vs fp16 / "无 GPU" vs "2060S" / Recall 口径 / 数据量）。
  - 任务清单：
    1. **《策略总览.md》**：项目一页纸 + 7 层全景 mermaid + 12 策略速览表 + 技术栈锁定 + 训练清单 + 11 + 2 条冲突裁决 + 甘特 + 分工 + 风险 + 开题对齐。
    2. **《一页纸决策清单.md》**：给用户 3 分钟看完的执行清单，含"用户必须立刻拍板"、"Coordinator 48h 内推进"、"组员今明两天硬动作"、"不做清单"。
    3. 主动抓出新冲突：合稿时发现 **C12（bf16→fp16）** 和 **C13（Recall 口径）** 并加入裁决表。
  - 禁止：不改任何原策略文档；只在合稿层对齐口径。
- **产出资产**：
  - `docs/strategies/策略总览.md`
  - `docs/strategies/一页纸决策清单.md`
- **人工把关点**：
  - 用户拍板 GPU=RTX 2060S 8GB + 数据口径=4,233/14,740，"8,000" 废弃（SynthesisAgent 执行前的关键输入）。
  - 最终 13 条冲突裁决全部留痕（见《策略总览.md》§6）。

---

## 3. 第三批 · M1 执行 Agents（仍在进行）

### Agent 14 · M1a 工程整备
- **模型**：Opus 4.7
- **派出时间**：2026-04-22（合稿后）
- **运行时长**：进行中
- **核心职责**：路径清理 + `08 bf16→fp16` patch + commit & push。
- **Prompt 结构**（精炼版）：
  - 任务：
    1. 替换 `08-response-training-strategy.md` 中 `bnb_4bit_compute_dtype=bfloat16` → `float16`。
    2. 同步 `configs/qlora_config.json`：`per_device_batch 4→2`，`grad_accum 4→8`。
    3. 创建 5 个 feature 分支（`feat/intent-v2-xu` 等）。
    4. commit message 遵循 `<type>: <desc>` 约定式提交。
  - 禁止：不动别的策略文档内容；不合并主干。
- **产出资产**：`configs/qlora_config.json` 补丁 + git commits
- **人工把关点**：bf16→fp16 的 patch 规则来自裁决 C12。

---

### Agent 15 · M1b 意图分类器 v2 扩增+训练
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：进行中
- **核心职责**：从 7×30=210 条扩到 7×500=**3,500 条**（LLM 扩增 + 人工 spotcheck），训练 Macro-F1 ≥ 0.92。
- **Prompt 结构**（精炼版）：
  - 任务：
    1. 基于 30 条/类种子，用 LLM 批量生成 470 条/类变体，8 维扰动（同义替换/语序/冗余词等）。
    2. 跑 `scripts/train_intent_classifier_v2.py`，TF-IDF+LR baseline 先跑，MiniLM+LR 为主。
    3. holdout 700 条验证 Macro-F1 ≥ 0.92；P95 延迟 ≤ 5ms。
- **产出资产**：
  - `configs/intent_examples_v2.json`
  - `artifacts/models/intent_classifier_v2.pkl`
- **人工把关点**：扩增后必须人工抽检 5% 防止 LLM 漂移。

---

### Agent 16 · M1c 同义词表扩增（已完成）
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：~200 秒（✅ 完成）
- **核心职责**：从 60 短语扩到 **930 短语**，覆盖证监会违规术语域。
- **Prompt 结构**（精炼版）：
  - 任务：
    1. 输入：`docs/strategies/03-query-rewrite-strategy.md` 附录 A 的 60 条样例模板。
    2. 领域扩展：内幕交易 / 信披违规 / 操纵市场 / 短线交易 / 未依法披露权益变动 / 从业人员违规 / 基金违规 / 保荐违规 / 并购重组等。
    3. 每组给出 1 主词 + 3–8 同义 / 近义 / 简称，分"强/弱"级。
    4. 输出 `configs/synonyms.json` + `configs/synonyms_stats.md` 统计。
- **产出资产**：
  - `configs/synonyms.json`（930 短语）
  - `configs/synonyms_stats.md`
- **人工把关点**：强/弱分级由 Agent C 裁决 C4 决定；本 agent 只做扩增，不改权重规则。

---

### Agent 17 · M1d QLoRA smoke test
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：进行中
- **核心职责**：在 **本机 RTX 2060 SUPER 8GB + fp16** 下验证 Qwen2.5-0.5B / 1.5B 可跑 QLoRA 不 OOM。
- **Prompt 结构**（精炼版）：
  - 任务：
    1. 准备 200 条样本 mini-set（从 `rag_qa_train.jsonl` 抽）。
    2. 先跑 0.5B 跑通（降级基座），再跑 1.5B（主目标）。
    3. 观察显存峰值 < 7GB，单 batch 前向 + 反向不 OOM。
    4. 若 1.5B OOM → 降 seq 到 1536，或切 0.5B，或切 Colab T4。
  - 产出要求：smoke 日志 + 显存峰值截图 + 是否通过的判断。
- **产出资产**：
  - `artifacts/smoke/qlora_*.log`
  - 结论：是否进入 M2 主训阶段
- **人工把关点**：
  - 基座锁 1.5B 来自裁决 C3。
  - fp16 配置来自裁决 C12。

---

## 4. 第四批 · ArchiveAgent-Meta（本 agent）

### Agent 18 · ArchiveAgent-Meta
- **模型**：Opus 4.7
- **派出时间**：2026-04-22
- **运行时长**：当次执行
- **核心职责**：把前 17 个 agent 归档成论文《AI Contribution Statement》章节的证据链。
- **Prompt 结构**（精炼版）：
  - 背景：课程要求论文必须附 AI Contribution Statement；需要可追溯的"用了哪些工具 / 做了什么 / 人工把关了什么"。
  - 任务清单：
    1. 产出 `docs/meta-agents/README.md`（索引 ≤ 800 字 + 18 agent 表 + 启动时序 mermaid）。
    2. 产出 `docs/meta-agents/all-prompts.md`（≈ 3000 字，每个 agent 一小节：模型 / 时长 / 职责 / Prompt 结构 / 产出 / 人工把关）。
    3. 引用模板章节，教论文作者怎么在 §7 AI Contribution Statement 里引用本文档。
  - 禁止：
    - 不要捏造 prompt 原文（明确标"精炼再创作版"）。
    - 不要 commit。
    - 不要动 `prompts/` 目录（那是 System-Agent 的职责）。
- **产出资产**：
  - `docs/meta-agents/README.md`
  - `docs/meta-agents/all-prompts.md`
- **人工把关点**：
  - 用户强调 Meta-Agent 与 System-Agent **必须概念区分**，已在 README §1 明确。
  - 引用模板由 Coordinator 审阅后方可进入论文。

---

## 5. 汇总统计

| 维度 | 数值 |
|------|------|
| 元 agent 总数 | 18 |
| 使用模型 | Anthropic Claude Opus 4.7（全部 18 个） |
| 派出批次 | 4 批（并行 12 + 合稿 1 + 执行 4 + 归档 1） |
| 产出策略文档 | 13 份（00–11 + 99） |
| 产出合稿文档 | 2 份（策略总览 / 一页纸决策清单） |
| 执行产出 | 配置文件 / 训练脚本 / git commits / smoke 日志 |
| 人工把关裁决 | **13 条冲突**（C1–C13） |
| 论文可复用章节 | §7 AI Contribution Statement + §3.1 数据集口径 + §附录 风险登记 |

---

## 6. Prompt 模板抽象规律

回看 18 个 agent 的 prompt，**4 条共性规律**：
1. **共享上下文先行**：所有第 1 批 agent 派出前，先推送 `00-agent-team-shared-context.md`，锁定 5 硬约束。
2. **产出格式对齐**：每个策略 agent 必须产出"500–1500 字 + ≥ 1 张 mermaid + JSON schema + 风险兜底"。
3. **明确禁止**：几乎每个 prompt 都含 "不改别人文件 / 不 commit / 不改评估口径 / 不改数据切分"。
4. **冲突留给 Coordinator**：agent 产出的冲突点（chunk 长度 / bf16 / top_k / prompt 严格度）一律由 Leader + Synthesis + Coordinator 三层裁决，不在 agent 内部消解。

---

## 7. 与 `prompts/` 的边界声明

本文档记录的是**开发时期**的 Meta-Agent。项目的 **System-Agent**（运行时智能角色：Planner / Rewriter / Responder / Validator）**另行**在 `prompts/` 目录归档，且会作为项目核心资产在论文正文（§3 方法论 / §4 实验）详细描述。两者的关系是**师徒**——Meta-Agent 设计了 System-Agent 的规则，但不在产品里执行。

---

## 8. 复现性说明

- 所有 Meta-Agent 的调用均通过 **Claude Code harness（Opus 4.7）** 在本机完成。
- 没有任何 Meta-Agent 的 prompt 或输出包含用户隐私数据（数据来自公开的 CNRDS 证监会处罚信息表）。
- 各 agent 之间**不直接通信**，统一由 Coordinator（执行员角色）做消息路由与合稿。
- 每份策略文档已纳入 git 版本控制（`feature/track-b-finetune` 分支），可按 git log 追溯产出顺序。

---

## 9. AI Contribution Statement 引用模板

> 论文 §7（AI Contribution Statement）建议直接使用以下段落（可根据最终提交情况调整工具列表与数字）：

```
本项目在方案设计阶段使用 Anthropic Claude Opus 4.7 派出 12 个专业 agent
并行产出策略文档（见 docs/meta-agents/all-prompts.md），由人类 Coordinator
对所有产出做冲突裁决、口径校对、技术路线把关（共裁决 13 条冲突，见
docs/strategies/策略总览.md §6）。所有最终方案均经过人工审阅后才进入代码
实现阶段。

具体而言：
- **策略设计阶段**（12 Meta-Agent + 1 Leader + 1 Synthesis）：Opus 4.7 并行产出
  13 份策略文档（拒答 / 数据 / 查询改写 / Planner / 检索 / 重排 / 生成 / QLoRA /
  评估 / 模型选型 / 前后端 / 总统筹 / 合稿），由人类 Coordinator 裁决 13 条
  跨 agent 冲突（含 bf16→fp16 适配 Turing 架构、chunk 长度、top_k、Prompt
  严格度等）。
- **执行阶段**（4 个 M1 子 agent）：Opus 4.7 辅助执行意图分类器数据扩增
  （210→3500 条）、同义词表扩增（60→930 短语）、QLoRA 配置修正、RTX 2060S
  可行性 smoke test。
- **数据构造辅助**：DeepSeek-Chat API 作为 G4 上限对照与部分 QA 精修，不进入
  模型训练数据主干（详见 docs/strategies/08-response-training-strategy.md §3）。
- **人工把关**：所有 agent 产出经 Coordinator 审阅后才进入代码仓库；关键
  决策（GPU 选型、数据口径、切分规则、最终基座）均由 5 名组员共议 + Coordinator
  拍板，AI 不参与最终决策。
- **归档证据链**：Meta-Agent 与其 prompt 结构归档于 docs/meta-agents/；产品
  运行时的 System-Agent 归档于 prompts/ 目录，两者边界清晰。
```

---

> **结语**：18 个 Meta-Agent 不是论文作者，是加速器。最终方案的责任人始终是 Coordinator + 5 名组员。AI Contribution Statement 的本质不是"感谢 AI"，而是"透明披露 AI 在哪里被用、在哪里被拒、在哪里被人工否决"。本文档即为这一披露的证据链底本。
