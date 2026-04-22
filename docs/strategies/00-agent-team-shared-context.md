# Agent Team 共享上下文（所有 agent 必读）

## 项目一句话
赛道 B 独立研究 · 基于 RAG 的证监会违规案例智能检索与问答系统 · 组员 5 人 · 2-4 周答辩 + 交稿。

## 核心约束（不能违反）
1. **必须展示 LoRA 微调前后对比**（赛道 B 硬性要求）
2. **必须做幻觉缓解**（有/无 RAG + 有/无强证据约束三组对照）
3. **必须做 baseline 消融**（≥ 5 组）
4. **必须交付 8-12 页论文 + 代码仓库 + ≤15 页 PPT + 真实 Demo**
5. **必须含 Team Contributions + AI Contribution Statement 章节**
6. **数据不能泄漏**：按 EventID 切分 + 时间切分（Train 1994-2021 / Val 2022-2023 / Test 2024-2025）
7. **PunishmentMeasure 不能进模型输入**（会泄漏标签）

## 现有资产（可以直接用）

### 数据
- 原始：`证监会处罚信息表.xlsx`（14,740 行 × 24 列）
- `data/processed/event_corpus.jsonl`（4,233 事件级文档）
- `data/processed/event_chunks.jsonl`（29,314 chunks）
- `data/processed/party_samples.jsonl`（14,740 当事人样本）
- `data/processed/chunk_embeddings.npy`（旧版 MiniLM embedding，89MB，待替换）

### 代码
- `src/csrc_rag/retrieval/{bm25,dense,hybrid,engine,chunking,tokenizer}.py` 检索层
- `src/csrc_rag/orchestration/{intents,intent_model,topic_guard}.py` 意图路由
- `src/csrc_rag/response/responder.py` 回复模板
- `src/csrc_rag/training/{sklearn_baseline,hf_finetune,data,metrics}.py` 训练脚手架
- `scripts/*.py` 若干运行脚本
- `web/{index.html,app.js,style.css}` 聊天前端
- `scripts/run_demo_server.py` FastAPI/http.server demo

### 环境
- **Windows，Python 3.12**（已装 torch/transformers/sentence_transformers/sklearn）
- 旧的 `.venv311` 不存在，不要用
- 本机**无 GPU**（需用 Colab/Kaggle 跑 LoRA，或本机 QLoRA 4-bit 跑 CPU 可行但慢）

### 仓库
- GitHub 私有仓 `Mindse-Tt/Deeplearning-Rag-Test`
- 当前分支 `feature/track-b-finetune`（基于 main）

## 系统端到端链路（七层）

```
用户输入
  ↓
L0 预处理
  ↓
L1 意图识别（7 类：greeting/chitchat/out_of_scope/case_retrieval/law_grounding/sanction_recommendation/trend_analysis）
  ↓
L2 Query 改写（共指消解 + 同义词扩展 + 槽位抽取）
  ↓
L3 检索（BM25 + Dense(bge-small-zh) + RRF + Reranker(bge-reranker-v2-m3)）
  ↓
L4 证据组装
  ↓
L5 生成（Qwen2.5-1.5B + LoRA，强证据约束 prompt）
  ↓
L6 趋势分析层（可选）
  ↓
L7 后处理：引证校验
  ↓
返回前端
```

## 开题报告承诺
- 赛道 B + 独立研究
- 数据：CNRDS 14,743 行 去重后 ~8,000 条有效（**实际执行统一口径为 14,740 → 4,233 事件 + 14,740 当事人**）
- 五层架构：意图识别 + 双路检索 + 重排融合 + 领域微调 + 多维评估
- 分工：许浩财（架构+意图+集成）/ 贾彤（数据+向量库）/ 戴一鑫（问答对+拒答）/ 张彦扬（问答对+prompt）/ 王怡菲（模型+评估）

## 本轮你要做什么

每个 agent 产出两件事：
1. **策略文档**：`docs/strategies/<你的策略名>.md`，500-1500 字，必须含：
   - 策略目标
   - 输入 / 输出
   - 关键算法 / 模型选型 / 超参
   - 与上下游的接口约定（JSON schema）
   - 评估指标
   - 风险与兜底
   - 至少一张 mermaid 流程图
2. **代码 stub**：在合适的位置新建或留空函数签名，**不动别人的现有文件**。

禁止：
- 不要动 `src/csrc_rag/` 下别人的现有文件（除非明确是你的模块）
- 不要 commit
- 不要自作主张把评估口径或数据切分规则改掉

所有 agent 返回内容**汇总给 Coordinator（我）合稿**。
