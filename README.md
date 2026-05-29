# CSRC-RAG · 证监会违规案例检索增强问答系统

> 一个面向中国证监会行政处罚公告的端到端 RAG 问答系统。
> 采用 **Multi-Agent Team** 协作开发，在 **Qwen2.5-0.5B-Instruct** 基座上以 **QLoRA** 指令微调，
> 将幻觉数字率从 **20% 压到 3.3%**（相对下降 83%），
> 格式合规率从 **0 提升至 76.7%**，全流程在 **8 GB 消费级 GPU** 上跑通。

<p align="center">
  <img src="https://img.shields.io/badge/Track-B-blue" />
  <img src="https://img.shields.io/badge/Base-Qwen2.5--0.5B-orange" />
  <img src="https://img.shields.io/badge/Method-QLoRA-green" />
  <img src="https://img.shields.io/badge/GPU-RTX_2060S_8GB-lightgrey" />
  <img src="https://img.shields.io/badge/Agents-Multi--Agent_Team-purple" />
  <img src="https://img.shields.io/badge/Course-Deep_Learning-red" />
</p>

**深度学习课程设计 · 赛道 B（垂直领域智能问答）** · 2026.04

作者：许浩财 · 贾彤 · 戴一鑫 · 张彦扬 · 王怡菲

---

## 核心亮点

| 维度 | 成果 |
|------|------|
| **幻觉缓解** | 20.0% → 3.3%（-83%），三层防线：RAG + 对抗训练 + 规则校验 |
| **格式合规** | 0% → 76.7%，证明 <1B 小模型必须靠微调习得结构化输出 |
| **检索提升** | Recall@5: 0.073 → 0.388（+431%），BM25 ⊕ bge ⊕ RRF 混合检索 |
| **Agent Team** | 多智能体协作开发（Leader-Scout-Worker 模式），策略层 → 执行层分离 |
| **成本可控** | 52 min 训练、34 MB adapter、+0.7s 延迟，8GB 显存全流程 |

---

## Multi-Agent Team 架构

本项目采用 **Agent Team** 协作模式开发，通过多智能体分工提升开发效率和决策质量：

```
┌─────────────────────────────────────────────────────┐
│              Leader Agent (总调度)                    │
│    负责：任务拆解、进度追踪、质量把关                  │
└─────────────────┬───────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌─────────┐  ┌─────────────┐
│ Scout  │  │ Worker  │  │  Evaluator  │
│ (侦察) │  │ (执行)  │  │   (评估)    │
└────────┘  └─────────┘  └─────────────┘
 策略研究     代码实现      指标评测
 方案设计     训练调优      消融对比
 文献调研     Bug修复       报告生成
```

### Agent 策略文档

所有 Agent 的共享上下文和分工策略详见 [`docs/strategies/`](docs/strategies/)：

| 策略 | 文件 | 内容 |
|------|------|------|
| 共享上下文 | `00-agent-team-shared-context.md` | 项目约束、资产清单、系统链路 |
| 拒答策略 | `01-reject-strategy.md` | OUT_OF_SCOPE 判定规则 |
| 数据策略 | `02-data-strategy.md` | 训练数据构造、切分、反幻觉负例 |
| 查询改写 | `03-query-rewrite-strategy.md` | 共指消解、同义词扩展 |
| 检索策略 | `05-retrieval-strategy.md` | 混合检索、RRF 融合 |
| 重排策略 | `06-reranking-strategy.md` | Cross-encoder 精排 |
| 回复策略 | `07-response-strategy.md` | 生成约束、格式要求 |
| 评估策略 | `09-evaluation-strategy.md` | 三层指标体系 |
| 总调度 | `99-leader-orchestration.md` | Agent 任务编排 |

---

## 评测指标体系

本项目采用三层指标覆盖完整 RAG 管线，详见 [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md)：

| 指标 | 层 | 本项目结果 | 学术引用 |
|------|-----|-----------|---------|
| **Hallucinated Number Rate** | 生成层 | G0: 20% → G3: **3.3%** | RAGTruth (ACL 2024), RAGAS (EACL 2024) |
| **Event ID Hit Rate** | 检索层 | G0: 0% → G3: **20%** | Manning IR, Practical RAG Eval (2024) |
| **Format Compliance** | 格式层 | G0: 0% → G3: **76.7%** | StructEval (2025) |

### 与外部基准对标

| 对比项 | 本项目 G3 | 业界参考 |
|--------|-----------|---------|
| 幻觉率 | 3.3% | GPT-4: 2-6%, Llama-2-7B: 30-40% |
| 格式合规 | 76.7% | GPT-4o (StructEval): 76% |
| 检索 Recall@5 | 0.388 | Hybrid+Rerank 典型: 0.80-0.92 |

---

## 系统架构：七层流水线

```
用户 query
    │
    ▼
┌─────────────────────────────────────────────┐
│  L1  意图分类 (TF-IDF + LR, F1 = 0.9989)   │
│       7 类意图路由                           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  L2  查询改写（共指消解 + 同义词扩展）       │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────┬──────────────────────────────┐
│ L3a BM25     │ L3b bge-small-zh-v1.5        │
│ jieba + 领域词│ 512 维 cosine               │
└──────┬───────┴──────┬───────────────────────┘
       └── RRF(k=60) ──┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  L4  交叉编码器精排 (bge-reranker-v2-m3)    │
└─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  L5  Qwen2.5-0.5B + QLoRA 生成              │
│      r=16, α=32, 4-bit NF4                  │
└─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  L6  趋势聚合器 (SQL-like groupby)          │
└─────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  L7  引证校验 (8 条 YAML 规则)              │
└─────────────────────────────────────────────┘
               │
               ▼
        结构化回答 + [EventID=xxx] 引证
```

---

## 四组对照实验

严格控制变量，四组只改一个维度：

| 组 | RAG 证据 | 强约束 prompt | LoRA | EID 命中率 | 格式合规 | 幻觉率 |
|---|---|---|---|---:|---:|---:|
| **G0** | ❌ | ❌ | ❌ | 0% | 0% | 20.0% |
| **G1** | ✅ | ❌ | ❌ | 0% | 0% | 10.0% |
| **G2** | ✅ | ✅ | ❌ | 0% | 0% | 10.0% |
| **G3** | ✅ | ✅ | ✅ | **20%** | **76.7%** | **3.3%** |

**核心结论**：
1. RAG 只能将幻觉砍一半（20% → 10%），prompt 工程无法继续改善
2. 格式学习必须微调——<1B 模型无法零样本遵循 `[EventID=xxx]` 格式
3. LoRA 成本极低：+0.7s 延迟、+1.7% 参数量

---

## 数据集

| 数据集 | 样本数 | 类别 | 用途 |
|--------|--------|------|------|
| `data/processed/rag_qa_train.jsonl` | 960 | A/B/C/D 各 240 | QLoRA 训练 |
| `data/processed/rag_qa_val.jsonl` | 120 | A/B/C/D 各 30 | 验证 |
| `data/processed/rag_qa_test.jsonl` | 120 | A/B/C/D 各 30 | 最终评测 |
| `data/eval/intent_eval_1211.jsonl` | 1,211 | IN_SCOPE/OUT_OF_SCOPE | 意图分类评测 |
| `data/eval/gold_130.jsonl` | 130 | — | 检索层 gold set |
| `data/eval/gold_trend_30.jsonl` | 30 | — | 趋势聚合评测 |

**数据格式** (Alpaca-style)：
```json
{
  "event_id_source": "401444",
  "category": "A",
  "split": "train",
  "instruction": "根据检索到的证监会处罚案例...",
  "input": "用户问题：... [检索证据] ...",
  "output": "根据检索证据，查询到 N 条... [EventID=xxx]"
}
```

---

## 快速上手

### 环境要求

- Python 3.11 / 3.12
- RTX 2060 SUPER 8 GB（或同级别 GPU）
- 磁盘约 5 GB

### 安装

```bash
pip install -e .
# 关键版本：transformers==4.44.2, peft==0.13.2, bitsandbytes==0.43.3
```

### 5 步复现

```bash
# 1. 构建知识库
PYTHONPATH=src python scripts/build_corpora.py
PYTHONPATH=src python scripts/build_event_chunks.py

# 2. 离线检索评测
PYTHONPATH=src python scripts/evaluate_retrieval_sanity.py

# 3. LoRA 训练（~52 min）
python scripts/train_qlora_m4.py

# 4. G0-G3 四组评测
python scripts/evaluate_generation_m4_4.py --n-samples 30

# 5. 生成论文图表
python scripts/build_paper_figures.py
python scripts/build_paper_docx.py
```

### 运行 Demo

```bash
python scripts/run_demo_server.py
# 浏览器打开 http://127.0.0.1:8000
```

---

## 目录结构

```
Deeplearning-Rag-Test/
├── README.md
├── configs/                      # 配置文件
│   ├── qlora_config.json         # QLoRA 训练配置
│   ├── models.json               # 模型选择
│   ├── retrieval.json            # 检索参数
│   └── intents.json              # 意图定义
│
├── data/
│   ├── processed/                # 训练数据 (Alpaca 格式)
│   │   ├── rag_qa_train.jsonl    # 960 条
│   │   ├── rag_qa_val.jsonl      # 120 条
│   │   └── rag_qa_test.jsonl     # 120 条
│   └── eval/                     # 评测集
│       ├── gold_130.jsonl        # 检索 gold set
│       ├── intent_eval_1211.jsonl # 意图分类评测
│       └── gold_trend_30.jsonl   # 趋势评测
│
├── docs/
│   ├── evaluation_metrics.md     # 评测指标学术文档
│   ├── strategies/               # Agent Team 策略文档
│   ├── reports/                  # 实验报告
│   ├── showcase/                 # 静态展示页
│   └── visuals/                  # 图表
│
├── src/csrc_rag/                 # 核心代码
│   ├── retrieval/                # L3-L4 检索层
│   ├── orchestration/            # L1-L2 意图+改写
│   ├── response/                 # L5-L7 生成+校验
│   └── training/                 # 训练相关
│
├── scripts/                      # 运行脚本
│   ├── train_qlora_m4.py         # LoRA 主训练
│   ├── evaluate_generation_m4_4.py # 四组评测
│   └── run_demo_server.py        # Demo 服务器
│
├── tools/                        # Agent Team 工具
│   ├── agent_team_runner.py      # 多智能体运行器
│   └── bootstrap_codex_workspace.py
│
├── web/                          # 前端
│   ├── index.html                # 聊天界面
│   └── compare.html              # G0 vs G3 对比
│
└── artifacts/
    └── models/qwen_lora_csrc/    # LoRA adapter (34 MB)
```

---

## 技术栈

| 层 | 技术 | 关键参数 |
|---|---|---|
| 基座 | Qwen/Qwen2.5-0.5B-Instruct | 494M 参数 |
| 量化 | bitsandbytes 4-bit NF4 | compute_dtype = fp16 |
| 微调 | QLoRA r=16 α=32 | 全 attention + FFN 层 |
| 检索 | BM25 + bge-small-zh-v1.5 + RRF | k=60 |
| 精排 | bge-reranker-v2-m3 | cross-encoder |
| 意图 | TF-IDF + LogReg | Macro-F1 = 0.9989 |
| Agent | OpenAI Agents SDK | Multi-Agent Team |

---

## 已知局限与未来工作

| 问题 | 根因 | 修复路径 |
|------|------|---------|
| EID 命中率只有 20% | 检索 Recall@5 天花板 | Reranker 领域 LoRA |
| 评测 30 条 CI 宽 | 样本量不足 | 扩展到 100+ 条 |
| 仅覆盖数字幻觉 | 正则局限 | NER + 人工标注 |

详见 [`docs/reports/bad_cases.md`](docs/reports/bad_cases.md)

---

## 文档索引

| 类型 | 文件 | 说明 |
|------|------|------|
| 评测指标 | [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md) | 三层指标 + 学术引用 + 对标 |
| Agent 策略 | [`docs/strategies/`](docs/strategies/) | 12 份 Agent 分工文档 |
| 核心评测 | [`docs/reports/m4_4_generation_eval.md`](docs/reports/m4_4_generation_eval.md) | G0-G3 详细结果 |
| 检索消融 | [`docs/reports/m3e_ablation_report.md`](docs/reports/m3e_ablation_report.md) | 混合检索对比 |
| Bad Cases | [`docs/reports/bad_cases.md`](docs/reports/bad_cases.md) | 已修 + 未修问题 |
| 答辩要点 | [`docs/答辩要点.md`](docs/答辩要点.md) | 答辩准备 |

---

## 联系

本项目为深度学习课程赛道 B 的课程作业。

**仓库**: <https://github.com/Mindse-Tt/Deeplearning-Rag-Test>

<p align="center">
  <sub>Built with Qwen2.5, QLoRA, Multi-Agent Team, and hallucination-aware design.</sub>
</p>
