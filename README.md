# CSRC-RAG · 证监会违规案例智能检索与问答系统

<p align="center">
  <img src="https://img.shields.io/badge/Track-B-blue" />
  <img src="https://img.shields.io/badge/Base-Qwen2.5--0.5B-orange" />
  <img src="https://img.shields.io/badge/Method-QLoRA-green" />
  <img src="https://img.shields.io/badge/GPU-RTX_2060S_8GB-lightgrey" />
  <img src="https://img.shields.io/badge/Agents-Multi--Agent_Team-purple" />
</p>

**深度学习课程设计 · 赛道 B（垂直领域智能问答）** · 2026.04

作者：许浩财 · 贾彤 · 戴一鑫 · 张彦扬 · 王怡菲

---

## 一、我们做了什么

**一句话**：把 4,233 条证监会行政处罚公告变成一个**会精确引用、拒绝编造**的智能问答助手。

<p align="center">
  <img src="docs/visuals/png/paper/fig2_g0_g3.png" width="85%" alt="G0-G3 四组对比" />
  <br/>
  <sub><b>Figure 1</b> · G0-G3 四组核心指标对比 —— 只有 G3(+LoRA) 实现了格式合规和幻觉控制</sub>
</p>

### 背景问题

通用大模型在中国证券违规案例问答场景下有三个致命短板：

| 问题 | 表现 | 后果 |
|------|------|------|
| **幻觉** | 编造不存在的法条、罚款金额、公司名称 | 误导合规从业人员 |
| **格式不合规** | 不遵守结构化引用格式 | 无法追溯、不可审计 |
| **检索召回低** | 领域术语向量化质量差 | 错过关键案例 |

### 我们的解决方案

用 **RAG + QLoRA 指令微调 + 规则校验** 三层防线系统性解决，在 **<1B 参数量、8GB 消费级 GPU** 的硬约束下实现：

| 指标 | 微调前 (G0) | 微调后 (G3) | 提升 |
|------|:-----------:|:-----------:|:----:|
| 幻觉数字率 ↓ | 20.0% | **3.3%** | -83% |
| 格式合规率 ↑ | 0% | **76.7%** | 从无到有 |
| 事件ID命中率 ↑ | 0% | **20.0%** | 从无到有 |
| 指令遵循率 ↑ | 0% | **76.7%** | 从无到有 |

---

## 二、技术架构：七层流水线

<p align="center">
  <img src="docs/visuals/png/paper/fig1_architecture.png" width="90%" alt="七层 RAG 架构" />
  <br/>
  <sub><b>Figure 2</b> · 七层 RAG 流水线架构</sub>
</p>

### 2.1 整体架构

```
用户 query: "谭光华因违规买卖股票被处罚的详情?"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ L1 意图分类 (TF-IDF + LogReg, Macro-F1 = 0.9989)       │
│     7 类: greeting / chitchat / out_of_scope /          │
│           case / law / sanction / trend                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ L2 查询改写（共指消解 + 同义词扩展 + 多约束拆分）        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┬───────────────────────────────────────┐
│ L3a BM25        │ L3b bge-small-zh-v1.5 (512维 cosine)  │
│ jieba + 领域词典 │ 每路召回 top-100                      │
└────────┬────────┴────────┬──────────────────────────────┘
         └─── RRF(k=60) ────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ L4 交叉编码器精排 (bge-reranker-v2-m3, top-5)           │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ L5 Qwen2.5-0.5B + QLoRA 生成                            │
│     r=16, α=32, 4-bit NF4 量化, 全 attention+FFN 层     │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ L6 趋势聚合器 (SQL-like groupby, 按年/类型分面)          │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ L7 引证校验 (Validator, 8 条 YAML 规则)                  │
│     • EID 必须在证据中 • 法条必须在证据中                │
│     • 不得出现证据外的罚款金额 • 失败→降级话术           │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
        "参考历史相似案例... [EventID=40111147]..."
```

### 2.2 各层技术选型

| 层 | 组件 | 技术 | 关键参数 |
|---|---|---|---|
| L1 | 意图分类 | TF-IDF + Logistic Regression | Macro-F1 = 0.9989 |
| L2 | 查询改写 | 规则 + LLM fallback | 257 规范词 / 673 别名 |
| L3 | 混合检索 | BM25 + bge-small-zh-v1.5 + RRF | k=60, top-100 each |
| L4 | 精排 | bge-reranker-v2-m3 | cross-encoder, top-5 |
| L5 | 生成 | Qwen2.5-0.5B + QLoRA | r=16, α=32, NF4 4-bit |
| L6 | 聚合 | SQL-like groupby | facet: year/vtype/ptype |
| L7 | 校验 | YAML 规则引擎 | 8 条规则，确定性解析 |

### 2.3 Multi-Agent Team 开发架构

本项目采用多智能体协作模式，通过 Agent 分工提升开发效率：

```
          ┌─────────────────────────┐
          │    Leader Agent (总调度) │
          └────────┬────────────────┘
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌────────┐   ┌─────────┐   ┌──────────┐
│ Scout  │   │ Worker  │   │Evaluator │
│ 策略研究│   │ 代码实现│   │ 评测分析 │
└────────┘   └─────────┘   └──────────┘
```

策略文档：[`docs/strategies/`](docs/strategies/)（12 份 Agent 分工和策略文档）

---

## 三、数据：从哪来、怎么用

### 3.1 数据来源

<p align="center">
  <img src="docs/visuals/png/paper/fig6_corpus.png" width="90%" alt="知识库分布" />
  <br/>
  <sub><b>Figure 5</b> · 知识库语料分布（左：按年度，右：Top-8 违规子类，n=4,233）</sub>
</p>

| 数据源 | 说明 | 规模 |
|--------|------|------|
| CNRDS 证监会处罚信息表 | 原始结构化数据 | 14,740 行 × 24 列 |
| 事件级文档 | 按 EventID 聚合的完整案例 | 4,233 篇 |
| 事件 Chunks | 按段落切分的检索单元 | 29,314 段 |

### 3.2 训练数据

**来源**：基于原始案例自动构造 + 人工校验，Alpaca 格式

| 文件 | 样本数 | 类别分布 | 用途 |
|------|--------|---------|------|
| `data/processed/rag_qa_train.jsonl` | 960 | A/B/C/D 各 240 | QLoRA 训练 |
| `data/processed/rag_qa_val.jsonl` | 120 | A/B/C/D 各 30 | 验证集 |
| `data/processed/rag_qa_test.jsonl` | 120 | A/B/C/D 各 30 | 最终评测 |

**四类任务**：
- **A** - 案例检索：按主体/机构/证券代码查找处罚案例
- **B** - 信息确认：确认/否认特定案例的具体细节
- **C** - 处罚推荐：基于案例对比推荐处罚方式
- **D** - 综合分析：跨案例对比、法条引用、趋势分析

### 3.3 评测数据

| 文件 | 样本数 | 用途 |
|------|--------|------|
| `data/eval/gold_130.jsonl` | 130 | 检索层精度评测 |
| `data/eval/intent_eval_1211.jsonl` | 1,211 | 意图分类独立评测 |
| `data/eval/gold_trend_30.jsonl` | 30 | 趋势聚合评测 |

### 3.4 数据样例

**训练数据格式** (Alpaca-style JSONL)：

```json
{
  "event_id_source": "401444",
  "category": "A",
  "split": "train",
  "instruction": "根据检索到的证监会处罚案例，回答用户关于主体、人员、监管机构或证券代码的查询。只能使用证据中出现的案例，并逐条引用 [EventID=xxx]。",
  "input": "用户问题：股票代码000677对应主体有什么处罚案例？\n\n[检索证据]\n案例1：\n  EventID=403932\n  标题：中国证监会行政处罚决定书（陈宝庆、李文静）\n  公告日期：2012-12-21\n  违规类型：内幕交易\n  ...",
  "output": "根据检索证据，查询到 3 条与证券代码「000677」相关的处罚记录：「中国证监会行政处罚决定书（陈宝庆、李文静）」（2012-12-21），违规类型：内幕交易，处罚类型：其他、罚款 [EventID=403932]；... 以上结论仅基于当前证据。"
}
```

**评测数据格式** (意图分类)：

```json
{"question": "若将人工智能辅助工具引入退市听证流程，可能带来哪些治理效能提升？", "label": "IN_SCOPE"}
{"question": "今天天气冷不冷", "label": "OUT_OF_SCOPE"}
```

---

## 四、如何验证模型好坏：6 项评测指标

我们设计了 **6 项指标**，覆盖系统全链路和微调效果两个维度：

### 4.1 系统评估指标（3 项）

| 指标 | 定义 | 学术引用 |
|------|------|---------|
| **Hallucinated Number Rate** | 生成的数值声明中无证据支撑的比例 | RAGTruth (ACL 2024), RAGAS (EACL 2024) |
| **Event ID Hit Rate** | 回答中包含正确 EventID 的比例 | Manning IR, Practical RAG Eval (2024) |
| **Format Compliance** | 输出通过 L7 校验规则的比例 | StructEval (2025) |

### 4.2 微调效果指标（3 项）

| 指标 | 定义 | 学术引用 |
|------|------|---------|
| **Task Accuracy** | 关键字段与标准答案完全匹配的比例 | QLoRA (Dettmers et al., NeurIPS 2023) |
| **Entity F1** | 领域实体（公司/金额/违规类型）的 Micro-F1 | CoNLL-2003 NER, 金融NER文献 |
| **Instruction Following** | 同时满足格式+字段+结构约束的比例 | Vicuna (Chiang 2023), StructEval (2025) |

### 4.3 实验结果

#### 四组对照实验（严格控制变量）

| 组 | RAG | 强 prompt | LoRA | 幻觉率↓ | 格式合规↑ | EID命中↑ | 指令遵循↑ |
|---|:---:|:---:|:---:|---:|---:|---:|---:|
| G0 | ❌ | ❌ | ❌ | 20.0% | 0% | 0% | 0% |
| G1 | ✅ | ❌ | ❌ | 10.0% | 0% | 0% | 0% |
| G2 | ✅ | ✅ | ❌ | 10.0% | 0% | 0% | 0% |
| **G3** | ✅ | ✅ | ✅ | **3.3%** | **76.7%** | **20.0%** | **76.7%** |

#### 与外部基准对标

| 指标 | 本项目 G3 | 业界参考 | 说明 |
|------|-----------|---------|------|
| 幻觉率 3.3% | GPT-4: 2-6% | 接近 GPT-4 水平 |
| 格式合规 76.7% | GPT-4o: 76% (StructEval) | 与 GPT-4o 持平 |
| 指令遵循 76.7% | 微调后典型: 90%+ | 受检索天花板制约 |

#### 核心结论

<p align="center">
  <img src="docs/visuals/png/paper/fig3_hallucination.png" width="75%" alt="幻觉率逐层下降" />
  <br/>
  <sub><b>Figure 3</b> · 幻觉数字率逐层下降 20.0% → 10.0% → 10.0% → 3.3%</sub>
</p>

1. **RAG 只能砍一半幻觉**：20% → 10%，prompt 工程不再改善
2. **格式必须微调习得**：<1B 模型完全无法零样本遵循 `[EventID=xxx]`
3. **成本极低**：52 min 训练，+34MB adapter，+0.7s 延迟

<p align="center">
  <img src="docs/visuals/png/paper/fig5_loss.png" width="70%" alt="训练收敛" />
  <br/>
  <sub><b>Figure 4</b> · QLoRA 训练收敛曲线（loss 2.52 → 0.70, 274 steps）</sub>
</p>

详细评测：[`docs/evaluation_metrics.md`](docs/evaluation_metrics.md) | [`docs/reports/m4_4_generation_eval.md`](docs/reports/m4_4_generation_eval.md)

---

## 五、我们的优势

| 维度 | 优势 | 对比 |
|------|------|------|
| **资源约束** | 8GB GPU 全流程跑通 | 主流方案需 24GB+ |
| **幻觉控制** | 三层防线（RAG+训练+规则），3.3% | 开源 7B 模型: 20-30% |
| **可审计性** | L7 确定性校验器，每条回答可追溯 | 通用 LLM 黑盒输出 |
| **方法论** | 严格四组消融，控制变量实验设计 | 多数项目只报最终结果 |
| **工程完整度** | 七层解耦流水线，各层独立可测 | 端到端黑盒系统 |
| **Agent Team** | 多智能体策略-执行分离 | 单人开发 |

---

## 六、如何启动项目

### 6.1 环境要求

```
Python 3.11 / 3.12
RTX 2060 SUPER 8 GB（或同级 GPU）
磁盘 ~5 GB（含模型下载）
```

### 6.2 安装

```bash
git clone https://github.com/Mindse-Tt/Deeplearning-Rag-Test.git
cd Deeplearning-Rag-Test
git checkout feature/track-b-finetune
pip install -e .
```

关键依赖版本：
```
transformers==4.44.2
peft==0.13.2
bitsandbytes==0.43.3
accelerate==0.33.0
sentence-transformers>=2.5
```

### 6.3 从零复现（5 步）

```bash
# Step 1: 构建知识库 (~2 min)
PYTHONPATH=src python scripts/build_corpora.py
PYTHONPATH=src python scripts/build_event_chunks.py

# Step 2: 离线检索评测 (~3 min)
PYTHONPATH=src python scripts/evaluate_retrieval_sanity.py

# Step 3: QLoRA 训练 (~52 min)
python scripts/train_qlora_m4.py

# Step 4: G0-G3 四组评测 (~30 min)
python scripts/evaluate_generation_m4_4.py --n-samples 30

# Step 5: 微调指标评测
python scripts/evaluate_finetune_metrics.py
```

### 6.4 运行 Demo

```bash
python scripts/run_demo_server.py
# 浏览器打开 http://127.0.0.1:8000
```

**推荐试的 query**：

| 类型 | 问题 | 预期行为 |
|------|------|---------|
| 直查 | `谭光华因违规买卖股票被处罚的详情` | 命中 [EventID=40111147] |
| 相似案例 | `帮我找和内幕交易类似的处罚案例` | 多条 EID + 证据展开 |
| 幻觉陷阱 | `2022年董事长因内幕交易被罚款的案件` | 保守引用，不编造 |
| 越界拒答 | `帮我预测明天股价` | out_of_scope 拒答 |

---

## 七、项目结构

```
Deeplearning-Rag-Test/
├── data/
│   ├── processed/          # 训练数据 (Alpaca 格式, 已跟踪)
│   │   ├── rag_qa_train.jsonl   (960 条)
│   │   ├── rag_qa_val.jsonl     (120 条)
│   │   └── rag_qa_test.jsonl    (120 条)
│   └── eval/               # 评测集
│       ├── gold_130.jsonl       (检索层)
│       └── intent_eval_1211.jsonl (意图分类)
│
├── src/csrc_rag/           # 核心代码 (七层流水线)
│   ├── retrieval/          # L3-L4: 检索+精排
│   ├── orchestration/      # L1-L2: 意图+改写
│   ├── response/           # L5-L7: 生成+校验
│   └── training/           # 训练相关
│
├── scripts/                # 运行脚本
│   ├── train_qlora_m4.py   # LoRA 主训练
│   ├── evaluate_generation_m4_4.py  # G0-G3 评测
│   ├── evaluate_finetune_metrics.py # 6指标评测
│   └── run_demo_server.py  # Demo 服务器
│
├── docs/
│   ├── evaluation_metrics.md  # 6项指标学术文档
│   ├── strategies/         # Agent Team 策略 (12份)
│   └── reports/            # 实验报告
│
├── tools/                  # Agent Team 工具
│   └── agent_team_runner.py
│
├── web/                    # 前端 (聊天 + G0/G3 对比)
├── configs/                # 配置
└── artifacts/models/       # LoRA adapter (34 MB)
```

---

## 八、引用

如果本仓库对你的研究或工程实践有帮助，欢迎引用：

```bibtex
@misc{xu2026csrcrag,
  title={CSRC-RAG: A RAG-based Intelligent Q\&A System for Securities Regulatory Penalty Cases with QLoRA Fine-tuning and Multi-layer Hallucination Control},
  author={Xu, Haocai and Jia, Tong and Dai, Yixin and Zhang, Yanyang and Wang, Yifei},
  year={2026},
  howpublished={\url{https://github.com/Mindse-Tt/Deeplearning-Rag-Test}},
  note={Deep Learning Course Project, Track B}
}
```

**仓库**：https://github.com/Mindse-Tt/Deeplearning-Rag-Test

**分支**：`feature/track-b-finetune`

---

<p align="center">
  <sub>Built with Qwen2.5 · QLoRA · Multi-Agent Team · Hallucination-aware Design</sub>
</p>
