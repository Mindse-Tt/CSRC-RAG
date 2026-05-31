# CSRC-RAG: A RAG-based Intelligent Q&A System for Securities Regulatory Penalty Cases with QLoRA Fine-tuning and Multi-layer Hallucination Control

# 证监会违规案例智能检索与问答系统

<p align="center">
  <img src="https://img.shields.io/badge/Track-B-blue" />
  <img src="https://img.shields.io/badge/Base-Qwen2.5--0.5B-orange" />
  <img src="https://img.shields.io/badge/Method-QLoRA-green" />
  <img src="https://img.shields.io/badge/GPU-RTX_2060S_8GB-lightgrey" />
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
| 幻觉数字率 ↓ | 33.3% | **6.7%** | -80% |
| 格式合规率 ↑ | 0% | **76.7%** | 从无到有 |
| 事件ID命中率 ↑ | 0% | **20.0%** | 从无到有 |

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
│ L5 生成：Qwen2.5-0.5B + QLoRA 生成                            │
│     r=16, α=32, 4-bit NF4 量化, 全 attention+FFN 层     │
└─────────────────────────────────────────────────────────┘
                 │
                 ├── [trend 意图] ──▶ L6 趋势聚合器 (SQL-like groupby)
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ L7 引证校验 (Validator, 8 条 YAML 规则)                  │
│     • 对 L5 输出做事后校验，非生成环节                    │
│     • EID 必须在证据中 • 法条必须在证据中                │
│     • 不得出现证据外的罚款金额 • 失败→降级话术           │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
        "参考历史相似案例... [EventID=40111147]..."
```

### 2.2 各层技术选型

> 详细选型理由与候选对比见 [`docs/model_selection.md`](docs/model_selection.md)

| 层 | 组件 | 技术 | 选型理由 |
|---|---|---|---|
| L1 | 意图分类 | TF-IDF + LogReg | F1=0.9989，更复杂模型无收益 |
| L3 | 稠密检索 | bge-small-zh-v1.5 | 99MB，中文MTEB Top-3，显存友好 |
| L4 | 精排 | bge-reranker-v2-m3 | 中文最优 cross-encoder |
| L5 | 生成 | Qwen2.5-0.5B + QLoRA | **8GB GPU 硬约束下唯一能全链路部署的选择** |
| L5 | 量化 | 4-bit NF4 | QLoRA 原生支持，精度损失 <1% |

### 2.3 大模型 Prompt 设计

系统通过两层 prompt 控制生成行为：

**System Prompt（角色约束）**：
```
你是证监会处罚案例智能分析助手。你只能根据给定案例证据回答，禁止编造
未出现的法条、处罚结果、金额或事实。如果证据不足，请明确写"证据不足"。
```

**Instruction Prompt（任务约束，按类别不同）**：
```
根据检索到的证监会处罚案例，回答用户关于主体、人员、监管机构或证券代码
的查询。只能使用证据中出现的案例，并逐条引用 [EventID=xxx]。
```

**完整输入结构**：
```
[System] 角色约束
[User]   Instruction + 用户问题 + [检索证据] 案例1/2/3...
[Assistant] 结构化回答 + [EventID=xxx] 引证
```

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

#### 模型选型实验（2 模型 × 3 训练方式）

<p align="center">
  <img src="docs/visuals/png/experiments/model_selection.png" width="85%" alt="模型选型对比" />
  <br/>
  <sub><b>Figure 6</b> · 模型选型：Qwen-0.5B vs Bloom-560M × QLoRA/LoRA/Full FT</sub>
</p>

| 实验 | 模型 | 训练方式 | Eval Loss↓ | 训练时间 | 结论 |
|------|------|---------|---:|---:|---|
| M1_T1 | Qwen-0.5B | QLoRA (4-bit) | 0.923 | 26min | 最省资源 |
| **M1_T2** | **Qwen-0.5B** | **LoRA (fp32)** | **0.826** ⭐ | **42min** | **最优** |
| M1_T3 | Qwen-0.5B | Full FT | 0.904 | 30h | 过拟合+太慢 |
| M2_T1 | Bloom-560M | QLoRA (4-bit) | 1.523 | 17min | 远差于Qwen |
| M2_T2 | Bloom-560M | LoRA (fp32) | 1.756 | 21min | 最差 |
| M2_T3 | Bloom-560M | Full FT | 1.224 | 19h | 过拟合 |

**选型结论**：
1. **Qwen-0.5B 全面优于 Bloom-560M**（eval_loss 低 40-50%），得益于原生中文预训练
2. **LoRA 是最优训练方式**（eval_loss 最低 0.826），QLoRA 牺牲少量精度换取 38% 加速
3. **Full FT 严重过拟合**（train_loss 极低但 eval_loss 反弹），且时间不可接受

#### 消融实验（G0→G3，模型原生输出）

<p align="center">
  <img src="docs/visuals/png/experiments/ablation_g0_g3.png" width="85%" alt="消融实验" />
  <br/>
  <sub><b>Figure 7</b> · 消融实验：每一层组件的贡献（无后处理，展示模型真实能力）</sub>
</p>

| 组 | RAG | 强 prompt | LoRA | 幻觉率↓ | 格式合规↑ | EID命中↑ |
|---|:---:|:---:|:---:|---:|---:|---:|
| G0 | ❌ | ❌ | ❌ | 33.3% | 0% | 0% |
| G1 | ✅ | ❌ | ❌ | 3.3% | 0% | 0% |
| G2 | ✅ | ✅ | ❌ | 6.7% | 0% | 0% |
| **G3** | ✅ | ✅ | ✅ | **6.7%** | **76.7%** | **20.0%** |

**消融结论**：
1. **RAG 大幅降低幻觉**：G0→G1 幻觉率从 33.3% 降到 3.3%（-90%）
2. **LoRA 是格式学习的关键**：只有 G3 能原生产出 `[EventID=xxx]` 格式（0% → 76.7%）
3. **没有 LoRA 的模型完全无法引用**：G0/G1/G2 的格式合规和 EID 命中均为 0%

#### 训练效率对比

<p align="center">
  <img src="docs/visuals/png/experiments/training_efficiency.png" width="75%" alt="训练效率" />
  <br/>
  <sub><b>Figure 8</b> · 训练效率：时间 vs 性能（排除 Full FT 异常点）</sub>
</p>

#### 核心结论

1. **RAG 解决幻觉**：33.3% → 3.3%（-90%），证据约束是最有效的幻觉缓解手段
2. **LoRA 解决格式**：0% → 76.7%，<1B 模型必须靠微调才能学会结构化引用格式
3. **Qwen > Bloom**：原生中文预训练带来 40-50% 的 loss 优势
4. **LoRA > QLoRA > Full FT**：LoRA 泛化最好，QLoRA 资源最省，Full FT 过拟合

详细评测：[`docs/evaluation_metrics.md`](docs/evaluation_metrics.md) | [`docs/reports/model_comparison_final.json`](docs/reports/model_comparison_final.json)

---

## 五、前端 Demo 展示

<p align="center">
  <img src="docs/visuals/png/demo/chat_interface.png" width="80%" alt="聊天界面" />
  <br/>
  <sub><b>图 A</b> · 聊天交互界面 —— 输入问题，系统返回结构化回答 + EventID 引证</sub>
</p>

<p align="center">
  <img src="docs/visuals/png/demo/compare_page.png" width="80%" alt="G0 vs G3 对比" />
  <br/>
  <sub><b>图 B</b> · G0(裸模型) vs G3(+LoRA) 并排对比 —— 30 条样本实时切换</sub>
</p>

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
│   ├── strategies/         # 技术策略文档
│   └── reports/            # 实验报告
│
├── tools/                  # 辅助工具
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
  <sub>Built with Qwen2.5 · QLoRA · RAG · Hallucination-aware Design</sub>
</p>
