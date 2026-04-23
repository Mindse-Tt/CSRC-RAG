# CSRC-RAG · 证监会违规案例检索增强问答系统

> **深度学习课程设计 · 赛道 B** · 2026.04
> 作者:许浩财 · 贾彤 · 戴一鑫 · 张彦扬 · 王怡菲

![badge](https://img.shields.io/badge/Track-B-blue) ![badge](https://img.shields.io/badge/Base-Qwen2.5--0.5B-orange) ![badge](https://img.shields.io/badge/Method-QLoRA-green) ![badge](https://img.shields.io/badge/GPU-RTX_2060S_8GB-lightgrey)

---

## 一、项目是什么

本项目面向**中国证监会行政处罚公告**构建了一个端到端 RAG(检索增强生成)问答系统,在 Qwen2.5-0.5B-Instruct 基座上用 QLoRA 做指令微调,**把幻觉数字率从 20% 压到 3.3%(-83%),格式合规率从 0 提升到 76.7%**,延迟只增加 +0.7s。

```
  用户问: "谭光华因违规买卖股票被处罚的详情?"
      ↓
  ┌──── 七层 RAG 流水线 ────┐
  │ L1 意图分类              │
  │ L2 查询改写(多约束拆分) │
  │ L3 BM25 + bge 向量 + RRF │
  │ L4 bge-reranker-v2 精排  │
  │ L5 Qwen + LoRA 生成       │
  │ L6 趋势聚合(SQL groupby)│
  │ L7 引证校验               │
  └───────────────────────────┘
      ↓
  "参考历史案例... [EventID=40111147],
   处罚方式包括没收非法所得、罚款..."
```

---

## 二、核心成果

| 指标 | G0 (裸模型) | **G3 (+LoRA)** | 绝对增量 |
|---|---:|---:|---:|
| EventID 引证命中率 | 0% | **20.0%** | +20.0 pp |
| 格式合规率 | 0% | **76.7%** | +76.7 pp |
| 幻觉数字率 | 20.0% | **3.3%** | **-16.7 pp (-83%)** |
| 端到端延迟 | 9.84 s | 10.54 s | +0.7 s (+7%) |

**实验配置**:n=30 gold 分层抽样,同证据、同 prompt、同采样参数,仅换模型。控制变量严格。

---

## 三、交付物清单(答辩用)

| 类型 | 路径 | 说明 |
|---|---|---|
| 📘 **论文** | [`docs/paper/csrc_rag_v2.docx`](docs/paper/csrc_rag_v2.docx) | 16 页,6 图 + 6 表,ACL/EMNLP 学术风 |
| 🌐 **Showcase 主页** | [`docs/showcase/index.html`](docs/showcase/index.html) | 静态页,KPI + 架构图 + 消融表 |
| 🔀 **G0 vs G3 对比** | [`docs/showcase/compare.html`](docs/showcase/compare.html) | 30 条样本并排对比 ⭐ 最直观 |
| 💻 **老版聊天 Demo** | `scripts/run_demo_server.py` | 真实跑模型的交互式前端 |
| 📊 **评测报告** | [`docs/reports/m4_4_generation_eval.md`](docs/reports/m4_4_generation_eval.md) | G0-G3 × 5 指标总表 + 定性样例 |
| 🧪 **LoRA Adapter** | `artifacts/models/qwen_lora_csrc/` | 34 MB,Qwen2.5-0.5B + QLoRA r=16 |
| 🚀 **Colab 1.5B 备选** | [`notebooks/qwen_1_5b_qlora_colab.ipynb`](notebooks/qwen_1_5b_qlora_colab.ipynb) | T4 / P100 上的 1.5B 重训 |
| 🎨 **学术风图源码** | `scripts/build_paper_figures.py` | matplotlib 300 DPI,6 张 PNG |

---

## 四、怎么快速看成果

### 方式 A · 静态 Showcase(最快,无需跑模型)

```bash
python -m http.server 8765 --directory docs
```

浏览器打开:
- **主页**: http://127.0.0.1:8765/showcase/index.html
- **G0 vs G3 对比**: http://127.0.0.1:8765/showcase/compare.html

### 方式 B · 真实交互 Demo(需装依赖、会下模型)

```bash
python scripts/run_demo_server.py
# 浏览器打开 http://127.0.0.1:8000
```

试这几个 query:
- `帮我找和内幕交易类似的处罚案例`
- `谭光华因违规买卖股票被处罚的详情`
- `2022 年董事长因内幕交易被罚款的案件有哪些?`
- `近五年内幕交易案件是否呈上升趋势?`

### 方式 C · 在线访问(需要你开 GitHub Pages)

1. 浏览器打开 https://github.com/Mindse-Tt/Deeplearning-Rag-Test/settings/pages
2. Source → Branch: `feature/track-b-finetune`,Folder: `/docs`
3. Save,等 1-2 分钟后访问:
 `https://mindse-tt.github.io/Deeplearning-Rag-Test/showcase/`

---

## 五、技术栈一览

| 层 | 技术 | 关键参数 |
|---|---|---|
| **基座** | Qwen/Qwen2.5-0.5B-Instruct | 0.5 B 参数 |
| **量化** | bitsandbytes 4-bit NF4 + double quant | compute_dtype=fp16 |
| **微调** | QLoRA r=16 α=32 | target: q/k/v/o/gate/up/down_proj |
| **稀疏检索** | BM25 + jieba | k1=1.2 b=0.75 + 领域 user_dict |
| **稠密检索** | BAAI/bge-small-zh-v1.5 | 512 维 cosine |
| **精排** | BAAI/bge-reranker-v2-m3 | cross-encoder |
| **融合** | 三层 Reciprocal Rank Fusion | k=60 |
| **意图分类** | TF-IDF + Logistic Regression | Macro-F1 = 0.9989 |
| **辅线分类** | MacBERT-base multilabel | Micro-F1 = 0.680 |
| **训练硬件** | RTX 2060 SUPER 8 GB | 52 min, 2 epoch |

---

## 六、目录结构

```
Deeplearning-Rag-Test/
├── configs/                  # 模型/QLoRA 配置
├── data/
│   ├── processed/            # 事件级 (4,233) + 当事人级 (14,740)
│   └── eval/                 # gold_130, gold_trend_30
├── docs/
│   ├── paper/                # ← 论文 docx 最终交付
│   ├── showcase/             # ← 静态前端(GitHub Pages)
│   ├── reports/              # M3e/M4.x/M5 各阶段实验报告
│   └── visuals/png/paper/    # ← 6 张学术风图
├── src/csrc_rag/
│   ├── retrieval/            # 7 层流水线核心
│   ├── orchestration/        # 意图路由 + L6 趋势聚合
│   ├── response/             # Qwen + LoRA 生成
│   └── training/             # QLoRA / MacBERT 训练
├── scripts/                  # 所有可执行入口
│   ├── train_qlora_m4.py        # LoRA 主训练
│   ├── evaluate_generation_m4_4.py  # G0-G3 评测
│   ├── build_paper_figures.py   # 6 张图
│   ├── build_paper_docx.py      # 论文 docx
│   └── run_demo_server.py       # 聊天前端
├── notebooks/                # Colab 1.5B 备选
├── artifacts/
│   ├── models/qwen_lora_csrc/   # ← LoRA adapter
│   └── macbert_csrc/            # ← MacBERT 权重
└── web/                      # 老版交互前端(聊天式)
```

---

## 七、复现步骤

### 本机(2060S 8 GB)

```bash
# 1. 数据构建
python scripts/build_corpora.py
python scripts/build_event_chunks.py

# 2. 检索 baseline
python scripts/evaluate_retrieval.py --dataset gold_130

# 3. LoRA 训练(52 min)
python scripts/train_qlora_m4.py

# 4. G0-G3 四组评测(约 30 min)
python scripts/evaluate_generation_m4_4.py --n-samples 30

# 5. 出图 + 出论文
python scripts/build_paper_figures.py
python scripts/build_paper_docx.py
```

### Colab(T4 GPU,免费档)

打开 `notebooks/qwen_1_5b_qlora_colab.ipynb`,整本跑完 ≈ 3-5 小时,产出 1.5 B 版 adapter。

---

## 八、已知局限(诚实声明)

1. **基座规模**:受 8 GB 显存约束,主训练用 0.5 B 而非 1.5 B;1.5 B 仅提供 Colab 方案
2. **EID 命中率 20%**:检索器 Recall@5=0.388 的天花板,生成层已尽力
3. **评测样本 30 条**:95% 置信区间 ±18%,需扩到 100 条
4. **幻觉检测仅覆盖数字**:人名/公司名幻觉需人工标注
5. **辅线 MacBERT 未串接**:Micro-F1=0.68 已训完,尚未合入主流程 engine

---

## 九、相关文档

- [完整方案 · 总纲](docs/完整方案-总纲.md)
- [微调方案](docs/微调方案.md)
- [M4.4 评测报告(核心)](docs/reports/m4_4_generation_eval.md)
- [M4.3 LoRA 训练报告](docs/reports/m4_3_lora_training_report.md)
- [M3e 检索消融报告](docs/reports/m3e_ablation_report.md)
- [M5 MacBERT 辅线报告](docs/reports/m5_macbert_report.md)

---

## 十、License

仅作课程交付使用。CNRDS 原始数据版权归数据提供方所有。
