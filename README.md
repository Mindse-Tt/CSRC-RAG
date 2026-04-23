# 证监会处罚案例 RAG 项目

本仓库用于实现一个面向证监会处罚案例的课程项目，核心目标是构建：

- 事件级案例检索与法规依据 RAG 系统
- 当事人级处罚类型预测模型
- 可用于答辩展示的端到端 Demo

## 🎯 交付物快速入口 (Track B, M6)

| 产物 | 路径 | 说明 |
|---|---|---|
| 📘 **论文 (.docx)** | [`docs/paper/csrc_rag_v1.docx`](docs/paper/csrc_rag_v1.docx) | 8-12 页,8 节,含 4 表 + 3 图,数据全部实测 |
| 🌐 **项目 Showcase** | [`docs/showcase/index.html`](docs/showcase/index.html) | GitHub Pages 首页 (KPI + 架构 + 消融) |
| 🔀 **G0 vs G3 对比** | [`docs/showcase/compare.html`](docs/showcase/compare.html) | 30 条样本并排逐条对比,可按筛选条件切换 |
| 📊 **M4.4 评测报告** | [`docs/reports/m4_4_generation_eval.md`](docs/reports/m4_4_generation_eval.md) | G0-G3 四组 × 5 指标总表 + 定性样例 |
| 🧪 **LoRA Adapter (0.5B 本地)** | `artifacts/models/qwen_lora_csrc/` | 34 MB,Qwen2.5-0.5B-Instruct + QLoRA r=16 |
| 🚀 **Colab 1.5B 重训** | [`notebooks/qwen_1_5b_qlora_colab.ipynb`](notebooks/qwen_1_5b_qlora_colab.ipynb) | T4 / P100 上的 1.5B QLoRA 备选方案 |

**核心结论 (一句话)**: 在 Qwen2.5-0.5B 上,LoRA 微调达到断崖式提升 —— 格式合规率 0 → **76.7%**,幻觉数字率 20% → **3.3%** (-83%),延迟只增加 +0.7s (+7%)。

### 启动 Showcase 本地预览

```bash
python -m http.server 8765 --directory docs
# 打开 http://127.0.0.1:8765/showcase/index.html
```

或在 GitHub 上开启 Pages (Settings → Pages → Source: main / docs) 后,访问 `https://<you>.github.io/<repo>/showcase/`。


当前仓库已经完成两层基础建设：

- 项目总体框架文档
- 第一版工程骨架与数据处理入口
- 第一版 BM25 检索 baseline
- 第一版本地前端 Demo
- 第一版 hybrid retrieval（当前默认 dense backend 为 `svd_tfidf`）
- 第一版可训练处罚类型 baseline
- `transformers + sentence-transformers` 训练环境：Windows Python 3.12（系统级）
- 本地意图分类模型（`TF-IDF + Logistic Regression`）
- 本地回复模型接入（默认 `Qwen/Qwen2.5-0.5B-Instruct`，失败时模板回退）
- 聊天式前端改版（参考 `LKWCoach` 的左侧会话栏 + 中间对话区布局）
- 前端折叠式结果展示：
  - 每次回答可折叠查看“本次调用参数”
  - 每次回答可折叠查看“检索证据”
  - 每条案例证据可单独展开

## 当前推荐执行顺序

不要一开始就写前端，也不要先上生成模型。当前仓库已经把这两块接上，但实验推进仍然推荐按下面顺序做：

1. 数据工程
   先把 Excel 稳定读通，拆成事件级知识库语料和当事人级训练样本。
2. 检索 baseline
   先做 BM25 / Dense / Hybrid 的离线检索验证。
3. 意图路由
   定义系统支持的问题类型、检索范围和回答模板。
4. 预测 baseline
   先跑 `PunishmentType` 多标签分类基线。
5. RAG 生成
   把检索结果接到生成模型，控制回答只基于证据输出。
6. Demo
   最后再做网页或演示层。

这个顺序比“先做一个页面再往里补功能”稳，因为每一步都能独立验证。

## 目录结构

```text
configs/                 项目配置（知识库字段、意图、实验设计）
data/processed/          处理后的事件级/当事人级数据
docs/                    项目文档与执行方案
scripts/                 命令行入口脚本
src/csrc_rag/            Python 包代码
  data/                  数据读取与样本构造
  orchestration/         意图路由与意图分类模型
  response/              本地回复模型与模板回退
  retrieval/             检索相关逻辑
  training/              训练任务定义
  evaluation/            评估方案定义
artifacts/               中间产物和输出
```

## 当前系统链路

当前网页端和后端采用如下链路：

`用户问题 -> 意图分类模型 -> Query Plan -> Hybrid Retrieval -> 本地回复模型 -> 折叠式证据展示`

其中：

- 意图分类模型：`TF-IDF + LogisticRegression`
- 检索模式：`hybrid`
- 当前 dense backend：`svd_tfidf`
- 回复模型：本地 `Qwen2.5-0.5B-Instruct`
- 回答失败回退：模板 responder

## 前端现在能看到什么

新版前端不再把全部证据直接铺满页面，而是按“先看答案，再按需展开”的方式组织。

每条系统回答现在包含：

- 简要回答文本
- 顶部摘要参数
  - 意图
  - 路由方式
  - 检索单元
  - top_k
  - 回复后端
- 可折叠的“本次调用参数”
  - 意图置信度
  - 过滤条件
  - 回复模型路径
  - 意图打分
- 可折叠的“检索证据”
  - 命中案例数量
  - 每条案例的标题、时间、机构、分数
  - 每条案例的处罚方式、法规依据、证据片段

## 你做测试时应该重点看哪些参数

推荐优先看这几个：

- `intent`
  - 问题是否被正确分到 `case_retrieval / law_grounding / sanction_recommendation / trend_analysis`
- `intent_confidence`
  - 如果特别低，说明意图路由不稳
- `retrieval_unit`
  - 当前是否按事件级检索
- `top_k`
  - 当前问题召回了多少候选案例
- `metadata_filters`
  - 是否触发了年份、上市公司、监管机构等过滤
- `response_backend`
  - 当前是否真的走了 `local_hf`
- `response_model`
  - 当前是否使用本地模型路径，而不是远程 API

## 推荐测试方式

### 1. 前端人工测试

启动：

```bash
.venv311/bin/python scripts/run_demo_server.py
```

> Windows 下改为：`python scripts\run_demo_server.py`（或直接双击 `start.bat`）。

打开：

```text
http://127.0.0.1:8000
```

先测下面几类问题：

- 相似案例检索
  - `帮我找和内幕交易类似的处罚案例`
- 法条依据
  - `这类行为通常违反哪些法条`
- 处罚推荐
  - `根据这段违规行为推荐处罚方式，并说明法条依据`
- 趋势分析
  - `近五年内幕交易相关案件的处罚是否更严`

### 2. 接口测试

如果你想直接看 JSON 返回：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{"query":"根据这段违规行为推荐处罚方式，并说明法条依据。","history":[]}'
```

你应该重点关注：

- `intent`
- `intent_confidence`
- `intent_method`
- `response_backend`
- `response_model`
- `query_plan`
- `events`

## 第一批建议执行命令

查看数据概况：

```bash
PYTHONPATH=src python3 scripts/profile_dataset.py
```

构建事件级知识库和当事人级训练样本：

```bash
PYTHONPATH=src python3 scripts/build_corpora.py
```

构建检索 chunk 库：

```bash
PYTHONPATH=src python3 scripts/build_event_chunks.py
```

运行第一版检索查询：

```bash
PYTHONPATH=src python3 scripts/query_retrieval.py 上市公司董事利用内幕信息买入股票获利，历史上通常对应哪些案例
```

运行 hybrid retrieval：

```bash
.venv311/bin/python scripts/query_hybrid_retrieval.py 上市公司董事利用内幕信息买入股票获利，历史上通常对应哪些案例
```

训练本地意图分类模型：

```bash
.venv311/bin/python scripts/train_intent_classifier.py
```

运行第一版检索 sanity benchmark：

```bash
PYTHONPATH=src python3 scripts/evaluate_retrieval_sanity.py
```

构建 dense backend 摘要：

```bash
.venv311/bin/python scripts/build_dense_index.py --backend svd_tfidf
```

运行处罚类型可训练 baseline：

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_punishment_baseline.py
```

运行 Hugging Face 微调 smoke test：

```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv311/bin/python scripts/train_punishment_finetune.py \
  --model-name hf-internal-testing/tiny-random-bert \
  --output-dir artifacts/hf_smoke \
  --max-train-samples 64 \
  --max-valid-samples 16 \
  --max-test-samples 16 \
  --batch-size 2 \
  --num-train-epochs 1 \
  --max-length 128
```

启动本地前端 Demo：

```bash
.venv311/bin/python scripts/run_demo_server.py
```

> Windows 下改为：`python scripts\run_demo_server.py`。

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

查看意图路由示例：

```bash
PYTHONPATH=src .venv311/bin/python -c "from csrc_rag.orchestration.intents import load_registry, route_query; r=load_registry(); print(route_query('根据这段违规行为推荐处罚方式', r))"
```

## 当前已有文档

- 总体框架：[docs/项目总体框架.md](docs/项目总体框架.md)
- 执行与验证路线：[docs/执行路线与验证计划.md](docs/执行路线与验证计划.md)
- 检索策略与知识库设计：[docs/检索策略与知识库设计.md](docs/检索策略与知识库设计.md)
- 前端测试问题：[docs/前端测试问题.md](docs/前端测试问题.md)
- 本地模型接入与前端改版说明：[docs/本地模型接入与前端改版说明.md](docs/本地模型接入与前端改版说明.md)
- 微调方案（赛道 B 核心）：[docs/微调方案.md](docs/微调方案.md)

## 后续优先级

第一优先级：

- 跑通数据处理
- 产出 `event_corpus.jsonl`
- 产出 `party_samples.jsonl`
- 产出 `event_chunks.jsonl`

第二优先级：

- 在 BM25 baseline 上继续补 Dense / Hybrid / Reranker
- 确定 `PunishmentType` 的多标签建模方案
- 把 `svd_tfidf` dense backend 升级为真实 sentence-transformer 中文 embedding 模型

第三优先级：

- 把 `svd_tfidf` dense backend 升级为真实中文 embedding 模型
- 把本地回复模型从接入状态推进到稳定实验状态
- 补趋势分析与处罚预测的前端可视化

## 注意事项

- `PunishmentMeasure` 不能进入处罚预测输入，否则会泄漏标签。
- 检索知识库不止 `Activity`，但 `Activity` 仍然是主证据字段。
- 训练和评估必须按 `EventID` 防泄漏切分，优先采用时间切分。
- 第一版检索 benchmark 目前只是 sanity check，不代表最终跨案例相似检索效果，后续还需要人工标注评测集。
- 当前已经有两条训练路径：
  - Windows Python 3.12 的 `sklearn` baseline
  - Windows Python 3.12 的 `transformers` 微调路径
- 当前推荐所有“本地模型推理”命令都使用 `python`（Python 3.12）
- 当前前端已经适合第一轮人工验证，但回复模型质量和 dense 检索质量仍需继续优化
