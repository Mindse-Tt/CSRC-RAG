# Progress

## 2026-04-04

### Completed

- Re-read `project-memory/` and `memory-bank/` to restore the full project operating context at session start.
- Verified that the local delivery skill definition still exists at `.agents/skills/project-delivery-team/SKILL.md`.
- Verified that the structured runner entrypoint still works with `zsh run_agent_team.sh --help`.
- Confirmed that the current workspace contains the expected local scaffolding under `.agents/` and `tools/`.
- Inspected `证监会处罚信息表.xlsx` and confirmed that the workbook metadata is misleading: the XML dimension says `A1:A14743`, but the actual sheet contains 24 fields from `EventID` through `SumPenalty`.
- Quantified the dataset shape for task selection: 14,740 rows, 4,233 unique events, 2,109 multi-row events, long-form factual text in `Activity`, and complete multi-label `PunishmentType` values.
- Captured the newly clarified project direction: RAG is the intended main mode, `M` column is the primary evidence field, and the final deliverable may include a web frontend and backend.
- Parsed the full course brief and aligned the project with Track `B`, including the required deliverables, evaluation emphasis, and timeline constraints.
- Wrote `docs/项目总体框架.md` as the first formal project artifact, consolidating the project positioning, scope boundary, technical route, retrieval strategy, baselines, validation plan, and evidence basis in Chinese.
- Created the first executable project scaffold:
  - `README.md`
  - `pyproject.toml`
  - `configs/knowledge_base.json`
  - `configs/intents.json`
  - `configs/experiments.json`
  - `docs/执行路线与验证计划.md`
  - `src/csrc_rag/...`
  - `scripts/profile_dataset.py`
  - `scripts/build_corpora.py`
- Built the first processed outputs from the Excel file:
  - `data/processed/event_corpus.jsonl`
  - `data/processed/party_samples.jsonl`
  - `data/processed/build_summary.json`
- Fixed the initial intent-routing priority so mixed questions like “推荐处罚并说明法条依据” route to the sanction recommendation pipeline instead of the law-grounding-only pipeline.
- Implemented the first retrieval baseline stack:
  - chunk builder
  - Chinese-friendly BM25 tokenizer
  - BM25 index
  - retrieval engine
  - retrieval sanity evaluation script
- Implemented a minimal local frontend demo served from `scripts/run_demo_server.py` and `web/`.
- Added documentation for retrieval strategy, knowledge-base construction, current baseline position, and frontend test questions.
- Implemented the dense/hybrid retrieval extension:
  - `configs/models.json`
  - `src/csrc_rag/retrieval/dense.py`
  - `src/csrc_rag/retrieval/hybrid.py`
  - `scripts/build_dense_index.py`
  - `scripts/query_hybrid_retrieval.py`
- Implemented the punishment-type training stack:
  - data split utilities
  - multilabel metrics
  - TF-IDF + OneVsRest logistic regression baseline
  - Hugging Face fine-tuning script
- Created `.venv311` and installed:
  - `numpy`
  - `scikit-learn`
  - `torch`
  - `transformers`
  - `datasets`
  - `sentence-transformers`
  - `accelerate`

### Verified

- Local memory files are present and readable from the project root.
- The current Codex session exposes no MCP memory resources, so direct markdown reads are required in this session.
- `zsh run_agent_team.sh --help` exits successfully and prints the runner interface.
- `git rev-parse --is-inside-work-tree` fails with `not a git repository`, confirming the current workspace is not under git.
- The dataset spans years `1994` through `2025`.
- `Activity` is information-rich enough for NLP modeling with median length about 1,503 characters and p90 about 4,170 characters.
- `PunishmentType` has 45 unique label combinations and is mostly multi-label: 10,231 rows contain two labels and 1,998 rows contain three labels.
- `PYTHONPATH=src PYTHONPYCACHEPREFIX=artifacts/pycache python3 -m py_compile ...` succeeds for the new scaffold modules.
- `PYTHONPATH=src python3 scripts/build_corpora.py` succeeds and writes the expected processed files.
- `data/processed/build_summary.json` reports `14740` raw rows, `4233` event documents, and `14740` party samples.
- The generated `party_samples.jsonl` input text excludes `PunishmentMeasure`, keeping the prediction path free from the main leakage field.
- The intent router now maps the query “根据这段违规行为推荐处罚方式，并说明相关法条依据” to `sanction_recommendation`.
- `PYTHONPATH=src python3 scripts/build_event_chunks.py` succeeds and writes `data/processed/event_chunks.jsonl` with `29314` chunks.
- `PYTHONPATH=src python3 scripts/query_retrieval.py 上市公司董事利用内幕信息买入股票获利，历史上通常对应哪些案例` returns relevant内幕交易案例.
- `PYTHONPATH=src python3 scripts/evaluate_retrieval_sanity.py` returns:
  - `Recall@5 = 0.0933`
  - `MRR = 0.0811`
  - `nDCG@10 = 0.0822`
- The local demo server starts successfully with escalation, `/api/health` returns `{"status":"ok"}`, the root page serves HTML, and `/api/query` returns structured retrieval results.
- `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/train_punishment_baseline.py` succeeds and reports:
  - `Micro-F1 = 0.8644`
  - `Macro-F1 = 0.2759`
  - `HammingLoss = 0.0691`
  - `SubsetAccuracy = 0.5961`
- `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv311/bin/python` can import:
  - `numpy`
  - `sklearn`
  - `torch`
  - `transformers`
  - `datasets`
  - `sentence_transformers`
- `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv311/bin/python scripts/query_hybrid_retrieval.py ...` succeeds with the current `svd_tfidf` dense backend.
- `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv311/bin/python scripts/build_dense_index.py --backend svd_tfidf` succeeds and reports `29314` indexed documents.
- `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE .venv311/bin/python scripts/train_punishment_finetune.py --model-name hf-internal-testing/tiny-random-bert ...` succeeds as a smoke test and produces a full transformer training/evaluation run on a small sample.

### Notes

- The project workflow is ready for the next task and should continue using the fixed loop: requirements -> plan -> execution -> verification -> memory update.
- For this workspace state, future change tracking should not assume git metadata is available.
- The main modeling boundary should separate event-level retrieval/generation from party-level punishment prediction, otherwise evaluation will mix different sample granularities.
- `PunishmentMeasure` should be excluded from model input features in predictive tasks because it leaks the target decision.
- The project explanation should now be organized around RAG use cases, retrieval strategy, model comparison baselines, and the path to an end-to-end demo.
- The final project must satisfy course-specific checkpoints: one-page proposal by week 8, one-page midterm report by week 12, classroom presentation in week 15, and final paper plus code in week 16.
- The write-up must include `Team Contributions` and `AI Contribution Statement`, and the system design should support a strong baseline comparison plus deployment/cost discussion.
- The framework is now stable enough to support the next execution steps without re-deciding the core topic in each session.
- The repository is now ready to move into retrieval baselines and prediction baselines instead of further high-level topic discussion.
- The repository now has a working baseline retrieval demo, but the current retrieval quality is only the first floor and still needs Dense / Hybrid / Reranker upgrades plus manual evaluation data.
- The repository now has both a runnable classical baseline and a verified transformer training path, so the project is no longer blocked on “can we train models here?”.

## 2026-04-06

### Completed

- Confirmed that the repository still lacked two critical layers after the initial scaffold:
  - a real response-model abstraction
  - a model-based intent router
- Inspected the local frontend and verified that the previous UI was still a basic single-page query form rather than the intended chat workspace.
- Used the locally provided `LKWCoach-main.zip` as the visual reference source and extracted its main template/CSS structure.
- Implemented a model-based intent-routing stack:
  - added `configs/intent_examples.json`
  - added `src/csrc_rag/orchestration/intent_model.py`
  - added `scripts/train_intent_classifier.py`
  - trained and wrote `artifacts/intent_classifier/intent_model.pkl`
  - trained and wrote `artifacts/intent_classifier/intent_report.json`
- Implemented a local response-model stack:
  - added `src/csrc_rag/response/responder.py`
  - added local Hugging Face backend support
  - kept a template fallback backend for failure cases
- Updated `src/csrc_rag/orchestration/intents.py` so `route_query(...)` now returns structured intent decisions with:
  - predicted intent
  - confidence
  - routing method
  - per-intent scores
- Updated `src/csrc_rag/retrieval/engine.py` so search responses now include:
  - intent confidence
  - intent method
  - intent scores
  - response backend
  - response model
- Updated `scripts/run_demo_server.py` to:
  - default to `hybrid` retrieval mode
  - expose `POST /api/query`
  - accept query history
  - return richer metadata needed by the chat UI
- Rebuilt the web frontend into a chat-style workspace with:
  - left-side session list
  - new-chat button
  - top status badges
  - central message timeline
  - bottom composer
  - evidence cards per assistant answer
- Downloaded the local response model into the repository with the China-friendly mirror path:
  - `artifacts/models/Qwen2.5-0.5B-Instruct`
- Switched the response-model config from a remote Hugging Face repo ID to the local project path.
- Added `docs/本地模型接入与前端改版说明.md`.
- Updated `README.md` so the recommended local stack now uses `.venv311/bin/python`.

### Verified

- `python3 -m compileall src scripts web` succeeds after the new model/router/frontend changes.
- `python3 scripts/train_intent_classifier.py` succeeds and writes a new intent-classifier artifact.
- `.venv311/bin/python scripts/train_intent_classifier.py` also succeeds, avoiding cross-environment skew for the saved classifier.
- The intent-classifier report currently shows synthetic-template holdout accuracy `1.0` on the augmented intent dataset.
- `HF_ENDPOINT=https://hf-mirror.com ... snapshot_download(...)` succeeds and writes `Qwen/Qwen2.5-0.5B-Instruct` into `artifacts/models/`.
- `.venv311/bin/python` can now run a full end-to-end search with:
  - `intent = sanction_recommendation`
  - `intent_method = tfidf_logistic_regression`
  - `response_backend = local_hf`
  - `response_model = artifacts/models/Qwen2.5-0.5B-Instruct`
- The local demo server now serves the upgraded frontend and responds on:
  - `GET /api/health` -> `{"status":"ok","retrieval_mode":"hybrid"}`
  - `POST /api/query` -> structured JSON with local-hf answer and retrieved evidence

### Notes

- The local response-model path is now real and runnable, but its answer quality is still only a small-model baseline and can drift into generic phrasing.
- The frontend is now presentation-ready enough for team testing, but the best demo experience still depends on improving retrieval quality and response grounding.
- The current dense component inside `hybrid` is still `svd_tfidf`; the next major technical upgrade should be a real Chinese embedding model plus reranking.

## 2026-03-31

### Completed

- Researched the main GitHub options for Codex-adjacent skills, agent-team orchestration, and memory.
- Installed `memory-bank-mcp` into `vendor/memory-bank-mcp`.
- Built the Memory Bank MCP server locally with `npm install` and `npm run build`.
- Registered `memory-bank` as a global Codex MCP server.
- Patched the local Memory Bank MCP source so it auto-detects an existing `memory-bank/` directory.
- Created a reusable Codex skill: `project-delivery-team`.
- Created project memory files for user profile, intake workflow, and the six memory-bank documents.
- Created a Python 3.11 virtual environment and installed `openai-agents` successfully.
- Built `tools/agent_team_runner.py` to generate requirement, plan, execution brief, verification, and durable session summary artifacts with `SQLiteSession`.
- Built `tools/bootstrap_codex_workspace.py` to copy this Codex workflow into other projects.
- Added `run_agent_team.sh` as a terminal wrapper for the agent runner.

### Verified

- `codex mcp list` shows `memory-bank` as enabled.
- The patched Memory Bank MCP build succeeded.
- `openai-agents` imports successfully from `.codex-agent-venv311`.
- `python3.11 -m py_compile tools/agent_team_runner.py tools/bootstrap_codex_workspace.py` succeeds.
- `zsh run_agent_team.sh --help` succeeds.
- `env -u OPENAI_API_KEY zsh run_agent_team.sh 测试任务` fails fast with a clear API key error.
- `python3.11 tools/bootstrap_codex_workspace.py /private/tmp/codex-bootstrap-test --force` successfully copies the scaffold.
- A live `zsh run_agent_team.sh --model gpt-5-mini ...` call reaches the OpenAI API but fails with `401 invalid_api_key`, confirming the current environment key is invalid rather than missing.

### Notes

- The Python 3.9 environment can install some packages but is not the recommended runtime for `openai-agents`.
- Session persistence in Codex is affected by the current global config and should be overridden during launch when needed.
- In this Downloads workspace, shell wrappers should be launched with `zsh ...` instead of `./...`.
