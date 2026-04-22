# Active Context

## Current Focus

Turn the CSRC punishment project from planning into execution with:

- a knowledge-backed system design
- a fine-tuning-friendly prediction task
- a defensible metric set
- clear project boundaries that avoid label leakage and invalid evaluation
- an actual project scaffold that the team can implement against

## Current State

- `memory-bank` MCP is installed and registered in Codex.
- The current Codex session does not expose any MCP memory resources, so direct markdown reads are the active fallback path.
- Local memory files exist in `memory-bank/`.
- A reusable Codex skill exists in `.agents/skills/project-delivery-team/`.
- A Python 3.11 environment exists for `openai-agents`.
- A runnable agent-team workflow exists at `tools/agent_team_runner.py`.
- A reusable project bootstrap script exists at `tools/bootstrap_codex_workspace.py`.
- This workspace is now a git repository with `main` as the default branch and `origin` set to `git@github.com:Mindse-Tt/Deeplearning-Rag-Test.git`.
- The current project input is [证监会处罚信息表.xlsx](证监会处罚信息表.xlsx), which contains one visible sheet with 14,740 data rows and 24 actual fields despite broken Excel dimension metadata.
- The dataset mixes event-level text with party-level punishment rows: 14,740 rows map to 4,233 unique `EventID`s, and 2,109 events have multiple rows.
- The user confirmed that the project should be framed in `RAG` mode rather than a generic knowledge-based pipeline.
- `M` column (`Activity` / `违规行为`) is the primary evidence field and should be treated as the core retrieval and modeling text.
- The likely final deliverable is an end-to-end demo system, potentially with a simple frontend and backend web application.
- The course requirements confirm that this project should target Track `B` (large-model fine-tuning and application / RAG) and must finally deliver a paper, code repository, and presentation.
- The course explicitly requires pre/post fine-tuning comparison, hallucination mitigation, deployment considerations, baseline comparisons, and clear AI usage disclosure.
- A formal project framework document now exists at `docs/项目总体框架.md`, covering direction, scope boundary, technical route, retrieval strategy, baselines, validation, and evidence.
- A concrete engineering scaffold now exists with `README.md`, `pyproject.toml`, `configs/`, `scripts/`, and `src/csrc_rag/`.
- The first processed artifacts have been generated successfully:
  - `data/processed/event_corpus.jsonl`
  - `data/processed/party_samples.jsonl`
  - `data/processed/build_summary.json`
- The current routing design gives sanction recommendation higher priority than law-grounding when a query asks for both recommendation and legal basis.
- The first retrieval baseline is now implemented with:
  - `configs/retrieval.json`
  - `configs/models.json`
  - `src/csrc_rag/retrieval/tokenizer.py`
  - `src/csrc_rag/retrieval/bm25.py`
  - `src/csrc_rag/retrieval/chunking.py`
  - `src/csrc_rag/retrieval/engine.py`
  - `src/csrc_rag/retrieval/dense.py`
  - `src/csrc_rag/retrieval/hybrid.py`
  - `scripts/build_event_chunks.py`
  - `scripts/query_retrieval.py`
  - `scripts/query_hybrid_retrieval.py`
  - `scripts/evaluate_retrieval_sanity.py`
- A minimal local demo frontend is now implemented with:
  - `scripts/run_demo_server.py`
  - `web/index.html`
  - `web/app.js`
  - `web/style.css`
- The frontend has now been upgraded from a simple form page into a chat-style workspace inspired by `LKWCoach`, with:
  - left-side session list
  - top status badges
  - central conversation stream
  - bottom composer
  - evidence cards under assistant answers
- `event_chunks.jsonl` has been generated successfully with `29314` chunks.
- A training stack now exists with:
  - `src/csrc_rag/training/data.py`
  - `src/csrc_rag/training/metrics.py`
  - `src/csrc_rag/training/sklearn_baseline.py`
  - `src/csrc_rag/training/hf_finetune.py`
  - `scripts/train_punishment_baseline.py`
  - `scripts/train_punishment_finetune.py`
- A local Python 3.11 training environment now exists at `.venv311` with `torch`, `transformers`, `datasets`, and `sentence-transformers` installed.
- Current dense retrieval is wired through a pluggable backend. The default backend is `svd_tfidf` for immediate local execution; the architecture is ready to switch to a real `sentence-transformer` model later.
- A model-based intent router is now implemented with:
  - `configs/intent_examples.json`
  - `src/csrc_rag/orchestration/intent_model.py`
  - `scripts/train_intent_classifier.py`
  - artifact at `artifacts/intent_classifier/intent_model.pkl`
- A local response-model layer is now implemented with:
  - `src/csrc_rag/response/responder.py`
  - default backend `local_hf`
  - local model path `artifacts/models/Qwen2.5-0.5B-Instruct`
  - template fallback when local model loading fails
- The local Qwen response model has been downloaded into the project directory through `HF_ENDPOINT=https://hf-mirror.com`.
- The demo server now exposes `POST /api/query` with query history support and returns:
  - intent
  - intent confidence and scores
  - response backend/model
  - query plan
  - retrieved events
- The current recommended way to run the full local stack is `.venv311/bin/python scripts/run_demo_server.py`.
- The project has now been published to GitHub as a private repository:
  - `https://github.com/Mindse-Tt/Deeplearning-Rag-Test`

## Next Actions

- Replace the current `svd_tfidf` dense backend with a real Chinese sentence-transformer model and compare hybrid retrieval quality.
- Improve the local response model prompt and/or swap to a stronger local model if answer quality is not adequate for the report/demo.
- Promote the current punishment-type baseline into a formal model comparison table: TF-IDF baseline vs Chinese transformer fine-tuning.
- Add a reranker stage and a manual evaluation set for retrieval quality.
- Keep the intent classifier, query-plan layer, and local responder as the protocol between frontend questions and backend pipelines.
- Keep all future design choices aligned with the course deliverables: opening proposal, midterm check, final paper, demo, and reproducible code.
- Start Codex with `zsh start_codex_team.sh`.
- Use `zsh run_agent_team.sh "<task>"` when a task needs structured requirement and planning artifacts before Codex execution.
- If the GitHub repository needs to become easier to clone or publish publicly, move oversized processed data files to Git LFS or replace them with smaller sample artifacts first.
- Copy this setup into other projects with `python3.11 tools/bootstrap_codex_workspace.py <target-dir>`.
- If long-term chat storage is important, keep launching Codex with `disable_response_storage=false` or change the global config intentionally later.
- If programmable multi-agent orchestration is needed, add a small `openai-agents` runner script on top of the Python 3.11 environment.

## Open Risks

- External SDK-based orchestration needs an API key.
- The currently configured `OPENAI_API_KEY` is invalid for live API calls.
- The memory bank captures distilled project facts, not every raw chat turn.
- Global Codex settings may still override desired persistence unless the local launcher is used.
- The current Downloads directory behaves like `noexec` for shell scripts, so wrapper scripts should be launched through `zsh`.
- `PunishmentMeasure` directly describes sanctions and should not be used as a model input for punishment prediction because it creates label leakage.
- Python compilation in this workspace should set `PYTHONPYCACHEPREFIX` into the project directory to avoid sandbox permission issues under the default cache path.
- Running the local demo server from the sandbox required escalation because binding `127.0.0.1:8000` is restricted in the current environment.
- The current GitHub repository includes two processed data files larger than GitHub's recommended `50MB` threshold:
  - `data/processed/event_chunks.jsonl`
  - `data/processed/party_samples.jsonl`
- The system Python 3.9 environment can run `numpy/sklearn`, but `torch/transformers` were unstable there due OpenMP/shared-memory issues; `.venv311` is the stable path for transformer training.
- The local Qwen 0.5B model is now runnable, but first-load latency is non-trivial and answer quality is still below a production-grade legal assistant; this should be presented as a course-project local baseline, not a final oracle.
