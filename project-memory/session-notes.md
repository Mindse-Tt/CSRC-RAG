# Session Notes

Use this file for distilled session memory, not raw transcripts.

For each meaningful session, append:

- Date
- Task
- Key decisions
- What changed
- What remains open

## 2026-03-31

- Task: Set up a reusable Codex workflow for skill-driven delivery, multi-agent expansion, and project memory.
- Key decisions:
  - Use a local Codex skill as the default workflow controller.
  - Use `memory-bank-mcp` plus markdown files for durable project memory.
  - Keep external multi-agent frameworks optional.
  - Use Python 3.11 for `openai-agents`.
- What changed:
  - Installed and registered `memory-bank`.
  - Created project memory files and the `project-delivery-team` skill.
  - Added a launcher script that starts Codex with response storage enabled for this project.
- What remains open:
  - Decide whether to change the global Codex config instead of relying on the local launcher.
  - Add an `openai-agents` runner script if programmable orchestration becomes necessary.

## 2026-03-31 03:00:00

- Task: Turn the workflow skeleton into executable tooling and make it reusable across other projects.
- Key decisions:
  - Use a deterministic multi-agent pipeline instead of relying on ad hoc delegation.
  - Keep session persistence in `SQLiteSession`.
  - Add a bootstrap script so the structure can be copied into future projects.
  - Document the `Downloads` noexec behavior and use `zsh script.sh` as the stable invocation path.
- What changed:
  - Added `tools/agent_team_runner.py`.
  - Added `tools/bootstrap_codex_workspace.py`.
  - Added `run_agent_team.sh`.
  - Updated memory files with the new workflow and environment constraints.
- What remains open:
  - Replace the invalid `OPENAI_API_KEY` and rerun a live agent-team workflow.
  - Decide whether to promote the launcher behavior into global Codex defaults.

## 2026-04-04

- Task: Take over the project in a new Codex session, reload memory, verify the local delivery workflow, and enter standby.
- Key decisions:
  - Follow the local `project-delivery-team` workflow manually even though the skill was not surfaced in the current session's skill list.
  - Use direct markdown reads as the session fallback because no MCP memory resources were exposed.
  - Treat the workspace as non-git-backed unless repository metadata is added later.
- What changed:
  - Re-read all project memory files and the six `memory-bank` documents.
  - Verified the local skill file and the `run_agent_team.sh` entrypoint.
  - Updated durable memory to reflect the current session constraints and standby state.
- What remains open:
  - Wait for the next concrete user requirement and process it through the fixed workflow.

## 2026-04-04 Dataset Review

- Task: Inspect the CSRC punishment Excel file and determine which deep-learning project direction is feasible.
- Key decisions:
  - Treat the workbook as valid despite broken Excel dimension metadata and read it with a parser that does not trust the dimension header.
  - Distinguish event-level text retrieval tasks from party-level punishment prediction tasks.
  - Exclude `PunishmentMeasure` from predictive inputs because it leaks the punishment result directly.
- What changed:
  - Measured the dataset structure and label distribution from `证监会处罚信息表.xlsx`.
  - Confirmed that the dataset contains rich factual text in `Activity` and usable target labels in `PunishmentType`.
  - Narrowed the likely project direction toward knowledge-backed case retrieval plus punishment-type prediction instead of direct penalty-amount prediction.
- What remains open:
  - Confirm the final topic boundary with the user and then implement the project skeleton, modeling plan, and evaluation pipeline.

## 2026-04-04 RAG Direction

- Task: Clarify the intended course-project framing and delivery shape.
- Key decisions:
  - Use `RAG` as the central system pattern.
  - Treat `M` column (`Activity`) as the main evidence text for retrieval and downstream modeling.
  - Keep the option open for a lightweight frontend/backend web demo as the final presentation form.
- What changed:
  - Refined the target from a generic knowledge-backed system to a RAG-oriented case analysis and recommendation system.
  - Added end-to-end system design, baseline comparison, and retrieval strategy selection to the active planning scope.
- What remains open:
  - Write the formal requirement document and technical framework after the user confirms the proposed project boundary.

## 2026-04-04 Course Alignment

- Task: Parse the deep-learning course final-project brief and align the project scope with the formal grading rubric.
- Key decisions:
  - Target Track `B` as the primary lane because the project now centers on RAG and fine-tuning.
  - Treat the course constraints as hard requirements for later design: baseline comparison, hallucination mitigation, cost/latency discussion, reproducibility, and formal paper structure.
- What changed:
  - Converted the course brief into project constraints and deliverables.
  - Added the course timeline and required report sections into the active planning context.
- What remains open:
  - Turn the aligned constraints into a formal requirement document, technical framework, and milestone plan for the team.

## 2026-04-04 Project Framework Draft

- Task: Produce the first formal Chinese project framework document for the team.
- Key decisions:
  - Fix the project boundary around a RAG-centered case analysis system rather than an open-ended chatbot.
  - Use a dual-track architecture: event-level RAG retrieval/QA plus party-level punishment-type prediction.
  - Emphasize course-aligned validation, including baselines, ablations, hallucination control, and deployment discussion.
- What changed:
  - Added `docs/项目总体框架.md`.
  - Consolidated the project direction, boundary, technical route, retrieval strategy, model comparison plan, validation plan, and evidence basis into one reusable artifact.
- What remains open:
  - Convert the framework into concrete implementation artifacts: proposal draft, data scripts, baseline experiments, and demo skeleton.

## 2026-04-04 Execution Scaffold

- Task: Move from macro planning into executable project setup.
- Key decisions:
  - Reorder execution as data engineering -> retrieval baseline -> intent routing -> punishment prediction baseline -> RAG generation -> demo.
  - Treat the intent registry as the protocol layer between user questions and backend retrieval/generation paths.
  - Generate two canonical processed datasets first: event-level corpus for retrieval and party-level samples for prediction.
- What changed:
  - Added the first project engineering scaffold (`README.md`, `pyproject.toml`, `configs/`, `src/csrc_rag/`, `scripts/`).
  - Added `docs/执行路线与验证计划.md`.
  - Generated `data/processed/event_corpus.jsonl`, `data/processed/party_samples.jsonl`, and `data/processed/build_summary.json`.
  - Fixed the intent-routing priority for mixed recommendation-plus-law questions.
- What remains open:
  - Implement retrieval baselines, prediction baselines, and then the RAG answer chain on top of the generated data artifacts.

## 2026-04-04 Retrieval Baseline + Demo

- Task: Implement the first retrieval baseline, produce retrieval-specific docs, and make the project testable through a local frontend.
- Key decisions:
  - Start retrieval with a Chinese-friendly BM25 baseline instead of jumping directly to dense retrieval.
  - Build the knowledge base at chunk level (`summary`, `activity`, `law`) on top of event-level documents.
  - Keep the current answer layer template-based for now so retrieval quality can be tested before a real generation model is added.
- What changed:
  - Added retrieval modules, retrieval scripts, retrieval docs, and frontend test-question docs.
  - Generated `data/processed/event_chunks.jsonl`.
  - Added a minimal local web demo (`scripts/run_demo_server.py` + `web/`) and verified `/api/health`, `/api/query`, and the root page.
  - Verified a real query about insider trading and obtained relevant returned cases.
- What remains open:
  - Upgrade retrieval to Dense / Hybrid / Reranker.
  - Add a manual evaluation set for cross-case similarity, not only the current sanity benchmark.
  - Implement the punishment-type baseline and then connect retrieval to a real RAG generation model.

## 2026-04-04 Embedding + Training

- Task: Add dense retrieval and training-capable model paths to the repository.
- Key decisions:
  - Use a pluggable dense backend and start with `svd_tfidf` as the immediate executable dense retrieval baseline.
  - Keep a real `sentence-transformer` path ready instead of hard-coding the project to a single local approximation.
  - Support both a classical baseline and a transformer fine-tuning path for `PunishmentType`.
- What changed:
  - Added dense retrieval modules and hybrid retrieval query scripts.
  - Added a full training stack for punishment-type prediction, including data splitting, metrics, sklearn baseline, and Hugging Face fine-tuning.
  - Created `.venv311`, installed the transformer ecosystem, and verified a tiny transformer fine-tuning smoke test end to end.
- What remains open:
  - Switch from `svd_tfidf` to a real Chinese sentence-transformer model for dense retrieval experiments.
  - Run a formal Chinese transformer fine-tuning experiment such as `hfl/chinese-macbert-base`.
  - Add retrieval-quality comparisons across BM25, dense, hybrid, and reranker setups.

## 2026-04-06 Local Models + Frontend Upgrade

- Task: Replace the remaining heuristic/template layers with local-model components and upgrade the demo UI into a chat-style workspace.
- Key decisions:
  - Keep the intent-router model lightweight and fully local by using `TF-IDF + LogisticRegression` over domain-specific intent examples.
  - Keep the response-model path fully local and non-API by downloading `Qwen/Qwen2.5-0.5B-Instruct` into the repository and calling it through `transformers`.
  - Use the locally provided `LKWCoach-main.zip` only as a UI style reference, not as a code dependency.
  - Keep template response generation as a fallback path in case the local Hugging Face model fails to load.
- What changed:
  - Added the intent-classifier training config, training script, and runtime loader.
  - Added the local response-model abstraction and wired it into `RetrievalEngine`.
  - Upgraded the demo API to support `POST /api/query` plus richer metadata.
  - Rebuilt the frontend into a session-based chat interface with evidence cards and model/routing badges.
  - Downloaded `Qwen/Qwen2.5-0.5B-Instruct` into `artifacts/models/` and changed the config to point at the local path.
  - Added `docs/本地模型接入与前端改版说明.md` and updated `README.md`.
- What remains open:
  - Improve local response quality or swap to a stronger local model before final report/demo polishing.
  - Replace `svd_tfidf` with a real Chinese embedding model and add reranking.
  - Connect the punishment-type prediction model into the frontend as a separate comparison or supporting signal.

## 2026-04-06 Frontend Test + GitHub Repository Setup

- Task: Start the local frontend for immediate testing and publish the current project to GitHub as `Deeplearning-Rag-Test`.
- Key decisions:
  - Start the existing demo server in place instead of changing the frontend stack before testing.
  - Create the GitHub repository as `private` by default because visibility was not specified.
  - Exclude the local virtual environment, local Hugging Face model directory, caches, and secret files from version control before publishing.
- What changed:
  - Verified the demo frontend is reachable at `http://127.0.0.1:8000`.
  - Added `.gitignore` for repo-safe publishing boundaries.
  - Initialized git in the workspace, created the initial commit, and pushed `main` to `Mindse-Tt/Deeplearning-Rag-Test`.
  - Recorded that the repository now exists on GitHub and the workspace is no longer non-git.
- What remains open:
  - Decide whether the GitHub repository should remain private or be changed to public.
  - Reduce or externalize the oversized processed data files if the repository should be lighter to clone or easier to maintain.
