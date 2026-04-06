---
name: project-delivery-team
description: Use when the user wants Codex to handle work in this workspace end-to-end with a stable delivery workflow: clarify requirements, decompose the task, execute, verify, update project memory, and leave reusable artifacts for future sessions or other projects.
---

# Project Delivery Team

Use this skill for real project work, not for one-off trivia.

## Load Context First

Before planning or editing, read only the files that matter:

- `project-memory/user-profile.md`
- `project-memory/task-intake.md`
- `project-memory/session-notes.md` when recent session context matters
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`
- `memory-bank/techContext.md`
- `memory-bank/systemPatterns.md` when architecture or workflow design may change

If the `memory-bank` MCP server is available, prefer it for search or retrieval. If it is not available, read the markdown files directly.

## Standard Workflow

Run tasks with this fixed order:

1. Requirement contract
Write a compact contract with `Goal`, `Inputs`, `Constraints`, `Assumptions`, `Deliverable`, and `Verification`.

2. Plan
Break the task into executable phases. Prefer concrete work packages over abstract brainstorming.

3. Execute
Make changes, run commands, or produce artifacts. Do not stay in analysis mode longer than necessary.

4. Verify
Run the smallest trustworthy validation available. If validation is blocked, say exactly what is missing.

5. Update memory
After meaningful work:
- update `memory-bank/activeContext.md` with the current focus and next steps
- update `memory-bank/progress.md` with what changed and how it was verified
- append a concise session summary to `project-memory/session-notes.md` when the work changed project understanding or task status
- update `memory-bank/systemPatterns.md` or `memory-bank/techContext.md` if architecture, tooling, or workflow changed
- update `project-memory/user-profile.md` only when the user states a stable preference that should persist

## Working Rules

- Do not ask the user to restate context that already exists in project memory.
- If the request is underspecified, make bounded assumptions and surface them explicitly.
- Default to Chinese for user-facing communication in this workspace.
- Optimize for reusable outputs that can be copied into other projects.
- When reviewing, present findings and risks before summary.
- Only propose or use subagents when the user explicitly asks for parallel or delegated agent work.

## Expected Output Shape

For substantial tasks, keep the work legible:

- Short requirement contract
- Short execution plan
- Actual implementation or commands
- Verification result
- Memory updates
