# Task Intake

When a new request is vague, translate it into this contract before execution:

- Goal
- Scope
- Inputs
- Constraints
- Unknowns
- Deliverable
- Verification

If some fields are missing, infer the minimum necessary assumptions and proceed.

## Recommended Starter Prompt

Use this when starting a new session in this project:

`$project-delivery-team 接管当前项目，先读取记忆，再按 需求 -> 计划 -> 执行 -> 验证 -> 记忆更新 的流程推进。`

## Memory Update Rule

After meaningful work, sync the result into:

- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

Update `systemPatterns.md` or `techContext.md` only if the stable architecture or tooling changes.
