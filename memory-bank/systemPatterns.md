# System Patterns

## Operating Model

This workspace uses a layered pattern instead of a single monolithic agent framework.

## Layer 1: Codex Skill

`project-delivery-team` is the main workflow controller. It standardizes:

- requirement intake
- planning
- execution
- verification
- memory updates

## Layer 2: Project Memory

The memory bank stores durable project state in markdown files. This gives both AI and the user an inspectable, editable memory surface.

## Layer 3: MCP Access

`memory-bank` is registered as an MCP server in Codex. It provides retrieval and document update capabilities for the six standard memory files.

## Layer 4: Optional External Agent Stack

External frameworks are optional and should not replace the default Codex workflow:

- `openai-agents` for programmable multi-agent orchestration and session storage when an API key is available
- `MetaGPT` as a reference project for software-company-style role decomposition

## Standard Task Loop

1. Read current memory.
2. Convert the user request into a compact requirement contract.
3. Plan execution.
4. Execute the smallest useful slice.
5. Verify.
6. Persist durable context updates.

## Memory Pattern

Do not store raw conversation transcripts as the primary memory artifact. Store distilled facts:

- stable preferences
- current focus
- technical decisions
- completed work
- next steps

This keeps memory searchable and reusable.
