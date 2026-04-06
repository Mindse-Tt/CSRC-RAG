# User Profile

## Identity

- Primary working mode: terminal-first, uses Codex as the main execution agent.
- Main need: a reusable project operating framework rather than one-off chat answers.
- Core expectation: AI should help drive the full loop from requirement intake to execution and verification.

## Communication Preferences

- Use Chinese by default.
- Be direct, structured, and implementation-oriented.
- State assumptions explicitly instead of hiding them.
- Prefer doing the work over long abstract planning.

## Workflow Preferences

- Prefer stable reusable processes: `requirements -> plan -> execution -> verification -> memory update`.
- Do not require the user to repeatedly explain who they are, their style, or the same project background.
- Store durable preferences and project decisions in local files so future sessions can reload them.
- Keep solutions easy to port into other projects.

## Tooling Preferences

- Codex is the default worker.
- MCP and local markdown memory are preferred for project continuity.
- External multi-agent frameworks are optional enhancements, not the default path.

## Stable Constraints

- Update project memory after meaningful work.
- Keep important artifacts local and inspectable.
- Avoid hidden magic. The workflow should remain understandable and reproducible.
