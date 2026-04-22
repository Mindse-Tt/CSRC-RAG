# Tech Context

## Host Environment

- OS family: macOS
- Shell: `zsh`
- Codex CLI: `0.117.0`
- Node.js: `v25.8.1`
- npm: `11.11.0`
- Python default: `3.9.6`
- Additional Python: `3.11.15`

## Installed Components

- Memory Bank MCP source:
  `vendor/memory-bank-mcp`
- Memory Bank MCP runtime entry:
  `vendor/memory-bank-mcp/dist/index.js`
- OpenAI Agents SDK virtual environment:
  `.codex-agent-venv311`
- Older Python 3.9 virtual environment kept for compatibility testing:
  `.codex-agent-venv`
- OpenAI Agents team runner:
  `tools/agent_team_runner.py`
- Workspace bootstrap script:
  `tools/bootstrap_codex_workspace.py`

## Codex Integration

- Global MCP server name: `memory-bank`
- Launch command:
  `node <memory-bank-mcp>/vendor/memory-bank-mcp/dist/index.js`
- Memory Bank MCP was patched locally to auto-detect an existing `memory-bank/` folder from the current project directory.

## Important Constraints

- The global Codex config currently has `disable_response_storage = true`.
- Use a local launcher or a `-c disable_response_storage=false` override when session persistence is desired.
- `openai-agents` requires an `OPENAI_API_KEY` to run real workflows.
- The current shell environment exposes an `OPENAI_API_KEY`, but the live API test returned `401 invalid_api_key` on 2026-03-31, so the key must be replaced before real agent runs will work.

## Operational Guidance

- Use Codex skill + memory bank as the default path.
- Use `openai-agents` only when programmable orchestration is needed.
- Use `zsh start_codex_team.sh` and `zsh run_agent_team.sh` in this workspace because direct `./script.sh` execution is blocked in the current Downloads location.
- Keep environment-sensitive paths documented here after tooling changes.
