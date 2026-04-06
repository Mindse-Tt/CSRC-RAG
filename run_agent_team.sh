#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [ -x ".codex-agent-venv311/bin/python" ]; then
  py="./.codex-agent-venv311/bin/python"
elif [ -x "$HOME/.codex/project-ops/.codex-agent-venv311/bin/python" ]; then
  py="$HOME/.codex/project-ops/.codex-agent-venv311/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  py="python3.11"
else
  echo "Python 3.11 is required to run agent_team_runner.py." >&2
  exit 1
fi

if ! "$py" -c 'import agents' >/dev/null 2>&1; then
  echo "openai-agents is not installed for $py. Create .codex-agent-venv311 or install the package first." >&2
  exit 2
fi

exec "$py" tools/agent_team_runner.py "$@"
