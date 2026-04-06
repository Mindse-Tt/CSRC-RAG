#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
  prompt="\$project-delivery-team $*"
else
  prompt='$project-delivery-team 接管当前项目。先读取 project-memory 和 memory-bank，再按 需求 -> 计划 -> 执行 -> 验证 -> 记忆更新 的流程待命。'
fi

exec codex -c disable_response_storage=false "$prompt"
