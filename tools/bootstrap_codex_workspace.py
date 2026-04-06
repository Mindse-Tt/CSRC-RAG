#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


TEMPLATE_FILES = [
    ".agents/skills/project-delivery-team/SKILL.md",
    "project-memory/user-profile.md",
    "project-memory/task-intake.md",
    "project-memory/session-notes.md",
    "memory-bank/.byterules",
    "memory-bank/projectbrief.md",
    "memory-bank/productContext.md",
    "memory-bank/systemPatterns.md",
    "memory-bank/techContext.md",
    "memory-bank/activeContext.md",
    "memory-bank/progress.md",
    "start_codex_team.sh",
    "run_agent_team.sh",
    "tools/agent_team_runner.py",
    "tools/bootstrap_codex_workspace.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the Codex workflow and memory scaffold into another project."
    )
    parser.add_argument("target_dir", type=Path, help="Target project directory.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent / "template",
        help="Template source root. Defaults to this project root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--install-openai-agents",
        action="store_true",
        help="Create a Python 3.11 virtual environment in the target and install openai-agents.",
    )
    return parser.parse_args()


def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def copy_file(source_root: Path, target_root: Path, relative_path: str, force: bool) -> str:
    source = source_root / relative_path
    target = target_root / relative_path

    if not source.exists():
        return f"missing template: {relative_path}"

    if target.exists() and not force:
        return f"skipped existing: {relative_path}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    if target.suffix == ".sh":
        ensure_executable(target)

    return f"copied: {relative_path}"


def install_openai_agents(target_root: Path) -> None:
    venv_path = target_root / ".codex-agent-venv311"
    subprocess.run(["python3.11", "-m", "venv", str(venv_path)], check=True)
    subprocess.run([str(venv_path / "bin" / "pip"), "install", "openai-agents"], check=True)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for relative_path in TEMPLATE_FILES:
        print(copy_file(source_root, target_root, relative_path, args.force))

    if args.install_openai_agents:
        if shutil.which("python3.11") is None:
            print("python3.11 is required for --install-openai-agents.", file=sys.stderr)
            return 2
        install_openai_agents(target_root)
        print("installed: .codex-agent-venv311 with openai-agents")

    print(f"Bootstrap complete: {target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
