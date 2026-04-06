#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from agents import Agent, Runner, SQLiteSession
except ImportError as exc:  # pragma: no cover
    print(
        "openai-agents is not available. Install it in Python 3.11 first, or use "
        ".codex-agent-venv311/bin/python to run this script.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


DEFAULT_MODEL = os.environ.get("OPENAI_AGENTS_MODEL", "gpt-5.2")


@dataclass
class ProjectFiles:
    project_root: Path
    project_name: str
    user_profile: str
    session_notes: str
    projectbrief: str
    product_context: str
    system_patterns: str
    tech_context: str
    active_context: str
    progress: str


def read_text(path: Path) -> str:
    if not path.exists():
        return f"[Missing] {path.name}"
    return path.read_text(encoding="utf-8").strip()


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip().lower())
    value = re.sub(r"[^a-z0-9\-_]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reusable multi-agent planning workflow with the OpenAI Agents SDK."
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task description. You can also provide --task-file instead.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Read the task description from a UTF-8 text or markdown file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root that contains project-memory/ and memory-bank/.",
    )
    parser.add_argument(
        "--project-name",
        help="Logical project name. Defaults to the project-root directory name.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name. Default: %(default)s",
    )
    parser.add_argument(
        "--session-id",
        help="Persistent session id. Defaults to agent-team::<project-name>.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="Path to the SQLite session DB. Defaults to project-memory/openai_agents_sessions.sqlite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the full run report to this file. Defaults to project-memory/agent-runs/<timestamp>_<slug>.md.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Do not write files. Print the run report to stdout only.",
    )
    return parser.parse_args()


def load_task(args: argparse.Namespace) -> str:
    if args.task_file:
        return args.task_file.read_text(encoding="utf-8").strip()
    task = " ".join(args.task).strip()
    if not task:
        raise SystemExit("Provide a task string or use --task-file.")
    return task


def load_project_files(project_root: Path, project_name: str) -> ProjectFiles:
    return ProjectFiles(
        project_root=project_root,
        project_name=project_name,
        user_profile=read_text(project_root / "project-memory" / "user-profile.md"),
        session_notes=read_text(project_root / "project-memory" / "session-notes.md"),
        projectbrief=read_text(project_root / "memory-bank" / "projectbrief.md"),
        product_context=read_text(project_root / "memory-bank" / "productContext.md"),
        system_patterns=read_text(project_root / "memory-bank" / "systemPatterns.md"),
        tech_context=read_text(project_root / "memory-bank" / "techContext.md"),
        active_context=read_text(project_root / "memory-bank" / "activeContext.md"),
        progress=read_text(project_root / "memory-bank" / "progress.md"),
    )


def build_context_bundle(files: ProjectFiles) -> str:
    sections = [
        ("Project Name", files.project_name),
        ("User Profile", files.user_profile),
        ("Recent Session Notes", files.session_notes),
        ("Project Brief", files.projectbrief),
        ("Product Context", files.product_context),
        ("System Patterns", files.system_patterns),
        ("Tech Context", files.tech_context),
        ("Active Context", files.active_context),
        ("Progress", files.progress),
    ]
    return "\n\n".join(f"## {name}\n{content}" for name, content in sections)


def run_agent(agent: Agent, prompt: str, session: SQLiteSession) -> str:
    result = Runner.run_sync(agent, prompt, session=session)
    return str(result.final_output).strip()


def append_session_note(project_root: Path, task: str, report_path: Path, model: str) -> None:
    session_notes = project_root / "project-memory" / "session-notes.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = (
        f"\n\n## {timestamp}\n\n"
        f"- Task: {task}\n"
        f"- Key decisions:\n"
        f"  - Generated a structured agent-team run artifact.\n"
        f"  - Used model `{model}` with persistent SQLite session storage.\n"
        f"- What changed:\n"
        f"  - Wrote run report to `{report_path.relative_to(project_root)}`.\n"
        f"- What remains open:\n"
        f"  - Execute the resulting Codex brief or refine the plan if needed.\n"
    )
    session_notes.parent.mkdir(parents=True, exist_ok=True)
    with session_notes.open("a", encoding="utf-8") as handle:
        handle.write(note)


def main() -> int:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required to run the OpenAI Agents SDK workflow.", file=sys.stderr)
        return 2

    task = load_task(args)
    project_root = args.project_root.resolve()
    project_name = args.project_name or project_root.name
    files = load_project_files(project_root, project_name)
    context_bundle = build_context_bundle(files)

    session_id = args.session_id or f"agent-team::{slugify(project_name)}"
    db_path = args.db_path or project_root / "project-memory" / "openai_agents_sessions.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    session = SQLiteSession(session_id=session_id, db_path=db_path)

    requirements_agent = Agent(
        name="Requirements Lead",
        model=args.model,
        instructions=(
            "You are the requirements lead. Work in Chinese. Convert the incoming task into a compact, "
            "execution-ready markdown contract. Use exactly these sections: "
            "`# 需求合同`, `## 目标`, `## 范围`, `## 输入`, `## 约束`, `## 假设`, "
            "`## 交付物`, `## 验证标准`. Be concrete and avoid fluff."
        ),
    )
    planner_agent = Agent(
        name="Planning Lead",
        model=args.model,
        instructions=(
            "You are the planning lead. Work in Chinese. Produce an actionable markdown plan. "
            "Use exactly these sections: `# 执行计划`, `## 阶段拆解`, `## 关键依赖`, "
            "`## 风险`, `## 验证路径`, `## 最小可交付切片`. Keep steps executable."
        ),
    )
    executor_brief_agent = Agent(
        name="Execution Lead",
        model=args.model,
        instructions=(
            "You are preparing a Codex-ready execution brief, not writing code directly. Work in Chinese. "
            "Use exactly these sections: `# Codex 执行简报`, `## 目标`, `## 建议修改`, "
            "`## 建议命令`, `## 关键文件`, `## 风险控制`, `## 完成定义`. "
            "Be specific enough that Codex can act on it immediately."
        ),
    )
    verifier_agent = Agent(
        name="Verification Lead",
        model=args.model,
        instructions=(
            "You are the verification lead. Work in Chinese. Critique the proposed delivery flow before execution. "
            "Use exactly these sections: `# 验证清单`, `## 必测项`, `## 失败信号`, "
            "`## 回滚点`, `## 是否可以开始执行`. If the plan is weak, say so plainly."
        ),
    )
    memory_scribe_agent = Agent(
        name="Memory Scribe",
        model=args.model,
        instructions=(
            "You write distilled project memory, not raw transcript. Work in Chinese. "
            "Use exactly these sections: `# 会话摘要`, `## 任务`, `## 关键决策`, "
            "`## 产出`, `## 下一步`. Keep it short and durable."
        ),
    )

    common_header = (
        f"Project root: {project_root}\n"
        f"Project name: {project_name}\n"
        f"Current task: {task}\n\n"
        f"## Project context\n{context_bundle}\n"
    )

    try:
        requirements = run_agent(
            requirements_agent,
            f"{common_header}\n\nTurn the current task into an execution contract.",
            session,
        )
        plan = run_agent(
            planner_agent,
            (
                f"{common_header}\n\n"
                f"Use this requirements contract as the planning input:\n\n{requirements}"
            ),
            session,
        )
        execution_brief = run_agent(
            executor_brief_agent,
            (
                f"{common_header}\n\n"
                f"Requirements contract:\n\n{requirements}\n\n"
                f"Execution plan:\n\n{plan}\n\n"
                "Prepare a Codex-facing execution brief."
            ),
            session,
        )
        verification = run_agent(
            verifier_agent,
            (
                f"{common_header}\n\n"
                f"Requirements contract:\n\n{requirements}\n\n"
                f"Execution plan:\n\n{plan}\n\n"
                f"Codex execution brief:\n\n{execution_brief}"
            ),
            session,
        )
        session_summary = run_agent(
            memory_scribe_agent,
            (
                f"{common_header}\n\n"
                f"Requirements contract:\n\n{requirements}\n\n"
                f"Execution plan:\n\n{plan}\n\n"
                f"Codex execution brief:\n\n{execution_brief}\n\n"
                f"Verification:\n\n{verification}\n\n"
                "Write a durable session summary for project memory."
            ),
            session,
        )
    except Exception as exc:
        message = str(exc)
        if "Incorrect API key provided" in message or "invalid_api_key" in message:
            print("OPENAI_API_KEY is set but invalid. Update it and rerun.", file=sys.stderr)
            return 1
        print(f"Agent workflow failed: {message}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = project_root / "project-memory" / "agent-runs" / f"{timestamp}_{slugify(task)[:48]}.md"
    output_path = (args.output or default_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = (
        "# Agent Team Run\n\n"
        f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Project: {project_name}\n"
        f"- Model: `{args.model}`\n"
        f"- Session ID: `{session_id}`\n"
        f"- Session DB: `{db_path}`\n\n"
        "## Input Task\n\n"
        f"{task}\n\n"
        "## Requirements Lead\n\n"
        f"{requirements}\n\n"
        "## Planning Lead\n\n"
        f"{plan}\n\n"
        "## Execution Lead\n\n"
        f"{execution_brief}\n\n"
        "## Verification Lead\n\n"
        f"{verification}\n\n"
        "## Memory Scribe\n\n"
        f"{session_summary}\n"
    )

    if args.print_only:
        print(report)
        return 0

    output_path.write_text(report, encoding="utf-8")
    append_session_note(project_root, task, output_path, args.model)

    print(f"Run report written to: {output_path}")
    print(f"Session DB: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
