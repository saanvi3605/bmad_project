"""
agents_impl/trigger_skill.py
────────────────────────────────────────────────────────────────────────────────
BMAD Pipeline Trigger Skill — CLI helper to trigger an Orchestrator agent run.

Skill definition : skills/pipeline_trigger.yaml
Agent            : agents_impl/orchestrator_agent.py

Usage:
    python agents_impl/trigger_skill.py "Build a task tracker with add/delete/list"
    python agents_impl/trigger_skill.py "Build a RAG service for PDFs" --mode fastapi_rag
    python agents_impl/trigger_skill.py --status
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
from filelock import FileLock, Timeout

CONTROL_FILE = Path(__file__).parent.parent / "control.yaml"
LOCK_FILE    = Path(__file__).parent.parent / "control.yaml.lock"
LOCK_TIMEOUT = 5


def _read() -> dict:
    defaults = {
        "run": False, "prompt": "", "project_mode": "auto",
        "status": "idle", "last_run": None, "last_result": None, "last_prompt": "",
    }
    try:
        data = yaml.safe_load(CONTROL_FILE.read_text(encoding="utf-8")) or {}
        return {**defaults, **data}
    except Exception:
        return defaults


def _write(data: dict) -> None:
    try:
        with FileLock(str(LOCK_FILE), timeout=LOCK_TIMEOUT):
            CONTROL_FILE.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
    except Timeout:
        print("⚠  Could not acquire file lock — try again.")
        sys.exit(1)


def show_status() -> None:
    data = _read()
    status      = data.get("status", "idle")
    last_run    = data.get("last_run") or "never"
    last_prompt = (data.get("last_prompt") or "")[:80]
    last_result = (data.get("last_result") or "")[:120]
    run_flag    = data.get("run", False)

    icon = {"idle": "[idle]", "running": "[running]", "complete": "[complete]", "failed": "[failed]"}.get(status, "[?]")

    sep = "-" * 54
    print(f"\n{sep}")
    print(f"  BMAD Orchestrator Agent — Status")
    print(sep)
    print(f"  Status     : {icon}")
    print(f"  run flag   : {run_flag}")
    print(f"  Last run   : {last_run}")
    if last_prompt:
        suffix = "..." if len(data.get("last_prompt", "")) > 80 else ""
        print(f"  Last prompt: {last_prompt}{suffix}")
    if last_result:
        suffix = "..." if len(data.get("last_result", "")) > 120 else ""
        print(f"  Last result: {last_result}{suffix}")
    print(f"{sep}\n")


def trigger(prompt: str, project_mode: str, sprint_item: str = "") -> None:
    """Write a run request to control.yaml (the event bus)."""
    data = _read()

    if data.get("run"):
        print("⚠  A run is already queued (run=true). Wait for it to start, then try again.")
        sys.exit(1)

    if data.get("status") == "running":
        print("⚠  Pipeline is already running. Wait for it to finish, then try again.")
        sys.exit(1)

    data["run"]          = True
    data["prompt"]       = prompt
    data["project_mode"] = project_mode
    data["sprint_item"]  = sprint_item
    data["last_result"]  = "Pending..."
    data["last_error"]   = None
    _write(data)

    suffix = "..." if len(prompt) > 80 else ""
    print(f"\n[OK] Triggered!")
    print(f"   Prompt      : {prompt[:80]}{suffix}")
    print(f"   Mode        : {project_mode}")
    if sprint_item:
        print(f"   Sprint item : {sprint_item}")
    print(f"   The orchestrator agent will pick this up within 10 seconds.")
    print(f"   Watch:  python agents_impl/orchestrator_agent.py")
    print(f"   Status: python agents_impl/trigger_skill.py --status\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger the BMAD Orchestrator agent from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agents_impl/trigger_skill.py "Build a recipe manager app"
  python agents_impl/trigger_skill.py "Build a RAG service for PDFs" --mode fastapi_rag
  python agents_impl/trigger_skill.py --status
        """,
    )
    parser.add_argument("prompt", nargs="?", help="The prompt to send to the BMAD pipeline.")
    parser.add_argument(
        "--mode", "-m",
        default="auto",
        choices=["auto", "streamlit_crud", "fastapi_rag"],
        help="Project mode (default: auto).",
    )
    parser.add_argument(
        "--item", "-i",
        default="",
        help="BMAD sprint item label, e.g. 'story-1-2' (optional).",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current orchestrator status and exit.",
    )

    args = parser.parse_args()

    if args.status or not args.prompt:
        show_status()
        if not args.status and not args.prompt:
            parser.print_help()
        return

    trigger(args.prompt.strip(), args.mode, args.item.strip())


if __name__ == "__main__":
    main()
