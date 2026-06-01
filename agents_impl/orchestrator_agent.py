"""
agents_impl/orchestrator_agent.py
────────────────────────────────────────────────────────────────────────────────
BMAD Choreography Polling Loop — Orchestrator Agent

Agent definition : agents/orchestrator.md
Skill definition : skills/pipeline_trigger.yaml

Implements the "agent loop" choreography pattern:

  Every POLL_INTERVAL seconds:
    1. Read control.yaml
    2. If run == true:
         a. Immediately flip run → false, status → running (prevents re-triggers)
         b. Run the full BMAD pipeline with the prompt
         c. Update status → complete / failed with results
    3. If run == false:
         Print a waiting heartbeat and sleep again

The control.yaml file is the "event bus".  Nobody calls this loop directly —
it watches shared state and self-triggers.  That's choreography.

Usage:
    python agents_impl/orchestrator_agent.py            # starts watching (Ctrl+C to stop)
    python agents_impl/orchestrator_agent.py --once     # run one check-and-execute cycle then exit
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml
from filelock import FileLock, Timeout

# ── Configuration ─────────────────────────────────────────────────────────────

POLL_INTERVAL   = 10          # seconds between "is run=true?" checks
CONTROL_FILE    = Path(__file__).parent.parent / "control.yaml"
LOCK_FILE       = Path(__file__).parent.parent / "control.yaml.lock"
LOCK_TIMEOUT    = 5           # seconds to wait for the file lock

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ts() -> str:
    """Short timestamp for console output."""
    return datetime.now().strftime("%H:%M:%S")


def _read_control() -> dict:
    """Read control.yaml; return defaults if missing or corrupt."""
    defaults = {
        "run": False,
        "prompt": "",
        "project_mode": "auto",
        "sprint_item": "",
        "status": "idle",
        "iteration": 0,
        "last_run": None,
        "last_result": None,
        "last_error": None,
        "last_prompt": "",
    }
    try:
        data = yaml.safe_load(CONTROL_FILE.read_text(encoding="utf-8")) or {}
        return {**defaults, **data}
    except Exception:
        return defaults


def _write_control(data: dict) -> None:
    """Atomically write control.yaml under a file lock."""
    try:
        with FileLock(str(LOCK_FILE), timeout=LOCK_TIMEOUT):
            CONTROL_FILE.write_text(
                yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
    except Timeout:
        print(f"[{_ts()}] ⚠  Could not acquire file lock — skipping write.")


def _set_fields(**kwargs) -> None:
    """Read → update fields → write control.yaml under lock."""
    data = _read_control()
    data.update(kwargs)
    _write_control(data)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(prompt: str, project_mode: str) -> tuple[bool, str]:
    """
    Run the full BMAD pipeline synchronously.

    Returns
    -------
    (success: bool, result_message: str)
    """
    try:
        from main import run as bmad_run
        bmad_run(prompt, project_mode=project_mode)
        return True, f"Pipeline completed. Output: outputs/generated_app.py"
    except Exception as exc:
        tb = traceback.format_exc()
        short = str(exc)[:300]
        print(f"\n[{_ts()}] ❌ Pipeline error:\n{tb}\n")
        return False, f"Error: {short}"


# ── Main agent function ───────────────────────────────────────────────────────

def orchestrator_agent(once: bool = False) -> None:
    """
    Main entry point for the Orchestrator agent.
    Runs forever unless once=True (single check-and-execute cycle).
    """
    print(f"\n{'='*62}")
    print(f"  BMAD Orchestrator Agent — Choreography Polling Pattern")
    print(f"{'='*62}")
    print(f"  Watching: {CONTROL_FILE}")
    print(f"  Interval: {POLL_INTERVAL}s   |   Mode: {'single-shot' if once else 'continuous'}")
    print(f"  Trigger:  python agents_impl/trigger_skill.py \"<your prompt>\"")
    print(f"  Stop:     Ctrl+C")
    print(f"{'='*62}\n")

    try:
        while True:
            _tick()
            if once:
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Orchestrator agent stopped.")
        _set_fields(status="idle")


def _tick() -> None:
    """One polling cycle: read, decide, act."""
    data = _read_control()

    status    = data.get("status", "idle")
    last_run  = data.get("last_run")
    last_ago  = ""
    if last_run:
        try:
            elapsed = (datetime.now() - datetime.strptime(last_run, "%Y-%m-%d %H:%M:%S")).seconds
            last_ago = f" | Last run: {elapsed//60}m {elapsed%60}s ago"
        except Exception:
            last_ago = f" | Last run: {last_run}"

    if not data.get("run"):
        iteration = data.get("iteration", 0)
        print(f"[{_ts()}] Iteration {iteration} | Status: {status}{last_ago} | Waiting for run=true …")
        return

    # ── Run triggered ────────────────────────────────────────────────────────
    prompt       = (data.get("prompt") or "").strip()
    project_mode = data.get("project_mode") or "auto"
    sprint_item  = (data.get("sprint_item") or "").strip()
    iteration    = int(data.get("iteration") or 0)

    if not prompt:
        print(f"[{_ts()}] ⚠  run=true but prompt is empty — resetting.")
        _set_fields(run=False, status="failed",
                    last_run=_now(), last_result="Error: prompt was empty",
                    last_error="prompt was empty")
        return

    print(f"\n[{_ts()}] {'='*58}")
    print(f"[{_ts()}] 🚀  RUN TRIGGERED  (iteration {iteration + 1})")
    print(f"[{_ts()}]     Prompt:  {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"[{_ts()}]     Mode:    {project_mode}")
    if sprint_item:
        print(f"[{_ts()}]     Sprint:  {sprint_item}")
    print(f"[{_ts()}] {'='*58}")

    _set_fields(
        run=False,
        status="running",
        last_prompt=prompt,
        last_run=_now(),
        last_result="Running...",
        last_error=None,
    )
    print(f"[{_ts()}] ✔  run flipped to false — control.yaml updated.")
    print(f"[{_ts()}] ⏳  Pipeline starting …\n")

    t0 = time.time()
    success, result = _run_pipeline(prompt, project_mode)
    elapsed = time.time() - t0

    final_status = "complete" if success else "failed"
    icon         = "✅" if success else "❌"

    _set_fields(
        status=final_status,
        iteration=iteration + 1 if success else iteration,
        last_run=_now(),
        last_result=result,
        last_error=None if success else result,
    )

    print(f"\n[{_ts()}] {icon}  Pipeline {final_status} in {elapsed:.1f}s")
    print(f"[{_ts()}]     Result:  {result[:100]}")
    print(f"[{_ts()}] Watching for next trigger …\n")


if __name__ == "__main__":
    once = "--once" in sys.argv
    orchestrator_agent(once=once)
