"""
agents_impl/executor_agent.py
────────────────────────────────────────────────────────────────────────────────
Executor agent — pure subprocess, no LLM call.

Writes the generated code to disk and attempts to launch it as a child process.
Waits for the configured startup period, then checks whether the process is
still alive to determine success vs crash.

Changes from original:
  - Output path read from core.llm_factory.get_executor_config() instead of
    hardcoded "generated_app.py" in the current working directory.
  - startup_wait_seconds read from config (default 8s) instead of hardcoded 5s.
  - Optional session archiving: when session.archive_outputs is True, the
    generated file is copied into outputs/sessions/{session_id}/ after writing.
  - All subprocess logic, process.poll() pattern, error handling, and state
    field mutations preserved verbatim from original.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from typing import Any

from core.llm_factory import get_executor_config, get_session_config


def _find_free_port(preferred: int = 8502, scan_limit: int = 20) -> int:
    """Return preferred port if free, otherwise the next available one."""
    for port in range(preferred, preferred + scan_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred  # fallback — Streamlit will show its own port error


def executor_agent(state: dict[str, Any]) -> dict[str, Any]:
    project_mode = state.get("project_mode", "streamlit_crud")

    if project_mode == "fastapi_rag":
        return _execute_rag(state)
    return _execute_streamlit(state)


def _execute_streamlit(state: dict[str, Any]) -> dict[str, Any]:
    """Original Streamlit executor — unchanged."""
    code = state.get("code", "")
    code = code.replace("```python", "").replace("```", "").strip()

    exec_cfg = get_executor_config()
    output_path: str = exec_cfg.get("output_file", "generated_app.py")
    startup_wait: int = int(exec_cfg.get("startup_wait_seconds", 8))

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[Executor] Code saved to {output_path}")

    session_cfg = get_session_config()
    if session_cfg.get("archive_outputs", False):
        session_id = state.get("session_id")
        if session_id:
            archive_dir = os.path.join(session_cfg.get("output_dir", "outputs/sessions/"), session_id)
            os.makedirs(archive_dir, exist_ok=True)
            shutil.copy2(output_path, os.path.join(archive_dir, "generated_app.py"))
            state["output_dir"] = archive_dir

    try:
        import sys
        port = _find_free_port(8502)
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", output_path,
             "--server.headless", "true", "--server.port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(startup_wait)

        if process.poll() is None:
            state["execution_result"] = f"App started successfully on port {port}."
            state["execution_error"] = ""
            state["runtime_error"] = ""
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        else:
            _, stderr = process.communicate()
            error_text = stderr.strip() if stderr else "Process exited before startup."
            state["execution_result"] = "Process exited before startup completed."
            state["execution_error"] = error_text
            state["runtime_error"] = error_text
            state["runtime_fix_attempts"] = (state.get("runtime_fix_attempts") or 0) + 1
            print(f"[Executor] Crash detected (attempt {state['runtime_fix_attempts']}). "
                  f"Routing to Developer for runtime fix.")
    except Exception as e:
        error_text = str(e)
        state["execution_result"] = ""
        state["execution_error"] = error_text
        state["runtime_error"] = error_text
        state["runtime_fix_attempts"] = (state.get("runtime_fix_attempts") or 0) + 1
        print(f"[Executor] Exception (attempt {state['runtime_fix_attempts']}): {error_text}")

    return state


def _execute_rag(state: dict[str, Any]) -> dict[str, Any]:
    """
    RAG mode executor — saves all generated files then launches with uvicorn.

    File layout:
      outputs/rag/{session_id}/main.py
      outputs/rag/{session_id}/requirements.txt
      outputs/rag/{session_id}/docker-compose.yml
      outputs/rag/{session_id}/.env.example
    """
    import sys

    session_id = state.get("session_id", "rag_output")
    rag_dir = os.path.join("outputs", "rag", session_id)
    os.makedirs(rag_dir, exist_ok=True)

    # ── Save main.py ──────────────────────────────────────────────────────
    main_code = state.get("code", "")
    main_path = os.path.join(rag_dir, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_code)
    print(f"[Executor-RAG] main.py saved → {main_path}")

    # ── Save extra files (requirements.txt, docker-compose.yml, etc.) ─────
    extra_files: dict[str, str] = state.get("extra_files") or {}
    for filename, content in extra_files.items():
        file_path = os.path.join(rag_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[Executor-RAG] {filename} saved → {file_path}")

    state["output_dir"] = rag_dir

    # Also write main path to the canonical output location for download
    canonical = os.path.join("outputs", "generated_app.py")
    try:
        shutil.copy2(main_path, canonical)
    except Exception:
        pass

    # ── Launch uvicorn ────────────────────────────────────────────────────
    startup_wait = int(get_executor_config().get("startup_wait_seconds", 8))
    port = _find_free_port(8000)

    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app",
             "--host", "0.0.0.0", "--port", str(port), "--reload"],
            cwd=rag_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(startup_wait)

        if process.poll() is None:
            state["execution_result"] = (
                f"RAG API started on http://localhost:{port} | "
                f"Docs: http://localhost:{port}/docs | "
                f"Files: {rag_dir}"
            )
            state["execution_error"] = ""
            state["runtime_error"] = ""
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"[Executor-RAG] uvicorn started on port {port}.")
        else:
            _, stderr = process.communicate()
            error_text = stderr.strip() if stderr else "uvicorn exited before startup."
            state["execution_result"] = "uvicorn exited before startup completed."
            state["execution_error"] = error_text
            state["runtime_error"] = error_text
            state["runtime_fix_attempts"] = (state.get("runtime_fix_attempts") or 0) + 1
            print(f"[Executor-RAG] Crash (attempt {state['runtime_fix_attempts']}): {error_text[:200]}")

    except Exception as e:
        error_text = str(e)
        state["execution_result"] = ""
        state["execution_error"] = error_text
        state["runtime_error"] = error_text
        state["runtime_fix_attempts"] = (state.get("runtime_fix_attempts") or 0) + 1
        print(f"[Executor-RAG] Exception: {error_text}")

    return state


# ---------------------------------------------------------------------------
# Routing function — consumed by orchestration/graph.py
# ---------------------------------------------------------------------------


def should_runtime_fix(state: dict[str, Any]) -> str:
    """
    Called by LangGraph after every Executor run.

    Returns
    -------
    "success"      — app started cleanly; proceed to TestWriter
    "runtime_fix"  — crash detected, attempt limit not yet reached; send to Developer
    "max_attempts" — crash but attempt limit exceeded; proceed to TestWriter anyway
    """
    runtime_error = state.get("runtime_error") or ""
    if not runtime_error:
        return "success"

    from core.llm_factory import get_config
    cfg = get_config()
    max_retries = int(cfg.get("retry_limits", {}).get("executor_max_retries", 2))
    attempts = state.get("runtime_fix_attempts") or 0

    if attempts <= max_retries:
        return "runtime_fix"
    print(f"[Executor] Max runtime fix attempts ({max_retries}) reached — proceeding to TestWriter.")
    return "max_attempts"
