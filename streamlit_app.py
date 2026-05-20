"""
streamlit_app.py
────────────────────────────────────────────────────────────────────────────────
BMAD Pipeline — Streamlit Frontend

Runs the LangGraph pipeline in a background thread so the UI never blocks.
A queue is used for thread-safe communication between the worker thread and
the Streamlit main thread.

Architecture:
  - Main thread: Streamlit UI rendering, polling for updates
  - Worker thread: app.invoke() → final state pushed to queue on completion
  - Queue: single-item queue; worker posts (final_state, error) tuple when done

Features:
  - Non-blocking UI with live status indicator
  - Tabbed results: Generated Code | Test File | Execution | Agent Log | Observability
  - Download buttons for generated_app.py and test_generated_app.py
  - Pipeline reset button
  - Error display
  - Per-agent token/cost summary table
  - Pipeline summary cards

Requirements:
  - app.invoke() MUST NOT run on the Streamlit main thread
  - generated_app.py and test_generated_app.py must be written to disk
    (handled by executor_agent and test_writer_agent respectively)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pathlib
import queue
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMAD Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Port helper (mirrors executor_agent logic, uses 8503+ to avoid clash) ────

def _find_free_port(preferred: int = 8503, scan_limit: int = 20) -> int:
    """Return preferred port if free, otherwise the next available one."""
    for port in range(preferred, preferred + scan_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


# ── Pipeline History DB ───────────────────────────────────────────────────────

_HISTORY_DB = str(pathlib.Path(__file__).parent / "pipeline_history.db")


def init_history_db() -> None:
    """Create the runs table if it doesn't exist. Called once at startup."""
    try:
        conn = sqlite3.connect(_HISTORY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT    NOT NULL,
                prompt           TEXT    NOT NULL,
                status           TEXT,
                functional_spec  TEXT,
                technical_design TEXT,
                code             TEXT,
                test_cases       TEXT,
                review_feedback  TEXT,
                review_approved  INTEGER,
                validation_passed INTEGER,
                execution_result TEXT,
                execution_error  TEXT,
                test_file_path   TEXT,
                total_tokens     INTEGER,
                total_cost_usd   REAL,
                total_latency_ms REAL,
                agents_called    TEXT,
                created_at       TEXT    DEFAULT (datetime('now'))
            )
        """)
        # Migrate existing DBs that pre-date the refined_prompt column
        try:
            conn.execute("ALTER TABLE runs ADD COLUMN refined_prompt TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        conn.close()
    except Exception:
        pass


def save_run_to_history(
    original_prompt: str,
    final_state: Optional[dict],
    error: Optional[str],
) -> None:
    """Persist one pipeline run to pipeline_history.db. Never raises."""
    try:
        from core.observability import get_pipeline_summary
        summary = get_pipeline_summary(final_state) if final_state else None
        status = "complete" if final_state and not error else "failed"
        fs = final_state or {}
        conn = sqlite3.connect(_HISTORY_DB)
        conn.execute("""
            INSERT INTO runs (
                session_id, prompt, status, refined_prompt,
                functional_spec, technical_design,
                code, test_cases, review_feedback, review_approved, validation_passed,
                execution_result, execution_error, test_file_path,
                total_tokens, total_cost_usd, total_latency_ms, agents_called
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fs.get("session_id", ""),
            original_prompt,
            status,
            fs.get("refined_prompt"),
            fs.get("functional_spec"),
            fs.get("technical_design"),
            fs.get("code"),
            fs.get("test_cases"),
            fs.get("review_feedback"),
            int(bool(fs.get("review_approved"))),
            int(bool(fs.get("validation_passed"))),
            fs.get("execution_result"),
            fs.get("execution_error") or error,
            fs.get("test_file"),
            summary.total_tokens if summary else 0,
            summary.total_cost_usd if summary else 0.0,
            summary.total_latency_ms if summary else 0.0,
            ",".join(summary.agents_called) if summary else "",
        ))
        conn.commit()
        conn.close()
        load_history.clear()  # bust cache so History tab shows the new row
    except Exception:
        pass


@st.cache_data(ttl=30)
def load_history() -> list[dict]:
    """Return all runs from history DB, newest first. Cached for 30 s."""
    try:
        conn = sqlite3.connect(_HISTORY_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ── Lazy imports (avoid triggering LLM build at startup) ─────────────────────


@st.cache_resource
def _load_app():
    """Load the compiled LangGraph app once, cached across reruns."""
    from orchestration.graph import app as langgraph_app
    return langgraph_app


@st.cache_resource
def _load_pipeline_rules() -> str:
    """Load pipeline rules suffix from config/pipeline_rules.yaml."""
    try:
        import yaml, pathlib
        cfg_path = pathlib.Path(__file__).parent / "config" / "pipeline_rules.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("prompt_suffix", "")
    except Exception:
        return ""


# ── Session state initialisation ─────────────────────────────────────────────

init_history_db()  # idempotent — runs on every Streamlit rerun but is a no-op after first call


def _init_session():
    defaults = {
        "running": False,
        "final_state": None,
        "error": None,
        "start_time": None,
        "result_queue": queue.Queue(maxsize=1),
        # Quick-launch state
        "app_process": None,   # subprocess.Popen for the running generated app
        "app_port": None,      # port it was started on
        "test_output": None,   # captured pytest output
        # History re-run
        "rerun_prompt": None,  # set by History tab to pre-fill the text area
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ── Worker thread ─────────────────────────────────────────────────────────────


def _run_pipeline_worker(
    user_request: str,
    result_q: queue.Queue,
) -> None:
    """
    Execute the full LangGraph pipeline in a background thread.

    Pushes (final_state, None) on success or (None, str(error)) on failure.
    Always saves the run to pipeline_history.db regardless of outcome.
    """
    final_state: Optional[dict] = None
    error_msg: Optional[str] = None
    try:
        from core.observability import create_session
        from core.agent_runner import build_initial_state, reset_llm_singletons

        reset_llm_singletons()
        session_id, langfuse_handler = create_session("BMAD Streamlit Run")

        rules_suffix = _load_pipeline_rules()
        full_request = user_request + rules_suffix

        initial_state = build_initial_state(
            user_request=full_request,
            session_id=session_id,
            langfuse_handler=langfuse_handler,
        )

        # ── Clean up stale outputs from previous run ──────────────────
        for stale in [
            "outputs/generated_app.py",
            "outputs/test_generated_app.py",
            "app.db",
            "test_app.db",
        ]:
            try:
                pathlib.Path(stale).unlink(missing_ok=True)
            except Exception:
                pass

        langgraph_app = _load_app()
        final_state = langgraph_app.invoke(initial_state)

        # Flush Langfuse traces after pipeline completes
        try:
            from core.observability import flush
            flush(langfuse_handler)
        except Exception:
            pass

        result_q.put((final_state, None))

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        result_q.put((None, error_msg))

    finally:
        # Save to history regardless of outcome — never let this crash the thread
        save_run_to_history(user_request, final_state, error_msg)


# ── UI helpers ────────────────────────────────────────────────────────────────


def _render_sidebar():
    with st.sidebar:
        st.title("⚙️ BMAD Config")
        st.markdown("---")

        try:
            import yaml, pathlib
            cfg = yaml.safe_load(
                (pathlib.Path(__file__).parent / "config" / "models.yaml").read_text()
            )
            llm_cfg = cfg.get("llm", {})
            st.subheader("LLM")
            st.caption(f"Model: `{llm_cfg.get('model', 'N/A')}`")
            st.caption(f"Temperature: `{llm_cfg.get('temperature', 'N/A')}`")
            st.caption(f"Max tokens: `{llm_cfg.get('max_tokens', 'N/A')}`")

            exec_cfg = cfg.get("executor", {})
            st.subheader("Executor")
            st.caption(f"Startup wait: `{exec_cfg.get('startup_wait_seconds', 8)}s`")
            st.caption(f"Output: `{exec_cfg.get('output_file', 'outputs/generated_app.py')}`")

            retry_cfg = cfg.get("retry_limits", {})
            st.subheader("Retry Limits")
            st.caption(f"Validator loop: `{retry_cfg.get('validator_max_retries', 2)}`")
            st.caption(f"Reviewer loop: `{retry_cfg.get('reviewer_max_retries', 2)}`")
        except Exception as e:
            st.warning(f"Could not load config: {e}")

        st.markdown("---")
        st.subheader("Pipeline")
        st.markdown("""
```
Planner → Architect → Developer
              ↓
          Validator ⟳ (2x)
              ↓
          Tester → Reviewer ⟳ (2x)
              ↓
          Executor → TestWriter
```
""")


def _render_agent_log_table(agent_log: list):
    """Render per-agent token/cost/latency table."""
    if not agent_log:
        st.info("No agent log entries.")
        return

    import pandas as pd
    rows = []
    for rec in agent_log:
        if hasattr(rec, "to_dict"):
            rows.append(rec.to_dict())
        elif isinstance(rec, dict):
            rows.append(rec)

    if rows:
        df = pd.DataFrame(rows)
        # Format columns
        if "timestamp_utc" in df.columns:
            df = df.drop(columns=["timestamp_utc"])
        if "cost_usd" in df.columns:
            df["cost_usd"] = df["cost_usd"].apply(lambda x: f"${x:.6f}")
        if "latency_ms" in df.columns:
            df["latency_ms"] = df["latency_ms"].apply(lambda x: f"{x:.0f}ms")
        st.dataframe(df, use_container_width=True)


def _render_pipeline_summary(final_state: dict):
    """Render pipeline summary metric cards."""
    from core.observability import get_pipeline_summary
    summary = get_pipeline_summary(final_state)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tokens", f"{summary.total_tokens:,}")
    col2.metric("Estimated Cost", f"${summary.total_cost_usd:.4f}")
    col3.metric("Total Latency", f"{summary.total_latency_ms/1000:.1f}s")
    col4.metric("Agents Called", str(summary.agent_count))

    if summary.agents_called:
        st.caption("Agent execution order: " + " → ".join(summary.agents_called))


def _render_results(final_state: dict):
    """Render all pipeline results in a tabbed layout."""
    tab_code, tab_tests, tab_exec, tab_log, tab_obs, tab_hist = st.tabs([
        "📄 Generated Code",
        "🧪 Test File",
        "▶️ Execution",
        "📋 Agent Log",
        "📊 Observability",
        "📜 History",
    ])

    # ── Tab 1: Generated Code ──────────────────────────────────────────────
    with tab_code:
        st.subheader("Generated Application")
        code = final_state.get("code") or ""
        if code:
            st.code(code, language="python", line_numbers=True)
            st.download_button(
                label="⬇️ Download generated_app.py",
                data=code,
                file_name="generated_app.py",
                mime="text/x-python",
            )
        else:
            st.warning("No code was generated.")

        # Functional spec and technical design in expanders
        with st.expander("🔍 Refined Prompt"):
            st.markdown(final_state.get("refined_prompt") or "_Not available_")
        with st.expander("📋 Functional Spec"):
            st.markdown(final_state.get("functional_spec") or "_Not available_")
        with st.expander("🏗️ Technical Design"):
            st.markdown(final_state.get("technical_design") or "_Not available_")

    # ── Tab 2: Test File ───────────────────────────────────────────────────
    with tab_tests:
        st.subheader("Generated pytest File")
        test_file_path = final_state.get("test_file") or ""
        test_code = ""

        # Try to read from disk first (most up-to-date)
        if test_file_path:
            try:
                with open(test_file_path, "r", encoding="utf-8") as fh:
                    test_code = fh.read()
            except FileNotFoundError:
                pass

        # Fallback: try default path
        if not test_code:
            try:
                with open("outputs/test_generated_app.py", "r", encoding="utf-8") as fh:
                    test_code = fh.read()
            except FileNotFoundError:
                pass

        if test_code:
            st.code(test_code, language="python", line_numbers=True)
            st.download_button(
                label="⬇️ Download test_generated_app.py",
                data=test_code,
                file_name="test_generated_app.py",
                mime="text/x-python",
            )
            st.info(f"Run with: `python -m pytest {test_file_path or 'outputs/test_generated_app.py'} -v`")
        else:
            st.warning("No test file was generated.")

        # Test cases in expander
        with st.expander("📝 Test Case Descriptions"):
            st.markdown(final_state.get("test_cases") or "_Not available_")

    # ── Tab 3: Execution ───────────────────────────────────────────────────
    with tab_exec:
        st.subheader("Execution Result")
        exec_result = final_state.get("execution_result") or ""
        exec_error = final_state.get("execution_error") or ""

        if exec_result and "successfully" in exec_result.lower():
            st.success(f"✅ {exec_result}")
        elif exec_result:
            st.info(exec_result)
        else:
            st.warning("No execution result recorded.")

        if exec_error:
            st.error("Execution stderr output:")
            st.code(exec_error, language="text")

        output_dir = final_state.get("output_dir")
        if output_dir:
            st.caption(f"Session archive: `{output_dir}`")

        # Reviewer feedback
        review_approved = final_state.get("review_approved")
        review_feedback = final_state.get("review_feedback") or ""
        with st.expander("🔍 Code Review Result"):
            if review_approved:
                st.success("APPROVED: YES")
            elif review_approved is False:
                st.error("APPROVED: NO")
            else:
                st.info("Review status unknown")
            if review_feedback:
                st.markdown(f"**Feedback:** {review_feedback}")

        # Validation info
        val_passed = final_state.get("validation_passed")
        val_error = final_state.get("validation_error")
        with st.expander("✅ Validation Result"):
            if val_passed:
                st.success("All validation checks passed.")
            elif val_passed is False:
                st.error(f"Validation failed: {val_error}")
            else:
                st.info("Validation status unknown")

    # ── Tab 4: Agent Log ───────────────────────────────────────────────────
    with tab_log:
        st.subheader("Per-Agent Execution Log")
        agent_log = final_state.get("agent_log") or []
        _render_agent_log_table(agent_log)

    # ── Tab 5: Observability ──────────────────────────────────────────────
    with tab_obs:
        st.subheader("Pipeline Summary")
        _render_pipeline_summary(final_state)

        session_id = final_state.get("session_id", "N/A")
        st.caption(f"Session ID: `{session_id}`")
        st.caption(f"Pipeline status: `{final_state.get('pipeline_status', 'N/A')}`")

        st.markdown("---")
        st.info(
            "Full traces are available in the Langfuse dashboard if "
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set in .env"
        )

    # ── Tab 6: History ────────────────────────────────────────────────────
    with tab_hist:
        import pandas as pd
        st.subheader("Pipeline Run History")
        runs = load_history()

        if not runs:
            st.info("No pipeline runs recorded yet. Complete a pipeline run to see history here.")
        else:
            # ── Aggregate metrics ──────────────────────────────────────
            total_runs = len(runs)
            success_count = sum(1 for r in runs if r.get("status") == "complete")
            success_rate = (success_count / total_runs * 100) if total_runs else 0
            agg_tokens = sum(r.get("total_tokens") or 0 for r in runs)
            agg_cost = sum(r.get("total_cost_usd") or 0.0 for r in runs)

            hm1, hm2, hm3, hm4 = st.columns(4)
            hm1.metric("Total Runs", total_runs)
            hm2.metric("Success Rate", f"{success_rate:.0f}%")
            hm3.metric("Total Tokens Used", f"{agg_tokens:,}")
            hm4.metric("Total Cost", f"${agg_cost:.4f}")

            # ── Past runs table ────────────────────────────────────────
            st.markdown("#### All Runs")
            table_rows = [{
                "ID": r["id"],
                "Prompt": (r["prompt"] or "")[:60] + ("…" if len(r["prompt"] or "") > 60 else ""),
                "Status": "✅ complete" if r.get("status") == "complete" else "❌ failed",
                "Tokens": r.get("total_tokens") or 0,
                "Cost ($)": round(r.get("total_cost_usd") or 0.0, 5),
                "Latency (s)": round((r.get("total_latency_ms") or 0) / 1000, 1),
                "Created At": r.get("created_at", ""),
            } for r in runs]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # ── Run detail viewer ──────────────────────────────────────
            st.markdown("#### Inspect a Run")
            run_labels = [
                f"#{r['id']} — {(r['prompt'] or '')[:40]}{'…' if len(r['prompt'] or '') > 40 else ''}"
                for r in runs
            ]
            sel_idx = st.selectbox(
                "Select a run to inspect",
                range(len(runs)),
                format_func=lambda i: run_labels[i],
                key="history_selected_run",
            )
            sel = runs[sel_idx]

            d1, d2, d3, d4, d5, d6 = st.tabs([
                "📋 Prompt", "📄 Functional Spec", "🏗️ Technical Design",
                "💻 Generated Code", "🧪 Test Cases", "📊 Metrics",
            ])
            with d1:
                st.text_area(
                    "Original Prompt",
                    value=sel.get("prompt") or "",
                    height=150,
                    disabled=True,
                    key=f"hist_prompt_{sel['id']}",
                )
                refined = sel.get("refined_prompt") or ""
                if refined:
                    st.markdown("**Refined Prompt** _(PromptRefiner output)_")
                    st.markdown(refined)
            with d2:
                st.markdown(sel.get("functional_spec") or "_Not available_")
            with d3:
                st.markdown(sel.get("technical_design") or "_Not available_")
            with d4:
                code_val = sel.get("code") or ""
                if code_val:
                    st.code(code_val, language="python", line_numbers=True)
                else:
                    st.info("No code recorded for this run.")
            with d5:
                st.markdown(sel.get("test_cases") or "_Not available_")
            with d6:
                dm1, dm2, dm3 = st.columns(3)
                dm1.metric("Tokens", f"{sel.get('total_tokens') or 0:,}")
                dm2.metric("Cost", f"${sel.get('total_cost_usd') or 0.0:.5f}")
                dm3.metric("Latency", f"{(sel.get('total_latency_ms') or 0) / 1000:.1f}s")
                agents_str = sel.get("agents_called") or ""
                if agents_str:
                    st.caption("Agents: " + " → ".join(agents_str.split(",")))
                val_lbl = "Passed" if sel.get("validation_passed") else "Failed"
                rev_lbl = "Approved" if sel.get("review_approved") else "Rejected"
                st.caption(f"Validation: {val_lbl}  |  Review: {rev_lbl}")

            # ── Action buttons ─────────────────────────────────────────
            st.markdown("")
            ac1, ac2 = st.columns(2)
            with ac1:
                if st.button(
                    "🔁 Re-run this prompt",
                    type="primary",
                    use_container_width=True,
                    key=f"rerun_{sel['id']}",
                ):
                    st.session_state["rerun_prompt"] = sel.get("prompt") or ""
                    st.rerun()
            with ac2:
                code_dl = sel.get("code") or ""
                st.download_button(
                    "⬇️ Download Code",
                    data=code_dl or " ",
                    file_name="generated_app.py",
                    mime="text/x-python",
                    disabled=not code_dl,
                    use_container_width=True,
                    key=f"hist_dl_{sel['id']}",
                )

    # ── Quick Launch ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Launch")
    col_app, col_tests = st.columns(2)

    # ── Left column: Generated App ────────────────────────────────────────
    with col_app:
        app_file = "outputs/generated_app.py"
        proc = st.session_state.get("app_process")
        app_running = proc is not None and proc.poll() is None

        if app_running:
            port = st.session_state["app_port"]
            st.link_button(
                "Open Generated App",
                f"http://localhost:{port}",
                use_container_width=True,
                type="primary",
            )
            if st.button("Stop App", use_container_width=True):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                st.session_state["app_process"] = None
                st.session_state["app_port"] = None
                st.rerun()
        else:
            can_launch = pathlib.Path(app_file).exists()
            if st.button(
                "Launch Generated App",
                use_container_width=True,
                type="primary",
                disabled=not can_launch,
            ):
                # Clean up any stale process
                if proc is not None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        pass
                port = _find_free_port(8503)
                new_proc = subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", app_file,
                     "--server.headless", "true", "--server.port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(5)  # wait for Streamlit to be ready
                if new_proc.poll() is None:
                    st.session_state["app_process"] = new_proc
                    st.session_state["app_port"] = port
                    st.rerun()
                else:
                    st.error("Generated app failed to start. Check the code for errors.")

    # ── Right column: Tests ───────────────────────────────────────────────
    with col_tests:
        test_file = final_state.get("test_file") or "outputs/test_generated_app.py"
        can_test = pathlib.Path(test_file).exists()

        if st.button("Run Tests", use_container_width=True, disabled=not can_test):
            with st.spinner("Running pytest..."):
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file,
                     "-v", "--tb=short", "--no-header"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            output = result.stdout
            if result.stderr.strip():
                output += "\n" + result.stderr
            st.session_state["test_output"] = output

        test_out = st.session_state.get("test_output")
        if test_out:
            low = test_out.lower()
            if "failed" in low or "error" in low:
                st.error("Some tests failed")
            else:
                st.success("All tests passed")
            with st.expander("Test Output", expanded=True):
                st.code(test_out, language="text")


# ── Main layout ───────────────────────────────────────────────────────────────

_render_sidebar()

st.title("🤖 BMAD AI Orchestration Pipeline")
st.caption("Planner → Architect → Developer → Validator → Tester → Reviewer → Executor → TestWriter")
st.markdown("---")

# ── Input section ─────────────────────────────────────────────────────────────
# If the History tab requested a re-run, pre-fill the text area widget
if st.session_state.get("rerun_prompt"):
    st.session_state["user_request_input"] = st.session_state.pop("rerun_prompt")

col_input, col_actions = st.columns([3, 1])

with col_input:
    user_request = st.text_area(
        label="Describe the application to build:",
        placeholder=(
            "e.g. Build a task management app with add/delete/list tasks. "
            "Each task has a title and due date."
        ),
        height=120,
        disabled=st.session_state["running"],
        key="user_request_input",
    )

with col_actions:
    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button(
        "🚀 Run Pipeline",
        type="primary",
        disabled=st.session_state["running"] or not (user_request or "").strip(),
        use_container_width=True,
    )
    reset_clicked = st.button(
        "🔄 Reset",
        disabled=st.session_state["running"],
        use_container_width=True,
    )

# ── Reset ─────────────────────────────────────────────────────────────────────
if reset_clicked:
    # Kill the launched generated app if still running
    proc = st.session_state.get("app_process")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    st.session_state["running"] = False
    st.session_state["final_state"] = None
    st.session_state["error"] = None
    st.session_state["start_time"] = None
    st.session_state["result_queue"] = queue.Queue(maxsize=1)
    st.session_state["app_process"] = None
    st.session_state["app_port"] = None
    st.session_state["test_output"] = None
    st.session_state["rerun_prompt"] = None
    st.rerun()

# ── Launch pipeline ───────────────────────────────────────────────────────────
if run_clicked and (user_request or "").strip():
    # Drain any leftover result from a previous run
    while not st.session_state["result_queue"].empty():
        try:
            st.session_state["result_queue"].get_nowait()
        except queue.Empty:
            break

    st.session_state["running"] = True
    st.session_state["final_state"] = None
    st.session_state["error"] = None
    st.session_state["start_time"] = time.time()

    worker = threading.Thread(
        target=_run_pipeline_worker,
        args=(user_request.strip(), st.session_state["result_queue"]),
        daemon=True,
    )
    worker.start()
    st.rerun()

# ── Poll for completion ───────────────────────────────────────────────────────
if st.session_state["running"]:
    try:
        result = st.session_state["result_queue"].get_nowait()
        final_state, error = result
        st.session_state["running"] = False
        st.session_state["final_state"] = final_state
        st.session_state["error"] = error
        if final_state:
            final_state["pipeline_status"] = "complete"
        st.rerun()
    except queue.Empty:
        # Still running — show spinner and schedule a rerun
        elapsed = time.time() - (st.session_state["start_time"] or time.time())
        with st.spinner(f"⏳ Pipeline running... ({elapsed:.0f}s elapsed)"):
            time.sleep(2)
        st.rerun()

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state["error"]:
    st.error(f"❌ Pipeline failed:\n\n{st.session_state['error']}")

if st.session_state["final_state"]:
    elapsed_total = time.time() - (st.session_state.get("start_time") or time.time())
    st.success(f"✅ Pipeline complete in {elapsed_total:.1f}s")
    st.markdown("---")
    _render_results(st.session_state["final_state"])
elif not st.session_state["running"] and not st.session_state["error"]:
    st.info("Enter a request above and click **Run Pipeline** to start.")
