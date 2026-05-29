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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMAD Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS injection ──────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Base ── */
        .stApp { background: #F1F5F9; }
        .main .block-container {
            padding: 2rem 2.5rem 3rem;
            max-width: 1200px;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }
        [data-testid="stDecoration"] { display: none; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
            border-right: none;
        }
        [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F1F5F9 !important;
            font-weight: 700;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stTextInput label { color: #94A3B8 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(255,255,255,0.07) !important;
            border-color: rgba(255,255,255,0.15) !important;
            color: #F1F5F9 !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #94A3B8 !important; }
        [data-testid="stSidebar"] .stCaption { color: #64748B !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
        [data-testid="stSidebar"] code {
            background: rgba(255,255,255,0.1) !important;
            color: #A5B4FC !important;
            border-radius: 4px;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
            transition: opacity 0.2s !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover { opacity: 0.9 !important; }
        [data-testid="stSidebar"] .stButton > button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed !important;
        }

        /* ── Cards / Metric containers ── */
        [data-testid="metric-container"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            padding: 1.1rem 1.4rem !important;
            box-shadow: 0 1px 4px rgba(15,23,42,0.06) !important;
        }
        /* Force ALL text inside metric cards to be visible */
        [data-testid="metric-container"],
        [data-testid="metric-container"] * {
            color: #0F172A !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricLabel"],
        [data-testid="metric-container"] label,
        [data-testid="metric-container"] p {
            color: #64748B !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em !important;
            text-transform: uppercase !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"],
        [data-testid="metric-container"] [data-testid="stMetricValue"] * {
            color: #0F172A !important;
            font-weight: 800 !important;
            font-size: 1.8rem !important;
        }

        /* ── All buttons — base rule ensures text is always visible ── */
        .stButton > button {
            color: #1E293B !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            height: 2.8rem !important;
            transition: all 0.2s !important;
        }

        /* ── Primary button ── */
        .stButton > button[kind="primary"],
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            border: none !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        }
        .stButton > button[kind="primary"]:hover,
        [data-testid="baseButton-primary"]:hover {
            box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
            transform: translateY(-1px) !important;
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:disabled,
        [data-testid="baseButton-primary"]:disabled {
            background: #C7D2FE !important;
            box-shadow: none !important;
            transform: none !important;
            color: #FFFFFF !important;
        }

        /* ── Secondary / default button ── */
        .stButton > button[kind="secondary"],
        [data-testid="baseButton-secondary"] {
            background: #FFFFFF !important;
            border: 1.5px solid #CBD5E1 !important;
            color: #374151 !important;
        }
        .stButton > button[kind="secondary"]:hover,
        [data-testid="baseButton-secondary"]:hover {
            border-color: #6366F1 !important;
            color: #4F46E5 !important;
            background: #EEF2FF !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 5px;
            gap: 4px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.45rem 1rem;
            color: #64748B;
            font-weight: 500;
            font-size: 0.85rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.3);
        }
        .stTabs [data-baseweb="tab-highlight"] { display: none; }
        .stTabs [data-baseweb="tab-border"] { display: none; }

        /* ── Text input / Text area ── */
        .stTextArea textarea, .stTextInput input {
            border-radius: 10px !important;
            border: 1.5px solid #E2E8F0 !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
            font-size: 0.95rem !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        }
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {
            color: #94A3B8 !important;
        }
        /* Labels for all form widgets */
        .stTextArea label,
        .stTextInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        label[data-testid="stWidgetLabel"] {
            color: #374151 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        /* Headings — keep them dark on the light background */
        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            color: #0F172A !important;
        }
        /* Caption text */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #64748B !important;
        }

        /* ── Code blocks ── */
        .stCode { border-radius: 10px !important; overflow: hidden; }
        pre { border-radius: 10px !important; }

        /* ── Expander ── */
        [data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }

        /* ── Data frame ── */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }

        /* ── Alerts ── */
        .stSuccess { border-radius: 10px !important; border-left: 4px solid #10B981 !important; }
        .stError   { border-radius: 10px !important; border-left: 4px solid #EF4444 !important; }
        .stWarning { border-radius: 10px !important; border-left: 4px solid #F59E0B !important; }
        .stInfo    { border-radius: 10px !important; border-left: 4px solid #6366F1 !important; }

        /* ── Progress bar ── */
        [data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
            border-radius: 9999px;
        }

        /* ── Custom stat card ── */
        .stat-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            box-shadow: 0 1px 4px rgba(15,23,42,0.05);
        }
        .stat-card .label {
            font-size: 0.72rem;
            font-weight: 700;
            color: #94A3B8;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .stat-card .value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }
        .stat-card .sub {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-top: 0.15rem;
        }

        /* ── Badge ── */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }
        .badge-indigo { background: #EEF2FF; color: #4F46E5; }
        .badge-green  { background: #D1FAE5; color: #065F46; }
        .badge-red    { background: #FEE2E2; color: #991B1B; }
        .badge-amber  { background: #FEF3C7; color: #92400E; }

        /* ── Divider ── */
        .light-divider {
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 1.5rem 0;
        }

        /* ── Pipeline step breadcrumb ── */
        .pipeline-steps {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            align-items: center;
            margin: 0.5rem 0 1.5rem;
        }
        .step-pill {
            background: #EEF2FF;
            color: #4F46E5;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .step-arrow { color: #CBD5E1; font-size: 0.75rem; }

        /* ── Eval score row ── */
        .eval-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            background: #F8FAFC;
            border-radius: 10px;
            margin-bottom: 8px;
        }
        .eval-label { font-weight: 600; color: #334155; font-size: 0.85rem; flex: 0 0 180px; }
        .eval-bar-wrap { flex: 1; background: #E2E8F0; border-radius: 9999px; height: 8px; overflow: hidden; }
        .eval-bar { height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #6366F1, #8B5CF6); }
        .eval-score { font-weight: 800; color: #0F172A; font-size: 0.85rem; flex: 0 0 38px; text-align: right; }

        /* ── Hero title ── */
        .hero-title {
            font-size: 2.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #A855F7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.15;
            margin-bottom: 0.2rem;
        }
        .hero-sub {
            color: #64748B;
            font-size: 0.95rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }

        /* ── Running pulse ── */
        @keyframes pulse {
            0%   { opacity: 1; }
            50%  { opacity: 0.45; }
            100% { opacity: 1; }
        }
        .pulse-dot {
            display: inline-block;
            width: 9px; height: 9px;
            border-radius: 50%;
            background: #6366F1;
            margin-right: 8px;
            animation: pulse 1.4s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_css()

# ── Port helper ───────────────────────────────────────────────────────────────

def _find_free_port(preferred: int = 8503, scan_limit: int = 20) -> int:
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

_HISTORY_DB    = str(pathlib.Path(__file__).parent / "pipeline_history.db")

# ── Agent Loop control file paths ─────────────────────────────────────────────
_CONTROL_FILE  = pathlib.Path(__file__).parent / "control.yaml"
_CONTROL_LOCK  = pathlib.Path(__file__).parent / "control.yaml.lock"


def init_history_db() -> None:
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
        try:
            conn.execute("ALTER TABLE runs ADD COLUMN refined_prompt TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
    except Exception:
        pass


def save_run_to_history(
    original_prompt: str,
    final_state: Optional[dict],
    error: Optional[str],
) -> None:
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
        load_history.clear()
    except Exception:
        pass


def _read_control_yaml() -> dict:
    """Read control.yaml; return defaults if missing or corrupt."""
    defaults = {
        "run": False, "prompt": "", "project_mode": "auto",
        "status": "idle", "last_run": None, "last_result": None, "last_prompt": "",
    }
    try:
        import yaml as _yaml
        data = _yaml.safe_load(_CONTROL_FILE.read_text(encoding="utf-8")) or {}
        return {**defaults, **data}
    except Exception:
        return defaults


def _write_control_yaml(data: dict) -> None:
    """Write control.yaml under a file lock (best-effort)."""
    try:
        import yaml as _yaml
        from filelock import FileLock, Timeout
        with FileLock(str(_CONTROL_LOCK), timeout=5):
            _CONTROL_FILE.write_text(
                _yaml.dump(data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
    except Exception:
        pass


@st.cache_data(ttl=30)
def load_history() -> list[dict]:
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


# ── Lazy imports ──────────────────────────────────────────────────────────────

@st.cache_resource
def _load_app():
    from orchestration.graph import app as langgraph_app
    return langgraph_app


@st.cache_data
def _load_prompt_templates() -> list[dict]:
    import json
    try:
        tpl_path = pathlib.Path(__file__).parent / "config" / "prompt_templates.json"
        return json.loads(tpl_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_pipeline_rules(mode: str = "streamlit_crud") -> str:
    try:
        import yaml
        if mode == "fastapi_rag":
            cfg_path = pathlib.Path(__file__).parent / "config" / "rag_rules.yaml"
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            return data.get("prompt_suffix", "")
        else:
            cfg_path = pathlib.Path(__file__).parent / "config" / "pipeline_rules.yaml"
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            return data.get("prompt_suffix", "")
    except Exception:
        return ""


# ── Session state ─────────────────────────────────────────────────────────────

init_history_db()


def _init_session():
    defaults = {
        "running": False,
        "final_state": None,
        "error": None,
        "start_time": None,
        "result_queue": queue.Queue(maxsize=1),
        "app_process": None,
        "app_port": None,
        "test_output": None,
        "rerun_prompt": None,
        "template_prompt": None,
        "project_mode": "streamlit_crud",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── Worker thread ─────────────────────────────────────────────────────────────

def _run_pipeline_worker(user_request: str, result_q: queue.Queue, project_mode: str = "streamlit_crud") -> None:
    final_state: Optional[dict] = None
    error_msg: Optional[str] = None
    try:
        from core.observability import create_session
        from core.agent_runner import build_initial_state, reset_llm_singletons

        reset_llm_singletons()
        session_id, langfuse_handler = create_session("BMAD Streamlit Run")

        rules_suffix = _load_pipeline_rules(project_mode)
        full_request = user_request + rules_suffix

        initial_state = build_initial_state(
            user_request=full_request,
            session_id=session_id,
            langfuse_handler=langfuse_handler,
            project_mode=project_mode,
        )

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
        # Do NOT pass the handler to app.invoke() — it is already stored in
        # state["langfuse_handler"] and gets passed to llm.invoke() inside
        # each agent via agent_runner.run_agent(). Passing it here too
        # creates duplicate callback registrations → orphaned spans → no
        # flowchart in Langfuse.
        final_state = langgraph_app.invoke(initial_state)
        result_q.put((final_state, None))

    except Exception as exc:
        error_msg = str(exc)
        result_q.put((None, error_msg))

    finally:
        try:
            from core.observability import flush
            flush()
        except Exception:
            pass
        save_run_to_history(user_request, final_state, error_msg)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding:1rem 0 0.5rem">
              <div style="font-size:1.5rem;font-weight:900;color:#F1F5F9;letter-spacing:-0.02em">
                ⚡ BMAD
              </div>
              <div style="font-size:0.72rem;color:#64748B;font-weight:500;
                          letter-spacing:0.08em;text-transform:uppercase;margin-top:2px">
                AI Orchestration Pipeline
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:0.5rem 0 1rem"/>', unsafe_allow_html=True)

        # ── Project mode selector ─────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#6366F1;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            '⚙️ Project Mode</div>',
            unsafe_allow_html=True,
        )
        mode_options = {
            "streamlit_crud": "🖥️ Streamlit CRUD App",
            "fastapi_rag":    "🔍 FastAPI RAG Service",
        }
        selected_mode = st.radio(
            "Project Mode",
            options=list(mode_options.keys()),
            format_func=lambda m: mode_options[m],
            index=0 if st.session_state.get("project_mode", "streamlit_crud") == "streamlit_crud" else 1,
            key="mode_radio",
            label_visibility="collapsed",
            disabled=st.session_state.get("running", False),
        )
        st.session_state["project_mode"] = selected_mode

        if selected_mode == "fastapi_rag":
            st.caption("Generates: FastAPI + ChromaDB + LangChain + Langfuse tracing")
        else:
            st.caption("Generates: Streamlit + SQLite single-file app")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:0.5rem 0 1rem"/>', unsafe_allow_html=True)

        # ── Template library ──────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#6366F1;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            '📚 Quick-Start Templates</div>',
            unsafe_allow_html=True,
        )
        templates = _load_prompt_templates()
        if templates:
            categories = ["All"] + sorted({t["category"] for t in templates})
            sel_cat = st.selectbox(
                "Category",
                categories,
                key="tpl_category",
                label_visibility="collapsed",
            )
            filtered = (
                templates if sel_cat == "All"
                else [t for t in templates if t["category"] == sel_cat]
            )

            def _tpl_label(t: dict) -> str:
                return t["name"] if sel_cat != "All" else f"[{t['category']}]  {t['name']}"

            option_names = [_tpl_label(t) for t in filtered]
            sel_idx = st.selectbox(
                "Template",
                range(len(filtered)),
                format_func=lambda i: option_names[i],
                key="tpl_selected",
                label_visibility="collapsed",
            )
            sel_tpl = filtered[sel_idx]
            st.caption(sel_tpl["description"])

            if st.button(
                "Use this template",
                use_container_width=True,
                disabled=st.session_state.get("running", False),
            ):
                st.session_state["template_prompt"] = sel_tpl["prompt"]
                st.rerun()
        else:
            st.caption("_(template file not found)_")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0"/>', unsafe_allow_html=True)

        # ── Config ────────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#6366F1;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            '⚙️ Config</div>',
            unsafe_allow_html=True,
        )
        try:
            import yaml
            cfg = yaml.safe_load(
                (pathlib.Path(__file__).parent / "config" / "models.yaml").read_text()
            )
            llm_cfg = cfg.get("llm", {})
            st.caption(f"**Model** `{llm_cfg.get('model', 'N/A')}`")
            st.caption(f"**Temp** `{llm_cfg.get('temperature', 'N/A')}` | **Max tokens** `{llm_cfg.get('max_tokens', 'N/A')}`")

            retry_cfg = cfg.get("retry_limits", {})
            st.caption(
                f"**Validator retries** `{retry_cfg.get('validator_max_retries', 2)}` | "
                f"**Reviewer retries** `{retry_cfg.get('reviewer_max_retries', 2)}`"
            )
        except Exception as e:
            st.caption(f"Could not load config: {e}")

        st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0"/>', unsafe_allow_html=True)

        # ── Pipeline diagram ──────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#6366F1;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem">'
            '🔗 Pipeline</div>',
            unsafe_allow_html=True,
        )
        steps = [
            ("PromptRefiner",    "#6366F1"),
            ("ComplexityScorer","#7C3AED"),
            ("Planner",         "#8B5CF6"),
            ("Architect",       "#A78BFA"),
            ("Developer",       "#6366F1"),
            ("Validator ⟳",    "#0EA5E9"),
            ("Tester",          "#0EA5E9"),
            ("Reviewer ⟳",     "#0EA5E9"),
            ("Executor",        "#10B981"),
            ("TestWriter",      "#10B981"),
            ("ReadmeWriter",    "#10B981"),
            ("EvalAgent",       "#F59E0B"),
        ]
        for i, (name, color) in enumerate(steps):
            connector = "" if i == 0 else (
                '<div style="width:2px;height:12px;background:rgba(255,255,255,0.12);'
                'margin:0 0 0 14px"></div>'
            )
            st.markdown(
                f'{connector}<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<span style="font-size:0.75rem;color:#CBD5E1;font-weight:500">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Langfuse dashboard button ──────────────────────────────────────
        st.markdown(
            '<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0"/>',
            unsafe_allow_html=True,
        )
        if _langfuse_configured():
            dashboard_url = _langfuse_base_url()
            st.link_button(
                "📊 Open Langfuse Dashboard",
                url=dashboard_url,
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;text-align:center;padding:0.5rem 0">'
                'Set <code style="color:#A5B4FC">LANGFUSE_PUBLIC_KEY</code> &amp; '
                '<code style="color:#A5B4FC">SECRET_KEY</code> in <code style="color:#A5B4FC">.env</code>'
                ' to enable tracing.</div>',
                unsafe_allow_html=True,
            )

        # ── 🔄 Agent Loop ─────────────────────────────────────────────────
        st.markdown(
            '<hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0"/>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#6366F1;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            '🔄 Agent Loop</div>',
            unsafe_allow_html=True,
        )

        # Read current loop state
        ctrl = _read_control_yaml()
        loop_status  = ctrl.get("status", "idle")
        loop_run     = bool(ctrl.get("run", False))
        loop_result  = (ctrl.get("last_result") or "")[:110]
        loop_prompt_ = (ctrl.get("last_prompt") or "")[:55]
        loop_last_run = ctrl.get("last_run") or ""
        eff_status   = "running" if loop_run else loop_status

        # Status badge
        _badge_map = {
            "idle":     ("💤", "#64748B", "rgba(100,116,139,0.12)"),
            "running":  ("⏳", "#D97706", "rgba(217,119,6,0.12)"),
            "complete": ("✅", "#059669", "rgba(5,150,105,0.12)"),
            "failed":   ("❌", "#DC2626", "rgba(220,38,38,0.12)"),
        }
        _icon, _fc, _bg = _badge_map.get(eff_status, _badge_map["idle"])
        ts_line = (
            f'<br><span style="font-size:0.65rem;opacity:0.75">{loop_last_run}</span>'
            if loop_last_run else ""
        )
        st.markdown(
            f'<div style="background:{_bg};border-radius:8px;'
            f'padding:0.45rem 0.75rem;margin-bottom:0.5rem">'
            f'<span style="color:{_fc};font-weight:700;font-size:0.8rem">'
            f'{_icon} {eff_status.upper()}</span>{ts_line}</div>',
            unsafe_allow_html=True,
        )

        if loop_prompt_:
            st.caption(
                f"Last prompt: _{loop_prompt_}{'…' if len(ctrl.get('last_prompt',''))>55 else ''}_"
            )
        if loop_result and eff_status != "idle":
            st.caption(
                f"Result: {loop_result}{'…' if len(ctrl.get('last_result',''))>110 else ''}"
            )

        # Trigger form
        _trigger_disabled = loop_run or eff_status == "running"
        with st.form("agent_loop_trigger_form", clear_on_submit=True):
            _loop_prompt_in = st.text_input(
                "Prompt",
                placeholder="Build a recipe manager app…",
                disabled=_trigger_disabled,
                label_visibility="collapsed",
            )
            _loop_mode_in = st.selectbox(
                "Mode",
                options=["auto", "streamlit_crud", "fastapi_rag"],
                format_func=lambda m: {"auto":"🔀 Auto-detect","streamlit_crud":"🖥️ Streamlit CRUD","fastapi_rag":"🔍 FastAPI RAG"}[m],
                disabled=_trigger_disabled,
                label_visibility="collapsed",
            )
            _submit = st.form_submit_button(
                "Waiting for loop…" if _trigger_disabled else "🚀 Trigger Pipeline",
                disabled=_trigger_disabled,
                use_container_width=True,
            )

        if _submit and _loop_prompt_in.strip():
            ctrl["run"]          = True
            ctrl["prompt"]       = _loop_prompt_in.strip()
            ctrl["project_mode"] = _loop_mode_in
            ctrl["last_result"]  = "Pending…"
            _write_control_yaml(ctrl)
            st.rerun()

        st.caption("Run `python agent_loop.py` in a terminal to start the watcher.")

        # Store flag so the bottom of the page can auto-refresh while loop runs
        st.session_state["_loop_needs_refresh"] = _trigger_disabled


# ── Langfuse URL helpers ──────────────────────────────────────────────────────

def _langfuse_base_url() -> str:
    """Return the Langfuse host from env (default: cloud.langfuse.com)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    # Ensure scheme is present
    if not host.startswith("http"):
        host = "https://" + host
    return host


def _langfuse_session_url(session_id: str) -> str:
    """Deep-link to a specific session in the Langfuse dashboard."""
    return f"{_langfuse_base_url()}/sessions/{session_id}"


def _langfuse_configured() -> bool:
    """True if both Langfuse API keys are set."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_agent_log_table(agent_log: list):
    if not agent_log:
        st.info("No agent log entries yet.")
        return
    import pandas as pd
    rows = []
    for rec in agent_log:
        rows.append(rec.to_dict() if hasattr(rec, "to_dict") else rec)
    if rows:
        df = pd.DataFrame(rows)
        if "timestamp_utc" in df.columns:
            df = df.drop(columns=["timestamp_utc"])
        if "cost_usd" in df.columns:
            df["cost_usd"] = df["cost_usd"].apply(lambda x: f"${x:.6f}")
        if "latency_ms" in df.columns:
            df["latency_ms"] = df["latency_ms"].apply(lambda x: f"{x:.0f} ms")
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_pipeline_summary(final_state: dict):
    from core.observability import get_pipeline_summary
    s = get_pipeline_summary(final_state)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tokens", f"{s.total_tokens:,}")
    c2.metric("Est. Cost", f"${s.total_cost_usd:.4f}")
    c3.metric("Latency", f"{s.total_latency_ms/1000:.1f}s")
    c4.metric("Agents", str(s.agent_count))
    if s.agents_called:
        st.markdown(
            '<div style="font-size:0.78rem;color:#64748B;margin-top:0.25rem">'
            + " &rarr; ".join(s.agents_called)
            + "</div>",
            unsafe_allow_html=True,
        )


def _color_for_score(v: float) -> str:
    if v >= 0.75:
        return "#10B981"
    if v >= 0.5:
        return "#F59E0B"
    return "#EF4444"


def _render_eval_scores(eval_scores: dict):
    ov = eval_scores.get("overall", {})
    ov_val = float(ov.get("value", 0.0))
    color = _color_for_score(ov_val)

    # Overall score card
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#6366F1,#8B5CF6);
                    border-radius:16px;padding:1.4rem 1.8rem;
                    margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem">
          <div>
            <div style="font-size:0.72rem;font-weight:700;color:rgba(255,255,255,0.7);
                        letter-spacing:0.1em;text-transform:uppercase">Overall Score</div>
            <div style="font-size:2.8rem;font-weight:900;color:#fff;line-height:1">
              {ov_val*100:.0f}<span style="font-size:1.2rem;font-weight:600">%</span>
            </div>
          </div>
          <div style="flex:1;font-size:0.8rem;color:rgba(255,255,255,0.8)">
            {ov.get('reason', '')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dimension breakdown
    DIMS = [
        ("feature_coverage",      "Feature Coverage",      "LLM judge: spec vs code"),
        ("requirement_alignment", "Requirement Alignment", "LLM judge: request vs code"),
        ("code_quality",          "Code Quality",          "Validation + review + execution"),
        ("semantic_overlap",      "Semantic Overlap",      "Jaccard similarity (NLP)"),
    ]
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:700;color:#64748B;'
        'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.75rem">'
        'Dimension Breakdown</div>',
        unsafe_allow_html=True,
    )
    for key, label, tooltip in DIMS:
        dim = eval_scores.get(key, {})
        val = float(dim.get("value", 0.0))
        bar_color = _color_for_score(val)
        reason = dim.get("reason", "")
        st.markdown(
            f"""
            <div class="eval-row" title="{tooltip}">
              <div class="eval-label">{label}</div>
              <div class="eval-bar-wrap">
                <div class="eval-bar" style="width:{val*100:.1f}%;background:{bar_color}"></div>
              </div>
              <div class="eval-score" style="color:{bar_color}">{val:.2f}</div>
            </div>
            <div style="font-size:0.72rem;color:#94A3B8;padding:0 14px;margin-bottom:4px">{reason}</div>
            """,
            unsafe_allow_html=True,
        )

    latency = eval_scores.get("eval_latency_ms", 0)
    if latency:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:0.5rem">'
            f'Eval completed in {latency/1000:.1f}s</div>',
            unsafe_allow_html=True,
        )


def _render_results(final_state: dict):
    tab_code, tab_tests, tab_exec, tab_log, tab_obs, tab_hist = st.tabs([
        "📄 Code",
        "🧪 Tests",
        "▶️ Execution",
        "📋 Agent Log",
        "📊 Observability",
        "📜 History",
    ])

    # ── Tab 1: Generated Code ──────────────────────────────────────────────
    with tab_code:
        code = final_state.get("code") or ""
        if code:
            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button(
                    "⬇️ Download generated_app.py",
                    data=code,
                    file_name="generated_app.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
            st.code(code, language="python", line_numbers=True)
        else:
            st.warning("No code was generated.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            with st.expander("🔮 Refined Prompt"):
                st.markdown(final_state.get("refined_prompt") or "_Not available_")
        with col_b:
            with st.expander("📋 Functional Spec"):
                st.markdown(final_state.get("functional_spec") or "_Not available_")
        with col_c:
            with st.expander("🏗️ Technical Design"):
                st.markdown(final_state.get("technical_design") or "_Not available_")

    # ── Tab 2: Tests ───────────────────────────────────────────────────────
    with tab_tests:
        test_file_path = final_state.get("test_file") or ""
        test_code = ""
        if test_file_path:
            try:
                test_code = pathlib.Path(test_file_path).read_text(encoding="utf-8")
            except FileNotFoundError:
                pass
        if not test_code:
            try:
                test_code = pathlib.Path("outputs/test_generated_app.py").read_text(encoding="utf-8")
            except FileNotFoundError:
                pass

        if test_code:
            col_dl2, _ = st.columns([1, 3])
            with col_dl2:
                st.download_button(
                    "⬇️ Download test file",
                    data=test_code,
                    file_name="test_generated_app.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
            st.code(test_code, language="python", line_numbers=True)
            st.info(f"Run with: `python -m pytest {test_file_path or 'outputs/test_generated_app.py'} -v`")
        else:
            st.warning("No test file was generated.")

        with st.expander("📝 Test Case Descriptions"):
            st.markdown(final_state.get("test_cases") or "_Not available_")

    # ── Tab 3: Execution ───────────────────────────────────────────────────
    with tab_exec:
        exec_result = final_state.get("execution_result") or ""
        exec_error  = final_state.get("execution_error") or ""

        if exec_result and "successfully" in exec_result.lower():
            st.success(f"✅ {exec_result}")
        elif exec_result:
            st.info(exec_result)
        else:
            st.warning("No execution result recorded.")

        if exec_error:
            st.error("Execution stderr:")
            st.code(exec_error, language="text")

        if final_state.get("output_dir"):
            st.caption(f"Session archive: `{final_state['output_dir']}`")

        colA, colB = st.columns(2)
        with colA:
            with st.expander("🔍 Code Review"):
                ra = final_state.get("review_approved")
                rf = final_state.get("review_feedback") or ""
                if ra:
                    st.success("Reviewer: APPROVED")
                elif ra is False:
                    st.error("Reviewer: REJECTED")
                else:
                    st.info("Review status unknown")
                if rf:
                    st.markdown(f"**Feedback:** {rf}")
        with colB:
            with st.expander("✅ Validation"):
                vp = final_state.get("validation_passed")
                ve = final_state.get("validation_error")
                if vp:
                    st.success("All static checks passed.")
                elif vp is False:
                    st.error(f"Failed: {ve}")
                else:
                    st.info("Validation status unknown")

        # ── Quick Launch ───────────────────────────────────────────────
        st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#64748B;'
            'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.75rem">'
            'Quick Launch</div>',
            unsafe_allow_html=True,
        )
        col_app, col_tests2 = st.columns(2)

        with col_app:
            app_file = "outputs/generated_app.py"
            proc = st.session_state.get("app_process")
            app_running = proc is not None and proc.poll() is None
            _mode = final_state.get("project_mode", "streamlit_crud") if final_state else st.session_state.get("project_mode", "streamlit_crud")

            if app_running:
                port = st.session_state["app_port"]
                open_label = "🌐 Open Generated App" if _mode == "streamlit_crud" else "🌐 Open API Docs"
                open_url = f"http://localhost:{port}" if _mode == "streamlit_crud" else f"http://localhost:{port}/docs"
                st.link_button(open_label, open_url, use_container_width=True, type="primary")
                if st.button("⏹ Stop App", use_container_width=True):
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
                launch_label = "🚀 Launch Generated App" if _mode == "streamlit_crud" else "🚀 Launch FastAPI Server"
                if st.button(
                    launch_label,
                    use_container_width=True,
                    type="primary",
                    disabled=not can_launch,
                ):
                    if proc is not None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except Exception:
                            pass
                    child_env = {
                        k: v for k, v in __import__("os").environ.items()
                        if not k.startswith("STREAMLIT_")
                    }
                    if _mode == "fastapi_rag":
                        # FastAPI: launch with uvicorn on a free port
                        port = _find_free_port(8000)
                        new_proc = subprocess.Popen(
                            [sys.executable, "-m", "uvicorn",
                             "outputs.generated_app:app",
                             "--host", "0.0.0.0",
                             "--port", str(port)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            env=child_env,
                        )
                        time.sleep(4)
                    else:
                        # Streamlit CRUD: launch with streamlit run
                        port = _find_free_port(8503)
                        new_proc = subprocess.Popen(
                            [sys.executable, "-m", "streamlit", "run", app_file,
                             "--server.headless", "true",
                             "--server.port", str(port),
                             "--server.address", "localhost"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            env=child_env,
                        )
                        time.sleep(9)
                    if new_proc.poll() is None:
                        st.session_state["app_process"] = new_proc
                        st.session_state["app_port"] = port
                        st.rerun()
                    else:
                        _, err = new_proc.communicate()
                        st.error(
                            "Generated app failed to start.\n\n"
                            f"**Error:**\n```\n{err.strip() or '(no stderr)'}\n```"
                        )

        with col_tests2:
            test_file_ql = final_state.get("test_file") or "outputs/test_generated_app.py"
            can_test = pathlib.Path(test_file_ql).exists()
            if st.button("🧪 Run Tests", use_container_width=True, disabled=not can_test):
                with st.spinner("Running pytest..."):
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", test_file_ql,
                         "-v", "--tb=short", "--no-header"],
                        capture_output=True, text=True, timeout=120,
                    )
                out = result.stdout
                if result.stderr.strip():
                    out += "\n" + result.stderr
                st.session_state["test_output"] = out

            test_out = st.session_state.get("test_output")
            if test_out:
                low = test_out.lower()
                if "failed" in low or "error" in low:
                    st.error("Some tests failed")
                else:
                    st.success("All tests passed")
                with st.expander("Test Output", expanded=True):
                    st.code(test_out, language="text")

    # ── Tab 4: Agent Log ───────────────────────────────────────────────────
    with tab_log:
        st.markdown(
            '<div style="font-size:0.78rem;color:#64748B;margin-bottom:1rem">'
            'Per-agent token usage, cost, and latency for this pipeline run.</div>',
            unsafe_allow_html=True,
        )
        _render_agent_log_table(final_state.get("agent_log") or [])

    # ── Tab 5: Observability ──────────────────────────────────────────────
    with tab_obs:
        # Summary metrics
        _render_pipeline_summary(final_state)

        sid = final_state.get("session_id", "N/A")
        _col_meta, _col_btn = st.columns([3, 1])
        with _col_meta:
            st.markdown(
                f'<div style="font-size:0.75rem;color:#94A3B8;margin-top:0.6rem">'
                f'Session <code style="background:#F1F5F9;padding:1px 6px;border-radius:4px">{sid}</code>'
                f' &nbsp;|&nbsp; Status <code style="background:#F1F5F9;padding:1px 6px;border-radius:4px">'
                f'{final_state.get("pipeline_status","N/A")}</code></div>',
                unsafe_allow_html=True,
            )
        with _col_btn:
            if _langfuse_configured() and sid != "N/A":
                st.link_button(
                    "📊 View in Langfuse",
                    url=_langfuse_session_url(sid),
                    use_container_width=True,
                )

        # Complexity
        cx = final_state.get("complexity_score")
        if cx is not None:
            st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748B;'
                'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.75rem">'
                'Complexity Analysis</div>',
                unsafe_allow_html=True,
            )
            cx_color = "#10B981" if cx <= 4 else ("#F59E0B" if cx <= 7 else "#EF4444")
            override = final_state.get("complexity_model_override")
            st.markdown(
                f"""
                <div class="stat-card" style="display:inline-flex;align-items:center;gap:1.5rem;width:100%">
                  <div>
                    <div class="label">Complexity</div>
                    <div class="value" style="color:{cx_color}">{cx}<span style="font-size:1rem;color:#94A3B8">/10</span></div>
                  </div>
                  <div style="flex:1;font-size:0.85rem;color:#475569">
                    {final_state.get('complexity_reason') or 'N/A'}
                    {"<br><span style='color:#6366F1;font-weight:600;font-size:0.78rem'>Light model used — tokens saved</span>" if override else ""}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Eval scores
        eval_scores = final_state.get("eval_scores")
        if eval_scores and "error" not in eval_scores:
            st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:0.7rem;font-weight:700;color:#64748B;'
                'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.75rem">'
                'Eval Scores</div>',
                unsafe_allow_html=True,
            )
            _render_eval_scores(eval_scores)
        elif eval_scores and "error" in eval_scores:
            st.warning(f"Eval failed: {eval_scores['error']}")

        st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)
        st.info(
            "Full traces with token-level breakdowns are visible in your "
            "Langfuse dashboard when LANGFUSE_PUBLIC_KEY / SECRET_KEY are set."
        )

    # ── Tab 6: History ────────────────────────────────────────────────────
    with tab_hist:
        _render_history_tab()


def _render_history_tab():
    import pandas as pd
    runs = load_history()

    if not runs:
        st.info("No pipeline runs recorded yet. Complete a run to see history here.")
        return

    # Aggregate metrics
    total_runs   = len(runs)
    success_cnt  = sum(1 for r in runs if r.get("status") == "complete")
    success_rate = success_cnt / total_runs * 100 if total_runs else 0
    agg_tokens   = sum(r.get("total_tokens") or 0 for r in runs)
    agg_cost     = sum(r.get("total_cost_usd") or 0.0 for r in runs)

    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric("Total Runs",   total_runs)
    hm2.metric("Success Rate", f"{success_rate:.0f}%")
    hm3.metric("Total Tokens", f"{agg_tokens:,}")
    hm4.metric("Total Cost",   f"${agg_cost:.4f}")

    st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)

    table_rows = [{
        "ID":         r["id"],
        "Prompt":     (r["prompt"] or "")[:60] + ("…" if len(r["prompt"] or "") > 60 else ""),
        "Status":     "✅ complete" if r.get("status") == "complete" else "❌ failed",
        "Tokens":     r.get("total_tokens") or 0,
        "Cost ($)":   round(r.get("total_cost_usd") or 0.0, 5),
        "Latency (s)":round((r.get("total_latency_ms") or 0) / 1000, 1),
        "Created At": r.get("created_at", ""),
    } for r in runs]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown('<hr class="light-divider"/>', unsafe_allow_html=True)
    run_labels = [
        f"#{r['id']} — {(r['prompt'] or '')[:45]}{'…' if len(r['prompt'] or '') > 45 else ''}"
        for r in runs
    ]
    sel_idx = st.selectbox(
        "Inspect a run",
        range(len(runs)),
        format_func=lambda i: run_labels[i],
        key="history_selected_run",
    )
    sel = runs[sel_idx]

    d1, d2, d3, d4, d5, d6 = st.tabs([
        "Prompt", "Functional Spec", "Technical Design",
        "Code", "Test Cases", "Metrics",
    ])
    with d1:
        st.text_area(
            "Original Prompt",
            value=sel.get("prompt") or "",
            height=140,
            disabled=True,
            key=f"hist_prompt_{sel['id']}",
        )
        refined = sel.get("refined_prompt") or ""
        if refined:
            with st.expander("Refined Prompt"):
                st.markdown(refined)
    with d2:
        st.markdown(sel.get("functional_spec") or "_Not available_")
    with d3:
        st.markdown(sel.get("technical_design") or "_Not available_")
    with d4:
        cv = sel.get("code") or ""
        if cv:
            st.code(cv, language="python", line_numbers=True)
        else:
            st.info("No code recorded for this run.")
    with d5:
        st.markdown(sel.get("test_cases") or "_Not available_")
    with d6:
        dm1, dm2, dm3 = st.columns(3)
        dm1.metric("Tokens",  f"{sel.get('total_tokens') or 0:,}")
        dm2.metric("Cost",    f"${sel.get('total_cost_usd') or 0.0:.5f}")
        dm3.metric("Latency", f"{(sel.get('total_latency_ms') or 0)/1000:.1f}s")
        agents_str = sel.get("agents_called") or ""
        if agents_str:
            st.caption(" → ".join(agents_str.split(",")))
        vl = "Passed" if sel.get("validation_passed") else "Failed"
        rl = "Approved" if sel.get("review_approved") else "Rejected"
        st.caption(f"Validation: {vl}  |  Review: {rl}")

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


# ── Main layout ───────────────────────────────────────────────────────────────

_render_sidebar()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-title">BMAD AI Pipeline</div>
    <div class="hero-sub">
      Describe any app in plain English — the pipeline writes, tests, and runs it for you.
    </div>
    <div class="pipeline-steps">
      <span class="step-pill">PromptRefiner</span><span class="step-arrow">›</span>
      <span class="step-pill">Planner</span><span class="step-arrow">›</span>
      <span class="step-pill">Architect</span><span class="step-arrow">›</span>
      <span class="step-pill">Developer</span><span class="step-arrow">›</span>
      <span class="step-pill">Validator ⟳</span><span class="step-arrow">›</span>
      <span class="step-pill">Reviewer ⟳</span><span class="step-arrow">›</span>
      <span class="step-pill">Executor</span><span class="step-arrow">›</span>
      <span class="step-pill">TestWriter</span><span class="step-arrow">›</span>
      <span class="step-pill">EvalAgent</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input section ─────────────────────────────────────────────────────────────
if st.session_state.get("rerun_prompt"):
    st.session_state["user_request_input"] = st.session_state.pop("rerun_prompt")
elif st.session_state.get("template_prompt"):
    st.session_state["user_request_input"] = st.session_state.pop("template_prompt")

col_input, col_actions = st.columns([4, 1])

with col_input:
    user_request = st.text_area(
        label="What would you like to build?",
        placeholder=(
            "e.g. Build a task management app with add / delete / list tasks. "
            "Each task has a title, priority, and due date."
        ),
        height=110,
        disabled=st.session_state["running"],
        key="user_request_input",
    )

with col_actions:
    st.markdown(
        '<div style="margin-top:1.85rem"></div>',
        unsafe_allow_html=True,
    )
    run_clicked = st.button(
        "⚡ Run Pipeline",
        type="primary",
        disabled=st.session_state["running"] or not (user_request or "").strip(),
        use_container_width=True,
    )
    reset_clicked = st.button(
        "↺ Reset",
        disabled=st.session_state["running"],
        use_container_width=True,
    )

# ── Reset ─────────────────────────────────────────────────────────────────────
if reset_clicked:
    proc = st.session_state.get("app_process")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    for k in ["running", "final_state", "error", "start_time",
              "app_process", "app_port", "test_output", "rerun_prompt"]:
        st.session_state[k] = None if k not in ("running",) else False
    st.session_state["running"] = False
    st.session_state["result_queue"] = queue.Queue(maxsize=1)
    st.rerun()

# ── Launch pipeline ───────────────────────────────────────────────────────────
if run_clicked and (user_request or "").strip():
    from core.guardrails import check_prompt, format_rejection
    _safe, _reason = check_prompt(user_request.strip())
    if not _safe:
        st.error(format_rejection(_reason))
        st.stop()

    while not st.session_state["result_queue"].empty():
        try:
            st.session_state["result_queue"].get_nowait()
        except queue.Empty:
            break

    st.session_state["running"]     = True
    st.session_state["final_state"] = None
    st.session_state["error"]       = None
    st.session_state["start_time"]  = time.time()

    worker = threading.Thread(
        target=_run_pipeline_worker,
        args=(user_request.strip(), st.session_state["result_queue"],
              st.session_state.get("project_mode", "streamlit_crud")),
        daemon=True,
    )
    worker.start()
    st.rerun()

# ── Poll for completion ───────────────────────────────────────────────────────
if st.session_state["running"]:
    try:
        result = st.session_state["result_queue"].get_nowait()
        final_state, error = result
        st.session_state["running"]     = False
        st.session_state["final_state"] = final_state
        st.session_state["error"]       = error
        if final_state:
            final_state["pipeline_status"] = "complete"
        st.rerun()
    except queue.Empty:
        elapsed = time.time() - (st.session_state["start_time"] or time.time())
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;
                        padding:1.2rem 1.6rem;margin-top:1rem;
                        box-shadow:0 1px 4px rgba(15,23,42,0.05);
                        display:flex;align-items:center;gap:1rem">
              <span class="pulse-dot"></span>
              <div>
                <div style="font-weight:700;color:#0F172A;font-size:0.95rem">Pipeline running…</div>
                <div style="font-size:0.78rem;color:#64748B;margin-top:2px">
                  {elapsed:.0f}s elapsed &nbsp;·&nbsp;
                  Agents are writing, testing, and reviewing your app
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(2)
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["error"]:
    st.error(f"Pipeline failed: {st.session_state['error']}")

if st.session_state["final_state"]:
    elapsed_total = time.time() - (st.session_state.get("start_time") or time.time())
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#ECFDF5,#D1FAE5);
                    border:1px solid #6EE7B7;border-radius:14px;
                    padding:0.85rem 1.4rem;margin:1rem 0;
                    display:flex;align-items:center;gap:0.75rem">
          <span style="font-size:1.3rem">✅</span>
          <div>
            <div style="font-weight:700;color:#065F46">Pipeline complete</div>
            <div style="font-size:0.78rem;color:#047857">Finished in {elapsed_total:.1f}s</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_results(st.session_state["final_state"])

elif not st.session_state["running"] and not st.session_state["error"]:
    st.markdown(
        """
        <div style="background:#FFFFFF;border:1px dashed #CBD5E1;border-radius:16px;
                    padding:3rem 2rem;text-align:center;margin-top:1.5rem">
          <div style="font-size:2.5rem;margin-bottom:0.75rem">⚡</div>
          <div style="font-size:1.05rem;font-weight:700;color:#0F172A;margin-bottom:0.4rem">
            Ready to build
          </div>
          <div style="font-size:0.85rem;color:#64748B;max-width:420px;margin:0 auto">
            Enter a description above and click <strong>Run Pipeline</strong>.
            The AI will plan, architect, code, test, and execute your app automatically.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Agent loop auto-refresh ───────────────────────────────────────────────────
# If the agent loop is actively running (or a run is queued) AND the main
# pipeline is idle, poll every 3 seconds so the sidebar badge updates.
if (
    st.session_state.get("_loop_needs_refresh")
    and not st.session_state["running"]
):
    time.sleep(3)
    st.rerun()
