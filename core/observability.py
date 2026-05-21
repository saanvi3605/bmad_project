"""
core/observability.py
────────────────────────────────────────────────────────────────────────────────
Observability layer: per-pipeline session IDs, per-agent token/cost logging,
and Langfuse tracing.

Tracing strategy — LangChain CallbackHandler (v2 + v3 compatible)
  create_session() builds a langfuse.callback.CallbackHandler and returns it
  as langfuse_handler.  Callers pass it to:
    • app.invoke(state, config={"callbacks": [handler]})
        → Langfuse auto-creates spans for every LangGraph node, producing
          the pipeline graph visualisation in the Langfuse dashboard.
    • llm.invoke(prompt, config={"callbacks": [handler]})
        → Langfuse auto-creates a nested ChatGroq generation with token
          counts and latency inside the node span above.

  flush(handler) ends and ships all pending events.  The handler is reset
  to None afterwards so the next run starts with a clean exporter.

Environment variables (set in .env):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_HOST          (default: https://cloud.langfuse.com)

All functions are safe to call even if Langfuse is disabled — they warn and no-op.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

from core.llm_factory import get_observability_config

# ---------------------------------------------------------------------------
# Per-run CallbackHandler singleton
# ---------------------------------------------------------------------------

_langfuse_handler: Optional[Any] = None  # langfuse.callback.CallbackHandler


def get_langfuse_handler() -> Optional[Any]:
    """Return the CallbackHandler for the current pipeline run, or None."""
    return _langfuse_handler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_env() -> bool:
    """
    Load .env and confirm the minimum Langfuse env vars are present.
    Returns True if Langfuse is properly configured, False otherwise.
    """
    load_dotenv()
    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec:
        warnings.warn(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set. "
            "Langfuse tracing is disabled for this run.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    os.environ["LANGFUSE_HOST"] = host
    return True


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentCallRecord:
    """
    Immutable record of a single agent's LLM invocation.
    Appended to BMADState["agent_log"] by log_agent_call().
    """

    agent_name: str
    timestamp_utc: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    prompt_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "timestamp_utc": self.timestamp_utc,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "prompt_key": self.prompt_key,
        }


@dataclass
class PipelineSummary:
    """Aggregated metrics across all agents in one pipeline run."""

    session_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    agent_count: int = 0
    agents_called: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "agent_count": self.agent_count,
            "agents_called": self.agents_called,
        }


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def create_session(
    run_name: str = "BMAD Pipeline Run",
) -> tuple[str, Optional[Any]]:
    """
    Generate a new pipeline session and build a Langfuse CallbackHandler for it.

    The CallbackHandler is the correct integration point for LangChain/LangGraph:
      • Pass to app.invoke() via config={"callbacks": [handler]}
          → Langfuse auto-instruments every LangGraph node, producing the
            pipeline graph visualisation (PromptRefiner → Planner → … → END).
      • Pass to llm.invoke() via config={"callbacks": [handler]}
          → Each ChatGroq call becomes a nested generation with token counts.

    Returns
    -------
    session_id : str
        UUID4 stored in BMADState["session_id"].
    handler : CallbackHandler | None
        Stored in BMADState["langfuse_handler"].
        None if Langfuse keys are missing or the SDK is unavailable.
    """
    global _langfuse_handler

    session_id = str(uuid.uuid4())

    if not _ensure_env():
        _langfuse_handler = None
        return session_id, None

    try:
        # Try the standard import path first; fall back to the older alias.
        try:
            from langfuse.callback import CallbackHandler  # type: ignore[import]
        except ImportError:
            from langfuse.langchain import CallbackHandler  # type: ignore[import]

        handler = CallbackHandler(
            session_id=session_id,
            trace_name=run_name,
        )
        _langfuse_handler = handler
        print(f"[Langfuse] CallbackHandler ready (session {session_id[:8]}…)")
        return session_id, handler

    except Exception as exc:
        warnings.warn(
            f"Failed to create Langfuse CallbackHandler: {exc}. "
            "Continuing without tracing.",
            RuntimeWarning,
            stacklevel=2,
        )
        _langfuse_handler = None
        return session_id, None


# ---------------------------------------------------------------------------
# Per-agent logging
# ---------------------------------------------------------------------------


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    obs_cfg: dict[str, Any],
) -> float:
    if not obs_cfg.get("track_cost", True):
        return 0.0
    rate_in = float(obs_cfg.get("cost_per_million_input_tokens", 0.59))
    rate_out = float(obs_cfg.get("cost_per_million_output_tokens", 0.79))
    return (input_tokens / 1_000_000 * rate_in) + (output_tokens / 1_000_000 * rate_out)


def log_agent_call(
    state: dict[str, Any],
    agent_name: str,
    prompt_key: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    model: str,
) -> AgentCallRecord:
    """
    Build an AgentCallRecord and append it to state["agent_log"].
    Called by agent_runner.py after every successful LLM invocation.
    """
    obs_cfg = get_observability_config()
    total = input_tokens + output_tokens
    cost = _estimate_cost(input_tokens, output_tokens, obs_cfg)

    record = AgentCallRecord(
        agent_name=agent_name,
        timestamp_utc=time.time(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cost_usd=cost,
        latency_ms=latency_ms,
        model=model,
        prompt_key=prompt_key,
    )

    if state.get("agent_log") is None:
        state["agent_log"] = []

    state["agent_log"].append(record)
    return record


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------


def get_pipeline_summary(state: dict[str, Any]) -> PipelineSummary:
    """
    Aggregate all AgentCallRecord entries in state["agent_log"] into a
    PipelineSummary. Safe to call even if agent_log is None or empty.
    """
    session_id = state.get("session_id", "unknown")
    summary = PipelineSummary(session_id=session_id)

    records: list[AgentCallRecord] = state.get("agent_log") or []
    for rec in records:
        summary.total_input_tokens += rec.input_tokens
        summary.total_output_tokens += rec.output_tokens
        summary.total_tokens += rec.total_tokens
        summary.total_cost_usd += rec.cost_usd
        summary.total_latency_ms += rec.latency_ms
        summary.agent_count += 1
        summary.agents_called.append(rec.agent_name)

    return summary


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


def flush(handler: Optional[Any] = None) -> None:
    """
    Flush all pending Langfuse events to the cloud.

    Must be called in a finally block after every pipeline run so traces
    appear in the Langfuse dashboard immediately.

    Parameters
    ----------
    handler : CallbackHandler | None
        Explicit handler to flush.  Falls back to the module-level
        _langfuse_handler if not provided (covers the Streamlit UI path
        where the handler is not passed to the call site).

    Safe to call even when Langfuse is unconfigured.
    """
    global _langfuse_handler

    h = handler or _langfuse_handler
    if h is None:
        return

    print("[Langfuse] Flushing...")
    try:
        h.flush()
        print("[Langfuse] Flush complete")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[Langfuse] Flush warning: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    finally:
        # Reset so the next pipeline run starts with a fresh handler and
        # exporter thread — avoids the drained-processor silent-drop bug.
        _langfuse_handler = None
