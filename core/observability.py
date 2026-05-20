"""
core/observability.py
────────────────────────────────────────────────────────────────────────────────
Replaces langfuse_setup.py with a richer observability layer that adds:

  • Per-pipeline session IDs (UUID4) so every run is uniquely identifiable
    in the Langfuse dashboard even when the run_name is the same.
  • Per-agent token and cost logging written into BMADState["agent_log"].
  • A cost estimator that uses the Groq token rates from models.yaml.
  • A pipeline-level summary aggregator for the Streamlit UI.
  • A safe flush() that uses the direct Langfuse client (not
    langfuse_context, which has no flush() in Langfuse v4.x).

Langfuse version compatibility
  This module targets langfuse>=4.0.0, which ships the CallbackHandler at
  langfuse.langchain.CallbackHandler.  The direct client is langfuse.Langfuse.
  Both are instantiated lazily so import errors only surface when the feature
  is actually used, not at module import time.

Environment variables (set in .env):
  LANGFUSE_PUBLIC_KEY
  LANGFUSE_SECRET_KEY
  LANGFUSE_HOST          (default: https://cloud.langfuse.com)

All functions are safe to call even if Langfuse is disabled or
the env vars are absent — they log a warning and return a no-op object.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from dotenv import load_dotenv

from core.llm_factory import get_observability_config

if TYPE_CHECKING:
    # Avoid hard import at module level so the whole codebase does not break
    # if langfuse is uninstalled.
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

# ---------------------------------------------------------------------------
# Singleton Langfuse client
# ---------------------------------------------------------------------------

_lf_client: Optional[Any] = None


def get_lf_client() -> Optional[Any]:
    """
    Return the process-level Langfuse singleton, creating it on first call.

    Using a singleton instead of Langfuse() per agent call prevents two bugs:
      1. Each Langfuse() instance starts its own background flush thread.
         When the local variable goes out of scope, GC can kill the thread
         mid-flight, silently dropping events.
      2. observability.flush() was creating a fresh Langfuse() with no
         pending events, making the final flush a no-op.
    """
    global _lf_client
    if _lf_client is None:
        if not _ensure_env():
            return None
        try:
            from langfuse import Langfuse  # type: ignore[import]
            _lf_client = Langfuse()
        except Exception:
            pass
    return _lf_client


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
    # Ensure LANGFUSE_HOST is exported so the SDK picks it up.
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

    Appended to BMADState["agent_log"] by ``log_agent_call()``.
    Serialisable to dict via ``to_dict()`` for JSON archiving.
    """

    agent_name: str
    timestamp_utc: float          # time.time() at call completion
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float               # estimated; 0.0 if track_cost is False
    latency_ms: float             # wall-clock ms for the LLM round-trip
    model: str
    prompt_key: str               # e.g. "developer_clean_v1"

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
) -> tuple[str, Optional[str]]:
    """
    Generate a new pipeline session and open a Langfuse trace for it.

    Returns
    -------
    session_id : str
        UUID4 string stored in BMADState["session_id"].
    trace_id : str | None
        Langfuse trace ID.  Stored in BMADState["langfuse_handler"] and
        passed to every _trace_langfuse() call so all agent generations are
        grouped under one trace in the dashboard.  None if Langfuse is
        unconfigured.
    """
    session_id = str(uuid.uuid4())

    lf = get_lf_client()
    if lf is None:
        return session_id, None

    try:
        trace = lf.trace(name=run_name, session_id=session_id)
        return session_id, trace.id
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to create Langfuse trace: {exc}. Continuing without tracing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return session_id, None


# ---------------------------------------------------------------------------
# Per-agent logging
# ---------------------------------------------------------------------------


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    obs_cfg: dict[str, Any],
) -> float:
    """
    Compute estimated USD cost from token counts using rates in models.yaml.

    Cost formula:
      cost = (input_tokens / 1_000_000 * cost_per_million_input)
           + (output_tokens / 1_000_000 * cost_per_million_output)
    """
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

    This function is called by agent_runner.py immediately after every
    successful LLM invocation.  The Langfuse CallbackHandler already handles
    uploading the raw trace; this function is purely for local bookkeeping
    so the Streamlit UI can display per-agent cost/token summaries without
    hitting the Langfuse API.

    Parameters
    ----------
    state        BMADState dict (mutated: agent_log is extended).
    agent_name   Human-readable agent identifier, e.g. "developer".
    prompt_key   Registry key of the prompt that was used, e.g. "developer_clean_v1".
    input_tokens Token count for the prompt (from response.usage_metadata).
    output_tokens Token count for the completion.
    latency_ms   Wall-clock round-trip time for llm.invoke().
    model        Model string used, e.g. "llama-3.3-70b-versatile".

    Returns
    -------
    AgentCallRecord — also appended to state["agent_log"].
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

    # Initialise the list if this is the first agent in the run.
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
    PipelineSummary.

    Safe to call even if agent_log is None or empty — returns zeroed summary.

    Used by:
      - apps/pipeline_ui.py  (renders cost/token summary cards)
      - main.py              (prints summary to console on pipeline complete)
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
    Flush all pending Langfuse events to the cloud using the singleton client.

    The ``handler`` parameter is kept for call-site compatibility but is
    ignored — the singleton already holds all pending events.

    Safe to call even when Langfuse is unconfigured.
    """
    lf = get_lf_client()
    if lf is None:
        return
    try:
        lf.flush()
        print("[Langfuse] Final flush complete")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[Langfuse] Flush warning: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
