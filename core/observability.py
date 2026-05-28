"""
core/observability.py
────────────────────────────────────────────────────────────────────────────────
Observability layer: per-pipeline session IDs, per-agent token/cost logging,
and Langfuse tracing via the Langfuse 4.x SDK directly.

How it works
  1. create_session() creates a Langfuse trace_id and returns (session_id, lf_context)
     where lf_context is:
       {"trace_id": "...", "session_id": "...", "trace_name": "..."}

  2. log_langfuse_generation() is called by agent_runner.run_agent() after every
     LiteLLM call.  It logs a generation directly via lf.start_observation()
     with real token counts, model name, input, and output — fixing the 0-token
     problem that plagued the old CallbackHandler approach.

  3. flush() calls lf.flush() to ship all pending events before the process exits.

Why direct SDK instead of LiteLLM's built-in "langfuse" callback
  LiteLLM 1.85.x is incompatible with both Langfuse 3.x and 4.x:
    - 3.x: TypeError: unexpected keyword argument 'sdk_integration'
    - 4.x: AttributeError: module 'langfuse' has no attribute 'version'
  Using the Langfuse 4.x SDK directly avoids this entirely and gives us
  full control over what appears in the dashboard.

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
# Singleton Langfuse client
# ---------------------------------------------------------------------------

_lf_client: Optional[Any] = None


def get_lf_client() -> Optional[Any]:
    """Return the process-level Langfuse singleton, creating it on first call."""
    global _lf_client
    if _lf_client is None:
        if not _ensure_env():
            return None
        try:
            from langfuse import Langfuse  # type: ignore[import]
            _lf_client = Langfuse()
        except Exception as exc:
            warnings.warn(
                f"Failed to initialise Langfuse client: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
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
) -> tuple[str, Optional[dict[str, str]]]:
    """
    Generate a new pipeline session and a Langfuse trace context.

    Returns
    -------
    session_id : str
        UUID4 stored in BMADState["session_id"].
    lf_context : dict | None
        Stored in BMADState["langfuse_handler"].  agent_runner.run_agent()
        passes this to log_langfuse_generation() after every LLM call.
        None if Langfuse keys are missing or the SDK is unavailable.
    """
    session_id = str(uuid.uuid4())

    lf = get_lf_client()
    if lf is None:
        return session_id, None

    try:
        trace_id = str(lf.create_trace_id())
        lf_context: dict[str, str] = {
            "trace_id": trace_id,
            "session_id": session_id,
            "trace_name": run_name,
        }
        print(f"[Langfuse] Session {session_id[:8]}… | trace {trace_id[:8]}…")
        return session_id, lf_context

    except Exception as exc:
        warnings.warn(
            f"Failed to create Langfuse trace: {exc}. Continuing without tracing.",
            RuntimeWarning,
            stacklevel=2,
        )
        return session_id, None


# ---------------------------------------------------------------------------
# Direct Langfuse generation logging  (called by agent_runner after each LLM call)
# ---------------------------------------------------------------------------


def log_langfuse_generation(
    lf_ctx: Optional[dict[str, str]],
    agent_name: str,
    model: str,
    messages: list[dict[str, str]],
    output: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """
    Log one LLM call as a Langfuse generation using the 4.x SDK directly.

    This fixes the 0-token problem: token counts are extracted from the
    LiteLLM response and passed explicitly — no LiteLLM callback needed.

    Parameters
    ----------
    lf_ctx      Dict returned by create_session(), or None to skip logging.
    agent_name  Name shown in the Langfuse dashboard (e.g. "developer").
    model       LiteLLM model string (e.g. "groq/llama-3.1-8b-instant").
    messages    The messages list sent to the LLM.
    output      The text content returned by the LLM.
    input_tokens / output_tokens  Token counts from the LiteLLM response.
    latency_ms  Wall-clock time for the LLM call in milliseconds.
    """
    if lf_ctx is None:
        return

    lf = get_lf_client()
    if lf is None:
        return

    try:
        # TraceContext tells Langfuse which trace this generation belongs to.
        trace_context = {
            "trace_id": lf_ctx["trace_id"],
            "session_id": lf_ctx["session_id"],
        }

        gen = lf.start_observation(
            trace_context=trace_context,       # type: ignore[arg-type]
            name=agent_name,
            as_type="generation",
            model=model,
            input=messages,
            output=output,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            metadata={
                "latency_ms": round(latency_ms, 1),
                "session_id": lf_ctx["session_id"],
                "trace_name": lf_ctx.get("trace_name", "BMAD Pipeline"),
            },
        )
        gen.end()

    except Exception as exc:
        warnings.warn(
            f"[Langfuse] Failed to log generation for '{agent_name}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Per-agent local logging
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
    Flush all pending Langfuse events so they appear in the dashboard
    before the process exits.  Safe to call even when Langfuse is unconfigured.
    """
    lf = get_lf_client()
    if lf is None:
        return

    print("[Langfuse] Flushing...")
    try:
        lf.flush()
        print("[Langfuse] Flush complete.")
    except Exception as exc:
        print(f"[Langfuse] Flush warning: {exc}")
