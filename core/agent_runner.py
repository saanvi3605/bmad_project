"""
core/agent_runner.py
────────────────────────────────────────────────────────────────────────────────
The single shared LLM invocation layer for every LLM-calling agent in the
BMAD pipeline.

Responsibilities
  1. Hold the single ChatGroq instance via a LAZY getter (_get_llm).
     The LLM is NOT instantiated at import time — only on the first call.
     This ensures tests that monkeypatch build_llm or set ar.llm = MagicMock()
     can do so before any LLM creation, even without GROQ_API_KEY set.
  2. Accept a filled prompt string + BMADState, invoke the LLM, and return
     the response content string.
  3. Extract token counts from the LangChain response and delegate to
     observability.log_agent_call() for local bookkeeping.
  4. Attach the Langfuse CallbackHandler stored in state["langfuse_handler"]
     so every call is automatically traced without each agent knowing about
     Langfuse at all.
  5. Expose a thin helper extract_tokens() so the exact extraction logic
     is tested in one place.

Test monkeypatching
  Tests can replace the module-level _llm variable OR monkeypatch build_llm:

      import core.agent_runner as ar
      ar._llm = MagicMock(...)

  Or pre-empt build_llm before import:

      from unittest.mock import patch, MagicMock
      with patch("core.llm_factory.build_llm", return_value=MagicMock()):
          import core.agent_runner
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

from core.llm_factory import (
    build_llm, build_llm_light, build_llm_for_config,
    get_llm_config, get_llm_light_config,
    clear_config_cache,
)
from core.observability import log_agent_call

# ---------------------------------------------------------------------------
# Lazy singletons — built on first call, NOT at import time.
# _llm_model / _llm_light_model track which model name each singleton was
# built with so we can detect stale instances when config changes mid-session.
# ---------------------------------------------------------------------------

_llm: Optional[Any] = None
_llm_light: Optional[Any] = None
_llm_model: Optional[str] = None
_llm_light_model: Optional[str] = None
# Cache for override LLMs keyed by model name (set by ComplexityScorer)
_llm_overrides: dict[str, Any] = {}


def _get_llm_for_override(model_name: str) -> Any:
    """Return a cached LLM instance for a complexity-scorer model override."""
    global _llm_overrides
    if model_name not in _llm_overrides:
        # Build from the light config but swap the model name so all other
        # settings (temperature, max_tokens, provider) remain consistent.
        base_cfg = dict(get_llm_light_config())
        base_cfg["model"] = model_name
        _llm_overrides[model_name] = build_llm_for_config(base_cfg)
        print(f"[LLM] Override model loaded: {model_name}")
    return _llm_overrides[model_name]


def _get_llm(light: bool = False) -> Any:
    global _llm, _llm_light, _llm_model, _llm_light_model
    if light:
        expected = get_llm_light_config()["model"]
        if _llm_light is not None and _llm_light_model != expected:
            print(f"[LLM] Light model config changed ({_llm_light_model} -> {expected}), rebuilding...")
            _llm_light = None
        if _llm_light is None:
            _llm_light = build_llm_light()
            _llm_light_model = expected
            print(f"[LLM] Light model loaded: {expected}")
        return _llm_light
    else:
        expected = get_llm_config()["model"]
        if _llm is not None and _llm_model != expected:
            print(f"[LLM] Heavy model config changed ({_llm_model} -> {expected}), rebuilding...")
            _llm = None
        if _llm is None:
            _llm = build_llm()
            _llm_model = expected
            print(f"[LLM] Heavy model loaded: {expected}")
        return _llm


def reset_llm_singletons() -> None:
    """
    Call once at the start of every pipeline run.

    Busts the models.yaml lru_cache and destroys both LLM singletons so the
    next _get_llm() call re-reads the config file and builds fresh instances.
    This ensures that config changes made between runs (e.g. switching models
    in models.yaml while the Streamlit UI is open) are always picked up.
    Also clears any complexity-scorer override LLMs from the previous run.
    """
    global _llm, _llm_light, _llm_model, _llm_light_model, _LLM_CFG, _llm_overrides
    clear_config_cache()
    _LLM_CFG = {}
    _llm = None
    _llm_light = None
    _llm_model = None
    _llm_light_model = None
    _llm_overrides = {}


# ---------------------------------------------------------------------------
# Token extraction helper
# ---------------------------------------------------------------------------


def extract_tokens(response: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LangChain AIMessage.

    LangChain / langchain-groq stores usage data in one of two places
    depending on the langchain-core version:

      response.usage_metadata           (langchain-core >= 0.2)
        {"input_tokens": N, "output_tokens": M, "total_tokens": N+M}

      response.response_metadata["token_usage"]  (older versions / fallback)
        {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}

    Returns (0, 0) if neither location is found — never raises.  A warning
    is emitted so missing usage data is visible in logs without crashing the
    pipeline.
    """
    # Attempt 1: usage_metadata (preferred, langchain-core >= 0.2)
    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        inp = usage_meta.get("input_tokens", 0)
        out = usage_meta.get("output_tokens", 0)
        if isinstance(inp, int) and isinstance(out, int):
            return inp, out

    # Attempt 2: response_metadata.token_usage (older langchain-groq)
    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage", {})
        if isinstance(token_usage, dict):
            inp = token_usage.get("prompt_tokens", 0)
            out = token_usage.get("completion_tokens", 0)
            if isinstance(inp, int) and isinstance(out, int):
                return inp, out

    warnings.warn(
        "Could not extract token counts from LLM response. "
        "Cost and token logging for this call will show 0.",
        RuntimeWarning,
        stacklevel=3,
    )
    return 0, 0


# ---------------------------------------------------------------------------
# Core invocation
# ---------------------------------------------------------------------------


def run_agent(
    prompt: str,
    agent_name: str,
    prompt_key: str,
    state: dict[str, Any],
    light: bool = False,
) -> str:
    """
    Invoke the shared LLM with ``prompt`` and return the response content.

    This is the ONLY function that calls ``llm.invoke()``.  Every LLM-calling
    agent_impl calls this function instead of holding its own LLM instance.

    Parameters
    ----------
    prompt       Fully rendered prompt string ready for the LLM.
    agent_name   Human-readable agent identifier for logging, e.g. "developer".
    prompt_key   Registry key of the prompt variant, e.g. "developer_clean_v1".
    state        BMADState dict.  Must already contain "langfuse_handler" if
                 Langfuse tracing is desired.

    Returns
    -------
    str  The LLM's response as a plain string (response.content).

    Raises
    ------
    Re-raises any exception from the LLM call unchanged.
    """
    # If ComplexityScorer set an override and this is a heavy-model call, use it.
    override_model = state.get("complexity_model_override") if not light else None
    if override_model:
        llm_instance = _get_llm_for_override(override_model)
        model_name = override_model
    else:
        llm_instance = _get_llm(light=light)
        model_name = (get_llm_light_config() if light else get_llm_config())["model"]

    # Build the Langfuse callback list from state so Langfuse auto-instruments
    # this LLM call and nests it under the current LangGraph node span.
    lf_handler = state.get("langfuse_handler")
    callbacks = (
        [lf_handler]
        if lf_handler is not None and hasattr(lf_handler, "on_llm_start")
        else []
    )

    t_start = time.perf_counter()
    if callbacks:
        response = llm_instance.invoke(prompt, config={"callbacks": callbacks})
    else:
        response = llm_instance.invoke(prompt)
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    input_tokens, output_tokens = extract_tokens(response)

    log_agent_call(
        state=state,
        agent_name=agent_name,
        prompt_key=prompt_key,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model=model_name,
    )

    return response.content


# ---------------------------------------------------------------------------
# State initialisation helper
# ---------------------------------------------------------------------------


def build_initial_state(
    user_request: str,
    session_id: str,
    langfuse_handler: Optional[Any],
) -> dict[str, Any]:
    """
    Construct the initial BMADState dict for a new pipeline run.

    This centralises the field list so streamlit_app.py and main.py
    do not each need to know every field name.
    """
    return {
        # Original 15 fields — preserved exactly from AgentState
        "user_request": user_request,
        "functional_spec": None,
        "technical_design": None,
        "code": None,
        "test_cases": None,
        "review_feedback": None,
        "review_approved": None,
        "review_attempts": 0,
        "validation_passed": None,
        "validation_error": None,
        "validation_attempts": 0,
        "execution_result": None,
        "execution_error": None,
        "langfuse_handler": langfuse_handler,
        "test_file": None,
        # New BMAD fields
        "session_id": session_id,
        "pipeline_status": "running",
        "agent_log": [],
        "output_dir": None,
        "refined_prompt": None,
        # Complexity Scorer fields
        "complexity_score": None,
        "complexity_reason": None,
        "complexity_model_override": None,
    }
