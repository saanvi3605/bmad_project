"""
core/agent_runner.py
────────────────────────────────────────────────────────────────────────────────
The single shared LLM invocation layer for every LLM-calling agent in the
BMAD pipeline.  Uses LiteLLM for provider-agnostic LLM calls with native
Langfuse observability.

Responsibilities
  1. Hold a lazy singleton LiteLLM config dict (_llm) — built on first call.
     NOT instantiated at import time so tests can inject mocks freely.
  2. Accept a filled prompt string + BMADState, call litellm.completion(),
     and return the response content string.
  3. Extract token counts from the LiteLLM response and delegate to
     observability.log_agent_call() for local bookkeeping.
  4. Pass Langfuse trace context (trace_id, session_id, generation_name)
     as LiteLLM metadata so every call is automatically observed without
     each agent knowing about Langfuse at all.
  5. Expose _call_litellm() as a thin wrapper that tests can monkeypatch.

Test monkeypatching
  Patch the LiteLLM call wrapper:

      from unittest.mock import patch, MagicMock
      mock_resp = MagicMock()
      mock_resp.choices[0].message.content = "mocked"
      mock_resp.usage.prompt_tokens = 10
      mock_resp.usage.completion_tokens = 20
      with patch("core.agent_runner._call_litellm", return_value=mock_resp):
          ...

  Or keep the old _llm singleton approach for lazy-init tests:

      import core.agent_runner as ar
      ar._llm = {"model": "groq/mock", "temperature": 0, "max_tokens": 1, "timeout": 1}
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any, Optional

from core.llm_factory import (
    build_llm, build_llm_light, build_llm_for_config,
    get_llm_config, get_llm_light_config,
    clear_config_cache,
)
from core.observability import log_agent_call, log_langfuse_generation

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
# LiteLLM call wrapper  (monkeypatch this in tests)
# ---------------------------------------------------------------------------


_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_FALLBACK_WAIT = 10.0  # seconds to wait when Groq doesn't tell us how long


def _parse_retry_after(error_message: str) -> float:
    """
    Extract the suggested wait time from a Groq rate-limit error message.

    Groq returns:  "Please try again in 4.189999999s."
    Falls back to _RATE_LIMIT_FALLBACK_WAIT if the pattern isn't found.
    """
    m = re.search(r'try again in ([\d.]+)s', error_message, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0   # add 2s buffer
    return _RATE_LIMIT_FALLBACK_WAIT


def _call_litellm(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    metadata: Optional[dict[str, Any]],
) -> Any:
    """
    Thin wrapper around litellm.completion — the only place LiteLLM is called.

    Includes automatic retry-with-backoff for RateLimitError.  Groq's free
    tier has a 6 000 TPM limit; multiple light-model agents in the same
    pipeline run can exceed it within the same minute window.  Rather than
    crashing the whole pipeline we wait the time Groq itself suggests and
    retry, up to _RATE_LIMIT_MAX_RETRIES times.

    Keeping it as a named function lets tests patch just this one symbol:

        with patch("core.agent_runner._call_litellm", return_value=mock_resp):
            ...
    """
    import litellm  # lazy import so missing litellm doesn't break imports

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if metadata:
        kwargs["metadata"] = metadata

    for attempt in range(1, _RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return litellm.completion(**kwargs)
        except litellm.RateLimitError as exc:
            if attempt == _RATE_LIMIT_MAX_RETRIES:
                raise   # exhausted all retries — propagate
            wait = _parse_retry_after(str(exc))
            print(
                f"\n  [LLM] Rate limit hit on {model} "
                f"(attempt {attempt}/{_RATE_LIMIT_MAX_RETRIES - 1}). "
                f"Waiting {wait:.1f}s before retry..."
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Token extraction helpers
# ---------------------------------------------------------------------------


def _extract_litellm_tokens(response: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LiteLLM ModelResponse.

    LiteLLM always returns an OpenAI-compatible response object:
        response.usage.prompt_tokens
        response.usage.completion_tokens
    """
    try:
        usage = response.usage
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        return int(inp), int(out)
    except Exception:
        warnings.warn(
            "Could not extract token counts from LiteLLM response. "
            "Cost and token logging for this call will show 0.",
            RuntimeWarning,
            stacklevel=3,
        )
        return 0, 0


def extract_tokens(response: Any) -> tuple[int, int]:
    """
    Extract (input_tokens, output_tokens) from a LangChain AIMessage.

    Kept for backwards compatibility with existing tests.  New code should
    use _extract_litellm_tokens() instead.

    LangChain stores usage data in one of two places:
      response.usage_metadata           (langchain-core >= 0.2)
      response.response_metadata["token_usage"]  (older versions / fallback)

    Returns (0, 0) if neither location is found — never raises.
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

    This is the ONLY function that calls the LLM.  Every LLM-calling
    agent_impl calls this function instead of holding its own LLM instance.

    Parameters
    ----------
    prompt       Fully rendered prompt string ready for the LLM.
    agent_name   Human-readable agent identifier for logging, e.g. "developer".
    prompt_key   Registry key of the prompt variant, e.g. "developer_clean_v1".
    state        BMADState dict.  Must already contain "langfuse_handler" (the
                 Langfuse context dict) if tracing is desired.
    light        If True, use the light model (llm_light: section).

    Returns
    -------
    str  The LLM's response as a plain string.

    Raises
    ------
    Re-raises any exception from the LLM call unchanged.
    """
    # Resolve which config to use
    override_model = state.get("complexity_model_override") if not light else None
    if override_model:
        llm_cfg = _get_llm_for_override(override_model)
        model_name = override_model
    else:
        llm_cfg = _get_llm(light=light)
        model_name = (get_llm_light_config() if light else get_llm_config())["model"]

    # state["langfuse_handler"] is a dict from observability.create_session():
    #   {"trace_id": "...", "session_id": "...", "trace_name": "..."}
    lf_ctx = state.get("langfuse_handler") if isinstance(state.get("langfuse_handler"), dict) else None

    messages = [{"role": "user", "content": prompt}]

    t_start = time.perf_counter()
    response = _call_litellm(
        model=llm_cfg["model"],
        messages=messages,
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        timeout=llm_cfg.get("timeout", 120),
        metadata=None,  # Langfuse is logged directly via log_langfuse_generation()
    )
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    input_tokens, output_tokens = _extract_litellm_tokens(response)
    content = response.choices[0].message.content

    # Log to Langfuse directly using the 4.x SDK — bypasses LiteLLM's
    # broken built-in Langfuse callback and guarantees real token counts.
    log_langfuse_generation(
        lf_ctx=lf_ctx,
        agent_name=agent_name,
        model=llm_cfg["model"],
        messages=messages,
        output=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )

    log_agent_call(
        state=state,
        agent_name=agent_name,
        prompt_key=prompt_key,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model=model_name,
    )

    return content


# ---------------------------------------------------------------------------
# State initialisation helper
# ---------------------------------------------------------------------------


def build_initial_state(
    user_request: str,
    session_id: str,
    langfuse_handler: Optional[Any],
    project_mode: str = "streamlit_crud",
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
        # Self-healing Executor fields
        "runtime_error": None,
        "runtime_fix_attempts": 0,
        # ReadmeWriter fields
        "readme_file": None,
        # EvalAgent fields
        "eval_scores": None,
        # Project mode
        "project_mode": project_mode,
        "extra_files": None,
    }
