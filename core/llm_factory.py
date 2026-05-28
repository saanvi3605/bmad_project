"""
core/llm_factory.py
────────────────────────────────────────────────────────────────────────────────
Responsible for ONE thing: building a LiteLLM-compatible config dict for every
LLM-calling agent in the pipeline.

build_llm() returns a plain dict:
    {"model": "groq/llama-3.3-70b-versatile", "temperature": 0.1,
     "max_tokens": 4096, "timeout": 120}

agent_runner.py holds this dict as a lazy singleton and passes it to
litellm.completion() on every invocation.  Tests can monkeypatch build_llm
before any call occurs.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "models.yaml"

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load config/models.yaml exactly once (cached)."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Required config file not found: {_CONFIG_PATH}\n"
            "Ensure config/models.yaml exists at the repository root."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config/models.yaml did not parse to a dict: {data!r}")
    return data


def get_config() -> dict[str, Any]:
    """Public accessor for the fully-parsed models.yaml dict."""
    return _load_config()


def get_llm_config() -> dict[str, Any]:
    """Return just the ``llm:`` sub-section of models.yaml."""
    return _load_config()["llm"]


def get_retry_limits() -> dict[str, int]:
    """Return the ``retry_limits:`` sub-section of models.yaml."""
    return _load_config()["retry_limits"]


def get_executor_config() -> dict[str, Any]:
    """Return the ``executor:`` sub-section of models.yaml."""
    return _load_config()["executor"]


def get_session_config() -> dict[str, Any]:
    """Return the ``session:`` sub-section of models.yaml."""
    return _load_config()["session"]


def get_observability_config() -> dict[str, Any]:
    """Return the ``observability:`` sub-section of models.yaml."""
    return _load_config()["observability"]


# ---------------------------------------------------------------------------
# LLM factory  — NOT cached here; agent_runner holds the singleton via lazy getter
# ---------------------------------------------------------------------------


def get_llm_light_config() -> dict[str, Any]:
    """Return the ``llm_light:`` section, falling back to ``llm:`` if absent."""
    return _load_config().get("llm_light") or _load_config()["llm"]


def _build_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Return a LiteLLM-compatible config dict from a models.yaml section.

    LiteLLM uses provider-prefixed model names:
      groq   → "groq/<model>"
      gemini → "gemini/<model>"

    The returned dict is stored as the agent_runner singleton and passed
    directly to litellm.completion() on every LLM call.
    """
    load_dotenv()
    provider = cfg.get("provider", "groq")
    model_name = cfg["model"]

    # Map provider → LiteLLM prefix
    _PROVIDER_PREFIX: dict[str, str] = {
        "groq": "groq",
        "gemini": "gemini",
        "openai": "openai",
    }
    prefix = _PROVIDER_PREFIX.get(provider)
    if prefix is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {list(_PROVIDER_PREFIX)}"
        )

    # Don't double-prefix if the model already starts with a known provider prefix.
    # Note: model names like "meta-llama/llama-4-scout-…" contain "/" but still
    # need the provider prefix (→ "groq/meta-llama/llama-4-scout-…").
    _KNOWN_PREFIXES = {f"{p}/" for p in _PROVIDER_PREFIX.values()}
    if any(model_name.startswith(p) for p in _KNOWN_PREFIXES):
        litellm_model = model_name  # already prefixed, e.g. "groq/llama-3.1-8b-instant"
    else:
        litellm_model = f"{prefix}/{model_name}"

    return {
        "model": litellm_model,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "timeout": cfg.get("request_timeout", 120),
    }


def build_llm() -> dict[str, Any]:
    """Return LiteLLM config dict for the heavy model (llm: section of models.yaml)."""
    return _build_from_config(get_llm_config())


def build_llm_light() -> dict[str, Any]:
    """Return LiteLLM config dict for the light model (llm_light: section)."""
    return _build_from_config(get_llm_light_config())


def build_llm_for_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a LiteLLM config dict from an arbitrary config dict (complexity override)."""
    return _build_from_config(cfg)


def clear_config_cache() -> None:
    """Bust the lru_cache so the next config read picks up models.yaml changes."""
    _load_config.cache_clear()
