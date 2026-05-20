"""
core/llm_factory.py
────────────────────────────────────────────────────────────────────────────────
Responsible for ONE thing: creating and returning the single shared ChatGroq
instance used by every LLM-calling agent in the pipeline.

build_llm() is NOT cached at the module level so imports do not trigger LLM
instantiation.  agent_runner.py calls _get_llm() lazily on the first real
invocation.  Tests can monkeypatch build_llm before any call occurs.
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


def _build_from_config(cfg: dict[str, Any]) -> Any:
    """Instantiate an LLM from a config dict. Supports providers: groq, gemini."""
    load_dotenv()
    provider = cfg.get("provider", "groq")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in .env")
        return ChatGroq(
            groq_api_key=api_key,
            model_name=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            model_kwargs={},
            request_timeout=cfg.get("request_timeout", 120),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set in .env")
        return ChatGoogleGenerativeAI(
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_tokens"],
            google_api_key=api_key,
        )

    raise ValueError(f"Unknown LLM provider '{provider}'. Supported: groq, gemini")


def build_llm() -> Any:
    """Build the heavy LLM (llm: section of models.yaml)."""
    return _build_from_config(get_llm_config())


def build_llm_light() -> Any:
    """Build the light LLM (llm_light: section, falls back to llm: if absent)."""
    return _build_from_config(get_llm_light_config())


def clear_config_cache() -> None:
    """Bust the lru_cache so the next config read picks up models.yaml changes."""
    _load_config.cache_clear()
