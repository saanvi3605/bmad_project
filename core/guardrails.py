"""
core/guardrails.py
────────────────────────────────────────────────────────────────────────────────
Input prompt safety checker for the BMAD pipeline.

Called before any LLM invocation so no tokens are wasted on blocked prompts.

Rules are loaded from config/guardrails.yaml and can be edited without
touching Python.  Three check layers run in order:

  1. Length check — rejects absurdly long prompts (prompt-injection vector)
  2. Hard-block check — any disallowed phrase → immediate rejection
  3. Intent-block check — creation verb + harmful target within 60 chars
                          unless an allow_override phrase is present

All checks are case-insensitive.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pathlib
import re
from functools import lru_cache
from typing import Any

import yaml

_CFG_PATH = pathlib.Path(__file__).parent.parent / "config" / "guardrails.yaml"

# Maximum character window for verb + target proximity matching
_PROXIMITY_WINDOW = 60


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    """Load and cache guardrails config. Re-call clear_guardrails_cache() to bust."""
    with open(_CFG_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("guardrails", {})


def clear_guardrails_cache() -> None:
    """Bust the lru_cache so config changes are picked up without restart."""
    _load_config.cache_clear()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def check_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate a user prompt against the guardrails config.

    Parameters
    ----------
    prompt : str
        The raw user request (before any pipeline_rules suffix is appended).

    Returns
    -------
    (is_safe, reason)
        is_safe = True  → prompt is allowed; reason is ""
        is_safe = False → prompt is blocked; reason explains why
    """
    cfg = _load_config()

    if not cfg.get("enabled", True):
        return True, ""

    lower = prompt.lower().strip()

    # ── 1. Length check ───────────────────────────────────────────────────────
    max_len = int(cfg.get("max_length", 4000))
    if len(prompt) > max_len:
        return False, (
            f"Prompt is too long ({len(prompt):,} characters). "
            f"Please keep it under {max_len:,} characters."
        )

    # ── Precompute allow-override flag ────────────────────────────────────────
    overrides: list[str] = cfg.get("allow_overrides", [])
    override_active = any(o.lower() in lower for o in overrides)

    # ── 2. Hard-block check ───────────────────────────────────────────────────
    hard_blocks: list[str] = cfg.get("hard_blocks", []) + cfg.get("custom_blocks", [])
    for phrase in hard_blocks:
        p = phrase.lower()
        if p in lower:
            # Some hard-block terms are waived if an override is present
            _waivable = {"virus", "malware", "exploit", "port scanner", "packet sniffer"}
            if override_active and any(w in p for w in _waivable):
                continue
            return False, (
                f"Prompt blocked: contains disallowed term \"{phrase}\". "
                "BMAD generates productivity apps — harmful or malicious "
                "software requests are not permitted."
            )

    # ── 3. Intent-block check ────────────────────────────────────────────────
    intent_cfg = cfg.get("intent_blocks", {})
    verbs: list[str] = [v.lower() for v in intent_cfg.get("verbs", [])]
    targets: list[str] = [t.lower() for t in intent_cfg.get("targets", [])]

    if verbs and targets:
        for verb in verbs:
            for match in re.finditer(re.escape(verb), lower):
                v_start, v_end = match.start(), match.end()
                # Extract a window around the verb to check for nearby targets
                window_start = max(0, v_start - _PROXIMITY_WINDOW)
                window_end = min(len(lower), v_end + _PROXIMITY_WINDOW)
                window = lower[window_start:window_end]

                for target in targets:
                    if target in window:
                        if override_active:
                            continue  # Defensive security tool — allow
                        return False, (
                            f"Prompt blocked: \"{verb} … {target}\" indicates a "
                            "request to build harmful software. "
                            "BMAD generates productivity apps only."
                        )

    return True, ""


# ---------------------------------------------------------------------------
# Convenience: pretty rejection message for display
# ---------------------------------------------------------------------------


def format_rejection(reason: str) -> str:
    """Return a user-facing rejection string with emoji and help text."""
    return (
        f"🚫 **Request blocked by guardrails**\n\n"
        f"{reason}\n\n"
        "_If you believe this is a false positive, update_ "
        "`config/guardrails.yaml` _to adjust the rules._"
    )
