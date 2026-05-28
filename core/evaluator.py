"""
core/evaluator.py
────────────────────────────────────────────────────────────────────────────────
Post-pipeline evaluation layer.

Scores every completed pipeline run across four dimensions and submits the
results as Langfuse trace-level scores so they appear as coloured metric bars
on every run in the Langfuse dashboard.

Dimensions
──────────
  feature_coverage      (0–1, LLM-as-judge)
    Does the generated code implement every feature described in the
    functional spec?  The LLM is given the spec bullet-list and the code
    and asked to count covered vs missing features.

  requirement_alignment (0–1, LLM-as-judge)
    How semantically aligned is the final app with the original user request?
    The LLM compares intent, scope, and key nouns from the request to the
    implemented behaviour.

  code_quality          (0–1, deterministic)
    Weighted signal from pipeline state:
      • validation_passed   → 0.40 pts (static checks all green)
      • review_approved     → 0.40 pts (LLM reviewer approved)
      • execution_success   → 0.20 pts (app started without crash)

  semantic_overlap      (0–1, NLP / no-LLM)
    Jaccard similarity of meaningful content words between the user_request
    and the functional_spec.  A fast proxy for "did the pipeline understand
    what was asked?"  No LLM call needed.

Usage
─────
  from core.evaluator import evaluate_pipeline_run
  eval_result = evaluate_pipeline_run(state)
  # → stores scores in state["eval_scores"] and submits to Langfuse
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    value: float          # 0.0 – 1.0
    reason: str           # one-sentence explanation
    dimension: str        # e.g. "feature_coverage"

    def to_dict(self) -> dict[str, Any]:
        return {
            "value":     round(self.value, 3),
            "reason":    self.reason,
            "dimension": self.dimension,
        }


@dataclass
class EvalResult:
    feature_coverage:      DimensionScore
    requirement_alignment: DimensionScore
    code_quality:          DimensionScore
    semantic_overlap:      DimensionScore
    overall:               DimensionScore
    eval_latency_ms:       float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_coverage":      self.feature_coverage.to_dict(),
            "requirement_alignment": self.requirement_alignment.to_dict(),
            "code_quality":          self.code_quality.to_dict(),
            "semantic_overlap":      self.semantic_overlap.to_dict(),
            "overall":               self.overall.to_dict(),
            "eval_latency_ms":       round(self.eval_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Semantic overlap — pure NLP, zero LLM calls
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    # Articles, conjunctions, prepositions
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "into", "through", "about",
    "above", "below", "between", "during", "before", "after", "against",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "us", "them", "who", "what", "which",
    # Demonstratives / quantifiers
    "that", "this", "these", "those", "each", "all", "any",
    "both", "few", "more", "most", "other", "some", "such",
    "every", "either", "neither", "enough", "several",
    # Common adverbs / connectors
    "not", "no", "so", "if", "as", "up", "out", "also", "just",
    "than", "then", "there", "here", "when", "where", "why", "how",
    "very", "too", "quite", "rather", "well", "already", "still",
    "yet", "even", "only", "however", "therefore", "thus",
    # Generic action verbs that carry no domain meaning
    "use", "using", "used", "include", "including",
    "provides", "provide", "provided",
    "build", "create", "add", "make", "get", "set",
    "show", "display", "allow", "allows", "ensure",
    "support", "supports", "need", "needs", "want", "wants",
    "enable", "enables", "give", "gives", "let", "lets",
    "keep", "keeps", "take", "takes", "put", "run", "runs",
    "return", "returns", "call", "calls", "send", "sends",
    "handle", "handles", "manage", "manages",
    # Generic app/project words that appear in almost every request
    "app", "application", "system", "feature", "page",
    "section", "panel", "view", "screen", "tab",
    "user", "users", "data", "field", "fields",
    "form", "forms", "table", "list", "item", "items",
    "new", "edit", "delete", "save", "submit", "cancel",
    "simple", "basic", "complete", "full", "good", "best",
    "able", "based", "following", "example", "way",
})


def semantic_overlap(text1: str, text2: str) -> DimensionScore:
    """
    Jaccard similarity of content-bearing words in two texts.

    Content words = lowercase alpha tokens of length ≥ 3 that are not
    stop words.  A score of 1.0 means identical vocabulary; 0.0 means
    no shared words at all.  Typical good scores are 0.3–0.6.
    """
    def _keywords(text: str) -> frozenset[str]:
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return frozenset(t for t in tokens if t not in _STOP_WORDS)

    kw1 = _keywords(text1)
    kw2 = _keywords(text2)

    if not kw1 or not kw2:
        return DimensionScore(0.0, "Insufficient text to compute overlap.", "semantic_overlap")

    intersection = kw1 & kw2
    union        = kw1 | kw2
    score = len(intersection) / len(union)

    top_shared = sorted(intersection)[:8]
    reason = (
        f"{len(intersection)}/{len(kw1)} request keywords present in spec. "
        f"Top shared: {', '.join(top_shared)}."
    )
    return DimensionScore(round(score, 3), reason, "semantic_overlap")


# ---------------------------------------------------------------------------
# Deterministic code quality score
# ---------------------------------------------------------------------------

def code_quality_score(state: dict[str, Any]) -> DimensionScore:
    """
    Weighted score from pipeline state signals — no LLM needed.

      validation_passed  → 0.40 pts
      review_approved    → 0.40 pts
      execution_success  → 0.20 pts
    """
    val      = bool(state.get("validation_passed"))
    review   = bool(state.get("review_approved"))
    exec_ok  = "successfully" in (state.get("execution_result") or "").lower()

    score  = 0.0
    parts  = []
    if val:
        score += 0.40
        parts.append("validation passed")
    else:
        parts.append("validation failed")

    if review:
        score += 0.40
        parts.append("reviewer approved")
    else:
        parts.append("reviewer rejected/skipped")

    if exec_ok:
        score += 0.20
        parts.append("app started successfully")
    else:
        parts.append("app did not start")

    reason = "; ".join(parts) + f" => {score:.0%}"
    return DimensionScore(round(score, 3), reason, "code_quality")


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

_FEATURE_COVERAGE_PROMPT = """\
You are a strict code auditor. Your task is to measure feature coverage.

FUNCTIONAL SPECIFICATION (what the app should do):
{functional_spec}

GENERATED CODE (what was actually implemented):
{code_snippet}

INSTRUCTIONS:
1. List every distinct feature bullet from the functional specification.
2. For each feature, check whether the code contains a corresponding \
implementation (function, UI widget, DB query, etc.).
3. Compute: covered_features / total_features.

Return ONLY valid JSON — no markdown, no explanation outside the JSON:
{{"score": 0.85, "covered": 6, "total": 7, \
"reason": "6 of 7 features implemented; missing export to CSV"}}
"""

_REQUIREMENT_ALIGNMENT_PROMPT = """\
You are a product manager reviewing a generated Streamlit application.

ORIGINAL USER REQUEST:
{user_request}

GENERATED CODE (first 3000 characters):
{code_snippet}

INSTRUCTIONS:
Score how well the generated app fulfils the original request (0.0 – 1.0).
Consider: correct problem domain, key nouns/entities present, primary \
user actions supported, appropriate scope.
0.0 = completely off-topic
0.5 = partially matches
1.0 = fully matches all stated requirements

Return ONLY valid JSON:
{{"score": 0.9, "reason": "All main requirements met; tracker with add/delete/list included"}}
"""


def _parse_llm_score(raw: str, fallback_reason: str) -> tuple[float, str]:
    """Extract (score, reason) from LLM JSON response; return (0.5, fallback) on error."""
    # Strip optional markdown fences
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(clean)
        score  = float(data.get("score", 0.5))
        reason = str(data.get("reason", fallback_reason))
        return round(max(0.0, min(1.0, score)), 3), reason
    except Exception:
        # Try to extract a bare decimal
        m = re.search(r'"score"\s*:\s*([\d.]+)', raw)
        if m:
            return round(float(m.group(1)), 3), fallback_reason
        return 0.5, f"Could not parse LLM response: {raw[:100]}"


def _invoke_light_llm(prompt: str, state: dict[str, Any]) -> str:
    """Call the light (8b) LLM via run_agent — uses LiteLLM, not LangChain."""
    from core.agent_runner import run_agent
    return run_agent(
        prompt=prompt,
        agent_name="eval_agent",
        prompt_key="eval_llm_judge",
        state=state,
        light=True,
    )


def score_feature_coverage(state: dict[str, Any]) -> DimensionScore:
    """LLM-as-judge: does the code implement every feature in the functional spec?"""
    spec = state.get("functional_spec") or ""
    code = state.get("code") or ""
    if not spec or not code:
        return DimensionScore(0.5, "Functional spec or code unavailable.", "feature_coverage")

    prompt = _FEATURE_COVERAGE_PROMPT.format(
        functional_spec=spec[:3000],
        code_snippet=code[:3000],
    )
    try:
        raw = _invoke_light_llm(prompt, state)
        value, reason = _parse_llm_score(raw, "Feature coverage assessed.")
        return DimensionScore(value, reason, "feature_coverage")
    except Exception as exc:
        return DimensionScore(0.5, f"Eval LLM call failed: {exc}", "feature_coverage")


def score_requirement_alignment(state: dict[str, Any]) -> DimensionScore:
    """LLM-as-judge: how well does the code match the original user request?"""
    user_req = state.get("user_request") or ""
    code     = state.get("code") or ""
    if not user_req or not code:
        return DimensionScore(0.5, "User request or code unavailable.", "requirement_alignment")

    prompt = _REQUIREMENT_ALIGNMENT_PROMPT.format(
        user_request=user_req[:1000],
        code_snippet=code[:3000],
    )
    try:
        raw = _invoke_light_llm(prompt, state)
        value, reason = _parse_llm_score(raw, "Requirement alignment assessed.")
        return DimensionScore(value, reason, "requirement_alignment")
    except Exception as exc:
        return DimensionScore(0.5, f"Eval LLM call failed: {exc}", "requirement_alignment")


# ---------------------------------------------------------------------------
# Langfuse score submission
# ---------------------------------------------------------------------------

def _submit_to_langfuse(
    session_id: str,
    eval_result: EvalResult,
    state: Optional[dict] = None,
) -> None:
    """
    Attach one score per evaluation dimension to the active Langfuse trace.

    Langfuse v3 API (current):
      • lf.get_current_trace_id()  → trace_id of the active OTEL span
      • lf.create_score(trace_id=…, name=…, value=…, comment=…) → numeric score

    Since eval_agent is the last node inside app.invoke(), the LangGraph OTEL
    context is still active here, so get_current_trace_id() reliably returns
    the pipeline trace ID.

    Falls back to creating a standalone score attached to the session_id if
    no active trace is found (e.g., when called from tests).

    Safe to call even when Langfuse is not configured — logs a warning and
    returns without raising.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not pub or not sec:
        return  # Langfuse not configured — skip silently

    try:
        from langfuse import get_client  # type: ignore[import]
        lf = get_client()

        # Resolve trace_id: prefer the active OTEL context (we're inside a
        # LangGraph node), fall back to the handler's last_trace_id.
        trace_id: Optional[str] = lf.get_current_trace_id()
        if not trace_id and state is not None:
            handler = state.get("langfuse_handler")
            trace_id = getattr(handler, "last_trace_id", None)

        dimensions = [
            eval_result.feature_coverage,
            eval_result.requirement_alignment,
            eval_result.code_quality,
            eval_result.semantic_overlap,
            eval_result.overall,
        ]

        if trace_id:
            for dim in dimensions:
                lf.create_score(
                    trace_id=trace_id,
                    name=dim.dimension,
                    value=dim.value,
                    comment=dim.reason,
                )
            print(f"[Evaluator] Scores attached to trace {trace_id[:8]}... (session {session_id[:8]}...)")
        else:
            # No active trace — best-effort score on session level
            for dim in dimensions:
                lf.create_score(
                    session_id=session_id,
                    name=dim.dimension,
                    value=dim.value,
                    comment=dim.reason,
                )
            print(f"[Evaluator] Scores submitted to Langfuse session {session_id[:8]}... (no active trace)")

    except Exception as exc:
        warnings.warn(
            f"[Evaluator] Langfuse score submission failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_pipeline_run(state: dict[str, Any]) -> EvalResult:
    """
    Run all four evaluations, compute an overall weighted score, submit to
    Langfuse, and store the result dict in state["eval_scores"].

    Weights:
      feature_coverage      35%
      requirement_alignment 35%
      code_quality          20%
      semantic_overlap      10%

    Returns the EvalResult dataclass (also stored in state).
    """
    t_start = time.perf_counter()

    print("\n  [Evaluator] Scoring pipeline outputs...")

    # 1. Fast deterministic scores (no LLM)
    qual_score = code_quality_score(state)

    user_req = state.get("user_request") or ""
    spec     = state.get("functional_spec") or ""
    sem_score = semantic_overlap(user_req, spec)

    # 2. LLM-as-judge scores (light model)
    feat_score = score_feature_coverage(state)
    req_score  = score_requirement_alignment(state)

    # 3. Weighted overall
    overall_value = round(
        feat_score.value  * 0.35
        + req_score.value * 0.35
        + qual_score.value * 0.20
        + sem_score.value  * 0.10,
        3,
    )
    overall = DimensionScore(
        overall_value,
        (
            f"Weighted: feature_coverage={feat_score.value:.2f} "
            f"req_alignment={req_score.value:.2f} "
            f"code_quality={qual_score.value:.2f} "
            f"semantic_overlap={sem_score.value:.2f}"
        ),
        "overall",
    )

    eval_latency_ms = (time.perf_counter() - t_start) * 1000.0

    result = EvalResult(
        feature_coverage=feat_score,
        requirement_alignment=req_score,
        code_quality=qual_score,
        semantic_overlap=sem_score,
        overall=overall,
        eval_latency_ms=eval_latency_ms,
    )

    # 4. Store in state
    state["eval_scores"] = result.to_dict()

    # 5. Submit to Langfuse
    session_id = state.get("session_id") or "unknown"
    _submit_to_langfuse(session_id, result, state=state)

    print(
        f"  [Evaluator] Overall score: {overall_value:.2f}  "
        f"(feat={feat_score.value:.2f} req={req_score.value:.2f} "
        f"qual={qual_score.value:.2f} sem={sem_score.value:.2f})"
    )
    return result
