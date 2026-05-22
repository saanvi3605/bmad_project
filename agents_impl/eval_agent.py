"""
agents_impl/eval_agent.py
────────────────────────────────────────────────────────────────────────────────
EvalAgent — final pipeline node that scores the run and pushes results to
Langfuse.

Runs after ReadmeWriter so it has access to every pipeline artifact:
  user_request, functional_spec, technical_design, code, validation_passed,
  review_approved, execution_result, test_file, readme_file.

Uses the light (8b) model for the two LLM-as-judge calls; the deterministic
and semantic-overlap dimensions require no LLM at all.

Scores appear in:
  • Langfuse dashboard — as coloured metric bars on the eval trace
  • Streamlit UI — Observability tab → Eval Scores section
  • Pipeline summary print at the end of main.py runs
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.evaluator import evaluate_pipeline_run

_AGENT_NAME = "eval_agent"


def eval_agent(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node — evaluate pipeline outputs and store scores in state."""
    try:
        evaluate_pipeline_run(state)
    except Exception as exc:
        # Never let an eval failure crash the pipeline
        import warnings
        warnings.warn(
            f"[EvalAgent] Evaluation failed and was skipped: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        state["eval_scores"] = {"error": str(exc)}
    return state
