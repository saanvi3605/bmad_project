"""
orchestration/graph.py
────────────────────────────────────────────────────────────────────────────────
LangGraph StateGraph construction — full BMAD pipeline.

Pipeline topology:

                                    ┌─ score ≤ 4 ──→ SimpleGenerator (A2A) ──┐
  PromptRefiner → ComplexityScorer ─┤                                         ↓
                                    └─ score > 4 ──→ Planner → Architect → Developer → Validator
                                                                                  ↑         │
                                                                 runtime_fix      │  passed / max_attempts
                                                                    └─────────────┘         │
                                                                                             ↓
                                                                                          Tester → Reviewer
                                                                                                      │
                                                          ┌── approved / max_retries ─────────────────┘
                                                          │
                                                          ↓
                                                       Executor
                                                          │
                                  ┌── success / max_attempts ──────────────────────────────────┐
                                  │                                                             ↓
                                  │  runtime_fix ──→ Developer     TestWriter → ReadmeWriter → EvalAgent → END
                                  └────────────────────────────────────────────────────────────────────────────┘

Three feedback loops:
  1. Validator ↔ Developer  (static analysis — syntax, imports, AST)
  2. Reviewer  ↔ Developer  (LLM quality gate — 8 correctness criteria)
  3. Executor  → Developer  (runtime self-healing — app crash stderr)

A2A routing:
  ComplexityScorer scores the prompt 1-10. Score ≤ simple_threshold (config) routes
  to SimpleGenerator which calls project_final_final via HTTP (POST /generate).
  Score > threshold runs the full BMAD pipeline as normal.
  Threshold is set in config/models.yaml::complexity_scorer.simple_threshold (default 4).
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from core.state import BMADState
from agents_impl.prompt_refiner_agent import prompt_refiner_agent
from agents_impl.complexity_scorer_agent import complexity_scorer_agent
from agents_impl.a2a_client import simple_generator_agent
from agents_impl.planner_agent import planner_agent
from agents_impl.architect_agent import architect_agent
from agents_impl.developer_agent import developer_agent
from agents_impl.validator_agent import validator_agent, should_fix
from agents_impl.tester_agent import tester_agent
from agents_impl.reviewer_agent import reviewer_agent, should_retry
from agents_impl.executor_agent import executor_agent, should_runtime_fix
from agents_impl.test_writer_agent import test_writer_agent
from agents_impl.readme_agent import readme_agent
from agents_impl.eval_agent import eval_agent
from core.llm_factory import get_config

# ---------------------------------------------------------------------------
# A2A routing — decides whether to call project_final_final or run full pipeline
# ---------------------------------------------------------------------------

def _should_use_a2a(state: dict) -> str:
    """
    After ComplexityScorer:
      score ≤ simple_threshold  →  "simple"  (call project_final_final via A2A)
      score >  simple_threshold  →  "complex" (run full BMAD pipeline)

    Threshold is read from config/models.yaml::complexity_scorer.simple_threshold
    so changing the value there affects both model routing AND A2A routing.
    """
    score     = state.get("complexity_score", 5)
    cfg       = get_config()
    threshold = int(cfg.get("complexity_scorer", {}).get("simple_threshold", 4))
    return "simple" if score <= threshold else "complex"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

workflow = StateGraph(BMADState)

workflow.add_node("PromptRefiner",     prompt_refiner_agent)
workflow.add_node("ComplexityScorer", complexity_scorer_agent)
workflow.add_node("SimpleGenerator",  simple_generator_agent)   # A2A path
workflow.add_node("Planner",          planner_agent)
workflow.add_node("Architect",     architect_agent)
workflow.add_node("Developer",     developer_agent)
workflow.add_node("Validator",     validator_agent)
workflow.add_node("Tester",        tester_agent)
workflow.add_node("Reviewer",      reviewer_agent)
workflow.add_node("Executor",      executor_agent)
workflow.add_node("TestWriter",    test_writer_agent)
workflow.add_node("ReadmeWriter",  readme_agent)
workflow.add_node("EvalAgent",     eval_agent)

workflow.set_entry_point("PromptRefiner")

workflow.add_edge("PromptRefiner",    "ComplexityScorer")

# A2A routing: simple prompts → project_final_final service; complex → full pipeline
workflow.add_conditional_edges(
    "ComplexityScorer",
    _should_use_a2a,
    {
        "simple":  "SimpleGenerator",  # score ≤ threshold → A2A call
        "complex": "Planner",          # score >  threshold → full BMAD pipeline
    },
)

# SimpleGenerator skips Planner/Architect/Developer — jumps straight to Validator
workflow.add_edge("SimpleGenerator", "Validator")

workflow.add_edge("Planner",          "Architect")
workflow.add_edge("Architect",     "Developer")

# Developer → Validator → fix loop — preserved verbatim
workflow.add_edge("Developer", "Validator")
workflow.add_conditional_edges(
    "Validator",
    should_fix,
    {
        "passed":       "Tester",
        "fix":          "Developer",    # send back with error for auto-fix
        "max_attempts": "Tester",       # give up and proceed after 2 retries
    },
)

workflow.add_edge("Tester", "Reviewer")
workflow.add_conditional_edges(
    "Reviewer",
    should_retry,
    {
        "approved":    "Executor",
        "retry":       "Developer",
        "max_retries": "Executor",
    },
)

workflow.add_conditional_edges(
    "Executor",
    should_runtime_fix,
    {
        "success":      "TestWriter",   # app started — move on
        "runtime_fix":  "Developer",    # crash → send stderr back for a fix
        "max_attempts": "TestWriter",   # gave up after N retries — still archive
    },
)
workflow.add_edge("TestWriter",   "ReadmeWriter")
workflow.add_edge("ReadmeWriter", "EvalAgent")
workflow.add_edge("EvalAgent",    END)

# ---------------------------------------------------------------------------
# Compiled graph — the public object consumed by pipeline_ui.py and main.py
# ---------------------------------------------------------------------------

app = workflow.compile()
