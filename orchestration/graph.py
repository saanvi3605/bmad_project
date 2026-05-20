"""
orchestration/graph.py
────────────────────────────────────────────────────────────────────────────────
LangGraph StateGraph construction — full BMAD pipeline.

Pipeline topology:

  PromptRefiner → Planner → Architect → Developer → Validator
                                                         │
                          ┌── passed / max_attempts ─────┘
                          ↓
                       Tester → Reviewer
                                   │
            ┌── approved / max_retries ──────────────────┐
            │                                            ↓
            └── retry ──────────────────────────→ Developer
                                                         ↓
                                                  Executor → TestWriter → END

PromptRefiner is the entry point.  It rewrites the raw user prompt into a
structured product brief stored in state["refined_prompt"].  Planner reads
this field instead of user_request, so all downstream agents benefit without
needing changes.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from core.state import BMADState
from agents_impl.prompt_refiner_agent import prompt_refiner_agent
from agents_impl.planner_agent import planner_agent
from agents_impl.architect_agent import architect_agent
from agents_impl.developer_agent import developer_agent
from agents_impl.validator_agent import validator_agent, should_fix
from agents_impl.tester_agent import tester_agent
from agents_impl.reviewer_agent import reviewer_agent, should_retry
from agents_impl.executor_agent import executor_agent
from agents_impl.test_writer_agent import test_writer_agent

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

workflow = StateGraph(BMADState)

workflow.add_node("PromptRefiner", prompt_refiner_agent)
workflow.add_node("Planner",       planner_agent)
workflow.add_node("Architect",     architect_agent)
workflow.add_node("Developer",     developer_agent)
workflow.add_node("Validator",     validator_agent)
workflow.add_node("Tester",        tester_agent)
workflow.add_node("Reviewer",      reviewer_agent)
workflow.add_node("Executor",      executor_agent)
workflow.add_node("TestWriter",    test_writer_agent)

workflow.set_entry_point("PromptRefiner")

workflow.add_edge("PromptRefiner", "Planner")
workflow.add_edge("Planner",       "Architect")
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

workflow.add_edge("Executor",    "TestWriter")
workflow.add_edge("TestWriter",  END)

# ---------------------------------------------------------------------------
# Compiled graph — the public object consumed by pipeline_ui.py and main.py
# ---------------------------------------------------------------------------

app = workflow.compile()
