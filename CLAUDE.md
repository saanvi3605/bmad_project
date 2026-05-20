# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BMAD is an AI-powered pipeline that generates complete Streamlit applications from natural language specifications. LangGraph orchestrates 9 sequential agents (PromptRefiner → Planner → Architect → Developer → Validator → Tester → Reviewer → Executor → TestWriter), each specializing in one phase of app development.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (CLI)
python main.py "Build a task tracker with add/delete/list"

# Run Streamlit UI
streamlit run streamlit_app.py

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_validator.py -v

# Run a single test
pytest tests/test_validator.py::test_validator_detects_syntax_error -v
```

## Environment

`.env` must contain:
```
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Architecture

### State

All agents read/write a single `BMADState` TypedDict (`core/state.py`). Key fields: `user_request`, `functional_spec`, `technical_design`, `code`, `validation_error`, `validation_attempts`, `review_feedback`, `review_approved`, `review_attempts`, `execution_result`, `session_id`, `agent_log`, `output_dir`.

### Two Feedback Loops

1. **Validator ↔ Developer** (max 2 attempts via `config/models.yaml::retry_limits.validator_max_retries`): Validator runs 6 deterministic checks (AST syntax, py_compile, required patterns, disallowed libs, NameError AST walk, Jinja2 syntax). On failure, Developer regenerates in `validation_fix` mode.

2. **Reviewer ↔ Developer** (max 2 attempts via `retry_limits.reviewer_max_retries`): Reviewer checks 8 correctness criteria via LLM. On rejection, Developer regenerates in `review_fix` mode.

Both loops are wired as conditional edges in `orchestration/graph.py`.

### Configuration-Driven Design

All tunable parameters live in `config/models.yaml` (LLM settings, retry limits, executor config, cost coefficients). Hard constraints appended to every user request (no auth, Streamlit only, SQLite only, single file) are in `config/pipeline_rules.yaml`.

### Code Generation Constraints

Generated apps must be:
- Single Python file, Streamlit-based
- SQLite via `sqlite3` stdlib only (no SQLAlchemy/ORMs)
- No auth unless explicitly requested

These are enforced by: prompt templates, `core/sanitizer.py` (post-processes LLM output), `validator_agent.py` (deterministic checks), and `reviewer_agent.py` (LLM checks).

### Key Modules

| File | Responsibility |
|------|----------------|
| `core/agent_runner.py` | Single LLM invocation point; lazy `ChatGroq` instantiation; token/cost logging |
| `core/llm_factory.py` | Loads `config/models.yaml`; exposes `build_llm()` and config accessors |
| `core/sanitizer.py` | Strips disallowed imports, injects required imports, normalizes code structure |
| `agents_impl/developer_agent.py` | Three modes: `clean`, `validation_fix`, `review_fix` |
| `agents_impl/validator_agent.py` | Deterministic 6-stage static analysis, no LLM |
| `agents_impl/test_writer_agent.py` | Generates pytest file; hardcoded fixture block + LLM-generated test functions |
| `orchestration/graph.py` | LangGraph StateGraph definition; conditional edges for both feedback loops |

### Lazy LLM Instantiation

`core/agent_runner._get_llm()` lazily instantiates `ChatGroq` on first call. This allows tests to set `ar._llm = MagicMock()` before any agent runs, enabling CI without a real `GROQ_API_KEY`.

### Backward Compatibility

- `workflow.py` re-exports `app` from `orchestration.graph`
- `state.py` re-exports `AgentState = BMADState`

## Testing Without Groq API Key

```python
from unittest.mock import MagicMock
import core.agent_runner as ar

ar._llm = MagicMock(
    invoke=MagicMock(return_value=MagicMock(
        content="mocked response",
        usage_metadata={"input_tokens": 10, "output_tokens": 20}
    ))
)
```
