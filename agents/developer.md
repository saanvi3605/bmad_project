# Agent: Developer

**Role:** Senior Python Developer  
**Module:** `agents_impl/developer_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Skills:** `code_generation` (see `skills/code_generation.yaml`)

---

## Responsibility

Generates a complete, single-file, directly runnable FastAPI + SQLite Python application from the Architect's technical design. Supports three invocation modes depending on the pipeline's feedback state.

---

## Invocation Modes

| Mode | Trigger Condition | `prompt_key` |
|---|---|---|
| Clean generation | No `validation_error`, no pending `review_feedback` | `developer_clean_v1` |
| Validation fix | `validation_error` set AND `validation_passed` is False | `developer_validation_fix_v1` |
| Review fix | `review_feedback` set AND `review_approved` is False | `developer_review_fix_v1` |

The mode detection logic is deterministic and evaluated at the start of every `developer_agent()` call.

---

## Inputs (from BMADState)

| Field | Type | Notes |
|---|---|---|
| `technical_design` | str | From Architect agent |
| `validation_error` | str\|None | Set by Validator on failure |
| `validation_passed` | bool\|None | Set by Validator |
| `review_feedback` | str\|None | Set by Reviewer on rejection |
| `review_approved` | bool\|None | Set by Reviewer |

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `code` | str | Sanitized Python source via `core.sanitizer.sanitize_code()` |

---

## Technology Constraints (hard-coded in prompt)

- Framework: FastAPI only
- Database: SQLite via `sqlite3` stdlib — no SQLAlchemy, no ORM
- Output: single Python file
- HTML: stored in Python string variables only — never raw HTML outside strings
- UI pattern: JavaScript `fetch()` calls to API endpoints — no Jinja2 template syntax
- Entry point: `uvicorn.run(...)` inside `if __name__ == '__main__':`

---

## Post-processing

Every LLM response is passed through `core.sanitizer.sanitize_code()` which enforces required imports, removes disallowed libraries, strips Jinja2 syntax, normalises `uvicorn.run()`, and injects `init_db()` if missing.

---

## Routing

- Receives from: `Architect` (clean), `Validator` (fix loop), `Reviewer` (retry loop)
- Routes to: `Validator` (unconditional after every generation)
