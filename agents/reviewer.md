# Agent: Reviewer

**Role:** Senior Code Reviewer  
**Module:** `agents_impl/reviewer_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Skills:** `code_review` (see `skills/code_review.yaml`)

---

## Responsibility

LLM-powered review of generated Python code against eight correctness and runnability criteria. Produces a structured `APPROVED: YES/NO` + `FEEDBACK:` verdict.

---

## Review Criteria

1. Valid, executable Python with no SyntaxErrors
2. Correct FastAPI usage
3. SQLite only (not MySQL or PostgreSQL)
4. `uvicorn.run(...)` inside `if __name__ == '__main__':`
5. GET `/` route returning `HTMLResponse`
6. All imports present and correct
7. All HTML inside Python strings — not raw HTML outside strings
8. Single self-contained file with no fake section headers

---

## Response Format

```
APPROVED: YES  (or NO)
FEEDBACK: Code meets all requirements.  (or list of specific issues)
```

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `review_approved` | bool | True iff response contains `APPROVED: YES` |
| `review_feedback` | str | Text after `FEEDBACK:` label |
| `review_attempts` | int | Incremented inside agent (not routing function) |

---

## Routing (`should_retry`)

```
review_approved == True             → "approved"    → Executor
review_approved == False, attempts < 2  → "retry"   → Developer
review_approved == False, attempts >= 2 → "max_retries" → Executor
```

**Note:** `review_attempts` is incremented inside `reviewer_agent()`, unlike `validation_attempts` which is incremented in `should_fix()`.
