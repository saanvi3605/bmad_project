# Agent: Validator

**Role:** Static Code Analyzer  
**Module:** `agents_impl/validator_agent.py`  
**LLM:** No — fully deterministic  
**Skills:** `code_validation` (see `skills/code_validation.yaml`)

---

## Responsibility

Runs six sequential, deterministic, pure-Python checks against the generated application code. No LLM is involved. Returns pass/fail with a human-readable reason fed back to the Developer on failure.

---

## Validation Stages (in order)

| # | Name | Method | Failure example |
|---|---|---|---|
| 1 | AST syntax parse | `ast.parse(code)` | `SyntaxError on line 42: invalid syntax` |
| 2 | py_compile subprocess | `python -m py_compile` with 15s timeout | `Compile error: ...` |
| 3 | Required pattern presence | String search for 4 patterns | `Missing GET / route` |
| 4 | Disallowed library detection | String search for 7 banned patterns | `sqlalchemy still present` |
| 5 | NameError detection (AST walk) | Module-level name collection + annotation check | `'Session' used as type annotation but not imported` |
| 6 | Jinja2 syntax in HTML | HTML section scan for `{%`, `%}`, `{{`, `}}` | `Jinja2 template syntax found: {{` |

---

## Required Patterns (Check 3)

```python
REQUIRED_PATTERNS = [
    ('@app.get("/"',  "Missing GET / route"),
    ("HTMLResponse",  "Missing HTMLResponse usage"),
    ("uvicorn.run(",  "Missing uvicorn.run(...)"),
    ("if __name__",   "Missing if __name__ == '__main__': block"),
]
```

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `validation_passed` | bool | True iff all six checks pass |
| `validation_error` | str\|None | Human-readable failure message, or None on pass |

---

## Routing (`should_fix`)

```
validation_passed == True           → "passed"  → Tester
validation_passed == False, attempts < 2  → "fix"     → Developer (increments validation_attempts)
validation_passed == False, attempts >= 2 → "max_attempts" → Tester (proceed anyway)
```

`validation_attempts` is incremented inside `should_fix()`, not inside the agent function.
