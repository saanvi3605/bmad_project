# Agent: Architect

**Role:** Software Architect  
**Module:** `agents_impl/architect_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Prompt key:** `architect_v1`

---

## Responsibility

Converts the Planner's functional specification into a concise technical design that the Developer agent will implement.

---

## Inputs

| Field | Type | Description |
|---|---|---|
| `functional_spec` | str | From Planner agent |

---

## Outputs

| Field | Type | Description |
|---|---|---|
| `technical_design` | str | Tech stack, component breakdown, data flow, file structure |

---

## Prompt Template

```
You are a software architect following the BMAD methodology.
Produce a concise technical design covering:
- Tech stack (language, framework, database, UI)
- Component breakdown (max 5 components)
- Data flow (simple step-by-step)
- File structure (tree format)

STRICT RULES:
- Use FastAPI for the web framework
- Use SQLite for the database
- Include a simple HTML dashboard route
- Do NOT design authentication or login unless user explicitly asked
- Only design components that are in the functional spec
- Maximum 300 words
- No external services
```

---

## Routing

`Architect → Developer` (unconditional edge)
