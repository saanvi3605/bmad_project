# Agent: Planner

**Role:** Software Business Analyst  
**Module:** `agents_impl/planner_agent.py`  
**LLM:** Yes (via `core.agent_runner.run_agent`)  
**Prompt key:** `planner_v1`

---

## Responsibility

Converts the user's raw application request into a structured, concise functional specification. Acts as the pipeline entry point after receiving the (potentially rules-augmented) user request.

---

## Inputs (from BMADState)

| Field | Type | Description |
|---|---|---|
| `user_request` | str | Raw request + pipeline constraints appended by main.py/streamlit_app.py |

---

## Outputs (mutations to BMADState)

| Field | Type | Description |
|---|---|---|
| `functional_spec` | str | Structured functional spec: overview, requirements, constraints |
| `review_attempts` | int | Reset to 0 at start of each pipeline run |

---

## Prompt Template

```
You are a software business analyst following the BMAD methodology.
Your job is to produce a clear, concise functional specification.

User Request:
{user_request}

Produce a functional specification covering:
- Project overview (2-3 sentences)
- Core functional requirements (numbered list, max 6)
- Key constraints and assumptions (max 4 bullet points)

STRICT RULES:
- Do NOT add authentication or login unless the user explicitly asked for it
- Do NOT add features beyond what the user requested
- Stick strictly to the user's requirement

Maximum 300 words. Be specific and actionable.
```

---

## Routing

`Planner → Architect` (unconditional edge)

---

## Notes

- Resets `review_attempts` to 0 at the start of every run.
- Keeps scope minimal: no gold-plating, no assumed features.
- 300-word cap prevents over-specification that inflates downstream prompts.
