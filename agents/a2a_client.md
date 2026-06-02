# Agent: SimpleGenerator (A2A Client)

**Role:** A2A Choreography Request Agent  
**Module:** `agents_impl/a2a_client.py`  
**LLM:** No (file I/O + polling)  
**Skill:** `a2a_request` (see `skills/a2a_request.yaml`)

---

## Responsibility

Routes simple prompts (complexity score ≤ `a2a_threshold`) to `project_final_final`
via the **A2A choreography pattern** — writing a request to a shared event bus file
and polling until the result is written back.

This is the **project_bmad side** of the A2A event bus:

```
project_bmad                              project_final_final
────────────────────────────────          ─────────────────────────────────
SimpleGenerator writes                    a2a_orchestrator_agent.py watches
  status: pending              →            a2a_control.yaml
  prompt: "..."                             sees status=pending
                               ←            sets status: running → runs pipeline
                               ←            writes result + status: complete
SimpleGenerator reads result
injects code/design/spec
into state → Executor
```

---

## Choreography vs HTTP

| Approach | Coupling | Error handling |
|---|---|---|
| **HTTP (old)** | Tight — server must be running and reachable | `ConnectionError` if service is down |
| **Choreography (new)** | Loose — both sides just watch a file | Graceful timeout with clear message |

---

## Inputs (from BMADState)

| Field | Notes |
|---|---|
| `user_request` | Raw prompt (scored before PromptRefiner) |
| `complexity_score` | Set by ComplexityScorer; must be ≤ `a2a_threshold` |

---

## Outputs (to BMADState)

| Field | Notes |
|---|---|
| `code` | Generated Streamlit app from project_final_final |
| `technical_design` | Architecture spec produced by project_final_final |
| `functional_spec` | Requirements spec produced by project_final_final |
| `a2a_used` | Set to `True` |

---

## Usage

Start project_final_final's orchestrator first:
```bash
cd project_final_final
python a2a_orchestrator_agent.py
```

Then run the BMAD pipeline normally — simple prompts route automatically.
