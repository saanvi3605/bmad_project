# Agent: Orchestrator

**Role:** Pipeline Choreography Agent  
**Module:** `agents_impl/orchestrator_agent.py`  
**LLM:** No (deterministic polling loop)  
**Skills:** `pipeline_trigger` (see `skills/pipeline_trigger.yaml`)

---

## Responsibility

Implements the choreography pattern for autonomous pipeline execution. Watches a shared
control file (`control.yaml`) as an event bus and self-triggers the BMAD pipeline whenever
a run is requested.

This agent does NOT use an LLM. It is a deterministic polling loop that:
1. Reads `control.yaml` every `POLL_INTERVAL` seconds
2. When `run: true` is detected, flips it to `false` immediately (prevents double-triggers)
3. Executes the full BMAD pipeline synchronously via `main.run()`
4. Writes the result (`complete` / `failed`) back to `control.yaml`

---

## Choreography vs Orchestration

| Pattern | How it works | Used where |
|---|---|---|
| **Orchestration** | Central controller calls agents in order | LangGraph StateGraph (`orchestration/graph.py`) |
| **Choreography** | Agents watch shared state and self-trigger | This agent (`agents_impl/orchestrator_agent.py`) |

The Orchestrator agent is the choreography layer — it sits outside the LangGraph pipeline
and decides *when* to run it, not *how*.

---

## Inputs (from control.yaml)

| Field | Type | Notes |
|---|---|---|
| `run` | bool | Set to `true` by the pipeline_trigger skill to start a run |
| `prompt` | str | The user's natural language request |
| `project_mode` | str | `auto` \| `streamlit_crud` \| `fastapi_rag` |
| `sprint_item` | str | Optional BMAD sprint tracking label |

---

## Outputs (to control.yaml)

| Field | Type | Notes |
|---|---|---|
| `status` | str | `idle` → `running` → `complete` \| `failed` |
| `last_run` | str | ISO timestamp of last execution |
| `last_result` | str | Short result message or error summary |
| `iteration` | int | Incremented on each successful run |

---

## Usage

```bash
# Start the orchestrator (runs continuously)
python agents_impl/orchestrator_agent.py

# Run a single check-and-execute cycle then exit
python agents_impl/orchestrator_agent.py --once
```

Trigger a run using the `pipeline_trigger` skill:
```bash
python agents_impl/trigger_skill.py "Build a recipe manager app"
```
