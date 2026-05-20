# BMAD AI Orchestration Pipeline

A production-grade AI pipeline that generates complete, runnable **Streamlit + SQLite applications** from a single natural language prompt. Describe what you want to build — the pipeline plans, architects, codes, validates, reviews, executes, and tests it automatically.

---

## How It Works

A 9-agent LangGraph pipeline where each agent has one job:

```
PromptRefiner → Planner → Architect → Developer → Validator
                                                      │
                                  ┌── pass ──────────┘
                                  ↓
                               Tester → Reviewer
                                           │
                              ┌── approved ┘
                              ↓
                           Executor → TestWriter → END
```

| Agent | Role | Model |
|---|---|---|
| **PromptRefiner** | Rewrites vague prompts into structured specs | Heavy (70b) |
| **Planner** | Writes functional specification | Heavy (70b) |
| **Architect** | Designs schema, forms, business logic | Heavy (70b) |
| **Developer** | Generates the full Streamlit app | Heavy (70b) |
| **Validator** | 10 deterministic static checks — no LLM | — |
| **Tester** | Writes TC-XX test case specifications | Light (8b) |
| **Reviewer** | Reviews code against 12 quality criteria | Heavy (70b) |
| **Executor** | Launches the app as a subprocess to verify it starts | — |
| **TestWriter** | Generates a runnable pytest file | Light (8b) |

**Two feedback loops** keep quality high:
- Validator → Developer (max 2 retries on static failures)
- Reviewer → Developer (max 2 retries on quality failures)

---

## Features

- **Natural language to working app** — describe it, get a fully runnable Streamlit app
- **Dual-model strategy** — heavy model (llama-3.3-70b) for planning and coding, light model (llama-3.1-8b) for test generation — two separate token quotas
- **10-check validator** — AST parse, py_compile, required patterns, disallowed libraries, NameError detection, Jinja2 detection, form checks, cache checks, stub detection, and more
- **Auto-sanitizer** — fixes common LLM mistakes: spaces in SQL table names, wrong `format=` strings, deprecated Streamlit APIs, disallowed imports
- **Pipeline History tab** — every run saved to SQLite with full artifacts, one-click re-run of any past prompt
- **Quick Launch buttons** — launch the generated app on a free port and open it in a new tab, or run pytest inline
- **Langfuse observability** — per-agent token counts, cost estimates, latency tracking, grouped traces per pipeline run
- **Session archiving** — every run's output saved to `outputs/sessions/{uuid}/`
- **Config-driven** — swap models, adjust retry limits, change timeouts — all in `config/models.yaml`, no code changes

---

## Project Structure

```
project_bmad/
├── streamlit_app.py          # Main UI — non-blocking, threading + queue
├── main.py                   # CLI entrypoint
├── workflow.py               # Compat shim → orchestration/graph.py
├── state.py                  # Compat shim → core/state.py
│
├── core/
│   ├── llm_factory.py        # Lazy LLM singleton, config-driven
│   ├── agent_runner.py       # Single run_agent() entry point for all LLM calls
│   ├── sanitizer.py          # sanitize_code() — auto-fixes LLM output
│   ├── observability.py      # Langfuse tracing, token/cost tracking
│   └── state.py              # BMADState TypedDict (19 fields)
│
├── agents_impl/              # 9 agent implementations
├── agents/                   # BMAD agent definition docs (.md)
├── skills/                   # BMAD skill definitions (.yaml)
├── templates/                # Output structure templates (.xml)
│
├── config/
│   ├── models.yaml           # LLM, retry limits, executor, cost coefficients
│   ├── workflow.yaml         # Graph topology documentation
│   └── pipeline_rules.yaml   # Constraints appended to every prompt
│
├── orchestration/
│   └── graph.py              # LangGraph StateGraph — 9 nodes, 2 feedback loops
│
├── tests/                    # Unit tests for sanitizer, validator, graph, agents
└── outputs/                  # Generated apps and session archives
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/saanvi3605/bmad_project
cd bmad_project
pip install -r requirements.txt
pip install plotly apscheduler  # for generated apps that use charts/scheduling
```

### 2. Set up environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Get your Groq API key free at [console.groq.com](https://console.groq.com).  
Langfuse keys are optional — observability is disabled gracefully if absent.

### 3. Run the pipeline UI

```bash
python -m streamlit run streamlit_app.py
```

### 4. Or run from CLI

```bash
python main.py "Build a task tracker with add, delete, and list tasks"
```

---

## Example Prompts

**Simple**
```
Build a personal expense tracker. Add expenses with amount, category and note.
Show monthly summary with total spent per category and a pie chart.
```

**Medium**
```
Build a job application tracker. Log jobs with company, role, date, status
(Applied/Interview/Offer/Rejected) and notes. Show a status board and analytics.
```

**Complex**
```
Build a restaurant order management system. Menu items have name, category,
price, and availability. Customers place orders with multiple items. Track
order status (Pending/Preparing/Ready/Delivered). Show live kitchen dashboard,
revenue by category, busiest hours chart, and average order value.
```

---

## Configuration

All runtime behaviour is controlled by `config/models.yaml` — no code changes needed:

```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile   # heavy agents
  temperature: 0.1
  max_tokens: 4096

llm_light:
  provider: groq
  model: llama-3.1-8b-instant      # light agents (tester, test_writer)
  max_tokens: 2048

retry_limits:
  validator_max_retries: 2
  reviewer_max_retries: 2

executor:
  startup_wait_seconds: 8
  output_file: outputs/generated_app.py
```

Supported providers: `groq`, `gemini` (set `provider:` and add the corresponding API key to `.env`).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover the sanitizer, validator, graph topology, agent runner lazy initialization, and test writer post-processing. All tests run without a real API key via monkeypatching.

---

## Generated App

After a pipeline run, the generated app lives at `outputs/generated_app.py`. Run it with:

```bash
streamlit run outputs/generated_app.py
```

Or use the **Launch Generated App** button in the pipeline UI — it starts the app on a free port and opens it in a new browser tab automatically.

Run the generated tests with:

```bash
pytest outputs/test_generated_app.py -v
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pipeline orchestration | LangGraph |
| LLM provider | Groq (llama-3.3-70b-versatile / llama-3.1-8b-instant) |
| LLM integration | LangChain |
| Observability | Langfuse |
| Pipeline UI | Streamlit |
| Generated apps | Streamlit + SQLite (sqlite3) |
| Charts in generated apps | Plotly Express |
| Config | YAML |
| Tests | pytest |
