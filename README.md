# BMAD AI Orchestration Pipeline

A production-grade AI pipeline that generates complete, runnable **Streamlit + SQLite applications** from a single natural language prompt. Describe what you want to build — the pipeline plans, architects, codes, validates, reviews, executes, and tests it automatically.

---

## How It Works

The pipeline scores the prompt first, then routes it:

```
ComplexityScorer
    ├── score ≤ 4 → SimpleGenerator (A2A) ──────────────────────────┐
    └── score > 4 → PromptRefiner → Planner → Architect → Developer │
                                                              │       │
                                              Validator ←────┘       │
                                                  │                   │
                                               Tester → Reviewer      │
                                                             │         │
                                                          Executor ←──┘
                                                             │
                                            TestWriter → ReadmeWriter → EvalAgent → END
```

**Three feedback loops:**
- Validator → Developer (static analysis, max 2 retries)
- Reviewer → Developer (LLM quality check, max 2 retries)
- Executor → Developer (runtime crash self-healing, max 2 retries)

| Agent | Role | Model |
|---|---|---|
| **ComplexityScorer** | Scores prompt 1-10, routes to A2A or full pipeline | Light (8b) |
| **SimpleGenerator** | Calls the A2A microservice for simple prompts | — |
| **PromptRefiner** | Expands vague prompts into structured specs | Light (8b) |
| **Planner** | Writes functional specification | Light (8b) |
| **Architect** | Designs schema, forms, business logic | Heavy (70b) |
| **Developer** | Generates the full Streamlit app | Heavy (70b) |
| **Validator** | Deterministic static checks — no LLM | — |
| **Tester** | Writes test case specifications | Light (8b) |
| **Reviewer** | Reviews code against quality criteria | Heavy (70b) |
| **Executor** | Launches the app subprocess; sends crashes back to Developer | — |
| **TestWriter** | Generates a runnable pytest file | Light (8b) |
| **ReadmeWriter** | Generates a README for the produced app | Light (8b) |
| **EvalAgent** | Scores the final output on multiple dimensions | Heavy (70b) |

---

## Features

- **Natural language to working app** — describe it, get a fully runnable Streamlit app
- **A2A routing** — simple prompts (score ≤ 4) are routed to a separate microservice; complex prompts run the full pipeline
- **Dual-model strategy** — heavy model (llama-3.3-70b) for planning and coding, light model (llama-3.1-8b) for lighter tasks
- **Self-healing executor** — if the generated app crashes, stderr is sent back to Developer for up to 2 automatic fixes
- **Deterministic validator** — AST parse, py_compile, required patterns, disallowed libraries, NameError detection, and more
- **Auto-sanitizer** — fixes common LLM mistakes: wrong imports, deprecated APIs, SQL naming issues
- **Langfuse observability** — per-agent token counts, cost estimates, latency, and grouped traces per run
- **MCP server** — exposes Langfuse analytics as tools inside Claude Code (`get_usage_summary`, `get_daily_trends`, `get_recent_traces`, `ask_langfuse`)
- **Pipeline History tab** — every run saved with full artifacts, one-click re-run of any past prompt
- **Config-driven** — swap models, adjust retry limits, change timeouts — all in `config/models.yaml`

---

## Project Structure

```
project_bmad/
├── streamlit_app.py          # Main UI
├── main.py                   # CLI entrypoint
├── langfuse_mcp.py           # Custom MCP server — Langfuse analytics tools
├── .mcp.json                 # MCP server registration for Claude Code
│
├── core/
│   ├── llm_factory.py        # LLM singleton, config-driven
│   ├── agent_runner.py       # Single entry point for all LLM calls
│   ├── sanitizer.py          # Auto-fixes LLM output
│   ├── observability.py      # Langfuse tracing, token/cost tracking
│   └── state.py              # BMADState TypedDict
│
├── agents_impl/              # All agent implementations
│   ├── complexity_scorer_agent.py
│   ├── a2a_client.py         # SimpleGenerator — calls project_final_final
│   ├── developer_agent.py    # Handles clean / validation_fix / review_fix modes
│   ├── validator_agent.py
│   └── ...
│
├── config/
│   ├── models.yaml           # LLM config, retry limits, A2A threshold
│   └── pipeline_rules.yaml   # Constraints appended to every prompt
│
├── orchestration/
│   └── graph.py              # LangGraph StateGraph
│
├── tests/
└── outputs/
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/saanvi3605/bmad_project
cd bmad_project
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Langfuse keys are optional — observability is disabled gracefully if absent.

### 3. Run

```bash
# Streamlit UI
streamlit run streamlit_app.py

# CLI
python main.py "Build a task tracker with add, delete, and list tasks"
```

### 4. A2A microservice (optional)

To enable A2A routing for simple prompts, start the microservice first:

```bash
cd path/to/project_final_final
python service.py        # runs on localhost:8001
```

Then run the main pipeline as normal — prompts scoring ≤ 4 will automatically route to it.

---

## Example Prompts

**Simple** (routes to A2A)
```
Show the current date and time
```

**Medium** (full pipeline)
```
Build a job application tracker. Log jobs with company, role, date, and status.
Show a status board and analytics.
```

**Complex** (full pipeline)
```
Build a restaurant order management system with menu management, order tracking,
a live kitchen dashboard, and revenue analytics.
```

---

## Configuration

All runtime behaviour is controlled by `config/models.yaml`:

```yaml
llm:
  model: llama-3.3-70b-versatile   # heavy agents

llm_light:
  model: llama-3.1-8b-instant      # light agents

retry_limits:
  validator_max_retries: 2
  reviewer_max_retries: 2
  executor_max_retries: 2

complexity_scorer:
  a2a_threshold: 4     # prompts scoring ≤ this go to A2A
  simple_threshold: 4  # prompts scoring ≤ this use the light model
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests run without a real API key via monkeypatching.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pipeline orchestration | LangGraph |
| LLM provider | Groq (llama-3.3-70b / llama-3.1-8b) |
| LLM integration | LangChain |
| Observability | Langfuse |
| MCP server | FastMCP (mcp library) |
| A2A communication | FastAPI + requests |
| Pipeline UI | Streamlit |
| Generated apps | Streamlit + SQLite |
| Config | YAML |
| Tests | pytest |
