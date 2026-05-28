"""
agents_impl/developer_agent.py
────────────────────────────────────────────────────────────────────────────────
Developer agent — generates production-quality apps.

Supports two modes controlled by state["project_mode"]:
  streamlit_crud  — single-file Streamlit + SQLite (original behaviour)
  fastapi_rag     — multi-file FastAPI + ChromaDB + LangChain + Langfuse RAG service

Feedback modes (runtime_fix > validation_fix > review_fix > clean) apply in
both project modes.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from core.agent_runner import run_agent
from core.sanitizer import sanitize_code, sanitize_rag_files
from core.multi_file_parser import parse_multi_file

_AGENT_NAME = "developer"

PROMPT_TEMPLATE = """You are a senior Python developer. Generate a COMPLETE, fully runnable Streamlit app as a SINGLE Python file.

Technical Design:
{technical_design}

{feedback_section}

RULES (all mandatory):
- Return ONLY raw Python code — no markdown, no backticks, no explanations
- Streamlit ONLY — no FastAPI, Flask, uvicorn
- SQLite via sqlite3 stdlib ONLY — no SQLAlchemy, no ORM
- Single file — no fake section headers like "# models.py"
- st.set_page_config() must be the very first Streamlit call
- init_db() called once at module level; use CREATE TABLE IF NOT EXISTS
- DB reads: @st.cache_data(ttl=5); DB writes: commit then st.cache_data.clear() + st.rerun()
- All data entry via st.form() + st.form_submit_button() — never bare st.button()
- Validate every form input; show st.error() on bad data
- st.metric() for KPIs; st.dataframe(df, use_container_width=True, hide_index=True) for tables
- Sidebar navigation with emoji labels; plotly.express for all charts
- Every section in the technical design must be fully implemented — no TODOs, no stubs
- Never use generic placeholder labels like "Subject 1", "Field 1", "Item 1" — always use descriptive real-world labels from the technical design (e.g. "Mathematics", "Science", "English" for a grade manager; "Amount", "Category", "Description" for an expense tracker)
- For integer inputs (marks, counts, quantities, IDs) always use format="%d" and step=1 — NEVER format="%.2f"
- Only use format="%.2f" for currency or decimal inputs like prices, rates, and ratios
- SQLite table names must use snake_case — never spaces (e.g. menu_items not "Menu Items", order_details not "Order Details")
"""

UI_RULES = ""

# ---------------------------------------------------------------------------
# RAG mode prompt template
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a senior Python backend engineer. Generate a COMPLETE, production-quality FastAPI RAG service.

Technical Design:
{technical_design}

{feedback_section}

OUTPUT FORMAT — mandatory:
Use ### FILE: <filename> ### before EVERY file. Generate ALL four files.

### FILE: main.py ###
<complete FastAPI application>

### FILE: requirements.txt ###
<pip requirements, one per line>

### FILE: docker-compose.yml ###
<docker-compose configuration>

### FILE: .env.example ###
<environment variable template>

RULES FOR main.py (ALL mandatory):
- Import: fastapi, chromadb, langchain, anthropic, langfuse, pydantic, dotenv
- ChromaDB: client = chromadb.PersistentClient(path="./chroma_db") at module level
- collection = client.get_or_create_collection("documents", metadata={{"hnsw:space":"cosine"}})
- Langfuse: lf = Langfuse() at module level; create trace per /query call
- CORS: app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ENDPOINTS:
1. GET /health  → {{"status": "ok", "doc_count": N, "version": "1.0.0"}}
2. POST /ingest → accepts UploadFile, supports .pdf .yaml .json .md .txt
   - Parse content based on file type
   - Split with RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
   - Embed with Anthropic (voyage-3 model) OR use a hash-based mock embedding if voyage unavailable
   - Store in ChromaDB with metadata: {{"filename": ..., "chunk_index": ..., "timestamp": ...}}
   - Return {{"status": "ok", "chunks_added": N, "filename": ...}}
3. POST /query → body: {{"question": str, "top_k": int = 5}}
   - Create Langfuse trace: lf.start_observation(name="rag_query", as_type="span", ...)
   - Child span "embedding": embed the question
   - Child span "retrieval": collection.query(query_embeddings=..., n_results=top_k)
   - Child span "prompt_build": assemble context + question into prompt
   - Child span "llm_call": anthropic client.messages.create(model="claude-3-5-sonnet-20241022", ...)
   - Child span "evaluation": score faithfulness (1-10) with a second LLM call
   - End trace; push faithfulness score to Langfuse
   - Return {{"answer": str, "citations": [{{"filename": str, "snippet": str, "score": float}}], "latency_ms": float, "faithfulness": float}}
4. GET /metrics → {{"total_docs": N, "total_chunks": N, "avg_query_latency_ms": float, "last_ingested": str}}

PYDANTIC MODELS required:
- QueryRequest(question: str, top_k: int = 5)
- Citation(filename: str, snippet: str, score: float)
- QueryResponse(answer: str, citations: list[Citation], latency_ms: float, faithfulness: float)
- IngestResponse(status: str, chunks_added: int, filename: str)

EMBEDDING STRATEGY:
- Try Anthropic voyage-3 first: client.embeddings.create(model="voyage-3", input=[text])
- Fallback: use a simple deterministic hash-based 384-dim float vector so the app works without voyage API access

LANGFUSE TRACING:
- lf = Langfuse() once at module level
- Per /query: root_span = lf.start_observation(name="rag_query", as_type="span", trace_context={{"session_id": request_id}})
- Child spans: root_span.start_observation(name="embedding", as_type="span")
- Always call span.end() in a finally block
- Push score: lf.score_current_span(name="faithfulness", value=score)

RULES FOR requirements.txt:
- fastapi>=0.110.0
- uvicorn[standard]>=0.27.0
- chromadb>=0.4.0
- langchain>=0.2.0
- langchain-community>=0.2.0
- anthropic>=0.25.0
- langfuse>=4.0.0
- python-multipart>=0.0.9
- python-dotenv>=1.0.0
- pydantic>=2.0.0
- pypdf>=4.0.0
- pyyaml>=6.0

RULES FOR docker-compose.yml:
- Single service "api" built from Dockerfile
- Port mapping 8000:8000
- Volume mounts: ./chroma_db:/app/chroma_db and ./uploads:/app/uploads
- env_file: .env

RULES FOR .env.example:
- ANTHROPIC_API_KEY=your_key_here
- LANGFUSE_PUBLIC_KEY=pk-lf-...
- LANGFUSE_SECRET_KEY=sk-lf-...
- LANGFUSE_HOST=https://cloud.langfuse.com
- CHROMA_PATH=./chroma_db

CRITICAL: Return ONLY the file contents with ### FILE: ### delimiters. No prose, no explanations.
"""


def developer_agent(state: dict[str, Any]) -> dict[str, Any]:
    project_mode = state.get("project_mode", "streamlit_crud")

    # ── Feedback-mode detection (runtime > validation > review > clean) ──────
    feedback_section = ""
    if state.get("runtime_error"):
        feedback_section = (
            "RUNTIME CRASH — the generated app passed validation but crashed "
            "at startup with this error:\n"
            + state["runtime_error"]
            + "\n\nDiagnose and fix this runtime error. "
            "Keep all other working functionality intact."
        )
        prompt_key = "developer_runtime_fix_v1"
        state["runtime_error"] = ""
    elif state.get("validation_error") and not state.get("validation_passed"):
        feedback_section = (
            "VALIDATION FAILED — your previous code had this error:\n"
            + state["validation_error"]
            + "\n\nFix this specific error. Keep everything else intact."
        )
        prompt_key = "developer_validation_fix_v1"
    elif state.get("review_feedback") and not state.get("review_approved"):
        feedback_section = (
            "CODE REVIEW FAILED — the reviewer rejected your code:\n"
            + state["review_feedback"]
            + "\n\nFix ALL issues listed above. Do not remove any working functionality."
        )
        prompt_key = "developer_review_fix_v1"
    else:
        prompt_key = "developer_clean_v1"

    # ── Route to the correct prompt template ─────────────────────────────────
    if project_mode == "fastapi_rag":
        prompt = RAG_PROMPT_TEMPLATE.format(
            technical_design=state["technical_design"],
            feedback_section=feedback_section,
        )
        content = run_agent(
            prompt=prompt,
            agent_name=_AGENT_NAME,
            prompt_key=f"rag_{prompt_key}",
            state=state,
        )
        # Parse multi-file output
        files = parse_multi_file(content)
        sanitized = sanitize_rag_files(files)
        # main.py → state["code"] (for validator); rest → state["extra_files"]
        state["code"] = sanitized.pop("main.py", "")
        state["extra_files"] = sanitized
        print(f"[Developer] RAG mode — generated files: {list(sanitized.keys()) + ['main.py']}")
    else:
        prompt = PROMPT_TEMPLATE.format(
            technical_design=state["technical_design"],
            feedback_section=feedback_section,
        ) + UI_RULES
        content = run_agent(
            prompt=prompt,
            agent_name=_AGENT_NAME,
            prompt_key=prompt_key,
            state=state,
        )
        state["code"] = sanitize_code(content)

    return state
