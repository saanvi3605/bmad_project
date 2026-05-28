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

COMMON MISTAKES — every item below has broken generated apps before. Do NOT repeat them:

## Function ordering
- Define EVERY helper function BEFORE it is called. If page A calls generate_answer(), score_chunk(),
  retrieve_top_chunks() — all three must appear above page A's code block, not below it.
- Never call a function at module level before its def appears in the file.

## Streamlit API versions
- NEVER use st.experimental_rerun() — it was removed in Streamlit 1.30. Use st.rerun() instead.
- NEVER use st.experimental_get_query_params() — use st.query_params instead.
- NEVER use st.experimental_set_query_params() — use st.query_params.update() instead.
- NEVER use st.experimental_memo or st.experimental_singleton — use st.cache_data / st.cache_resource.

## Form submit button placement
- st.form_submit_button() must be called UNCONDITIONALLY at the top level of the with st.form(): block.
  Do NOT put it inside an if/else branch within the form. Streamlit shows "This form has no submit
  button" whenever the condition is False.
  WRONG:
    with st.form("f"):
        x = st.text_input("x")
        if some_condition:
            submitted = st.form_submit_button("Save")   # hidden when condition is False!
  RIGHT:
    with st.form("f"):
        x = st.text_input("x")
        submitted = st.form_submit_button("Save")       # always visible
    if submitted and some_condition:
        ...

## Imports
- Write each import exactly once. Duplicate imports cause isort/black to emit warnings and make
  the file harder to read. If you need os and datetime, import them once at the top.
- Never import python_dotenv — the package is python-dotenv but the import name is dotenv:
    from dotenv import load_dotenv   # correct
    import python_dotenv             # WRONG — this package does not exist

## SQLite constraints
- Use CHECK(col != '') on TEXT NOT NULL columns that must not be empty strings. NOT NULL alone
  does NOT prevent empty strings in SQLite.
  Example: name TEXT NOT NULL CHECK(name != '')
- Use CHECK(score IN (0, 1)) or CHECK(value BETWEEN 0 AND 100) for bounded integers.
- For schema migrations (columns added after initial release), use PRAGMA table_info(table) to
  detect missing columns and ALTER TABLE … ADD COLUMN — never drop and recreate a table with data.

## Session state and reruns
- Always initialise every session_state key before reading it:
    if "my_key" not in st.session_state:
        st.session_state.my_key = default_value
- After a st.rerun() the entire script re-executes from line 1. Do not assume local variables
  survive a rerun — persist anything you need in st.session_state.
- To show feedback buttons (👍/👎) after a form submission, store the conversation/record id in
  st.session_state inside the if submitted: block, then read it OUTSIDE the form for rendering.

## Completeness
- Never leave placeholder implementations like "return 'This is a placeholder answer'" or
  "pass  # TODO". Every function must contain real, working logic.
- If an LLM call is part of the design, implement it fully using the Groq SDK:
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[...], max_tokens=1024)
    answer = resp.choices[0].message.content
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

LANGFUSE TRACING (Langfuse v4 SDK — use EXACTLY this API):
- lf = Langfuse() once at module level
- Per /query call:
    trace = lf.trace(name="rag_query", session_id=str(uuid.uuid4()))
    emb_span   = trace.span(name="embedding");   emb_span.end(output={{"dims": len(vec)}})
    ret_span   = trace.span(name="retrieval");   ret_span.end(output={{"n": top_k}})
    build_span = trace.span(name="prompt_build"); build_span.end()
    llm_span   = trace.span(name="llm_call");    llm_span.end(output={{"answer": answer[:100]}})
    eval_span  = trace.span(name="evaluation");  eval_span.end(output={{"faithfulness": score}})
    trace.score(name="faithfulness", value=float(score))
    lf.flush()
- NEVER use lf.start_observation() — that method does not exist in Langfuse v4
- NEVER use lf.score_current_span() — use trace.score() instead
- Import uuid at the top for session IDs

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

COMMON MISTAKES — every item below has broken generated RAG services before. Do NOT repeat them:

## LangChain import paths (these changed in LangChain 0.2+)
- NEVER use: from langchain.embeddings.anthropic import AnthropicEmbeddings
- NEVER use: from langchain.llms.anthropic import Anthropic
- NEVER use: from langchain.chat_models import ChatAnthropic
- These old paths no longer exist. Use the Anthropic SDK directly instead:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(model="claude-3-5-sonnet-20241022",
                                   max_tokens=1024,
                                   messages=[{{"role":"user","content":prompt}}])
    answer = resp.content[0].text
- For embeddings use the voyageai package or the hash-based fallback — never AnthropicEmbeddings

## Anthropic client usage
- The Anthropic() object itself has no .messages — call client.messages.create(), NOT llm.messages.create()
- WRONG:  llm = Anthropic(); response = llm.messages.create(...)
- RIGHT:  client = Anthropic(); response = client.messages.create(...)
- resp.content[0].text is the answer string — NOT resp.choices[0].message.content (that's OpenAI)

## Langfuse v4 API — lf.start_observation() does not exist
- WRONG (v2 API, completely broken in v4):
    root_span = lf.start_observation(name="rag_query", as_type="span", ...)
    child = root_span.start_observation(name="embedding", as_type="span")
    lf.score_current_span(name="faithfulness", value=score)
- RIGHT (v4 API):
    trace = lf.trace(name="rag_query", session_id=session_id)
    span  = trace.span(name="embedding")
    span.end()
    trace.score(name="faithfulness", value=float(score))
    lf.flush()

## QueryRequest must not reference fields that don't exist
- QueryRequest has ONLY: question (str) and top_k (int = 5)
- NEVER access request.session_id — it is not in the model
- Generate a session_id inside the endpoint: session_id = str(uuid.uuid4())

## ChromaDB — always store document text, not just metadata
- WRONG (stores metadata only — retrieval returns no readable text):
    collection.add(ids=[...], embeddings=[...], metadatas=[...])
- RIGHT (stores the actual chunk text so it can be returned in citations):
    collection.add(ids=[...], embeddings=[...], documents=[chunk_text], metadatas=[...])
- When querying, include_documents=True is the default; results["documents"][0] is the list of texts

## Citations must contain readable snippet text, not integers
- results["metadatas"][0] is a list of dicts; results["documents"][0] is a list of strings
- WRONG: Citation(snippet=result["chunk_index"])   ← chunk_index is an integer
- RIGHT: Citation(snippet=doc_text[:200])           ← doc_text from results["documents"][0][i]

## Faithfulness score must be a float, not a response object
- The second LLM call returns a Messages object — you must parse the number out of the text:
    raw = client.messages.create(...).content[0].text   # e.g. "8" or "Score: 7/10"
    import re
    m = re.search(r'\\d+', raw)
    faithfulness = float(m.group()) / 10.0 if m else 0.5
- NEVER pass the response object directly to int() or float()

## File reading in FastAPI sync endpoints
- file.read() returns a coroutine in async context; in sync endpoints use file.file.read():
    contents = file.file.read()          # sync endpoint
    contents = await file.read()         # async endpoint (def query → async def query)
- For PDF: pass the file object to PdfReader, not the raw bytes decoded as utf-8:
    from io import BytesIO
    reader = PdfReader(BytesIO(contents))

## Hash-based embedding fallback must produce valid float vectors
- Python's hash() returns a large integer — ChromaDB needs floats in a consistent range
- WRONG:  [[hash(chunk)] * 384]   ← single huge integer, not 384 floats
- RIGHT:
    import hashlib
    def hash_embed(text: str, dim: int = 384) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()  # 16 bytes
        rng = [b / 255.0 - 0.5 for b in digest]      # 16 floats in [-0.5, 0.5]
        # tile to fill dim dimensions
        return (rng * (dim // len(rng) + 1))[:dim]

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
