"""
core/mode_detector.py
────────────────────────────────────────────────────────────────────────────────
Detects the project mode from a user request string.

Two modes are supported:
  streamlit_crud  — the original BMAD mode: single-file Streamlit + SQLite app
  fastapi_rag     — new: multi-file FastAPI + ChromaDB + LangChain + Langfuse RAG service

Detection is keyword-based (fast, no LLM call).  Each keyword group is scored;
the group with the highest score wins.  Ties go to streamlit_crud (conservative).

Called by:
  main.py              — CLI entrypoint
  streamlit_app.py     — sets default in UI mode selector
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

_RAG_KEYWORDS: list[str] = [
    # Strong / unambiguous RAG signals (each worth 1 point)
    "rag", "retrieval augmented", "retrieval-augmented",
    "vector database", "vector db", "vector store",
    "semantic search", "similarity search",
    "chat with documents", "chat with pdf", "chat with files",
    "fastapi", "fast api", "rest api", "restful api",
    "chromadb", "chroma db", "pinecone", "weaviate", "qdrant", "milvus",
    "langchain", "llamaindex", "llama index",
    "llm-as-a-judge", "llm as a judge",
    "faithfulness", "relevance score",
    "evaluation pipeline",
    "/query endpoint", "/ingest endpoint",
    "openapi", "swagger",
    # Weaker signals — these alone should NOT trigger RAG mode; they need
    # company from stronger signals above to push the score high enough.
    "embedding", "embeddings",
    "chunking", "text chunking",
    "document ingestion", "document retrieval", "document search",
    "knowledge assistant",
    "ingest", "ingestion",
    "retrieval", "retrieve",
]

_STREAMLIT_KEYWORDS: list[str] = [
    "streamlit", "crud", "dashboard", "tracker", "manager",
    "form", "sqlite", "database app", "data entry",
    "admin panel", "inventory", "task list", "todo",
    "expense", "budget", "grade", "attendance",
    "point of sale", "pos system", "booking",
    "student", "employee", "product catalog",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MODES = ("streamlit_crud", "fastapi_rag")


def detect_mode(user_request: str) -> str:
    """
    Return "fastapi_rag" or "streamlit_crud" based on keyword scoring.

    Parameters
    ----------
    user_request : str
        Raw user request (before pipeline rules are appended).

    Returns
    -------
    str  One of the two mode strings.
    """
    lower = user_request.lower()

    rag_score = sum(1 for kw in _RAG_KEYWORDS if kw in lower)
    streamlit_score = sum(1 for kw in _STREAMLIT_KEYWORDS if kw in lower)

    # Require at least 3 RAG signals AND a clear margin over Streamlit signals.
    # This prevents single ambiguous words like "knowledge base", "embedding",
    # or "chunk" from accidentally triggering FastAPI RAG mode.
    if rag_score >= 3 and rag_score > streamlit_score + 1:
        return "fastapi_rag"
    return "streamlit_crud"


def mode_label(mode: str) -> str:
    """Human-readable label for display in UI."""
    return {
        "streamlit_crud": "Streamlit CRUD App",
        "fastapi_rag":    "FastAPI RAG Service",
    }.get(mode, mode)
