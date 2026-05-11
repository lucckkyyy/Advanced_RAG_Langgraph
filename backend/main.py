"""
FastAPI Backend — Advanced Cyclic RAG with LangGraph & Groq

Endpoints:
  POST /ask        — Run the full cyclic RAG pipeline
  GET  /health     — Health check
  GET  /docs-count — Number of documents in knowledge base
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from rag.knowledge_base import seed_knowledge_base
from graph.rag_graph import run_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — seeding knowledge base...")
    seed_knowledge_base()
    logger.info("✅ Ready!")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Advanced Cyclic RAG API",
    description="Production-grade RAG with LangGraph, Groq, and self-correction loops.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    question: str
    rewritten_question: str | None
    retrieval_attempts: int
    hallucination_flag: str
    num_docs_used: int


@app.get("/health")
async def health():
    return {"status": "ok", "service": "advanced-rag-langgraph"}


@app.post("/ask", response_model=RAGResponse)
async def ask(req: QuestionRequest):
    """
    Run the cyclic RAG pipeline.
    The graph will:
    1. Retrieve documents
    2. Grade relevance (filter noisy docs)
    3. Rewrite query + re-retrieve if needed (cyclic loop)
    4. Generate answer with Groq
    5. Check for hallucinations
    6. Regenerate if hallucination detected
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = run_rag(req.question)
        return RAGResponse(**result)
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs-count")
async def docs_count():
    """Return number of documents in the knowledge base."""
    import chromadb
    import os
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    try:
        col = client.get_collection("ai_ml_knowledge_base")
        return {"count": col.count(), "topic": "AI & ML Concepts"}
    except Exception:
        return {"count": 0, "topic": "AI & ML Concepts"}
