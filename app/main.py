"""
main.py
FastAPI application — exposes /ingest, /ask, /health, and static frontend.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import QueryType, route_query
from app.config import API_HOST, API_PORT, LOG_LEVEL
from app.llm import check_ollama_health, generate_answer
from app.rag import get_store_stats, ingest_documents, retrieve_context

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏥 MediAssist Healthcare AI starting up...")
    logger.info("Warming up embedding model...")
    from app.embeddings import get_embedding_model
    get_embedding_model()
    logger.info("Embedding model ready.")
    yield
    logger.info("MediAssist shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediAssist — Healthcare AI Assistant",
    description=(
        "A RAG-powered healthcare AI assistant for MediCare Health System. "
        "Answers patient questions from internal clinical and policy documents "
        "using local LLM inference via Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    data_dir: Optional[str] = Field(
        default=None,
        description="Path to documents directory. Defaults to config DATA_DIR."
    )


class IngestResponse(BaseModel):
    status: str
    ingested_files: List[str]
    total_chunks: int
    message: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Patient question")


class SourceCitation(BaseModel):
    document: str
    chunk: str
    relevance_distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    confidence: str
    query_type: str
    tool_used: str
    route_reason: str
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    ollama: Dict[str, Any]
    embedding_model: str
    version: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(request: IngestRequest):
    """
    Ingest healthcare documents into the vector store.
    Reads all .txt and .md files from the data directory,
    chunks them, generates embeddings, and stores in ChromaDB.
    """
    logger.info("POST /ingest — starting document ingestion.")
    try:
        from app.config import DATA_DIR
        data_dir = request.data_dir or DATA_DIR
        result = ingest_documents(data_dir)
        return IngestResponse(
            status="success",
            ingested_files=result["ingested_files"],
            total_chunks=result["total_chunks"],
            message=f"Successfully ingested {len(result['ingested_files'])} documents into {result['total_chunks']} chunks.",
        )
    except FileNotFoundError as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Question Answering"])
async def ask(request: AskRequest):
    """
    Answer a healthcare question using the RAG pipeline and local LLM.

    The agent first classifies the question:
    - Emergency → immediate safety response
    - Appointment → mock slot availability tool
    - Knowledge → RAG retrieval + LLM generation
    """
    start_time = time.time()
    question = request.question.strip()
    logger.info(f"POST /ask — question: '{question[:100]}'")

    # Step 1: Agent routing
    decision = route_query(question)
    logger.info(f"Agent decision: {decision.route_reason}")

    sources: List[SourceCitation] = []
    confidence = "high"

    # Step 2: Handle non-knowledge queries directly
    if decision.query_type != QueryType.KNOWLEDGE:
        answer = decision.tool_result
        confidence = "high" if decision.query_type == QueryType.EMERGENCY else "medium"
    else:
        # Step 3: RAG retrieval
        try:
            chunks, metadatas = retrieve_context(question)
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

        # Step 4: Build source citations
        for chunk, meta in zip(chunks, metadatas):
            sources.append(
                SourceCitation(
                    document=meta.get("source", "unknown"),
                    chunk=chunk[:300] + ("..." if len(chunk) > 300 else ""),
                    relevance_distance=meta.get("distance", 0.0),
                )
            )

        # Step 5: Determine confidence
        if not chunks:
            confidence = "none"
        elif len(chunks) >= 3:
            confidence = "high"
        elif len(chunks) >= 1:
            confidence = "medium"

        # Step 6: LLM generation
        try:
            answer = await generate_answer(question, chunks)
        except RuntimeError as e:
            logger.error(f"LLM error: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Response generated in {elapsed_ms}ms | confidence={confidence} | sources={len(sources)}")

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
        confidence=confidence,
        query_type=decision.query_type.value,
        tool_used=decision.tool_used,
        route_reason=decision.route_reason,
        response_time_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint. Returns status of vector store and Ollama LLM.
    """
    from app.config import EMBEDDING_MODEL
    store_stats = get_store_stats()
    ollama_stats = await check_ollama_health()

    overall = "healthy"
    if store_stats.get("status") == "empty":
        overall = "degraded — vector store empty, run POST /ingest"
    if not ollama_stats.get("ollama_running"):
        overall = "degraded — Ollama not running"
    if not ollama_stats.get("model_available"):
        overall = f"degraded — model '{ollama_stats['configured_model']}' not pulled"

    return HealthResponse(
        status=overall,
        vector_store=store_stats,
        ollama=ollama_stats,
        embedding_model=EMBEDDING_MODEL,
        version="1.0.0",
    )


# ── Static Frontend ────────────────────────────────────────────────────────────
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(static_dir / "index.html"))


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=True, log_level=LOG_LEVEL.lower())
