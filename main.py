"""
main.py
FastAPI backend for BIS Standards Recommendation Engine.
Endpoints: /recommend (main), /upload, /health, /session/{id}
"""

import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from retriever  import load_indexes, reload_indexes, hybrid_retrieve, indexes_loaded as _indexes_loaded
from agent      import run_rag_agent
from ingestion  import run_ingestion
from query_expansion import expanded_retrieve
from config     import DOCS_DIR, TOP_K, MAX_HISTORY_TURNS

sessions: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_indexes()
    except Exception as e:
        print(f"WARNING: Could not load indexes at startup: {e}")
    yield


app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description="AI-powered RAG system for BIS building material standards",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──

class RecommendRequest(BaseModel):
    query:          str           # Product description
    session_id:     str = "default"
    top_k:          int = TOP_K
    use_expansion:  bool = True   # Enable query expansion


class StandardResult(BaseModel):
    standard_number: str
    title:           str
    category:        str
    rationale:       str
    confidence:      float


class RecommendResponse(BaseModel):
    query:            str
    standards:        list[StandardResult]
    standard_codes:   list[str]   # Just IS codes, for easy copy
    answer:           str         # Full LLM rationale text
    retries_used:     int
    validation:       str
    latency_seconds:  float
    session_id:       str


# ── Endpoints ──

@app.get("/")
def home():
    return {"message": "BIS Standards RAG API 🏗️", "version": "2.0"}


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "indexes_loaded": _indexes_loaded(),
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Main endpoint: takes a product description, returns ranked BIS standards.
    """
    import time

    # Ensure indexes are loaded
    if not _indexes_loaded():
        try:
            load_indexes()
        except Exception:
            pass
    if not _indexes_loaded():
        raise HTTPException(
            status_code=503,
            detail="Indexes not ready. Upload the SP 21 PDF and index it first."
        )

    t0 = time.time()

    # Retrieval (with or without query expansion)
    if req.use_expansion:
        results = expanded_retrieve(req.query, top_k=req.top_k)
    else:
        results = hybrid_retrieve(req.query, top_k=req.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant standards found.")

    # Run corrective RAG agent for rationale
    history = sessions.get(req.session_id, [])
    answer, retries, verdict = run_rag_agent(req.query, results, history)

    # Update session memory
    history.append(HumanMessage(content=req.query))
    history.append(AIMessage(content=answer))
    sessions[req.session_id] = history[-(MAX_HISTORY_TURNS * 2):]

    latency = round(time.time() - t0, 3)

    # Parse individual standard rationales from answer
    standards_out = []
    for r in results:
        standards_out.append(StandardResult(
            standard_number = r.get("standard_number", ""),
            title           = r.get("title", ""),
            category        = r.get("category", "General"),
            rationale       = r.get("scope", ""),   # Short rationale from metadata
            confidence      = r.get("confidence", 0.0),
        ))

    return RecommendResponse(
        query           = req.query,
        standards       = standards_out,
        standard_codes  = [r.get("standard_number", "") for r in results],
        answer          = answer,
        retries_used    = retries,
        validation      = verdict,
        latency_seconds = latency,
        session_id      = req.session_id,
    )


# Keep /query for backward compatibility
@app.post("/query")
async def query_compat(req: RecommendRequest):
    return await recommend(req)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload SP 21 PDF and trigger re-ingestion."""
    allowed = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files allowed.")

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        run_ingestion(pdf_path=dest)
        reload_indexes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status":    "indexed",
        "filename":  file.filename,
        "message":   "SP 21 indexed successfully. Ready for queries.",
        "standards": "Check /health for index status.",
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))