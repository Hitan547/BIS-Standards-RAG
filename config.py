import os
import warnings
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    warnings.warn("GROQ_API_KEY not set — LLM calls will fail")

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── LLM ──
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Paths ──
DOCS_DIR         = os.path.join(_BASE, "docs")
DATA_DIR         = os.path.join(_BASE, "data")
FAISS_INDEX_PATH = os.path.join(_BASE, "faiss.index")
BM25_PATH        = os.path.join(_BASE, "bm25.pkl")
CHUNKS_PATH      = os.path.join(_BASE, "chunks.pkl")
SOURCES_PATH     = os.path.join(_BASE, "sources.pkl")
METADATA_PATH    = os.path.join(_BASE, "metadata.pkl")   # NEW: stores per-chunk IS metadata
STANDARDS_PATH   = os.path.join(_BASE, "standards.json") # NEW: full parsed standards list

# ── Models — upgraded for better retrieval ──
EMBEDDER_NAME  = "BAAI/bge-small-en-v1.5"          # Better than all-MiniLM for retrieval
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Retrieval ──
TOP_K         = 5
MAX_RETRIES   = 3
MAX_HISTORY_TURNS = 5

# ── BIS-specific ──
BIS_CATEGORIES = ["Cement", "Steel", "Concrete", "Aggregates", "Bricks",
                  "Tiles", "Glass", "Timber", "Paint", "Lime", "Gypsum",
                  "Waterproofing", "Insulation", "General"]

# IS code pattern: matches "IS 8112", "IS 8112:2013", "IS/ISO 8112" etc.
IS_CODE_PATTERN = r"IS\s*[/:·]?\s*\d{1,5}(?:\s*(?::|:)\s*\d{4})?"