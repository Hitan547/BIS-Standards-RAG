"""
retriever.py
Hybrid retrieval: FAISS dense + BM25 sparse + RRF fusion + Cross-encoder reranking.
Upgraded for BIS hackathon:
  - Loads standard_number + category metadata
  - Returns IS codes directly
  - Adds normalized confidence score (0-1) per result
  - BGE query prefix for better semantic search
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    FAISS_INDEX_PATH, BM25_PATH, CHUNKS_PATH,
    SOURCES_PATH, METADATA_PATH, EMBEDDER_NAME, RERANKER_MODEL
)

# ── Globals (loaded once at startup) ──
_faiss_index = None
_bm25_index  = None
_chunks      = None
_sources     = None
_metadata    = None   # List of {standard_number, title, category, scope}
_model       = None
_reranker    = None


def indexes_loaded() -> bool:
    return _faiss_index is not None


def load_indexes():
    global _faiss_index, _bm25_index, _chunks, _sources, _metadata, _model, _reranker

    if not os.path.exists(FAISS_INDEX_PATH):
        print("WARNING: No FAISS index found. Run ingestion first.")
        return

    _faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(BM25_PATH,    "rb") as f: _bm25_index = pickle.load(f)
    with open(CHUNKS_PATH,  "rb") as f: _chunks     = pickle.load(f)
    with open(SOURCES_PATH, "rb") as f: _sources    = pickle.load(f)

    # Load metadata (new file — fall back gracefully if missing)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f: _metadata = pickle.load(f)
    else:
        # Fallback: build minimal metadata from sources
        _metadata = [{"standard_number": s, "title": "", "category": "General", "scope": ""}
                     for s in _sources]

    _model    = SentenceTransformer(EMBEDDER_NAME)
    _reranker = CrossEncoder(RERANKER_MODEL)

    print(f"✅ Indexes loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} standards")


def reload_indexes():
    global _faiss_index, _bm25_index, _chunks, _sources, _metadata, _model, _reranker
    _faiss_index = _bm25_index = _chunks = _sources = _metadata = _model = _reranker = None
    load_indexes()


def _reciprocal_rank_fusion(lists: list[list], k: int = 60) -> dict:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict = {}
    for ranked_list in lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize a list of scores to [0, 1] range."""
    if not scores:
        return scores
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full hybrid retrieval pipeline for a single query.

    Args:
        query:  Product description or question
        top_k:  Number of top standards to return

    Returns:
        List of dicts with chunk, standard_number, title, category,
        rrf_score, ce_score, confidence (0-1)
    """
    if not indexes_loaded():
        raise RuntimeError("Indexes not loaded. Call load_indexes() first.")

    # ── 1. Dense retrieval (BGE needs query prefix) ──
    query_text = f"Represent this building material query for retrieval: {query}"
    q_emb = _model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype("float32")

    _, dense_ids = _faiss_index.search(q_emb, top_k * 4)
    dense_ranking = [int(i) for i in dense_ids[0] if i >= 0]

    # ── 2. Sparse retrieval (BM25) ──
    bm25_scores    = _bm25_index.get_scores(query.lower().split())
    sparse_ranking = np.argsort(bm25_scores)[::-1][: top_k * 4].tolist()

    # ── 3. RRF Fusion ──
    rrf_scores = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    fused_ids  = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: top_k * 2]

    # ── 4. Cross-encoder reranking ──
    candidates = [(query, _chunks[i]) for i in fused_ids]
    ce_scores  = _reranker.predict(candidates)

    # Sort by cross-encoder score
    ranked = sorted(
        zip(fused_ids, ce_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Normalize cross-encoder scores to confidence [0, 1]
    raw_ce  = [float(s) for _, s in ranked]
    normed  = _normalize_scores(raw_ce)

    # ── 5. Build result list ──
    results = []
    for (idx, ce_score), confidence in zip(ranked, normed):
        meta = _metadata[idx] if _metadata else {}
        results.append({
            "chunk":           _chunks[idx],
            "source":          _sources[idx],
            "standard_number": meta.get("standard_number", _sources[idx]),
            "title":           meta.get("title", ""),
            "category":        meta.get("category", "General"),
            "scope":           meta.get("scope", ""),
            "chunk_id":        idx,
            "rrf_score":       round(float(rrf_scores.get(idx, 0)), 4),
            "ce_score":        round(float(ce_score), 4),
            "confidence":      round(confidence, 4),   # 0.0 – 1.0
        })

    results = [r for r in results if r["confidence"] >= 0.60]
    return results


def get_standard_numbers(results: list[dict]) -> list[str]:
    """
    Extract IS code strings from retrieval results.
    Used by inference.py for the output JSON.
    """
    return [r["standard_number"] for r in results if r.get("standard_number")]