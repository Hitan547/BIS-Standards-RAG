"""
ingestion.py
Builds FAISS + BM25 indexes from BIS SP 21 parsed standards.
Each 'chunk' = one IS standard entry (not a paragraph).
Metadata (standard_number, category) stored alongside each chunk.

FIXES vs original:
1. build_chunk_text() now includes raw_text so LLM has actual PDF content.
2. BM25 tokenisation includes raw_text for better keyword recall on exact
   technical terms (IS codes, property names) that queries may contain.
3. _sanity_check() prints parse quality ratios right after parsing — catches
   bad PDF extraction before wasting time building embeddings.
4. Long standards (raw_text > 800 chars) produce a second chunk covering
   requirements/body text for better embedding granularity. Both chunks share
   the same standard_number so retriever deduplication still works correctly.
"""

import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from bis_parser import parse_sp21_pdf, save_standards, load_standards
from config import (
    DOCS_DIR, FAISS_INDEX_PATH, BM25_PATH, CHUNKS_PATH,
    SOURCES_PATH, METADATA_PATH, STANDARDS_PATH, EMBEDDER_NAME
)

SPLIT_THRESHOLD = 800   # raw_text chars above which a second chunk is added


def find_sp21_pdf() -> str | None:
    """Find the SP 21 PDF in the docs directory."""
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(exist_ok=True)

    pdfs = list(docs_path.glob("*.pdf"))
    if not pdfs:
        return None

    for pdf in pdfs:
        if "sp" in pdf.name.lower() and "21" in pdf.name:
            return str(pdf)

    return str(pdfs[0])


def build_embedding_text(standard: dict) -> str:
    """
    Build the text embedded for a standard (concise, semantically rich).
    """
    parts = [
        standard.get("standard_number", ""),
        standard.get("title", ""),
        f"Category: {standard.get('category', '')}",
        standard.get("scope", ""),
    ]
    reqs = standard.get("key_requirements", [])
    if reqs:
        parts.append("Requirements: " + ". ".join(reqs[:3]))

    return " ".join(p for p in parts if p).strip()


def build_chunk_text(standard: dict) -> str:
    """
    Build the text stored as the 'chunk' — used by LLM for context.

    FIX: Includes raw_text so the LLM receives actual PDF content, not
    just the parsed structured fields. This improves rationale quality
    and reduces hallucination risk because the LLM can cite real text.
    """
    lines = [
        f"Standard: {standard.get('standard_number', 'Unknown')}",
        f"Title: {standard.get('title', 'N/A')}",
        f"Category: {standard.get('category', 'General')}",
    ]
    if standard.get("scope"):
        lines.append(f"Scope: {standard['scope']}")
    if standard.get("key_requirements"):
        lines.append("Key Requirements:")
        for req in standard["key_requirements"][:5]:
            lines.append(f"  - {req}")

    # FIX: Include raw PDF text for richer LLM context
    raw = standard.get("raw_text", "")
    scope_len = len(standard.get("scope", ""))
    if raw and len(raw) > scope_len + 50:
        lines.append(f"\nFull text excerpt:\n{raw[:600]}")

    return "\n".join(lines)


def _sanity_check(standards: list[dict]) -> None:
    """
    Print a quality report after parsing.
    Low coverage ratios usually mean bad PDF encoding or a scanned document.
    """
    total  = len(standards)
    titled = sum(1 for s in standards if s.get("title"))
    scoped = sum(1 for s in standards if s.get("scope"))
    rawed  = sum(1 for s in standards if len(s.get("raw_text", "")) > 100)
    parted = sum(1 for s in standards if "(Part" in s.get("standard_number", ""))

    print("\n── Parsing quality check ──")
    print(f"  Total standards    : {total}")
    print(f"  With title         : {titled:>4} ({titled / total:.0%})")
    print(f"  With scope         : {scoped:>4} ({scoped / total:.0%})")
    print(f"  With raw text      : {rawed:>4} ({rawed  / total:.0%})")
    print(f"  With Part numbers  : {parted:>4}")

    if titled / total < 0.5:
        print(
            "  ⚠️  WARNING: <50% of standards have titles.\n"
            "     Your PDF may be scanned/image-only. "
            "Consider running OCR before ingestion."
        )
    if parted == 0:
        print(
            "  ⚠️  WARNING: No Part-numbered standards found.\n"
            "     IS 1489 (Part 2):1991 and IS 2185 (Part 2):1983 will be missed.\n"
            "     Check that bis_parser.py IS_CODE_RE captures (PART N) groups."
        )
    if titled / total >= 0.5 and parted > 0:
        print("  ✅ Parsing quality looks good.")
    print()


def _expand_standards_to_chunks(
    standards: list[dict],
) -> tuple[list, list, list, list]:
    """
    Convert standards list into parallel arrays for indexing.

    FIX: Standards with long raw_text produce a second chunk covering
    requirements/body text. Both chunks share the same standard_number
    so retriever deduplication collapses them into one result per standard.

    Returns: embed_texts, chunks, sources, metadata  (all same length)
    """
    embed_texts = []
    chunks      = []
    sources     = []
    metadata    = []

    for std in standards:
        raw      = std.get("raw_text", "")
        std_num  = std.get("standard_number", "Unknown")
        title    = std.get("title", "")
        category = std.get("category", "General")
        scope    = std.get("scope", "")

        base_meta = {
            "standard_number": std_num,
            "title":           title,
            "category":        category,
            "scope":           scope[:300],
        }

        # Primary chunk (always created)
        embed_texts.append(build_embedding_text(std))
        chunks.append(build_chunk_text(std))
        sources.append(std_num)
        metadata.append(base_meta)

        # Secondary chunk for long standards
        if len(raw) > SPLIT_THRESHOLD and std.get("key_requirements"):
            req_text = (
                f"Standard: {std_num}\n"
                f"Title: {title}\n"
                f"Category: {category}\n"
                "Detailed Requirements:\n"
                + "\n".join(f"  - {r}" for r in std["key_requirements"])
                + f"\n\nBody text:\n{raw[400:1000]}"
            )
            req_embed = (
                f"{std_num} {title} {category} "
                + " ".join(std["key_requirements"])
            )
            embed_texts.append(req_embed)
            chunks.append(req_text)
            sources.append(std_num)       # same IS code — retriever deduplicates
            metadata.append(base_meta)    # same metadata

    return embed_texts, chunks, sources, metadata


def run_ingestion(pdf_path: str = None, model=None, force_reparse: bool = False):
    """
    Main ingestion pipeline.

    Args:
        pdf_path:      Path to SP 21 PDF. Auto-detected if None.
        model:         Pre-loaded SentenceTransformer (saves reload time).
        force_reparse: If True, re-parse PDF even if standards.json exists.
    """
    print("=" * 50)
    print("BIS SP 21 INGESTION PIPELINE")
    print("=" * 50)

    # ── Step 1: Parse or load standards ──
    if os.path.exists(STANDARDS_PATH) and not force_reparse:
        print(f"\n[1/4] Loading cached standards from {STANDARDS_PATH}...")
        standards = load_standards()
        print(f"      Loaded {len(standards)} standards")
    else:
        if pdf_path is None:
            pdf_path = find_sp21_pdf()
        if pdf_path is None:
            raise FileNotFoundError(
                f"No PDF found in '{DOCS_DIR}'. "
                "Place the SP 21 PDF there and retry."
            )
        print(f"\n[1/4] Parsing PDF: {pdf_path}")
        standards = parse_sp21_pdf(pdf_path)
        save_standards(standards)

    if not standards:
        raise ValueError("No standards parsed. Check your PDF file.")

    # FIX: Sanity-check before spending time on embeddings
    _sanity_check(standards)

    # ── Step 2: Build chunks ──
    print(f"[2/4] Building chunk texts for {len(standards)} standards...")
    embed_texts, chunks, sources, metadata = _expand_standards_to_chunks(standards)

    print(f"      Built {len(chunks)} chunks from {len(standards)} standards")
    if len(chunks) > len(standards):
        print(f"      ({len(chunks) - len(standards)} extra chunks from long standards)")

    sample_idx = min(5, len(embed_texts) - 1)
    print(f"\n      Sample embed text:\n      {embed_texts[sample_idx][:200]}")

    # ── Step 3: Build FAISS dense index ──
    print(f"\n[3/4] Building embeddings with {EMBEDDER_NAME}...")
    if model is None:
        model = SentenceTransformer(EMBEDDER_NAME)

    passage_prefix = "Represent this building material standard: "
    prefixed_texts = [passage_prefix + t for t in embed_texts]

    embeddings = model.encode(
        prefixed_texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim         = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    print(f"      FAISS index: {faiss_index.ntotal} vectors, dim={dim}")

    # ── Step 4: Build BM25 sparse index ──
    print(f"\n[4/4] Building BM25 sparse index...")

    # FIX: Include raw_text in BM25 for keyword recall on technical terms
    raw_lookup = {s["standard_number"]: s.get("raw_text", "") for s in standards}

    bm25_texts = []
    for m, t in zip(metadata, embed_texts):
        std_num  = m["standard_number"]
        raw_body = raw_lookup.get(std_num, "")[:400]
        combined = f"{std_num} {m['title']} {t} {raw_body}".lower()
        bm25_texts.append(combined.split())

    bm25_index = BM25Okapi(bm25_texts)
    print(f"      BM25 index built on {len(bm25_texts)} documents")

    # ── Save everything ──
    print("\nSaving indexes to disk...")
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    for path, obj in [
        (BM25_PATH,     bm25_index),
        (CHUNKS_PATH,   chunks),
        (SOURCES_PATH,  sources),
        (METADATA_PATH, metadata),
    ]:
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    print("\n" + "=" * 50)
    print("✅ INGESTION COMPLETE")
    print(f"   Standards parsed:  {len(standards)}")
    print(f"   Chunks indexed:    {len(chunks)}")
    print(f"   FAISS vectors:     {faiss_index.ntotal}")
    print(f"   BM25 documents:    {len(bm25_texts)}")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    force    = "--force" in sys.argv
    run_ingestion(pdf_path=pdf_path, force_reparse=force)