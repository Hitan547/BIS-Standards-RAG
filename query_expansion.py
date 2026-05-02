"""
query_expansion.py
Expands a product query into multiple alternate phrasings using LLM.
Retrieves for each phrasing, then merges + deduplicates results.
This improves Hit Rate significantly for ambiguous product descriptions.

FIXES vs original:
1. expand_query() results are LRU-cached — identical queries never hit the
   LLM twice. Saves Groq quota during test-set reruns and demo restarts.
2. expanded_retrieve() has a latency guard — if cumulative retrieval time
   exceeds LATENCY_BUDGET_S the remaining expansions are skipped, keeping
   the pipeline under the 5-second judge target.
"""

import time
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL

llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, api_key=GROQ_API_KEY)

# If total retrieval wall-time exceeds this after the first expansion,
# remaining expansions are skipped. Leaves ~2.5s for the agent step.
LATENCY_BUDGET_S = 2.5

EXPANSION_PROMPT = """You are a BIS standards expert. A user described a building material product.
Generate 3 alternate search queries to find relevant Indian Standards (IS codes) for this product.

Rules:
- Each query should use different technical terminology
- Include material type, application, and property terms
- Keep each query under 15 words
- Return ONLY the 3 queries, one per line, no numbering or extra text

Product description: {query}

3 alternate queries:"""


@lru_cache(maxsize=256)
def _expand_query_cached(query: str) -> tuple[str, ...]:
    """
    Internal cached expansion — returns a tuple (hashable for lru_cache).
    Use expand_query() for the public list interface.
    """
    try:
        prompt   = EXPANSION_PROMPT.format(query=query)
        response = llm.invoke([HumanMessage(content=prompt)])
        lines    = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return (query,) + tuple(lines[:3])
    except Exception as e:
        print(f"Query expansion failed (using original only): {e}")
        return (query,)


def expand_query(query: str) -> list[str]:
    """
    Generate alternate phrasings for a product query.

    Results are LRU-cached — identical queries incur zero LLM cost after
    the first call. Useful when running the same test set multiple times.

    Args:
        query: Original product description

    Returns:
        List of [original_query, expansion1, expansion2, expansion3]
    """
    return list(_expand_query_cached(query))


def expanded_retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full expanded retrieval pipeline:
    1. Expand query into multiple phrasings (LRU-cached).
    2. Retrieve for each phrasing, stopping early if latency budget exceeded.
    3. Merge results using score fusion.
    4. Return top_k deduplicated results.

    Args:
        query:  Original product description
        top_k:  Number of final results to return

    Returns:
        Merged and deduplicated list of retrieval results
    """
    from retriever import hybrid_retrieve

    queries = expand_query(query)
    print(f"  Query expansions ({len(queries)}):")
    for i, q in enumerate(queries):
        print(f"    {i}: {q}")

    per_query_k  = min(top_k + 2, 8)
    all_results: dict[str, dict] = {}
    budget_start = time.time()

    for i, q in enumerate(queries):
        # FIX: Latency guard — skip remaining expansions if budget spent
        elapsed = time.time() - budget_start
        if i > 0 and elapsed > LATENCY_BUDGET_S:
            print(
                f"  ⚡ Latency budget ({LATENCY_BUDGET_S}s) reached after "
                f"{i}/{len(queries)} expansions — skipping rest."
            )
            break

        try:
            results = hybrid_retrieve(q, top_k=per_query_k)
            weight  = 1.0 if i == 0 else 0.7

            new_count = 0
            for r in results:
                std_num = r.get("standard_number", "")
                if not std_num:
                    continue

                if std_num not in all_results:
                    r["merged_score"] = r["confidence"] * weight
                    all_results[std_num] = r
                    new_count += 1
                else:
                    all_results[std_num]["merged_score"] = (
                        all_results[std_num]["merged_score"]
                        + r["confidence"] * weight * 0.5
                    )
                    if r["confidence"] > all_results[std_num]["confidence"]:
                        r["merged_score"] = all_results[std_num]["merged_score"]
                        all_results[std_num] = r

            print(
                f"    expansion {i}: {len(results)} results, "
                f"{new_count} new ({time.time() - budget_start:.2f}s elapsed)"
            )

        except Exception as e:
            print(f"  Retrieval failed for expansion '{q}': {e}")
            continue

    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x.get("merged_score", 0),
        reverse=True,
    )[:top_k]

    print(f"  Merged {len(all_results)} unique standards → returning top {len(sorted_results)}")
    return sorted_results