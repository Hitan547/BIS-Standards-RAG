"""
inference.py
Mandatory entry point for hackathon judges.
Usage: python inference.py --input hidden_private_dataset.json --output team_results.json

Input JSON format:
    [{"id": "q001", "query": "fly ash bricks for load bearing walls"}, ...]

Output JSON format (STRICT — do not change key names):
    [{"id": "q001", "retrieved_standards": ["IS 12894:2002", ...], "latency_seconds": 1.24}, ...]
"""

import argparse
import json
import time
import sys

# ── Load pipeline at module level (once, before query loop) ──
print("Loading indexes...", flush=True)
from retriever import load_indexes, hybrid_retrieve, indexes_loaded
from query_expansion import expanded_retrieve

load_indexes()

if not indexes_loaded():
    print("ERROR: Indexes not loaded. Run ingestion.py first.", file=sys.stderr)
    sys.exit(1)

print("Pipeline ready.\n", flush=True)


def run_single_query(query: str, use_expansion: bool = True, top_k: int = 5) -> tuple[list[str], float]:
    """
    Run the full retrieval pipeline for one query.

    Args:
        query:          Product description string
        use_expansion:  Whether to use query expansion (more accurate, slightly slower)
        top_k:          Number of standards to return

    Returns:
        (list_of_is_codes, latency_seconds)
    """
    start = time.time()

    try:
        if use_expansion:
            results = expanded_retrieve(query, top_k=top_k)
        else:
            results = hybrid_retrieve(query, top_k=top_k)

        # Extract IS code strings — these are what the judges evaluate
        standard_numbers = [
            r["standard_number"]
            for r in results
            if r.get("standard_number")
        ]

    except Exception as e:
        print(f"  ERROR on query '{query[:60]}...': {e}", file=sys.stderr)
        standard_numbers = []

    latency = time.time() - start
    return standard_numbers, round(latency, 4)


def main():
    parser = argparse.ArgumentParser(description="BIS Standards RAG Inference")
    parser.add_argument("--input",  required=True,  help="Path to input JSON file")
    parser.add_argument("--output", required=True,  help="Path to output JSON file")
    parser.add_argument("--top-k",  type=int, default=5, help="Number of standards to return")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Disable query expansion (faster but less accurate)")
    args = parser.parse_args()

    # ── Load input ──
    print(f"Reading input: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        print("ERROR: Input JSON must be a list of objects.", file=sys.stderr)
        sys.exit(1)

    total = len(queries)
    print(f"Processing {total} queries...\n")

    use_expansion = not args.no_expansion
    results_out   = []
    latencies     = []

    for i, item in enumerate(queries, 1):
        query_id = item.get("id", f"q{i:03d}")
        query    = item.get("query", "").strip()

        if not query:
            print(f"[{i}/{total}] SKIP: empty query for id={query_id}")
            results_out.append({
                "id":                  query_id,
                "retrieved_standards": [],
                "latency_seconds":     0.0,
            })
            continue

        print(f"[{i}/{total}] id={query_id}")
        print(f"  query: {query[:80]}{'...' if len(query) > 80 else ''}")

        standards, latency = run_single_query(
            query,
            use_expansion=use_expansion,
            top_k=args.top_k,
        )

        latencies.append(latency)
        print(f"  → {standards[:3]} ... ({latency:.2f}s)")

        # STRICT OUTPUT SCHEMA — do not change key names
        results_out.append({
            "id":                  query_id,
            "expected_standards":  item.get("expected_standards", []),  # ADD THIS LINE
            "retrieved_standards": standards,
            "latency_seconds":     latency,
        })

    # ── Write output ──
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n{'='*50}")
    print(f"✅ Done. Results saved to: {args.output}")
    print(f"   Queries processed: {total}")
    print(f"   Avg latency:       {avg_latency:.2f}s")
    print(f"   Under 5s target:   {'✅ YES' if avg_latency < 5 else '❌ NO'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()