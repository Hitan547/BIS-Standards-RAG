"""
app.py
Streamlit UI for BIS Standards Recommendation Engine.
Run: streamlit run app.py
"""
import re
import html
import time
import streamlit as st
# Page config
st.set_page_config(
    page_title="BIS Standards Finder",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1f36 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid #2d5a8e;
    }
    .main-header h1 { color: #e8f4fd; font-size: 2rem; margin: 0; }
    .main-header p  { color: #90b8d8; margin: 0.5rem 0 0; font-size: 1rem; }

    .standard-card {
        background: #0e1e2e; border: 1px solid #1e3a5f;
        border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem;
        border-left: 4px solid #2196F3;
    }
    .standard-card.rank-1 { border-left-color: #FFD700; }
    .standard-card.rank-2 { border-left-color: #C0C0C0; }
    .standard-card.rank-3 { border-left-color: #CD7F32; }

    .is-code    { font-size: 1.3rem; font-weight: 700; color: #4db8ff; font-family: monospace; }
    .std-title  { font-size: 1rem; color: #cde8ff; margin: 0.3rem 0; }
    .std-meta   { font-size: 0.82rem; color: #7a9bb5; margin-bottom: 0.5rem; }
    .rationale  { font-size: 0.88rem; color: #b0c8d8; line-height: 1.6; }

    .confidence-bar { height: 6px; border-radius: 3px; margin: 0.5rem 0; background: #0a1520; }

    .category-badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cat-cement        { background: #2d1f00; color: #ffaa00; border: 1px solid #5a3d00; }
    .cat-steel         { background: #002d1f; color: #00ff88; border: 1px solid #005a3d; }
    .cat-concrete      { background: #1f002d; color: #bb66ff; border: 1px solid #3d005a; }
    .cat-aggregates    { background: #002d2d; color: #00ffff; border: 1px solid #005a5a; }
    .cat-bricks        { background: #2d1500; color: #ff8833; border: 1px solid #5a2800; }
    .cat-general       { background: #1a1a1a; color: #888888; border: 1px solid #333333; }

    .metric-box {
        background: #0e1e2e; border: 1px solid #1e3a5f;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #4db8ff; }
    .metric-lbl { font-size: 0.75rem; color: #7a9bb5; text-transform: uppercase; }

    .pass-badge { color: #00cc66; font-weight: 700; }
    .fail-badge { color: #ff4444; font-weight: 700; }

    .example-btn {
        background: #0e1e2e; border: 1px solid #1e3a5f;
        border-radius: 8px; padding: 0.6rem 1rem;
        color: #90b8d8; cursor: pointer; font-size: 0.85rem;
        margin-bottom: 0.5rem; width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ──
@st.cache_resource(show_spinner="Loading BIS Standards Index...")
def load_pipeline():
    from retriever import load_indexes, indexes_loaded
    from query_expansion import expanded_retrieve
    from retriever import hybrid_retrieve
    from agent import run_rag_agent
    load_indexes()
    return indexes_loaded(), expanded_retrieve, hybrid_retrieve, run_rag_agent
def strip_html(text: str) -> str:
    """Remove HTML tags from chunk text before displaying."""
    return re.sub(r'<[^>]+>', '', text or '').strip()

def get_category_class(category: str) -> str:
    return f"cat-{category.lower().replace(' ', '-')}"


def confidence_bar_html(confidence: float) -> str:
    pct   = int(confidence * 100)
    color = "#00cc66" if pct >= 70 else "#ffaa00" if pct >= 40 else "#ff4444"
    return f"""
    <div class="confidence-bar">
      <div style="width:{pct}%; height:100%; background:{color}; border-radius:3px;
                  transition:width 0.5s ease;"></div>
    </div>
    <span style="font-size:0.75rem;color:{color};font-weight:600;">{pct}% confidence</span>
    """


def render_standard_card(result: dict, rank: int):
    std_num      = result.get("standard_number", "Unknown")
    title        = result.get("title", "")
    category     = result.get("category", "General")
    conf         = result.get("confidence", 0)
    chunk        = result.get("chunk", "")
    scope        = result.get("scope", "")
    display_text = (scope or re.sub(r'<[^>]+>', '', chunk).replace('\n', ' ').strip())[:300]

    rank_class = f"rank-{rank}" if rank <= 3 else ""
    cat_class  = get_category_class(category)
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

    st.markdown(f"""
    <div class="standard-card {rank_class}">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span class="is-code">{rank_emoji} {std_num}</span>
                <div class="std-title">{title or "—"}</div>
            </div>
            <span class="category-badge {cat_class}">{category}</span>
        </div>
        {confidence_bar_html(conf)}
    </div>
    """, unsafe_allow_html=True)
    st.caption(display_text + ("..." if len(display_text) == 300 else ""))
# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🏗️ BIS Finder")
    st.markdown("---")

    # Example queries
    st.markdown("### 💡 Example Queries")
    examples = [
        "43 grade ordinary Portland cement for concrete mixing",
        "TMT steel bars for RCC construction",
        "Fly ash bricks for load bearing wall construction",
        "Coarse aggregates for M25 grade concrete",
        "Ready mix concrete for foundation work",
        "Hollow concrete blocks for partition walls",
        "High strength steel wire for prestressed concrete",
        "Sand for plastering and masonry work",
    ]
    for ex in examples:
        if st.button(ex[:55] + ("..." if len(ex) > 55 else ""),
                     key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    use_expansion = st.toggle("Query Expansion", value=True,
                              help="Generates alternate phrasings for better recall")
    top_k = st.slider("Standards to retrieve", 3, 8, 5)

    st.markdown("---")
    st.markdown("### 📊 Stack")
    st.markdown("""
    - `BAAI/bge-small-en-v1.5` embeddings
    - FAISS dense + BM25 sparse
    - RRF fusion + Cross-encoder
    - LangGraph corrective agent
    - LLaMA 3.3 70B via Groq
    """)


# ── Main ──
st.markdown("""
<div class="main-header">
    <h1>🏗️ BIS Standards Recommendation Engine</h1>
    <p>AI-powered discovery of Indian Standards for building materials · Powered by RAG</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
indexes_ready, expanded_retrieve, hybrid_retrieve, run_rag_agent = load_pipeline()

if not indexes_ready:
    st.error("⚠️ Index not loaded. Run `python ingestion.py` first to index the SP 21 PDF.")
    st.stop()

# Query input
query = st.text_area(
    "Describe your building material product:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g. '43 grade ordinary Portland cement for use in concrete structures'",
    key="query_input"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_btn = st.button("🔍 Find BIS Standards", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["query_input"] = ""
        st.rerun()

# ── Results ──
if search_btn and query.strip():
    with st.spinner("Searching BIS standards..."):
        t0 = time.time()

        try:
            if use_expansion:
                results = expanded_retrieve(query.strip(), top_k=top_k)
            else:
                results = hybrid_retrieve(query.strip(), top_k=top_k)

            latency_retrieval = time.time() - t0

            if not results:
                st.warning("No relevant standards found. Try a different product description.")
                st.stop()

            # Run agent for rationale
            with st.spinner("Generating rationale..."):
                answer, retries, verdict = run_rag_agent(query, results)

            total_latency = time.time() - t0

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{len(results)}</div>
            <div class="metric-lbl">Standards Found</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{total_latency:.1f}s</div>
            <div class="metric-lbl">Total Latency</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        badge = "pass-badge" if verdict == "PASS" else "fail-badge"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val {badge}">{"✓ PASS" if verdict == "PASS" else "✗ FAIL"}</div>
            <div class="metric-lbl">Hallucination Check</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{retries}</div>
            <div class="metric-lbl">Retries Used</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Two-column layout: standards + rationale ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📋 Recommended Standards")
        for i, result in enumerate(results, 1):
            render_standard_card(result, i)

        # Copy-friendly IS codes
        codes = [r.get("standard_number", "") for r in results]
        st.markdown("**IS Codes (copy-ready):**")
        st.code(", ".join(codes), language=None)

    with col_right:
        st.markdown("### 🤖 AI Rationale")
        st.markdown(answer)

        st.markdown("---")
        st.markdown("### 📁 Query Details")
        with st.expander("Show expansion queries & retrieval details"):
            st.markdown(f"**Original query:** {query}")
            st.markdown(f"**Retrieval latency:** {latency_retrieval:.2f}s")
            st.markdown("**Retrieved chunks:**")
            for r in results:
                st.markdown(f"- `{r.get('standard_number')}` — CE score: `{r.get('ce_score', 0):.3f}` | Confidence: `{r.get('confidence', 0):.1%}`")

elif search_btn:
    st.warning("Please enter a product description.")

else:
    st.info("👆 Enter a product description above or pick an example from the sidebar.")