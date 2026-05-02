# BIS Standards Recommendation Engine 🏗️

AI-powered RAG system that turns building material product descriptions into accurate BIS (Bureau of Indian Standards) recommendations in seconds.

Built for the **BIS × SS Hackathon 2026** — Track: AI / Retrieval Augmented Generation (RAG)

---

## 🏆 Evaluation Results (Public Test Set)

| Metric | Score | Target |
|---|---|---|
| Hit Rate @3 | **100.00%** | >80% |
| MRR @5 | **1.0000** | >0.7 |
| Avg Latency | **3.70s** | <5 seconds |

---

## 🧠 System Architecture

```
Product Description (query)
        ↓
  Query Expansion (LLM generates 3 alternate phrasings)
        ↓
  Hybrid Retrieval
  ├── FAISS Dense Search (BGE embeddings)
  └── BM25 Sparse Search (keyword)
        ↓
  RRF Fusion (Reciprocal Rank Fusion)
        ↓
  Cross-Encoder Reranking
        ↓
  LangGraph Corrective RAG Agent
  ├── Generate rationale
  ├── Validate (regex + LLM hallucination check)
  └── Retry if hallucination detected (max 3 retries)
        ↓
  Ranked IS Standards + Rationale
```

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 4. Add the SP 21 PDF
```bash
mkdir docs
cp sp21.pdf docs/
```

### 5. Run ingestion (one-time, ~10 minutes)
```bash
python ingestion.py
```

### 6. Run inference (judges use this)
```bash
python inference.py --input public_test_set.json --output my_results.json
```

### 7. Evaluate results
```bash
python eval_script.py --results my_results.json
```

### 8. Launch Streamlit UI
```bash
streamlit run app.py
```

### 9. Launch FastAPI backend (optional)
```bash
uvicorn main:app --reload --port 8000
```

---

## 📁 Project Structure

```
├── agent.py              # LangGraph corrective RAG agent
├── app.py                # Streamlit UI
├── bis_parser.py         # SP 21 PDF parser
├── config.py             # Configuration and paths
├── eval_script.py        # Hackathon evaluation script
├── inference.py          # Judge entry point
├── ingestion.py          # FAISS + BM25 index builder
├── main.py               # FastAPI backend
├── query_expansion.py    # LLM query expansion
├── retriever.py          # Hybrid retrieval pipeline
├── requirements.txt      # Dependencies
├── docs/                 # Place sp21.pdf here
└── data/                 # Public test set results
    └── public_test_results.json
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Dense Index | FAISS (IndexFlatIP) |
| Sparse Index | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | LLaMA 3.3 70B via Groq |
| Agent Framework | LangGraph |
| PDF Parsing | pdfplumber |
| UI | Streamlit |
| API | FastAPI |

---

## 💡 Key Innovations

1. **Hybrid Retrieval** — Combines semantic (FAISS) and keyword (BM25) search, fused with RRF for best of both worlds
2. **Corrective RAG** — LangGraph agent self-corrects hallucinated IS codes via regex + LLM validation loop
3. **Query Expansion** — LLM generates 3 alternate phrasings to improve recall on ambiguous descriptions
4. **Dual Chunking** — Long standards produce two chunks (header + body) for better embedding granularity
5. **Part Number Parsing** — Correctly handles IS 1489 (Part 2):1991 and similar multi-part standards

---

## 🌍 Impact on MSEs

Indian MSEs currently spend weeks manually searching through hundreds of IS standards. This system reduces that to **seconds**, with verified, hallucination-free results — directly reducing compliance costs for small businesses.

---

## 📋 External APIs & Data Sources

- **Groq API** — LLaMA 3.3 70B inference (free tier)
- **BIS SP 21 PDF** — Official dataset provided by hackathon organizers
- **HuggingFace** — BGE embeddings and cross-encoder reranker (open source)