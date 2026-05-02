<div align="center">

# 🏗️ BIS Standards Recommendation Engine

**AI-powered RAG system that maps building material descriptions to accurate BIS (Bureau of Indian Standards) codes — in seconds.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6B35?style=flat)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-LLaMA_3.3_70B_via_Groq-F55036?style=flat)](https://groq.com)
[![FAISS](https://img.shields.io/badge/Index-FAISS_+_BM25-4B8BBE?style=flat)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

*Built for the **BIS × SS Hackathon 2026** — Track: AI / Retrieval Augmented Generation (RAG)*

</div>

---

## 🏆 Evaluation Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Hit Rate @3 | **100.00%** | > 80% | ✅ |
| MRR @5 | **0.9500** | > 0.7 | ✅ |
| Avg Latency | **3.46s** | < 5s | ✅ |

---

## 🧠 How It Works

A product description enters the pipeline and passes through five sequential stages before returning verified IS standard codes with rationale.

```
Product Description
        │
        ▼
┌───────────────────────┐
│   Query Expansion     │  LLM generates 3 alternate phrasings to improve recall
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐     ┌─────────────────────────┐
│  FAISS Dense Search   │ ──► │   RRF Fusion             │
│  (BGE Embeddings)     │     │   (Reciprocal Rank       │
├───────────────────────┤ ──► │    Fusion)               │
│  BM25 Sparse Search   │     └───────────┬─────────────┘
│  (Keyword Matching)   │                 │
└───────────────────────┘                 ▼
                              ┌───────────────────────┐
                              │  Cross-Encoder        │
                              │  Reranking            │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  Confidence Filter    │  Drops results < 60% confidence
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  LangGraph Corrective │
                              │  RAG Agent            │
                              │  ├─ Generate rationale│
                              │  ├─ Validate output   │
                              │  └─ Retry if needed   │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              Ranked IS Standards + Rationale
```

---

## 💡 Key Design Decisions

### Hybrid Retrieval + RRF
Dense embeddings catch semantic similarity; BM25 catches exact keyword matches (e.g. standard codes, material names). Reciprocal Rank Fusion merges both ranked lists without needing score normalization.

### Corrective RAG via LangGraph
A self-correcting agent validates generated IS codes with a two-layer check — regex for format correctness, then an LLM hallucination check. If either fails, the agent retries up to 3 times with a refined prompt.

### Query Expansion
Ambiguous product descriptions (e.g. *"OPC 43 grade cement bag"*) often miss relevant standards on the first pass. The LLM generates 3 rephrasings before retrieval, dramatically improving recall.

### Dual Chunking
Long standards are split into a **header chunk** (IS code, title, scope) and a **body chunk** (specifications). This ensures the embedding captures both the identifier and the substance of each standard independently.

### Confidence Filtering
Results below 60% normalized cross-encoder confidence are dropped post-reranking. This eliminates false positives caused by keyword collisions — e.g. "solvent cement" (IS 14182, a PVC adhesive) surfacing for Portland cement queries — without affecting true positives which consistently score above 70%.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Dense Index | FAISS (`IndexFlatIP`) |
| Sparse Index | BM25 (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | LLaMA 3.3 70B via Groq |
| Agent Framework | LangGraph |
| PDF Parsing | pdfplumber |
| UI | Streamlit |
| API | FastAPI |

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Create .env in project root
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

### 3. Add the SP 21 dataset

```bash
mkdir docs
cp /path/to/sp21.pdf docs/
```

### 4. Build the index (one-time, ~10 min)

```bash
python ingestion.py
```

This builds the FAISS dense index and BM25 sparse index from the SP 21 PDF.

### 5. Run inference

```bash
python inference.py --input public_test_set.json --output my_results.json
```

### 6. Evaluate

```bash
python eval_script.py --results my_results.json
```

### 7. Launch the UI

```bash
# Streamlit (recommended)
streamlit run app.py

# FastAPI backend (optional)
uvicorn main:app --reload --port 8000
```

---

## 📁 Project Structure

```
├── agent.py               # LangGraph corrective RAG agent
├── app.py                 # Streamlit UI
├── bis_parser.py          # SP 21 PDF parser + dual chunking
├── config.py              # Paths and hyperparameters
├── eval_script.py         # Hackathon evaluation (Hit Rate, MRR, Latency)
├── inference.py           # Entry point for judges
├── ingestion.py           # Builds FAISS + BM25 indexes
├── main.py                # FastAPI backend
├── query_expansion.py     # LLM query expansion (3 rephrasings)
├── retriever.py           # Hybrid retrieval + RRF + reranking + confidence filter
├── requirements.txt
├── docs/                  # Place sp21.pdf here
└── data/
    └── public_test_results.json
```

---

## 🌍 Why This Matters

Indian MSEs currently spend **days or weeks** manually searching through hundreds of IS standards to verify compliance for their materials. Errors lead to failed inspections, wasted inventory, and lost contracts.

This system reduces that process to **under 4 seconds** — with hallucination-checked, source-attributed results — directly lowering compliance costs for small businesses operating on thin margins.

---

## 📋 External Dependencies

| Service | Usage | Cost |
|---------|-------|------|
| [Groq API](https://console.groq.com) | LLaMA 3.3 70B inference | Free tier |
| [HuggingFace](https://huggingface.co) | BGE embeddings + cross-encoder | Open source |
| BIS SP 21 PDF | Official standard dataset | Provided by hackathon |

---

## 👤 Author

**Hitan K** — AI Systems Engineer  
Final-year CS undergrad (AI Specialization) · Bengaluru, India  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hitan_K-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/hitank)
[![GitHub](https://img.shields.io/badge/GitHub-Hitan547-181717?style=flat&logo=github)](https://github.com/Hitan547)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Hitan2004-FFD21F?style=flat)](https://huggingface.co/Hitan2004)

> *"Ship first, iterate always. Real systems, real users, real impact."*