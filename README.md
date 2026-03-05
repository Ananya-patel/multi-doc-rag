# 📚 Multi-Document RAG System

> Project 3 of my RAG Mastery Journey

A production-grade RAG system that queries across multiple documents 
simultaneously. Every answer comes with source attribution — which 
document, which page, how confident.

---

##  What This Project Taught Me

- **ChromaDB:** Persistent vector storage that survives restarts
- **Metadata:** Tagging every chunk with source, page, doc_id
- **Multi-doc retrieval:** Searching 510 chunks across 3 docs at once
- **Source attribution:** Every answer cites its exact source
- **Confidence thresholding:** Two-layer hallucination prevention
- **Incremental indexing:** Add new docs without rebuilding

---

##  Architecture
```
Multiple PDFs
     ↓
ingest.py   → extract pages → chunk with metadata → embed → ChromaDB
     ↓
query.py    → embed question → search all docs → filter by confidence
     ↓
generate    → grounded answer with source citations
     ↓
app.py      → chat UI with source cards + document filter
```

---

##  What Was Indexed

| Document | Pages | Chunks |
|---|---|---|
| japan_culture.pdf | 23 | 102 |
| france_culture.pdf | 33 | 167 |
| india_culture.pdf | 53 | 241 |
| **Total** | **109** | **510** |

---

##  Key Upgrades From Project 2

| Feature | Project 2 | Project 3 |
|---|---|---|
| Storage | FAISS (in-memory) | ChromaDB (persistent) |
| Documents | 1 PDF | Multiple PDFs |
| Metadata | chunk_id only | source + page + doc_id |
| Source attribution | ❌ | ✅ |
| Incremental indexing | ❌ | ✅ |
| Document filtering | ❌ | ✅ |
| Confidence threshold | Basic | Two-layer |

---

##  Tech Stack

| Component | Tool |
|---|---|
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector store | ChromaDB (persistent) |
| PDF parsing | PyPDF2 |
| UI | Streamlit |

---

## ⚙️ Setup & Run Locally

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/multi-doc-rag.git
cd multi-doc-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Environment variables**
```
GROQ_API_KEY=your-groq-key-here
```

**3. Ingest your PDFs**
```bash
# Put your PDFs in the project folder, then:
python ingest.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

##  Project Structure
```
project3/
├── ingest.py       # PDF processing + ChromaDB ingestion
├── query.py        # Retrieval + generation pipeline
├── app.py          # Streamlit multi-doc chat UI
├── requirements.txt
└── README.md
```

---

##  RAG Mastery Journey

| Project | Topic | Status |
|---|---|---|
| Project 1 | Document Analysis Using LLMs | ✅ Complete |
| Project 2 | RAG System From Scratch | ✅ Complete |
| **Project 3** | **Multi-Document RAG** | ✅ **Complete** |
| Project 4 | GraphRAG Pipeline | 🔄 Next |

---

LIVE LINK:
https://multi-doc-rag0301.streamlit.app/