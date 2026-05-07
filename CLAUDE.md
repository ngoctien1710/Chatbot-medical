# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese medical chatbot with Retrieval-Augmented Generation (RAG). Three-tier architecture: static frontend → Node.js gateway → Python FastAPI RAG service. Primary corpus is Vietnamese medical PDFs.

## Running the Services

All three services must run concurrently. Start them in this order:

**1. Python RAG Service** (authoritative core, port 8001)
```bash
# From repo root — venv is at root level, not inside python-rag-service
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Unix

cd python-rag-service
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

**2. Node.js Gateway** (port 3000)
```bash
cd backend
npm run dev
```

**3. Frontend** (port 5500)
```bash
cd frontend
python -m http.server 5500
# Visit http://127.0.0.1:5500/index.html
```

**Prerequisite:** Ollama must be running at `http://127.0.0.1:11434` with the `mistral` model pulled.

## Environment Configuration

Copy `python-rag-service/.env.example` to `python-rag-service/.env`. Key variables:

| Variable | Purpose |
|---|---|
| `OLLAMA_CHAT_MODEL` | Local LLM model name (default: mistral) |
| `OLLAMA_EMBED_MODEL` | Embedding model (default: Dqdung205/medical_vietnamese_embedding) |
| `OPENAI_API_KEY` / `OPENAI_CHAT_MODEL` | Enable GPT provider |
| `GEMINI_API_KEY` / `GEMINI_CHAT_MODEL` | Enable Gemini provider |
| `RAG_RETRIEVAL_MODE` | One of: `hybrid_original`, `dense_only`, `sparse_only`, `hybrid_rrf`, `cross_encoder_only` |
| `RAG_TOP_K` | Number of chunks to retrieve (default: 5) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | Chunking params (default: 900/180) |
| `GLM_CLEANUP_ENABLED` | Enable HuggingFace GLM semantic PDF cleanup |

## Corpus Ingestion

Run after adding/changing PDFs in `Documents/VietNam/`:
```bash
cd python-rag-service
python -m app.scripts.prepare_pdf_corpus
# Optional flags: --source-dir, --output-dir, --no-reset
```

Benchmark retrieval modes:
```bash
python -m app.scripts.benchmark_retrieval_modes
```

## Architecture

```
frontend/index.html (vanilla JS, ~870 lines)
    │  HTTP to port 3000
    ▼
backend/server.js (Express gateway, ~491 lines)
    │  Proxies to Python service
    ▼
python-rag-service/app/
    ├── routes/rag.py          — /rag/health, /rag/ingest, /rag/answer
    ├── routes/session.py      — session CRUD endpoints
    ├── services/rag_service.py        — RAG pipeline orchestration
    ├── services/session_service.py    — chat_log/ JSON persistence
    ├── services/pdf_markdown_service.py — PDF→Markdown (PyMuPDF + RapidOCR)
    ├── utils/hybrid_retriever.py      — BM25 + Chroma fusion
    ├── utils/sparse_retriever.py      — BM25 with Vietnamese tokenization
    ├── utils/reranker.py              — Cross-encoder reranking
    ├── schemas/models.py              — Pydantic request/response models
    └── settings.py                    — Pydantic BaseSettings (.env binding)
```

**Data directories:**
- `data/chroma/` — ChromaDB vector store (persisted)
- `data/raw/` — Converted markdown files
- `Documents/VietNam/` — Source PDF corpus
- `chat_log/DD-MM-YYYY/` — Session JSON files

## API Contracts (Do Not Break)

The frontend and Node gateway depend on these response shapes from the Python service:

**POST `/rag/answer`** response keys: `response`, `retrieval`, `diagnostics`

**Node gateway endpoints:** `/chat/start`, `/chat/feedback`

**Retrieval metadata fields** in `retrieval` are rendered by the frontend snippet viewer — do not remove or rename them.

Per-request overrides (sent from frontend): `model` (`mistral`|`gpt`|`gemini`) and `retrieval_mode` (one of the 5 modes above). These override `.env` defaults for that request only.

## Medical Domain Constraints

From `/.github/instructions/medical-rag.instructions.md`:
- Never present model output as a definitive medical diagnosis
- State uncertainty explicitly when retrieved context is weak
- Preserve Vietnamese medical terminology — do not translate or normalize
- All answers must be grounded in retrieved context; do not generate unsupported claims
- Preserve existing disclaimer behavior in the prompt templates

## Tech Stack

- **Python RAG service:** FastAPI, LangChain 0.3.x, ChromaDB, Sentence-Transformers, PyMuPDF, RapidOCR, `pyvi`/`underthesea` (Vietnamese NLP)
- **Node gateway:** Express 4.x (thin proxy, minimal logic)
- **Frontend:** Vanilla HTML/CSS/JS — no build step, no framework
- **LLM providers:** Ollama (local), OpenAI, Google Generative AI — selected per request via `model` field
