Python RAG service (FastAPI + LangChain + Chroma)

Run locally:
1. Create venv and install dependencies
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt
2. Ensure Ollama models are available
   - ollama pull mistral
   - Note: Embedding model runs via HuggingFace (`Dqdung205/medical_vietnamese_embedding`) and downloads automatically on first use.
3. Start service
   - uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
4. Convert PDF corpus to markdown and ingest (recommended)
   - python -m app.scripts.prepare_pdf_corpus
   - optional flags:
     - --source-dir ..\\Documents\\VietNam
     - --output-dir ..\\data\\raw
     - --no-reset
    - corpus refresh from scratch:
       - delete `data/raw/*.md` and `data/raw/.pdf_manifest.json`
       - rerun `python -m app.scripts.prepare_pdf_corpus`
5. Build index directly from markdown/text in raw dir
   - POST http://127.0.0.1:8001/rag/ingest

Default data paths:
- Source PDF directory: Documents/VietNam
- Raw markdown/text directory for ingestion: data/raw
- Chroma persist directory: data/chroma

PDF cleanup pipeline:
- Deterministic cleanup removes metadata/page markers/table-of-contents/references noise.
- Optional GLM semantic cleanup stage uses Hugging Face model `zai-org/GLM-5-FP8`.
- If HF inference is unavailable, converter falls back to deterministic cleanup only.

Key environment variables for cleanup:
- `OCR_MIN_TEXT_CHARS`, `OCR_DPI`
- `CLEAN_STRIP_TOC`, `CLEAN_STRIP_REFERENCES`
- `GLM_CLEANUP_ENABLED`, `GLM_CLEANUP_MODEL`
- `GLM_CLEANUP_TIMEOUT_SECONDS`, `GLM_CLEANUP_MAX_CHUNK_CHARS`, `GLM_CLEANUP_MAX_CHUNKS_PER_DOC`
- `HF_TOKEN` (optional, for authenticated HF inference)

API endpoints:
- POST /chat         {"query":"...", "model":"mistral"}
- POST /feedback     {"session_id":"...", "action":"agree|disagree", "feedback":"..."}
- GET /sessions?date=DD-MM-YYYY
- GET /sessions/{session_id}
- GET /rag/health
- POST /rag/ingest
- POST /rag/answer   {"query":"..."}
