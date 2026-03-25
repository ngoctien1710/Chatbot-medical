Python RAG service (FastAPI + LangChain + Chroma)

Run locally:
1. Create venv and install dependencies
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt
2. Ensure Ollama models are available
   - ollama pull nomic-embed-text
   - ollama pull llama3.2:3b
3. Start service
   - uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
4. Convert PDF corpus to markdown and ingest (recommended)
   - python -m app.scripts.prepare_pdf_corpus
   - optional flags:
     - --source-dir ..\\Documents\\VietNam
     - --output-dir ..\\data\\raw
     - --no-reset
5. Build index directly from markdown/text in raw dir
   - POST http://127.0.0.1:8001/rag/ingest

Default data paths:
- Source PDF directory: Documents/VietNam
- Raw markdown/text directory for ingestion: data/raw
- Chroma persist directory: data/chroma

API endpoints:
- GET /rag/health
- POST /rag/ingest
- POST /rag/answer   {"query":"..."}
