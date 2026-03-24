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
4. Build index from backend/data/raw
   - POST http://127.0.0.1:8001/rag/ingest

API endpoints:
- GET /rag/health
- POST /rag/ingest
- POST /rag/answer   {"query":"..."}
