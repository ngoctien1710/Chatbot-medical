## 2026-03-24 - RAG naive baseline implementation
- Added backend RAG module set under backend/rag (config, chunker, similarity, store, retriever, prompt builder, Ollama client, pipeline, ingest script).
- Refactored backend/server.js to answer via RAG pipeline and return retrieval metadata in /chat/start and /chat/feedback.
- Extended frontend/index.html to display retrieval status and top-k snippets for AI responses.
- Added baseline data scaffold under backend/data (raw sample document, index skeleton, quick-start README).
- Updated backend/package.json scripts with rag:ingest.

## 2026-03-24 - Server runtime update
- Updated backend/server.js to allow PORT from environment variables for parallel local testing.

## 2026-03-24 - Test data cleanup
- Removed temporary session records in backend/sessions.json that were created during implementation sanity checks.

## 2026-03-24 - Python RAG service migration
- Added new Python service at python-rag-service/ using FastAPI + LangChain + Chroma + Ollama integration.
- Added Python service modules for settings, schemas, semantic chunking, ingest, retrieval, and answer generation.
- Refactored backend/server.js into Node gateway mode to call Python endpoints (/rag/answer and /rag/ingest) via HTTP.
- Removed legacy JS RAG module files from backend/rag/.
- Updated backend/package.json to drop legacy rag:ingest script.
- Updated backend/data/README.md to document Python-service-based ingest flow.
- Added repository .gitignore for Python cache/venv and vector store artifacts.
- Removed empty legacy backend/rag directory after migration completion.
