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

## 2026-04-01 - HF embedding migration and corpus refresh
- Reviewed current RAG pipeline and confirmed retrieval flow: chunk -> embed -> vector similarity search -> threshold filter -> context to chat model.
- Migrated embedding runtime in python-rag-service/app/services/rag_service.py from OllamaEmbeddings to HuggingFaceEmbeddings.
- Set embedding model to Dqdung205/medical_vietnamese_embedding and enabled trust_remote_code for model loading.
- Updated python-rag-service/app/settings.py default embedding model to Dqdung205/medical_vietnamese_embedding.
- Updated python-rag-service/requirements.txt to include langchain-huggingface and pinned compatible stack:
	sentence-transformers==5.1.0, transformers==4.56.1, tokenizers==0.22.0, accelerate==1.10.1.
- Diagnosed model runtime incompatibility (IndexError during encode with newer transformer stack) and resolved via version pinning above.
- Removed excluded source files from corpus before re-ingest:
	Documents/VietNam/HuongDanChanDoanVaDieuTriMotSoBenhLyHuyetHoc.pdf and
	data/raw/HuongDanChanDoanVaDieuTriMotSoBenhLyHuyetHoc.md.
- Rebuilt Chroma index with full reset ingest using remaining corpus; successful ingest summary: 2 documents, 351 chunks.
- Verified retrieval after migration: retrieval_status=success and snippets only from remaining two medical documents.
- Updated python-rag-service/.env.example to match current code defaults and path comments (including HF embedding model value).
- Added repository memory note for this model compatibility at /memories/repo/hf-embedding-compat.md.

## 2026-04-01 - FastAPI backend cutover implementation (phase 1)
- Added new FastAPI chat/session endpoints for direct frontend integration:
	- POST /chat
	- POST /feedback
	- GET /sessions?date=DD-MM-YYYY
	- GET /sessions/{session_id}
- Added session persistence service in python-rag-service/app/services/session_service.py, porting Node logic for:
	- date-based folders (chat_log/DD-MM-YYYY)
	- atomic session file allocation (session_001.json...)
	- session id parsing and lookup
	- feedback follow-up prompt building
- Extended schemas in python-rag-service/app/schemas/models.py for chat, feedback, and session contracts.
- Updated python-rag-service/app/main.py to include new session router and CORS middleware for frontend static host origins.
- Updated frontend/index.html to call FastAPI directly at http://localhost:8001 and switched API paths from legacy /chat/start,/chat/feedback,/chat/sessions,/chat/:id to /chat,/feedback,/sessions,/sessions/:id.
- Updated python-rag-service/.env.example with CORS_ORIGINS and CHAT_LOG_ROOT.
- Updated python-rag-service/README.md to document new FastAPI-first API surface and runtime notes.

## 2026-04-01 - Corpus cleaning pipeline rewrite and direct-answer prompt tuning
- Rewrote python-rag-service/app/services/pdf_markdown_service.py to build cleaned markdown corpus with a clean-first pipeline:
	- PDF text extraction with OCR fallback (RapidOCR)
	- deterministic cleanup to remove metadata/page markers/table-of-contents noise/references tail
	- optional GLM semantic cleanup stage using Hugging Face model zai-org/GLM-5-FP8 with safe fallback when HF inference is unavailable
	- manifest versioning using cleaner_signature to force regenerate when cleaning config changes
- Added new cleanup settings in python-rag-service/app/settings.py and .env.example:
	- OCR_MIN_TEXT_CHARS, OCR_DPI
	- CLEAN_STRIP_TOC, CLEAN_STRIP_REFERENCES
	- GLM_CLEANUP_* and HF_TOKEN
- Added explicit huggingface-hub dependency in python-rag-service/requirements.txt for GLM cleanup client.
- Reset data/raw corpus (deleted old *.md and .pdf_manifest.json), then regenerated markdown and rebuilt Chroma index via python -m app.scripts.prepare_pdf_corpus.
- Verified regenerated markdown no longer contains legacy converter artifacts (Source PDF/Converted at/## Page markers).
- Tuned answer prompt in python-rag-service/app/services/rag_service.py to enforce:
	- direct answer in first sentence
	- no opener phrases like "bạn đang xem tài liệu" / "theo tài liệu" / "dựa trên tài liệu"
	- uncertainty disclosure when context is weak
- Smoke-tested /rag/answer after rebuild: retrieval_status=success for sample ALL definition query and response starts with direct definition.

## 2026-04-11 - Runtime model/retrieval controls, provider fallback, and diagnostics expansion
- Added chat model alias controls and provider resilience settings in python-rag-service/app/settings.py:
	- SUPPORTED_CHAT_MODELS: mistral, gpt, gemini
	- RAG_DEFAULT_CHAT_MODEL_ALIAS, OPENAI_CHAT_MODEL, GEMINI_CHAT_MODEL
	- OPENAI_API_KEY, GEMINI_API_KEY
	- RAG_PROVIDER_TIMEOUT_SECONDS, RAG_PROVIDER_RETRY_ATTEMPTS, RAG_RATE_LIMIT_COOLDOWN_SECONDS, RAG_FALLBACK_ON_PROVIDER_ERROR
- Extended API schemas in python-rag-service/app/schemas/models.py to support request-time model/retrieval selection and richer diagnostics:
	- model and retrieval_mode in AnswerRequest/ChatRequest/FeedbackRequest
	- retrieval_mode in retrieval payload
	- provider, model_alias, model_name, error_category, retry_count, request_id, fallback metadata in diagnostics
	- session history metadata fields (model_alias, model_provider, model_name, retrieval_mode, provider_metadata)
- Updated python-rag-service/app/routes/rag.py:
	- /rag/answer now accepts model and retrieval_mode overrides
	- added /rag/capabilities endpoint exposing model availability, retrieval modes, and defaults
- Updated python-rag-service/app/routes/session.py and python-rag-service/app/services/session_service.py:
	- normalize/validate incoming model and retrieval_mode
	- persist model/retrieval/provider metadata per history turn
	- return model_used and retrieval_mode_used in /chat and /feedback responses
	- include retrieval_mode in session summaries and default missing legacy values safely
- Reworked generation flow in python-rag-service/app/services/rag_service.py:
	- provider routing for mistral (Ollama), gpt (OpenAI), gemini (Google Generative AI)
	- transient error classification + retry loop + provider cooldown
	- optional fallback to mistral on provider failures
	- normalized diagnostics emitted for both success and fallback/error paths
	- retrieval_mode propagated in retrieval payload for frontend/gateway display
- Updated backend/server.js (Node gateway) to:
	- forward model and retrieval_mode to Python /rag/answer
	- persist retrieval_mode and provider metadata in session history
	- return diagnostics, model_used, retrieval_mode_used to frontend
- Updated frontend/index.html:
	- model selector now uses mistral/gpt/gemini aliases
	- added retrieval mode selector (hybrid_original, dense_only, sparse_only, hybrid_rrf, cross_encoder_only)
	- send model + retrieval_mode in start and feedback payloads
	- display retrieval mode in RAG badge and keep selector state synced with model_used/retrieval_mode_used from backend
- Updated dependency/config hygiene:
	- python-rag-service/requirements.txt: added openai==1.107.1, google-generativeai==0.8.6, tenacity==9.1.2
	- python-rag-service/.env.example expanded with provider/runtime settings
	- .gitignore now ignores local env files, node_modules, and top-level data/chroma artifacts
- Post-change validation executed:
	- no static diagnostics errors in edited Python/Node/frontend files
	- Python smoke tests passed for /rag/health, /rag/capabilities, /chat, /feedback, /sessions, /sessions/{id}
	- Node gateway smoke tests passed for /chat/start, /chat/feedback, /chat/:session_id with new metadata flow
	- verified fallback behavior when GPT/Gemini keys are unset: requests continue via mistral and return fallback diagnostics
