---
applyTo: "python-rag-service/**/*.py,backend/server.js,frontend/index.html"
description: "Use when editing this medical chatbot stack (FastAPI RAG service, Node gateway, frontend) to preserve API contracts, retrieval metadata, and medically safe response behavior as project guidance (deviations allowed with explicit rationale)."
---

# Medical Chatbot Project Conventions

These instructions are guidance defaults for this repository. If a task requires deviating, explain the reason and keep compatibility impacts explicit.

## Architecture contracts
- Keep the Python RAG service authoritative for retrieval and answer generation.
- Keep Node backend as a gateway/orchestration layer; avoid duplicating retrieval logic in Node unless explicitly requested.
- Preserve endpoint contracts unless the task explicitly asks for API changes:
  - Python: `/rag/health`, `/rag/ingest`, `/rag/answer`
  - Node: `/chat/start`, `/chat/feedback`, `/chat/:session_id`, `/rag/ingest`

## Response shape and compatibility
- For Python `answer` flow, preserve top-level keys: `response`, `retrieval`, `diagnostics`.
- Preserve retrieval metadata fields used by frontend/backend (`retrieval_status`, `top_k`, `snippets`, `score_summary`) unless task asks to version the contract.
- When changing response schemas, update all affected layers in the same change (Python schemas/routes, Node gateway mapping, frontend renderer).

## Medical safety and tone
- Do not present model output as definitive medical diagnosis.
- Prefer conservative phrasing when context is weak or absent; state uncertainty explicitly.
- Keep or strengthen the existing medical disclaimer behavior in generated answers.
- Avoid adding claims not grounded in retrieved context.

## Language and UX continuity
- Preserve existing user-facing language style already present in the codebase (currently Vietnamese text in prompts and fallback messages) unless asked to switch language.
- Keep fallback/error messages actionable and concise.

## Implementation style
- In Python service code, prefer typed return values and clear data models (dataclass/Pydantic where appropriate).
- Keep changes minimal and localized; avoid broad refactors unless explicitly requested.
- Validate cross-service changes by checking request/response fields at each boundary (Python service <-> Node gateway <-> frontend).
