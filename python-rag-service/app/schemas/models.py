from typing import Any
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)


class RetrievalSnippet(BaseModel):
    id: str | None = None
    doc_id: str | None = None
    score: float | None = None
    snippet: str


class RetrievalPayload(BaseModel):
    retrieval_status: str
    top_k: int
    snippets: list[RetrievalSnippet]
    score_summary: dict[str, float] | None = None


class DiagnosticsPayload(BaseModel):
    latency_ms: int
    context_used: int | None = None
    error: str | None = None


class AnswerResponse(BaseModel):
    response: str
    retrieval: RetrievalPayload
    diagnostics: DiagnosticsPayload


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    persisted_dir: str


class HealthResponse(BaseModel):
    status: str
    detail: dict[str, Any]
