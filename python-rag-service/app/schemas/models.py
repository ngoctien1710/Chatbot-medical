from typing import Any, Literal
from pydantic import BaseModel, Field


ModelAlias = Literal['mistral', 'gpt', 'gemini']
RetrievalMode = Literal['hybrid_original', 'dense_only', 'sparse_only', 'hybrid_rrf', 'cross_encoder_only']


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    model: ModelAlias | None = None
    retrieval_mode: RetrievalMode | None = None


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
    retrieval_mode: RetrievalMode | None = None


class DiagnosticsPayload(BaseModel):
    latency_ms: int
    context_used: int | None = None
    error: str | None = None
    provider: str | None = None
    model_alias: ModelAlias | None = None
    model_name: str | None = None
    error_category: str | None = None
    http_status: int | None = None
    retry_count: int | None = None
    request_id: str | None = None
    fallback_used: str | None = None
    fallback_reason: str | None = None
    quota_remaining: int | None = None
    rate_limit_reset_seconds: int | None = None


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


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    model: ModelAlias = Field(default='mistral')
    retrieval_mode: RetrievalMode | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    retrieval: RetrievalPayload
    diagnostics: DiagnosticsPayload | None = None
    model_used: ModelAlias | None = None
    retrieval_mode_used: RetrievalMode | None = None


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    action: Literal['agree', 'disagree']
    feedback: str | None = None
    model: ModelAlias | None = None
    retrieval_mode: RetrievalMode | None = None


class FeedbackResponse(BaseModel):
    status: Literal['agreed', 'pending']
    response: str | None = None
    retrieval: RetrievalPayload | None = None
    diagnostics: DiagnosticsPayload | None = None
    model_used: ModelAlias | None = None
    retrieval_mode_used: RetrievalMode | None = None


class SessionHistoryItem(BaseModel):
    llm: str
    client: str | None = None
    retrieval: RetrievalPayload | None = None
    diagnostics: DiagnosticsPayload | None = None
    model_alias: ModelAlias | None = None
    model_provider: str | None = None
    model_name: str | None = None
    retrieval_mode: RetrievalMode | None = None
    provider_metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    id: str
    date_folder: str
    sequence: int
    file_name: str
    model: str
    query: str
    status: str
    history: list[SessionHistoryItem]
    retrieval_mode: RetrievalMode | None = None
    created_at: str
    updated_at: str


class SessionSummary(BaseModel):
    session_id: str
    sequence: int
    file_name: str
    status: str
    model: str
    query: str
    retrieval_mode: RetrievalMode | None = None
    created_at: str
    updated_at: str
    history_count: int


class SessionListResponse(BaseModel):
    date: str
    sessions: list[SessionSummary]
