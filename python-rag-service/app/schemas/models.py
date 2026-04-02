from typing import Any, Literal
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


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str = Field(default='mistral')


class ChatResponse(BaseModel):
    session_id: str
    response: str
    retrieval: RetrievalPayload


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    action: Literal['agree', 'disagree']
    feedback: str | None = None


class FeedbackResponse(BaseModel):
    status: Literal['agreed', 'pending']
    response: str | None = None
    retrieval: RetrievalPayload | None = None


class SessionHistoryItem(BaseModel):
    llm: str
    client: str | None = None
    retrieval: RetrievalPayload | None = None
    diagnostics: DiagnosticsPayload | None = None


class SessionResponse(BaseModel):
    id: str
    date_folder: str
    sequence: int
    file_name: str
    model: str
    query: str
    status: str
    history: list[SessionHistoryItem]
    created_at: str
    updated_at: str


class SessionSummary(BaseModel):
    session_id: str
    sequence: int
    file_name: str
    status: str
    model: str
    query: str
    created_at: str
    updated_at: str
    history_count: int


class SessionListResponse(BaseModel):
    date: str
    sessions: list[SessionSummary]
