from fastapi import APIRouter

from app.schemas.models import AnswerRequest, AnswerResponse, HealthResponse, IngestResponse
from app.services.rag_service import rag_service
from app.settings import settings

router = APIRouter(prefix='/rag', tags=['rag'])


@router.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status='ok',
        detail={
            'vector_db_dir': str(settings.vector_db_dir),
            'raw_data_dir': str(settings.data_raw_dir),
            'ollama_base_url': settings.ollama_base_url,
            'chat_model': settings.chat_model,
            'embedding_model': settings.embedding_model,
        },
    )


@router.post('/ingest', response_model=IngestResponse)
def ingest() -> IngestResponse:
    documents, chunks = rag_service.ingest(reset=True)
    return IngestResponse(
        status='ok',
        documents=documents,
        chunks=chunks,
        persisted_dir=str(settings.vector_db_dir),
    )


@router.post('/answer', response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    payload = rag_service.answer(request.query)
    return AnswerResponse(**payload)
