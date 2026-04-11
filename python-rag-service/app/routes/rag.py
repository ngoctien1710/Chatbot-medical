from fastapi import APIRouter, HTTPException

from app.schemas.models import AnswerRequest, AnswerResponse, HealthResponse, IngestResponse
from app.services.rag_service import rag_service
from app.settings import SUPPORTED_CHAT_MODELS, SUPPORTED_RETRIEVAL_MODES, settings
from app.utils.text_chunking import SemanticChunkingError

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
            'default_model_alias': settings.default_chat_model_alias,
            'default_retrieval_mode': settings.retrieval_mode,
        },
    )


@router.get('/capabilities')
def capabilities() -> dict:
    return {
        'models': {
            'mistral': {
                'available': True,
                'provider': 'ollama',
                'model_name': settings.chat_model,
            },
            'gpt': {
                'available': bool(settings.openai_api_key and settings.openai_chat_model),
                'provider': 'openai',
                'model_name': settings.openai_chat_model,
            },
            'gemini': {
                'available': bool(settings.gemini_api_key and settings.gemini_chat_model),
                'provider': 'gemini',
                'model_name': settings.gemini_chat_model,
            },
        },
        'model_aliases': sorted(SUPPORTED_CHAT_MODELS),
        'retrieval_modes': sorted(SUPPORTED_RETRIEVAL_MODES),
        'defaults': {
            'model': settings.default_chat_model_alias,
            'retrieval_mode': settings.retrieval_mode,
        },
    }


@router.post('/ingest', response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        documents, chunks = rag_service.ingest(reset=True)
    except SemanticChunkingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        status='ok',
        documents=documents,
        chunks=chunks,
        persisted_dir=str(settings.vector_db_dir),
    )


@router.post('/answer', response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    payload = rag_service.answer(
        request.query,
        model=request.model,
        retrieval_mode=request.retrieval_mode,
    )
    return AnswerResponse(**payload)
