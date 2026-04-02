from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    SessionListResponse,
    SessionResponse,
)
from app.services.rag_service import rag_service
from app.services.session_service import session_service

router = APIRouter(tags=['chat'])


@router.post('/chat', response_model=ChatResponse)
def start_chat(request: ChatRequest) -> ChatResponse:
    rag_result = rag_service.answer(request.query)
    initial_response = str(rag_result.get('response', '')).strip()

    created_session = session_service.create_session_file(
        {
            'model': request.model,
            'query': request.query,
            'status': 'pending',
            'history': [
                {
                    'llm': initial_response,
                    'client': None,
                    'retrieval': rag_result.get('retrieval'),
                    'diagnostics': rag_result.get('diagnostics'),
                }
            ],
        }
    )

    retrieval = rag_result.get('retrieval')
    if not isinstance(retrieval, dict):
        raise HTTPException(status_code=500, detail='Invalid retrieval payload from RAG service')

    return ChatResponse(
        session_id=str(created_session['id']),
        response=initial_response,
        retrieval=retrieval,
    )


@router.post('/feedback', response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    session = session_service.read_session_by_id(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')

    history = session.get('history')
    if not isinstance(history, list):
        history = []
        session['history'] = history

    if request.action == 'agree':
        if history:
            history[-1]['client'] = request.feedback or 'Agreed (no comment)'
        session['status'] = 'agreed'
        session['updated_at'] = datetime.now().isoformat()
        session_service.save_session_by_id(session)
        return FeedbackResponse(status='agreed', response=None, retrieval=None)

    if history and request.feedback:
        history[-1]['client'] = request.feedback

    followup_query = session_service.build_prompt_from_feedback(str(session.get('query', '')), request.feedback)
    rag_result = rag_service.answer(followup_query)
    new_response = str(rag_result.get('response', '')).strip()

    history.append(
        {
            'llm': new_response,
            'client': None,
            'retrieval': rag_result.get('retrieval'),
            'diagnostics': rag_result.get('diagnostics'),
        }
    )
    session['status'] = 'pending'
    session['updated_at'] = datetime.now().isoformat()
    session_service.save_session_by_id(session)

    retrieval = rag_result.get('retrieval')
    if not isinstance(retrieval, dict):
        raise HTTPException(status_code=500, detail='Invalid retrieval payload from RAG service')

    return FeedbackResponse(status='pending', response=new_response, retrieval=retrieval)


@router.get('/sessions', response_model=SessionListResponse)
def list_sessions(date: str | None = Query(default=None)) -> SessionListResponse:
    requested_date = (date or session_service.today_date_key()).strip()
    if not session_service.is_valid_date_key(requested_date):
        raise HTTPException(status_code=400, detail='date must be DD-MM-YYYY')

    sessions = session_service.list_sessions_by_date(requested_date)
    return SessionListResponse(date=requested_date, sessions=sessions)


@router.get('/sessions/{session_id}', response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = session_service.read_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')

    if 'date_folder' not in session:
        parsed = session_service.parse_session_id(session_id)
        session['date_folder'] = parsed['date_key'] if parsed else ''

    return SessionResponse(**session)
