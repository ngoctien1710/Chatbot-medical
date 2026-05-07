from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

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
from app.settings import SUPPORTED_CHAT_MODELS, SUPPORTED_RETRIEVAL_MODES, settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=['chat'])


def _safe_model_alias(value: str | None) -> str:
    candidate = (value or settings.default_chat_model_alias).strip().lower()
    if candidate not in SUPPORTED_CHAT_MODELS:
        logger.warning('Unknown model alias %r, falling back to %s', candidate, settings.default_chat_model_alias)
        return settings.default_chat_model_alias
    return candidate


def _safe_retrieval_mode(value: str | None) -> str:
    candidate = (value or settings.retrieval_mode).strip().lower()
    if candidate not in SUPPORTED_RETRIEVAL_MODES:
        logger.warning('Unknown retrieval mode %r, falling back to %s', candidate, settings.retrieval_mode)
        return settings.retrieval_mode
    return candidate


def _provider_metadata_from_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(diagnostics, dict):
        return None
    metadata = {
        'request_id': diagnostics.get('request_id'),
        'retry_count': diagnostics.get('retry_count'),
        'fallback_used': diagnostics.get('fallback_used'),
        'fallback_reason': diagnostics.get('fallback_reason'),
        'error_category': diagnostics.get('error_category'),
        'http_status': diagnostics.get('http_status'),
        'quota_remaining': diagnostics.get('quota_remaining'),
        'rate_limit_reset_seconds': diagnostics.get('rate_limit_reset_seconds'),
    }
    compact = {k: v for k, v in metadata.items() if v is not None}
    return compact or None


def _build_history_entry(
    rag_result: dict[str, Any],
    model_alias: str,
    retrieval_mode: str,
    response_text: str,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct a history entry dict from a rag_service.answer() result."""
    provider_metadata = _provider_metadata_from_diagnostics(diagnostics)
    model_name = diagnostics.get('model_name') if diagnostics else None
    model_provider = diagnostics.get('provider') if diagnostics else None
    return {
        'llm_response': response_text,
        'client': None,
        'retrieval': rag_result.get('retrieval'),
        'diagnostics': diagnostics,
        'model_alias': model_alias,
        'model_provider': model_provider,
        'model_name': model_name,
        'retrieval_mode': retrieval_mode,
        'provider_metadata': provider_metadata,
    }


def _resolve_rag_result(rag_result: dict[str, Any], requested_model: str, requested_mode: str) -> tuple[str, str, str, dict | None, dict | None]:
    """
    Extract and validate the key fields from a rag_service.answer() result.
    Returns: (response_text, model_used, retrieval_mode_used, diagnostics, retrieval_payload)
    """
    response_text = str(rag_result.get('response', '')).strip()
    diagnostics = rag_result.get('diagnostics') if isinstance(rag_result.get('diagnostics'), dict) else None
    retrieval_payload = rag_result.get('retrieval') if isinstance(rag_result.get('retrieval'), dict) else None

    model_used = _safe_model_alias(
        str(diagnostics.get('model_alias') or requested_model) if diagnostics else requested_model
    )
    retrieval_mode_used = _safe_retrieval_mode(
        str(retrieval_payload.get('retrieval_mode') or requested_mode) if retrieval_payload else requested_mode
    )
    return response_text, model_used, retrieval_mode_used, diagnostics, retrieval_payload


@router.post('/chat', response_model=ChatResponse)
def start_chat(request: ChatRequest) -> ChatResponse:
    model_alias = _safe_model_alias(request.model)
    retrieval_mode = _safe_retrieval_mode(request.retrieval_mode)

    rag_result = rag_service.answer(request.query, model=model_alias, retrieval_mode=retrieval_mode)
    response_text, model_used, retrieval_mode_used, diagnostics, retrieval_payload = _resolve_rag_result(
        rag_result, model_alias, retrieval_mode
    )

    history_entry = _build_history_entry(rag_result, model_used, retrieval_mode_used, response_text, diagnostics)
    if model_alias != model_used:
        history_entry['provider_metadata'] = {**(history_entry.get('provider_metadata') or {}), 'requested_model_alias': model_alias}

    created_session = session_service.create_session_file(
        {
            'model': model_used,
            'retrieval_mode': retrieval_mode_used,
            'query': request.query,
            'status': 'pending',
            'history': [history_entry],
        }
    )

    if not isinstance(retrieval_payload, dict):
        raise HTTPException(status_code=500, detail='Invalid retrieval payload from RAG service')

    return ChatResponse(
        session_id=str(created_session['id']),
        response=response_text,
        retrieval=retrieval_payload,
        diagnostics=diagnostics,
        model_used=model_used,
        retrieval_mode_used=retrieval_mode_used,
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

    # action == 'disagree': record client comment and regenerate
    if history and request.feedback:
        history[-1]['client'] = request.feedback

    latest_history = history[-1] if history and isinstance(history[-1], dict) else {}
    current_model_alias = _safe_model_alias(
        request.model
        or str(latest_history.get('model_alias') or session.get('model') or settings.default_chat_model_alias)
    )
    current_retrieval_mode = _safe_retrieval_mode(
        request.retrieval_mode
        or str(latest_history.get('retrieval_mode') or session.get('retrieval_mode') or settings.retrieval_mode)
    )

    followup_query = session_service.build_prompt_from_feedback(str(session.get('query', '')), request.feedback)
    rag_result = rag_service.answer(followup_query, model=current_model_alias, retrieval_mode=current_retrieval_mode)
    response_text, model_used, retrieval_mode_used, diagnostics, retrieval_payload = _resolve_rag_result(
        rag_result, current_model_alias, current_retrieval_mode
    )

    history_entry = _build_history_entry(rag_result, model_used, retrieval_mode_used, response_text, diagnostics)
    if model_used != current_model_alias:
        history_entry['provider_metadata'] = {**(history_entry.get('provider_metadata') or {}), 'requested_model_alias': current_model_alias}

    history.append(history_entry)
    session['model'] = model_used
    session['retrieval_mode'] = retrieval_mode_used
    session['status'] = 'pending'
    session['updated_at'] = datetime.now().isoformat()
    session_service.save_session_by_id(session)

    if not isinstance(retrieval_payload, dict):
        raise HTTPException(status_code=500, detail='Invalid retrieval payload from RAG service')

    return FeedbackResponse(
        status='regenerated',
        response=response_text,
        retrieval=retrieval_payload,
        diagnostics=diagnostics,
        model_used=model_used,
        retrieval_mode_used=retrieval_mode_used,
    )


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
    if 'retrieval_mode' not in session:
        session['retrieval_mode'] = settings.retrieval_mode

    # Migrate legacy 'llm' key to 'llm_response' for old session files
    for entry in session.get('history', []):
        if isinstance(entry, dict) and 'llm' in entry and 'llm_response' not in entry:
            entry['llm_response'] = entry.pop('llm')

    return SessionResponse(**session)
