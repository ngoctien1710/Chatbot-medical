from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.rag import router as rag_router
from app.routes.session import router as session_router
from app.settings import settings

app = FastAPI(title='Python RAG Service', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.include_router(rag_router)
app.include_router(session_router)


@app.get('/')
def root() -> dict:
    return {
        'service': 'python-rag-service',
        'status': 'running',
        'health': '/rag/health',
        'chat': '/chat',
        'sessions': '/sessions',
        'feedback': '/feedback',
    }
