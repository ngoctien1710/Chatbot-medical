from fastapi import FastAPI

from app.routes.rag import router as rag_router

app = FastAPI(title='Python RAG Service', version='1.0.0')
app.include_router(rag_router)


@app.get('/')
def root() -> dict:
    return {
        'service': 'python-rag-service',
        'status': 'running',
        'health': '/rag/health',
    }
