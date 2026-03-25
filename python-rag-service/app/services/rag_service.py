from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.settings import settings
from app.utils.text_chunking import semantic_chunk_text


@dataclass
class RetrievalItem:
    doc_id: str
    chunk_id: str
    score: float
    text: str


class RagService:
    def __init__(self) -> None:
        self._embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self._chat = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    'Ban la tro ly y te su dung bo context duoc truy hoi de tra loi. '
                    'Chi dua tren context khi dua ra ket luan thuc te. '
                    'Neu context khong du, hay noi ro thong tin chua day du thay vi phan doan. '
                    'Luon nhac rang day khong thay the tu van bac si.',
                ),
                (
                    'human',
                    'Cau hoi nguoi dung: {query}\n\n'
                    'Context retrieve:\n{context_block}\n\n'
                    'Yeu cau: tra loi ngan gon, ro rang, uu tien thong tin trong context va khong bịa them su that.',
                ),
            ]
        )
        self._qa_chain = self._prompt | self._chat | StrOutputParser()

    def _vector_store(self) -> Chroma:
        settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name='medical_rag',
            embedding_function=self._embeddings,
            persist_directory=str(settings.vector_db_dir),
        )

    def _load_raw_documents(self) -> list[tuple[str, str]]:
        root = settings.data_raw_dir
        if not root.exists():
            return []

        items: list[tuple[str, str]] = []
        for file_path in sorted(root.iterdir()):
            if file_path.suffix.lower() not in {'.txt', '.md'}:
                continue
            content = file_path.read_text(encoding='utf-8').strip()
            if content:
                items.append((file_path.name, content))
        return items

    def ingest(self, reset: bool = True) -> tuple[int, int]:
        if reset and settings.vector_db_dir.exists():
            shutil.rmtree(settings.vector_db_dir)

        docs = self._load_raw_documents()
        vector_store = self._vector_store()
        chunks: list[Document] = []

        for doc_id, content in docs:
            parts = semantic_chunk_text(
                text=content,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embeddings=self._embeddings,
            )
            for idx, chunk in enumerate(parts):
                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            'doc_id': doc_id,
                            'chunk_id': f'{doc_id}:{idx}',
                        },
                    )
                )

        if chunks:
            vector_store.add_documents(chunks)

        return len(docs), len(chunks)

    def _context_block(self, contexts: list[RetrievalItem]) -> str:
        if not contexts:
            return 'Khong co context nao duoc retrieve tu kho tai lieu.'
        return '\n\n'.join(
            [
                f"[Context {idx + 1}] Source={item.doc_id}; Score={item.score:.3f}\n{item.text}"
                for idx, item in enumerate(contexts)
            ]
        )

    def _retrieve(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        vector_store = self._vector_store()
        raw = vector_store.similarity_search_with_relevance_scores(query, k=settings.top_k)

        selected: list[RetrievalItem] = []
        for doc, score in raw:
            if score < settings.score_threshold:
                continue
            selected.append(
                RetrievalItem(
                    doc_id=str(doc.metadata.get('doc_id', 'unknown_source')),
                    chunk_id=str(doc.metadata.get('chunk_id', 'unknown_chunk')),
                    score=float(score),
                    text=doc.page_content,
                )
            )

        if not raw:
            return [], 'empty_index', None
        if not selected:
            return [], 'no_match', None

        scores = [item.score for item in selected]
        summary = {
            'max': max(scores),
            'min': min(scores),
            'avg': sum(scores) / len(scores),
        }
        return selected, 'success', summary

    def answer(self, query: str) -> dict:
        started = time.time()
        try:
            selected, status, score_summary = self._retrieve(query)
            text = self._qa_chain.invoke(
                {
                    'query': query,
                    'context_block': self._context_block(selected),
                }
            )

            snippets = [
                {
                    'id': item.chunk_id,
                    'doc_id': item.doc_id,
                    'score': round(item.score, 4),
                    'snippet': item.text[:180].replace('\n', ' ').strip() + ('...' if len(item.text) > 180 else ''),
                }
                for item in selected
            ]

            return {
                'response': str(text).strip(),
                'retrieval': {
                    'retrieval_status': status,
                    'top_k': len(selected),
                    'snippets': snippets,
                    'score_summary': (
                        {
                            'max': round(score_summary['max'], 4),
                            'min': round(score_summary['min'], 4),
                            'avg': round(score_summary['avg'], 4),
                        }
                        if score_summary
                        else None
                    ),
                },
                'diagnostics': {
                    'latency_ms': int((time.time() - started) * 1000),
                    'context_used': len(selected),
                    'error': None,
                },
            }
        except Exception as exc:
            return {
                'response': 'He thong local LLM tam thoi chua san sang. Vui long kiem tra Python RAG service va Ollama.',
                'retrieval': {
                    'retrieval_status': 'fallback_error',
                    'top_k': 0,
                    'snippets': [],
                    'score_summary': None,
                },
                'diagnostics': {
                    'latency_ms': int((time.time() - started) * 1000),
                    'context_used': 0,
                    'error': str(exc),
                },
            }


rag_service = RagService()
