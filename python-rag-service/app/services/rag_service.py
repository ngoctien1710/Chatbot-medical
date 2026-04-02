from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from app.settings import settings
from app.utils.text_chunking import semantic_chunk_text


@dataclass
class RetrievalItem:
    doc_id: str
    chunk_id: str
    score: float
    text: str


class RagService:
    _ingest_batch_size = 1

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'trust_remote_code': True},
        )
        self._chat = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        system_prompt = (
            'Bạn là chuyên gia y khoa.\n\n'
            '1. NHIỆM VỤ:\n'
            '- Trả lời trực tiếp câu hỏi người dùng ngay ở câu đầu tiên.\n'
            '- Luôn đối chiếu thông tin trong context và không bịa thêm chi tiết ngoài context.\n'
            '- Nếu context yếu hoặc thiếu, nêu rõ mức độ chưa chắc chắn thay vì khẳng định tuyệt đối.\n'
            '2. Format trả lời\n'
            '- Trình bày rõ ràng, mạch lạc và tự nhiên.\n'
            '- Mở đầu bằng câu trả lời trực diện, sau đó mới mở rộng bằng các ý chính dạng gạch đầu dòng.\n'
            '- Không mở đầu bằng các cụm như: "bạn đang xem tài liệu", "theo tài liệu", "dựa trên tài liệu".\n'
            '- Không giải thích nguồn tài liệu ở phần mở đầu câu trả lời.\n'
        )
        human_prompt = (
            'CÂU HỎI NGƯỜI DÙNG:\n'
            '{query}\n\n'
            'CONTEXT:\n'
            '{context_block}\n\n'
            'TRẢ LỜI (câu đầu tiên phải trả lời trực diện câu hỏi):'
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                ('human', human_prompt),
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
                chunk_size=settings.chunk_token_size,
                chunk_overlap=settings.chunk_token_overlap,
                min_chunk_chars=settings.min_chunk_chars,
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
            for start in range(0, len(chunks), self._ingest_batch_size):
                vector_store.add_documents(chunks[start:start + self._ingest_batch_size])

        return len(docs), len(chunks)

    def _context_block(self, contexts: list[RetrievalItem]) -> str:
        if not contexts:
            return 'Khong co context nao duoc retrieve tu kho tai lieu.'
        return '\n\n'.join(
            [
                f"[Context {idx + 1}]\n{item.text}"
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
                    'snippet': item.text,
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
