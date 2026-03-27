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
        system_prompt = (
            'Bạn là chuyên gia y khoa.\n\n'
            'NHIỆM VỤ:\n'
            '- Trả lời câu hỏi người dùng dựa trên thông tin trong context.\n'
            '- CÓ thể sử dụng kiến thức bên ngoài nếu context không đề cập.\n\n'
            'QUY TẮC QUAN TRỌNG:\n\n'
            '1. Grounding\n'
            '- Mọi kết luận thực tế phải có cơ sở trong context.\n'
            '- Không suy đoán hoặc bịa thêm.\n\n'
            '2. Thiếu thông tin\n'
            '- Nếu context không đủ → nói rõ: "Thông tin trong tài liệu truy hồi chưa đủ để kết luận."\n\n'
            '3. Format trả lời\n'
            '- Đầy đủ, chi tiết\n'
            '- Sát với thông tin trong context\n'
            '4. Luôn kết thúc bằng:\n'
            '"Thông tin trên chỉ mang tính tham khảo và không thay thế tư vấn y khoa từ bác sĩ."'
        )
        human_prompt = (
            'CÂU HỎI:\n'
            '{query}\n\n'
            'CONTEXT:\n'
            '{context_block}\n\n'
            'HÃY THỰC HIỆN:\n\n'
            'Bước 1: Xác định thông tin liên quan trong context\n'
            'Bước 2: Tổng hợp câu trả lời sát với context\n'
            'TRẢ LỜI:'
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
<<<<<<< HEAD
                (
                    'system',
                    """Bạn là một trợ lý y tế AI chuyên nghiệp, cẩn trọng và đáng tin cậy. Nhiệm vụ của bạn là phân tích thông tin được cung cấp và trả lời câu hỏi của người dùng.

NGUYÊN TẮC HOẠT ĐỘNG:
1. SỰ THẬT LÀ TUYỆT ĐỐI: Chỉ sử dụng thông tin được cung cấp trong phần "Context" để trả lời. Tuyệt đối không sử dụng kiến thức tự có để bịa đặt, suy diễn hoặc thêm thắt thông tin.
2. XỬ LÝ THIẾU THÔNG TIN: Nếu "Context" không chứa đủ thông tin để trả lời trọn vẹn, hãy nói rõ: "Dựa trên dữ liệu hiện tại, tôi không có đủ thông tin để trả lời [phần cụ thể của câu hỏi]."
3. ĐỊNH DẠNG: Trình bày câu trả lời rõ ràng, mạch lạc. Sử dụng gạch đầu dòng cho các ý chính. Nếu context có nhiều ý phức tạp, hãy tổng hợp chúng một cách logic.
4. CẢNH BÁO BẮT BUỘC: Luôn kết thúc câu trả lời bằng dòng chữ: "*Lưu ý: Thông tin trên chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán hoặc tư vấn từ bác sĩ chuyên khoa.*\""""
                ),
                (
                    'human',
                    """Context được cung cấp:
{context_block}

Câu hỏi của người dùng: {query}

Hãy suy nghĩ từng bước để đối chiếu câu hỏi với Context trước khi đưa ra câu trả lời cuối cùng."""
                ),
=======
                ('system', system_prompt),
                ('human', human_prompt),
>>>>>>> 4178c4d (d:\BAI_LAB\Chatbot-medical\.venv\Scripts\activate.bat)
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
