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
from app.utils.Advance_RAG_chunking import MedicalDocumentChunker
from app.utils.sparse_retriever import SparseRetriever
from app.utils.hybrid_retriever import HybridRetriever
from app.utils.reranker import Reranker


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

        # Init Advanced RAG components
        self._sparse_retriever = SparseRetriever()
        self._hybrid_retriever = HybridRetriever(
            sparse_retriever_instance=self._sparse_retriever,
            retrieve_dense_callback=self._retrieve_dense
        )
        self._reranker = Reranker()

        # Prebuild BM25 index from existing ChromaDB if data is present
        try:
            vector_store = self._vector_store()
            db_data = vector_store.get()
            if db_data and db_data.get("documents"):
                existing_chunks = []
                for i in range(len(db_data["documents"])):
                    existing_chunks.append({
                        "content": db_data["documents"][i],
                        "metadata": db_data["metadatas"][i] if db_data.get("metadatas") else {}
                    })
                if existing_chunks:
                    self._sparse_retriever.build_index(existing_chunks)
        except Exception as e:
            print(f"[RagService] Could not preload BM25 index: {e}")

    def _retrieve_dense(self, query: str, top_k: int = 20, **kwargs) -> list[dict]:
        vector_store = self._vector_store()
        raw = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        results = []
        for doc, score in raw:
            results.append({
                "chunk_text": doc.page_content,
                "metadata": doc.metadata,
                "score_dense": float(score)
            })
        return results

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
        vector_store = self._vector_store()

        if reset:
            # Delete collection gracefully to avoid SQLite locking issues
            try:
                vector_store.delete_collection()
            except Exception:
                pass
            # Force re-initialize the collection wrapper
            vector_store = self._vector_store()
            
        docs = self._load_raw_documents()
        chunks: list[Document] = []
        sparse_chunks_data: list[dict] = []

        chunker = MedicalDocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        for doc_id, content in docs:
            processed_chunks = chunker.process_markdown(
                text=content,
                initial_metadata={'doc_id': doc_id}
            )
            
            for chunk_data in processed_chunks:
                idx = chunk_data["metadata"].get("chunk_index", 0)
                chunk_data["metadata"]["chunk_id"] = f"{doc_id}:{idx}"
                
                # Format raw dictionaries needed by SparseRetriever
                sparse_chunks_data.append(chunk_data)
                
                # Format Document class needed by vector_store
                chunks.append(
                    Document(
                        page_content=chunk_data["content"],
                        metadata=chunk_data["metadata"]
                    )
                )

        if chunks:
            for start in range(0, len(chunks), self._ingest_batch_size):
                vector_store.add_documents(chunks[start:start + self._ingest_batch_size])
                
        if sparse_chunks_data:
            self._sparse_retriever.build_index(sparse_chunks_data)

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
        if not self._sparse_retriever.bm25:
            # Fallback nếu index BM25 chưa load được
            return [], 'sparse_index_not_built', None

        # 1. Hybrid Search (Sparse + Dense) -> RRF Fusion
        hybrid_candidates = self._hybrid_retriever.hybrid_search(
            query=query, 
            top_k_candidate=20
        )
        
        # 2. Rerank bằng Cross-Encoder
        reranked_results = self._reranker.rerank_candidates(
            query=query, 
            candidates=hybrid_candidates, 
            top_n=settings.top_k
        )

        selected: list[RetrievalItem] = []
        for item in reranked_results:
            score = item.get("score_cross_encoder", 0.0)
            
            # Lưu ý BGE-Reranker M3 có thể trả về logit âm
            # Bạn có thể cân nhắc tắt `if score < settings.score_threshold` nếu kết quả bị chặn quá nhiều
            if score < settings.score_threshold:
                 continue
                
            selected.append(
                RetrievalItem(
                    doc_id=str(item.get("metadata", {}).get("doc_id", "unknown_source")),
                    chunk_id=str(item.get("metadata", {}).get("chunk_id", "unknown_chunk")),
                    score=float(score),
                    text=item.get("chunk_text", ""),
                )
            )

        if not hybrid_candidates:
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
