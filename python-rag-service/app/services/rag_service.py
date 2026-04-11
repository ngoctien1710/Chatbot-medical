from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from app.settings import SUPPORTED_CHAT_MODELS, SUPPORTED_RETRIEVAL_MODES, settings
from app.utils.Advance_RAG_chunking import MedicalDocumentChunker
from app.utils.hybrid_retriever import HybridRetriever
from app.utils.reranker import Reranker
from app.utils.sparse_retriever import SparseRetriever


@dataclass
class RetrievalItem:
    doc_id: str
    chunk_id: str
    score: float
    text: str


@dataclass
class ProviderCallResult:
    text: str
    provider: str
    model_alias: str
    model_name: str
    retry_count: int = 0
    error_category: str | None = None
    http_status: int | None = None
    request_id: str | None = None
    fallback_used: str | None = None
    fallback_reason: str | None = None
    quota_remaining: int | None = None
    rate_limit_reset_seconds: int | None = None
    source_error: str | None = None


class ProviderInvocationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_category: str,
        http_status: int | None,
        request_id: str | None,
        retry_count: int,
        rate_limit_reset_seconds: int | None,
    ) -> None:
        super().__init__(message)
        self.error_category = error_category
        self.http_status = http_status
        self.request_id = request_id
        self.retry_count = retry_count
        self.rate_limit_reset_seconds = rate_limit_reset_seconds


class RagService:
    _ingest_batch_size = 1
    _mode_aliases = {
        'hybrid': 'hybrid_rrf',
    }
    _transient_error_categories = {
        'timeout',
        'rate_limited',
        'provider_unavailable',
        'network_error',
    }

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'trust_remote_code': True},
        )
        self._chat_clients: dict[str, ChatOllama] = {}
        self._openai_client: Any = None
        self._gemini_client: Any = None
        self._provider_cooldown_until = {
            'gpt': 0.0,
            'gemini': 0.0,
        }

        self._system_prompt = (
            'Bạn là một chuyên gia y khoa tận tâm, chính xác và chuyên nghiệp.\n\n'
            'MỤC TIÊU:\n'
            '1) Trả lời trực diện: Câu đầu tiên phải trả lời thẳng vào câu hỏi của người dùng.\n'
            '2) Bao phủ thông tin liên quan: Sau câu mở đầu, tổng hợp ĐẦY ĐỦ các thông tin LIÊN QUAN TRỰC TIẾP từ Context.\n'
            '3) Trung thực dữ liệu: Chỉ dùng dữ kiện có trong Context, không suy diễn hoặc bịa thêm.\n'
            '4) An toàn y khoa: Không khẳng định chẩn đoán tuyệt đối khi Context chưa đủ; nêu rõ giới hạn dữ liệu khi cần.\n\n'
            'QUY TẮC TỔNG HỢP CONTEXT:\n'
            '- Ưu tiên thông tin liên quan trực tiếp câu hỏi; bỏ qua chi tiết không liên quan.\n'
            '- Gộp các ý trùng lặp, tránh lặp lại cùng một thông tin.\n'
            '- Nếu có thông tin mâu thuẫn giữa các đoạn context, nêu rõ mâu thuẫn và trả lời theo hướng thận trọng.\n\n'
            '- Không dùng các cụm như: "Theo tài liệu", "Dựa vào context được cung cấp", "Trong văn bản có nói".\n'
        )

        self._human_prompt = (
            'Hãy đọc kỹ toàn bộ context trước khi trả lời.\n\n'
            '=== CONTEXT ===\n'
            '{context_block}\n'
            '===============\n\n'
            'CÂU HỎI CỦA NGƯỜI DÙNG: {query}\n\n'
            'YÊU CẦU THỰC HIỆN:\n'
            '- Trả lời đúng trọng tâm câu hỏi.\n'
            '- Bao phủ đủ ý liên quan trong context, không bỏ sót ý quan trọng.\n'
            'TRẢ LỜI:'
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ('system', self._system_prompt),
                ('human', self._human_prompt),
            ]
        )
        self._output_parser = StrOutputParser()

        # Init retrieval components
        self._sparse_retriever = SparseRetriever()
        self._hybrid_retriever = HybridRetriever(
            sparse_retriever_instance=self._sparse_retriever,
            retrieve_dense_callback=self._retrieve_dense,
        )
        self._reranker = Reranker()

        # Prebuild BM25 index from existing ChromaDB if data is present
        try:
            vector_store = self._vector_store()
            db_data = vector_store.get()
            documents = db_data.get('documents') if db_data else None
            if documents:
                existing_chunks = []
                metadatas = db_data.get('metadatas') or []
                for i, content in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
                    existing_chunks.append(
                        {
                            'content': content,
                            'metadata': metadata,
                        }
                    )
                if existing_chunks:
                    self._sparse_retriever.build_index(existing_chunks)
        except Exception as exc:
            print(f'[RagService] Could not preload BM25 index: {exc}')

    def _resolve_mode(self, mode: str | None = None) -> str:
        selected = (mode or settings.retrieval_mode).strip().lower()
        selected = self._mode_aliases.get(selected, selected)
        if selected not in SUPPORTED_RETRIEVAL_MODES:
            return 'hybrid_original'
        return selected

    def _resolve_model_alias(self, model: str | None = None) -> str:
        selected = (model or settings.default_chat_model_alias).strip().lower()
        if selected not in SUPPORTED_CHAT_MODELS:
            return settings.default_chat_model_alias
        return selected

    @staticmethod
    def _extract_http_status(exc: Exception) -> int | None:
        status = getattr(exc, 'status_code', None)
        if isinstance(status, int):
            return status

        response = getattr(exc, 'response', None)
        if response is not None:
            status = getattr(response, 'status_code', None)
            if isinstance(status, int):
                return status
            status = getattr(response, 'status', None)
            if isinstance(status, int):
                return status
        return None

    @staticmethod
    def _extract_request_id(exc: Exception) -> str | None:
        request_id = getattr(exc, 'request_id', None)
        if isinstance(request_id, str) and request_id.strip():
            return request_id.strip()

        response = getattr(exc, 'response', None)
        headers = getattr(response, 'headers', None)
        if headers is not None and hasattr(headers, 'get'):
            for key in ('x-request-id', 'request-id', 'x-google-request-id'):
                value = headers.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> int | None:
        response = getattr(exc, 'response', None)
        headers = getattr(response, 'headers', None)
        if headers is None or not hasattr(headers, 'get'):
            return None

        retry_after = headers.get('retry-after')
        if retry_after is None:
            return None

        try:
            seconds = int(float(str(retry_after).strip()))
        except Exception:
            return None
        if seconds < 0:
            return None
        return seconds

    def _is_provider_in_cooldown(self, model_alias: str) -> bool:
        if model_alias not in self._provider_cooldown_until:
            return False
        return time.time() < self._provider_cooldown_until[model_alias]

    def _set_provider_cooldown(self, model_alias: str) -> int:
        seconds = max(0, int(settings.rate_limit_cooldown_seconds))
        self._provider_cooldown_until[model_alias] = time.time() + seconds
        return seconds

    def _classify_provider_error(self, exc: Exception) -> str:
        message = str(exc).lower()
        status = self._extract_http_status(exc)
        class_name = type(exc).__name__.lower()

        if status == 429 or 'rate limit' in message or 'too many requests' in message or 'resource exhausted' in message:
            if 'quota' in message or 'insufficient_quota' in message:
                return 'quota_exceeded'
            return 'rate_limited'

        if status in {401, 403} or 'api key' in message or 'unauthorized' in message or 'permission' in message:
            return 'auth_error'

        if status in {400, 404, 422} or 'invalid' in message:
            return 'invalid_request'

        if status in {408, 504} or 'timeout' in message or 'deadline exceeded' in message or 'apitimeouterror' in class_name:
            return 'timeout'

        if status is not None and status >= 500:
            return 'provider_unavailable'

        if 'connection' in message or 'network' in message or 'temporarily unavailable' in message:
            return 'network_error'

        return 'internal_error'

    def _invoke_with_retry(
        self,
        provider_alias: str,
        call: Callable[[], tuple[str, dict[str, Any]]],
    ) -> tuple[str, dict[str, Any], int, int | None]:
        max_attempts = max(1, int(settings.provider_retry_attempts) + 1)
        retry_count = 0

        for attempt in range(max_attempts):
            try:
                text, metadata = call()
                return text, metadata, retry_count, None
            except Exception as exc:
                error_category = self._classify_provider_error(exc)
                status = self._extract_http_status(exc)
                request_id = self._extract_request_id(exc)
                retry_after_seconds = self._extract_retry_after_seconds(exc)
                transient = error_category in self._transient_error_categories

                if (not transient) or attempt == max_attempts - 1:
                    raise ProviderInvocationError(
                        str(exc),
                        error_category=error_category,
                        http_status=status,
                        request_id=request_id,
                        retry_count=retry_count,
                        rate_limit_reset_seconds=retry_after_seconds,
                    ) from exc

                retry_count += 1
                wait_seconds = retry_after_seconds
                if wait_seconds is None:
                    wait_seconds = int(min(8.0, (2 ** attempt) + random.uniform(0.0, 0.5)))
                time.sleep(max(1, wait_seconds))

        raise ProviderInvocationError(
            f'{provider_alias} provider invocation failed unexpectedly.',
            error_category='internal_error',
            http_status=None,
            request_id=None,
            retry_count=retry_count,
            rate_limit_reset_seconds=None,
        )

    def _invoke_mistral(self, query: str, context_block: str) -> tuple[str, dict[str, Any]]:
        model_name = settings.chat_model
        chat = self._chat_clients.get(model_name)
        if chat is None:
            chat = ChatOllama(
                model=model_name,
                base_url=settings.ollama_base_url,
                temperature=0,
            )
            self._chat_clients[model_name] = chat

        qa_chain = self._prompt | chat | self._output_parser
        text = qa_chain.invoke(
            {
                'query': query,
                'context_block': context_block,
            }
        )
        return str(text).strip(), {
            'provider': 'ollama',
            'model_name': model_name,
            'request_id': None,
            'quota_remaining': None,
            'tokens_total': None,
        }

    def _invoke_openai(self, query: str, context_block: str) -> tuple[str, dict[str, Any]]:
        if not settings.openai_api_key or not settings.openai_chat_model:
            raise RuntimeError('OpenAI provider is not configured. Missing OPENAI_API_KEY or OPENAI_CHAT_MODEL.')

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError('OpenAI SDK is not installed. Please install openai package.') from exc

        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.provider_timeout_seconds,
            )

        response = self._openai_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {'role': 'system', 'content': self._system_prompt},
                {
                    'role': 'user',
                    'content': self._human_prompt.format(
                        query=query,
                        context_block=context_block,
                    ),
                },
            ],
            temperature=0,
        )

        content = ''
        if response.choices:
            content = response.choices[0].message.content or ''

        usage = getattr(response, 'usage', None)
        tokens_total = getattr(usage, 'total_tokens', None) if usage is not None else None
        return str(content).strip(), {
            'provider': 'openai',
            'model_name': settings.openai_chat_model,
            'request_id': getattr(response, 'id', None),
            'quota_remaining': None,
            'tokens_total': int(tokens_total) if isinstance(tokens_total, int) else None,
        }

    def _invoke_gemini(self, query: str, context_block: str) -> tuple[str, dict[str, Any]]:
        if not settings.gemini_api_key or not settings.gemini_chat_model:
            raise RuntimeError('Gemini provider is not configured. Missing GEMINI_API_KEY or GEMINI_CHAT_MODEL.')

        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError('Google Generative AI SDK is not installed. Please install google-generativeai package.') from exc

        if self._gemini_client is None:
            genai.configure(api_key=settings.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(settings.gemini_chat_model)

        prompt = (
            f'{self._system_prompt}\n\n'
            f"{self._human_prompt.format(query=query, context_block=context_block)}"
        )
        response = self._gemini_client.generate_content(
            prompt,
            generation_config={'temperature': 0},
            request_options={'timeout': settings.provider_timeout_seconds},
        )

        content = getattr(response, 'text', '') or ''
        usage = getattr(response, 'usage_metadata', None)
        tokens_total = getattr(usage, 'total_token_count', None) if usage is not None else None

        return str(content).strip(), {
            'provider': 'gemini',
            'model_name': settings.gemini_chat_model,
            'request_id': None,
            'quota_remaining': None,
            'tokens_total': int(tokens_total) if isinstance(tokens_total, int) else None,
        }

    def _invoke_provider(self, model_alias: str, query: str, context_block: str) -> ProviderCallResult:
        if model_alias == 'mistral':
            text, metadata = self._invoke_mistral(query, context_block)
            return ProviderCallResult(
                text=text,
                provider='ollama',
                model_alias='mistral',
                model_name=str(metadata.get('model_name') or settings.chat_model),
            )

        if model_alias == 'gpt':
            provider_name = 'openai'
            provider_call: Callable[[], tuple[str, dict[str, Any]]] = lambda: self._invoke_openai(query, context_block)
        else:
            provider_name = 'gemini'
            provider_call = lambda: self._invoke_gemini(query, context_block)

        if self._is_provider_in_cooldown(model_alias):
            if settings.fallback_on_provider_error:
                text, metadata = self._invoke_mistral(query, context_block)
                return ProviderCallResult(
                    text=text,
                    provider='ollama',
                    model_alias='mistral',
                    model_name=str(metadata.get('model_name') or settings.chat_model),
                    fallback_used='mistral',
                    fallback_reason=f'{model_alias}_cooldown',
                    error_category='rate_limited',
                )
            raise ProviderInvocationError(
                f'{provider_name} provider is cooling down after rate limit.',
                error_category='rate_limited',
                http_status=429,
                request_id=None,
                retry_count=0,
                rate_limit_reset_seconds=settings.rate_limit_cooldown_seconds,
            )

        try:
            text, metadata, retry_count, retry_after_seconds = self._invoke_with_retry(model_alias, provider_call)
            return ProviderCallResult(
                text=text,
                provider=str(metadata.get('provider') or provider_name),
                model_alias=model_alias,
                model_name=str(metadata.get('model_name') or ''),
                retry_count=retry_count,
                request_id=metadata.get('request_id'),
                quota_remaining=metadata.get('quota_remaining'),
                rate_limit_reset_seconds=retry_after_seconds,
            )
        except ProviderInvocationError as exc:
            if exc.error_category in {'rate_limited', 'quota_exceeded'}:
                self._set_provider_cooldown(model_alias)

            if settings.fallback_on_provider_error:
                text, metadata = self._invoke_mistral(query, context_block)
                return ProviderCallResult(
                    text=text,
                    provider='ollama',
                    model_alias='mistral',
                    model_name=str(metadata.get('model_name') or settings.chat_model),
                    retry_count=exc.retry_count,
                    error_category=exc.error_category,
                    http_status=exc.http_status,
                    request_id=exc.request_id,
                    fallback_used='mistral',
                    fallback_reason=f'{provider_name}_{exc.error_category}',
                    rate_limit_reset_seconds=exc.rate_limit_reset_seconds,
                    source_error=str(exc),
                )
            raise

    @staticmethod
    def _summarize_scores(items: list[RetrievalItem]) -> dict[str, float] | None:
        if not items:
            return None
        scores = [item.score for item in items]
        return {
            'max': max(scores),
            'min': min(scores),
            'avg': sum(scores) / len(scores),
        }

    @staticmethod
    def _build_retrieval_items(
        candidates: list[dict],
        score_key: str,
        apply_threshold: bool = False,
    ) -> list[RetrievalItem]:
        selected: list[RetrievalItem] = []
        for item in candidates:
            score = float(item.get(score_key, 0.0))
            if apply_threshold and score < settings.score_threshold:
                continue

            metadata = item.get('metadata') if isinstance(item.get('metadata'), dict) else {}
            doc_id = str(metadata.get('doc_id', 'unknown_source'))
            chunk_id = str(metadata.get('chunk_id', 'unknown_chunk'))
            chunk_text = str(item.get('chunk_text', ''))

            selected.append(
                RetrievalItem(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score,
                    text=chunk_text,
                )
            )
        return selected

    def _retrieve_dense(self, query: str, top_k: int = 20, **kwargs) -> list[dict]:
        vector_store = self._vector_store()
        raw = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        results: list[dict] = []
        for doc, score in raw:
            results.append(
                {
                    'chunk_text': doc.page_content,
                    'metadata': doc.metadata,
                    'score_dense': float(score),
                }
            )
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
            chunk_overlap=settings.chunk_overlap,
        )

        for doc_id, content in docs:
            processed_chunks = chunker.process_markdown(
                text=content,
                initial_metadata={'doc_id': doc_id},
            )

            for chunk_data in processed_chunks:
                idx = chunk_data['metadata'].get('chunk_index', 0)
                chunk_data['metadata']['chunk_id'] = f'{doc_id}:{idx}'

                # Format raw dictionaries needed by SparseRetriever
                sparse_chunks_data.append(chunk_data)

                # Format Document class needed by vector_store
                chunks.append(
                    Document(
                        page_content=chunk_data['content'],
                        metadata=chunk_data['metadata'],
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
                f'[Context {idx + 1}]\n{item.text}'
                for idx, item in enumerate(contexts)
            ]
        )

    def _is_sparse_ready(self) -> bool:
        return self._sparse_retriever.bm25 is not None

    def _full_corpus_candidates(self) -> tuple[list[dict], bool]:
        vector_store = self._vector_store()
        db_data = vector_store.get()
        documents = db_data.get('documents') if db_data else None
        if not documents:
            return [], False

        metadatas = db_data.get('metadatas') if db_data else None
        max_chunks = settings.cross_encoder_max_scan_chunks
        truncated = max_chunks > 0 and len(documents) > max_chunks
        upper_bound = min(len(documents), max_chunks) if max_chunks > 0 else len(documents)

        candidates: list[dict] = []
        for idx in range(upper_bound):
            metadata = {}
            if isinstance(metadatas, list) and idx < len(metadatas) and isinstance(metadatas[idx], dict):
                metadata = dict(metadatas[idx])

            metadata.setdefault('doc_id', str(metadata.get('source', 'unknown_source')))
            metadata.setdefault('chunk_id', f'fullscan:{idx}')
            candidates.append(
                {
                    'chunk_text': str(documents[idx]),
                    'metadata': metadata,
                }
            )

        return candidates, truncated

    def _retrieve_hybrid_original(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        if not self._is_sparse_ready():
            # Fallback nếu index BM25 chưa load được
            return [], 'sparse_index_not_built', None

        hybrid_candidates = self._hybrid_retriever.hybrid_search(
            query=query,
            top_k_candidate=settings.retrieval_candidate_k,
            rrf_k=settings.rrf_k,
        )

        if not hybrid_candidates:
            return [], 'empty_index', None

        reranked_results = self._reranker.rerank_candidates(
            query=query,
            candidates=hybrid_candidates,
            top_n=settings.top_k,
        )

        selected = self._build_retrieval_items(
            candidates=reranked_results,
            score_key='score_cross_encoder',
            apply_threshold=True,
        )
        if not selected:
            return [], 'no_match', None

        return selected, 'success', self._summarize_scores(selected)

    def _retrieve_dense_only(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        dense_results = self._retrieve_dense(query=query, top_k=settings.top_k)
        if not dense_results:
            return [], 'empty_index', None

        selected = self._build_retrieval_items(
            candidates=dense_results,
            score_key='score_dense',
        )
        if not selected:
            return [], 'no_match_dense_only', None

        return selected, 'success_dense_only', self._summarize_scores(selected)

    def _retrieve_sparse_only(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        if not self._is_sparse_ready():
            return [], 'sparse_index_not_built', None

        sparse_results = self._sparse_retriever.retrieve_bm25(query=query, top_k=settings.top_k)
        if not sparse_results:
            return [], 'no_match_sparse_only', None

        selected = self._build_retrieval_items(
            candidates=sparse_results,
            score_key='score_bm25',
        )
        if not selected:
            return [], 'no_match_sparse_only', None

        return selected, 'success_sparse_only', self._summarize_scores(selected)

    def _retrieve_hybrid_rrf(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        if not self._is_sparse_ready():
            return [], 'sparse_index_not_built', None

        hybrid_candidates = self._hybrid_retriever.hybrid_search(
            query=query,
            top_k_candidate=settings.retrieval_candidate_k,
            rrf_k=settings.rrf_k,
        )
        if not hybrid_candidates:
            return [], 'empty_index', None

        selected = self._build_retrieval_items(
            candidates=hybrid_candidates[: settings.top_k],
            score_key='score_rrf',
        )
        if not selected:
            return [], 'no_match_hybrid_rrf', None

        return selected, 'success_hybrid_rrf', self._summarize_scores(selected)

    def _retrieve_cross_encoder_only(self, query: str) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        candidates, truncated = self._full_corpus_candidates()
        if not candidates:
            return [], 'empty_index', None

        reranked_results = self._reranker.rerank_candidates(
            query=query,
            candidates=candidates,
            top_n=settings.top_k,
        )
        selected = self._build_retrieval_items(
            candidates=reranked_results,
            score_key='score_cross_encoder',
        )
        if not selected:
            return [], 'no_match_cross_encoder_only', None

        status = 'success_cross_encoder_only_limited' if truncated else 'success_cross_encoder_only'
        return selected, status, self._summarize_scores(selected)

    def retrieve(
        self,
        query: str,
        mode: str | None = None,
    ) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        mode_name = self._resolve_mode(mode)

        if mode_name == 'dense_only':
            return self._retrieve_dense_only(query)
        if mode_name == 'sparse_only':
            return self._retrieve_sparse_only(query)
        if mode_name == 'hybrid_rrf':
            return self._retrieve_hybrid_rrf(query)
        if mode_name == 'cross_encoder_only':
            return self._retrieve_cross_encoder_only(query)
        return self._retrieve_hybrid_original(query)

    # Backward-compatible alias for older call sites
    def _retrieve(self, query: str, mode: str | None = None) -> tuple[list[RetrievalItem], str, dict[str, float] | None]:
        return self.retrieve(query=query, mode=mode)

    def answer(self, query: str, model: str | None = None, retrieval_mode: str | None = None) -> dict:
        started = time.time()
        effective_model_alias = self._resolve_model_alias(model)
        effective_retrieval_mode = self._resolve_mode(retrieval_mode)

        try:
            selected, status, score_summary = self.retrieve(query, mode=effective_retrieval_mode)
            context_block = self._context_block(selected)
            generation = self._invoke_provider(
                model_alias=effective_model_alias,
                query=query,
                context_block=context_block,
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
                'response': generation.text,
                'retrieval': {
                    'retrieval_status': status,
                    'top_k': len(selected),
                    'snippets': snippets,
                    'retrieval_mode': effective_retrieval_mode,
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
                    'error': generation.source_error,
                    'provider': generation.provider,
                    'model_alias': generation.model_alias,
                    'model_name': generation.model_name,
                    'error_category': generation.error_category,
                    'http_status': generation.http_status,
                    'retry_count': generation.retry_count,
                    'request_id': generation.request_id,
                    'fallback_used': generation.fallback_used,
                    'fallback_reason': generation.fallback_reason,
                    'quota_remaining': generation.quota_remaining,
                    'rate_limit_reset_seconds': generation.rate_limit_reset_seconds,
                },
            }
        except Exception as exc:
            return {
                'response': 'He thong local LLM tam thoi chua san sang. Vui long kiem tra Python RAG service va Ollama.',
                'retrieval': {
                    'retrieval_status': 'fallback_error',
                    'top_k': 0,
                    'snippets': [],
                    'retrieval_mode': effective_retrieval_mode,
                    'score_summary': None,
                },
                'diagnostics': {
                    'latency_ms': int((time.time() - started) * 1000),
                    'context_used': 0,
                    'error': str(exc),
                    'provider': 'none',
                    'model_alias': effective_model_alias,
                    'model_name': None,
                    'error_category': 'internal_error',
                    'http_status': None,
                    'retry_count': 0,
                    'request_id': None,
                    'fallback_used': None,
                    'fallback_reason': None,
                    'quota_remaining': None,
                    'rate_limit_reset_seconds': None,
                },
            }


rag_service = RagService()
