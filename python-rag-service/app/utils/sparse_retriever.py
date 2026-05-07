import logging
import string
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

try:
    from pyvi import ViTokenizer
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện 'pyvi' để chạy module này (pip install pyvi)")

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    In-memory Sparse Retrieval (BM25) tối ưu cho tài liệu y khoa tiếng Việt.

    Sử dụng PyVi word segmentation trước khi tokenize để bảo toàn các cụm từ ghép y khoa
    (ví dụ: 'đái tháo đường', 'nhồi máu cơ tim') thành một token thay vì bị tách rời.

    Input format (từ chunker/DB): {"content": str, "metadata": dict}
    Output format (cho downstream): {"chunk_text": str, "metadata": dict, "score_bm25": float}
    Việc đổi tên content → chunk_text là intentional để thống nhất với Dense/Hybrid retriever output.
    """

    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.chunks_data: List[Dict[str, Any]] = []

    def _segment_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản: lowercase → xóa dấu câu → tách từ tiếng Việt.

        PyVi biến "nhồi máu cơ tim" thành "nhồi_máu cơ_tim" (cụm từ → 1 token).
        """
        if not text:
            return ''

        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return ViTokenizer.tokenize(text)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Xây dựng (hoặc rebuild) BM25 index từ danh sách chunks.

        Args:
            chunks: Danh sách dict với key 'content' (text) và 'metadata'.
        """
        if not chunks:
            logger.warning('SparseRetriever.build_index() nhận list chunks rỗng — index không được tạo.')
            return

        logger.info('Building BM25 index for %d chunks...', len(chunks))
        self.chunks_data = chunks
        tokenized_corpus = []

        for chunk in chunks:
            text = chunk.get('content', '')
            segmented_text = self._segment_text(text)
            tokenized_corpus.append(segmented_text.split())

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info('BM25 index built successfully.')

    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Tìm kiếm keyword-based, trả về top K chunks theo BM25 score.

        Args:
            query: Câu hỏi của người dùng (tự động được segment giống corpus).
            top_k: Số chunk trả về. Dùng oversampling (k=20) để nhường Cross-Encoder rerank.

        Returns:
            Danh sách dict với keys 'chunk_text', 'metadata', 'score_bm25'.

        Raises:
            ValueError: Nếu build_index() chưa được gọi trước.
        """
        if self.bm25 is None or not self.chunks_data:
            raise ValueError('BM25 chưa được khởi tạo. Vui lòng gọi build_index() trước.')

        query_tokens = self._segment_text(query).split()
        doc_scores = self.bm25.get_scores(query_tokens)

        results = [
            (score, self.chunks_data[i])
            for i, score in enumerate(doc_scores)
            if score > 0  # bỏ qua chunk không chứa từ khóa nào
        ]

        if not results:
            logger.debug('BM25 returned no matches for query: %r', query[:60])

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                'chunk_text': data.get('content', ''),
                'metadata': data.get('metadata', {}),
                'score_bm25': float(score),
            }
            for score, data in results[:top_k]
        ]
