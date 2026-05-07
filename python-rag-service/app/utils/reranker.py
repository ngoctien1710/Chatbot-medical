import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """
    Singleton class quản trị mô hình Cross-Encoder chuyên trách Reranking.
    Singleton bảo vệ hệ thống khỏi memory leak khi tải model nhiều lần vào RAM/VRAM.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'cpu') -> None:
        """
        Tải model Cross-Encoder từ HuggingFace.

        Args:
            model_name: Mặc định dùng BGE-Reranker-M3 (multilingual, tốt cho tiếng Việt).
            device: 'cuda' nếu có GPU NVIDIA, mặc định 'cpu'.
        """
        logger.info('Loading CrossEncoder model %s on %s...', model_name, device.upper())
        # max_length=512 phù hợp với chunk_size ~900 chars
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        logger.info('CrossEncoder model loaded successfully.')

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Chấm điểm cross-attention giữa query và từng chunk.

        Khác với Dual-Encoder (ChromaDB) tính cosine similarity giữa 2 vector riêng biệt,
        Cross-Encoder gộp query và doc vào cùng một network để có cross-attention đầy đủ,
        cho độ chính xác cao hơn nhưng chậm hơn — phù hợp để rerank tập nhỏ (~20-40 chunks).

        Args:
            query: Câu hỏi của người dùng.
            candidates: Output từ HybridRetriever (RRF), khoảng 20-40 tài liệu.
            top_n: Số tài liệu tốt nhất trả về làm input prompt cho LLM.

        Returns:
            Top N candidates đã sắp xếp lại, kèm trường 'score_cross_encoder'.
        """
        if not candidates:
            logger.debug('rerank_candidates called with empty candidates list.')
            return []

        # CrossEncoder yêu cầu format [[query, doc], [query, doc], ...]
        cross_inp = [[query, doc.get('chunk_text', '')] for doc in candidates]

        scores = self.model.predict(cross_inp)

        for idx, score in enumerate(scores):
            # Chuyển Float32 của NumPy sang float Python để JSON-serializable
            candidates[idx]['score_cross_encoder'] = float(score)

        reranked = sorted(candidates, key=lambda x: x['score_cross_encoder'], reverse=True)

        logger.debug('Reranked %d candidates, returning top %d.', len(candidates), top_n)
        return reranked[:top_n]
