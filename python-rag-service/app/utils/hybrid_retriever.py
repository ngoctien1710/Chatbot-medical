import hashlib
import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid Retrieval kết hợp Sparse (BM25) và Dense (Semantic Embedding) bằng RRF.

    Lợi thế trong y khoa: cân bằng tra cứu triệu chứng tổng quát (dense) và
    thuốc/tên bệnh cụ thể (sparse). RRF tránh bias khi một nhánh có điểm số vượt trội.
    """

    def __init__(self, sparse_retriever_instance, retrieve_dense_callback: Callable) -> None:
        """
        Args:
            sparse_retriever_instance: Đối tượng SparseRetriever cho BM25 search.
            retrieve_dense_callback: Hàm nhận (query, top_k, **kwargs) và trả về
                                     list[{"chunk_text": str, "metadata": dict, ...}].
        """
        self.sparse_retriever = sparse_retriever_instance
        self.retrieve_dense_chroma = retrieve_dense_callback

    def _generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """
        Tạo ID để deduplicate giữa kết quả sparse và dense.
        Dùng metadata source+chunk_index nếu có, fallback về MD5 hash của text.
        """
        meta = chunk.get('metadata', {})
        if 'source' in meta and 'chunk_index' in meta:
            return f"{meta['source']}_{meta['chunk_index']}"
        if 'chunk_index' in meta:
            return f"index_{meta['chunk_index']}"
        return hashlib.md5(chunk.get('chunk_text', '').encode('utf-8')).hexdigest()

    def reciprocal_rank_fusion(self, search_results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (Cormack et al. 2009).

        Điểm BM25 và cosine similarity có phương sai khác nhau, nên không thể cộng trực tiếp.
        RRF đánh giá theo THỨ HẠNG thay vì điểm số, đảm bảo công bằng giữa hai nhánh:

            RRF(D) = Σ [ 1 / (k + rank_i(D)) ]

        k=60 (smoothing factor) giảm ảnh hưởng của vị trí rank 1 so với rank 2.

        Args:
            search_results_list: [sparse_results, dense_results], mỗi list đã được sort theo score.
            k: Smoothing factor. Mặc định 60 theo bài báo gốc.

        Returns:
            Danh sách chunk đã deduplicate và sort theo RRF score giảm dần.
        """
        rrf_map: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for result_list in search_results_list:
            for rank, item in enumerate(result_list, start=1):
                chunk_id = self._generate_chunk_id(item)
                score_contribution = 1.0 / (k + rank)

                if chunk_id in rrf_map:
                    rrf_map[chunk_id] += score_contribution
                else:
                    rrf_map[chunk_id] = score_contribution
                    chunk_map[chunk_id] = item

        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for chunk_id, rrf_score in sorted_items:
            final_chunk = chunk_map[chunk_id].copy()
            final_chunk['score_rrf'] = rrf_score
            final_results.append(final_chunk)

        logger.debug('RRF merged %d candidates into %d unique chunks.', sum(len(r) for r in search_results_list), len(final_results))
        return final_results

    def hybrid_search(self, query: str, top_k_candidate: int = 20, rrf_k: int = 60, **kwargs) -> List[Dict[str, Any]]:
        """
        Chạy đồng thời sparse + dense retrieval, rồi merge bằng RRF.

        Args:
            query: Câu hỏi tìm kiếm.
            top_k_candidate: Số chunk lấy từ mỗi retriever (oversampling). Mặc định 20.
            rrf_k: Smoothing factor cho RRF. Mặc định 60.
            **kwargs: Tham số thêm pass xuống dense retriever (ví dụ: metadata_filter).

        Returns:
            Tối đa ~40 chunks sau dedup và RRF sort, sẵn sàng cho Cross-Encoder rerank.
        """
        sparse_results = self.sparse_retriever.retrieve_bm25(query, top_k=top_k_candidate)
        dense_results = self.retrieve_dense_chroma(query, top_k=top_k_candidate, **kwargs)

        return self.reciprocal_rank_fusion([sparse_results, dense_results], k=rrf_k)
