import hashlib
from typing import List, Dict, Any, Callable

class HybridRetriever:
    """
    Class quản trị In-memory Hybrid Retrieval. Trộn (Fusion) 2 kĩ thuật 
    tìm kiếm Sparse (Keyword-based) và Dense (Semantic Embedding-based) ưu việt 
    cho lĩnh vực y khoa (giúp cân bằng tra cứu triệu chứng tổng quát và thuốc đặc trị).
    """

    def __init__(self, sparse_retriever_instance, retrieve_dense_callback: Callable):
        """
        Khởi tạo hệ thống Hybrid RAG.
        
        Args:
            sparse_retriever_instance: Đối tượng `SparseRetriever` chuyên xử lý tìm theo text Tiếng Việt.
            retrieve_dense_callback: Hàm (hành vi) lấy Top K chunks từ ChromaDB, cấu trúc trả ra 
                                     cũng nên dạng `[{"chunk_text":..., "metadata":...}, ...]`.
        """
        self.sparse_retriever = sparse_retriever_instance
        self.retrieve_dense_chroma = retrieve_dense_callback
        
    def _generate_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """
        Tạo ID định danh để Deduplicate.
        Dùng metadata `[source]_[chunk_index]` (từ Phần 1) nếu có. Nếu không, băm toàn bộ text thành mã Hash.
        """
        meta = chunk.get("metadata", {})
        if "source" in meta and "chunk_index" in meta:
            return f"{meta['source']}_{meta['chunk_index']}"
        elif "chunk_index" in meta:
            return f"index_{meta['chunk_index']}"
        
        # Fallback Deduplication cực mạnh độ chính xác tuyệt đối
        text = chunk.get("chunk_text", "")
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def reciprocal_rank_fusion(self, search_results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Thuật toán Toán học RRF (Reciprocal Rank Fusion).
        
        [Cơ sở Toán học của RRF]:
        Điểm số BM25 (dựa trên tần suất TF-IDF) và điểm Cosine của VectorDB có phương sai hoàn toàn khác nhau.
        Linear Combination (nhân trọng số alpha*X + beta*Y) sẽ vô tình làm lu mờ nhánh kia. 
        RRF khắc phục hoàn toàn bằng việc đánh giá công bằng tuyệt đối dưạ trên THỨ HẠNG (RANK) thay cho (SCORE).
        
        Công thức RRF Score cho chunk D:
            RRF(D) = Σ [ 1 / ( k + rank_i(D) ) ]
            
        Giải thích:
        - `rank_i(D)`: Thứ hạng gốc của văn bản D do bộ tìm kiếm thứ i bầu chọn (tính từ 1).
        - `k`: Hyperparameter mượt mà (Smoothing factor). Bài gốc Cormack 2009 lấy K=60 để chống lại 
               sự độc tài của văn bản hạng 1. Hạng thấp dần sẽ chia tỉ lệ phạt chậm lại.
               
        Args:
            search_results_list: Danh sách chứa kết quả Dense List và Sparse List.
            k (int): Hằng số điều hoà. Mặc định 60.
            
Returns:
            List các tài liệu đã đuợc Deduplicate, phân hạng lại và chèn RRF Score mới.
        """
        rrf_map: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        
        # Duyệt qua danh sách 2 nhánh (Sparse và Dense)
        for result_list in search_results_list:
            # Duyệt danh sách được sort mặc định của một hệ thống truy xuất (vị trí 0 là rank 1)
            for rank, item in enumerate(result_list, start=1):
                chunk_id = self._generate_chunk_id(item)
                
                # Trọng số RRF
                score_contribution = 1.0 / (k + rank)
                
                if chunk_id in rrf_map:
                    # Nếu chunk đã được tìm thấy ở hệ thống khác, cộng dồn điểm RRF rank
                    rrf_map[chunk_id] += score_contribution
                else:
                    rrf_map[chunk_id] = score_contribution
                    # Lưu thể hiện vật lý của object chunk data
                    chunk_map[chunk_id] = item
                    
        # Sort danh sách các chunk đã deduplicate theo RRF score giảm dần
        sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, rrf_score in sorted_items:
            final_chunk = chunk_map[chunk_id].copy()
            # Đính kèm RRF score chuẩn
            final_chunk["score_rrf"] = rrf_score
            final_results.append(final_chunk)
            
        return final_results

    def hybrid_search(self, query: str, top_k_candidate: int = 20, **kwargs) -> List[Dict[str, Any]]:
        """
        Khởi chạy tra cứu đồng bộ, kết hợp và chuẩn bị tệp Data cho Cross-Encoder (Phần sau).
        
        Args:
            query (str): Lệnh tìm kiếm của Y/Bác sĩ.
            top_k_candidate (int): Mặc định là 20 chunks từ mỗi bộ Retriever (Rất sâu).
            **kwargs: Hỗ trợ pass tham số (chẳng hạn metadata_filter={'source':'phac_do'}) xuống Dense Retriever.
            
        Returns:
            List[Dict]: Khối ứng viên tối đa khoảng 40 chunks. Hoàn toàn sạch dấu trùng lặp.
        """
        # --- 1. Nhánh Sparse (Tìm theo từ vựng NLP) ---
        sparse_results = self.sparse_retriever.retrieve_bm25(query, top_k=top_k_candidate)
        
        # --- 2. Nhánh Dense (Tìm theo ngữ nghĩa AI) ---
        # Hàm callback gọi vào DB Vector giả định nhận thêm kwargs pass sang DB
        dense_results = self.retrieve_dense_chroma(query, top_k=top_k_candidate, **kwargs)
        
        # --- 3. Deduplicate và Trộn (RRF Fusion) ---
        # Nạp tất cả list nhánh tìm kiếm. RRF sẽ vắt kiệt và tìm ra best of both worlds.
        merged_candidates = self.reciprocal_rank_fusion([sparse_results, dense_results], k=60)
        
        return merged_candidates

# ====== Example Usage ======
# if __name__ == "__main__":
#     # Hàm Mock (Dummy) cho Chroma
#     def mock_chroma_func(q, top_k, **kwargs):
#         return [
#             {"chunk_text": "Bệnh đái tháo đường (tiểu đường) là gì?", "metadata": {"chunk_index": 1}},
#             {"chunk_text": "Thuốc insulin chữa tiểu đường.", "metadata": {"chunk_index": 5}}
#         ]
#     
#     # SparseRetrieval giả lập
#     class MockSparse:
#         def retrieve_bm25(self, q, top_k):
#             return [
#                 {"chunk_text": "Bệnh đái tháo đường type 2 là bệnh lý mạn tính.", "metadata": {"chunk_index": 2}},
#                 {"chunk_text": "Bệnh đái tháo đường (tiểu đường) là gì?", "metadata": {"chunk_index": 1}} # Trùng với Chroma
#             ]
#             
#     hybrid = HybridRetriever(MockSparse(), mock_chroma_func)
#     res = hybrid.hybrid_search("đái tháo đường", top_k_candidate=2)
#     print(f"Tổng chunks sau hòa trộn RRF: {len(res)}")
#     for r in res:
#         print(f"RRF={r['score_rrf']:.4f} | {r['chunk_text']}")