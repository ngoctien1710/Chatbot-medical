import string
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

try:
    from pyvi import ViTokenizer
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện 'pyvi' để chạy module này (uv pip install pyvi)")

class SparseRetriever:
    """
    Class quản trị In-memory Sparse Retrieval (BM25) tối ưu cho tài liệu Y khoa Tiếng Việt.
    Điểm cốt lõi là việc sử dụng Word Segmentation (PyVi) trước khi token hóa để bảo toàn 
    các cụm từ ghép y khoa (vd: 'đái tháo đường', 'nhồi máu cơ tim').
    """
    
    def __init__(self):
        # bm25: Chứa instance của BM25Okapi sau khi build index
        self.bm25: Optional[BM25Okapi] = None
        
        # chunks_data: Lưu trữ dữ liệu chunk nội bộ để tra cứu theo index
        self.chunks_data: List[Dict[str, Any]] = []

    def _segment_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản:
        1. Chữ thường (lowercase)
        2. Bỏ dấu câu (punctuation) để tối ưu index của thuật toán túi từ (BoW)
        3. Tách từ tiếng Việt thành các Token liền khối.
        """
        if not text:
            return ""
        
        text = text.lower()
        
        # Xóa các dấu câu vì Sparse Retrieval quan tâm chủ yếu đến danh từ độc lập
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # PyVi sẽ biến "nhồi máu cơ tim" thành "nhồi_máu cơ_tim" tuỳ từ điển nội bộ
        segmented = ViTokenizer.tokenize(text)
        return segmented

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Xây dựng (hoặc update) In-memory BM25 Index từ danh sách chunks.
        
        Quy trình chuẩn:
        - Nhận output format từ `chunking.py`.
        - Segmentation từng `content`.
        - Gửi mảng token vào `rank_bm25.BM25Okapi`.
        """
        if not chunks:
            print("[Warning] SparseRetriever nhận list chunks rỗng.")
            return

        self.chunks_data = chunks
        tokenized_corpus = []
        
        print(f"[SparseRetriever] Đang thực hiện Word Segmentation & Build BM25 cho {len(chunks)} chunks...")
        
        for chunk in chunks:
            text = chunk.get("content", "")
            segmented_text = self._segment_text(text)
            
            # Tạo list tokens để nạp vào Okapi
            tokens = segmented_text.split()
            tokenized_corpus.append(tokens)
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("[SparseRetriever] Đã hoàn tất Build BM25 In-memory Index.")

    def retrieve_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Dựa vào keyword, trả về danh sách Top K tài liệu (Oversampling).
        
        Args:
query (str): Câu hỏi của User (Sẽ tự động được tách từ giống doc).
            top_k (int): Số lượng chunk top trả về. Oversampling dùng K=20 để nhường sân cho Cross-Encoder rerank.
            
        Returns:
            List[Dict]: Mỗi kết quả chứa "chunk_text", "metadata", và "score_bm25"
        """
        if self.bm25 is None or not self.chunks_data:
            raise ValueError("BM25 chưa được khởi tạo. Vui lòng gọi build_index() trước.")

        # Xử lý câu query cùng một pipeline
        segmented_query = self._segment_text(query)
        query_tokens = segmented_query.split()
        
        # Lấy mảng score cho toàn bộ document
        doc_scores = self.bm25.get_scores(query_tokens)
        
        results = []
        for i, score in enumerate(doc_scores):
            # Bỏ qua những tài liệu trả về điểm 0 (không chứa bất kỳ từ khóa nào từ câu hỏi)
            if score > 0:
                results.append((score, self.chunks_data[i]))
                
        # Sắp xếp giảm dần theo điểm độ phụ thuộc (Relevance Score) của BM25
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Giới hạn Top K
        top_results = results[:top_k]
        
        # Chuẩn hóa Data Format Output theo specs
        formatted_results = []
        for score, data in top_results:
            formatted_results.append({
                "chunk_text": data.get("content", ""),
                "metadata": data.get("metadata", {}),
                "score_bm25": float(score)
            })
            
        return formatted_results

# ====== Example Usage ======
# if __name__ == "__main__":
#     fake_chunks = [
#         {"content": "Bệnh đái tháo đường type 2 là bệnh lý mạn tính.", "metadata": {"page": 1}},
#         {"content": "Dấu hiệu nhồi máu cơ tim bao gồm đau thắt ngực.", "metadata": {"page": 2}},
#         {"content": "Để điều trị tiểu đường cần tiêm insulin đúng liều", "metadata": {"page": 3}}
#     ]
# 
#     retriever = SparseRetriever()
#     retriever.build_index(fake_chunks)
#     
#     query = "dấu hiệu đái tháo đường"
#     res = retriever.retrieve_bm25(query, top_k=2)
#     for r in res:
#         print(f"Score: {r['score_bm25']:.3f} | {r['chunk_text']}")

