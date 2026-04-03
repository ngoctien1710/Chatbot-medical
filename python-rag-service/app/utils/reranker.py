import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Singleton Class quản trị mô hình Cross-Encoder chuyên trách Reranking.
    Cơ chế Singleton bảo vệ hệ thống khỏi các vòng rò rỉ bộ nhớ (Memory Leak)
    khi tải mô hình Heavy Deep Learning nhiều lần vào RAM/VRAM.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        """
        Khởi tạo và tải tạ cục bộ (Local Model Weight) từ HuggingFace.
        
        Args:
            model_name (str): Mặc định sử dụng BGE-Reranker-M3 (Multi-lingual 3 cực mạnh cho tiếng Việt).
            device (str): Gán "cuda" nếu máy trạm của bạn có GPU NVIDIA, mặc định "cpu".
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[Reranker] Đang tải mô hình nhúng {model_name} vào {device.upper()}... Vui lòng chờ!")
        print(f"[Reranker] Tải model {model_name} ({device}). Lần đầu chạy sẽ kéo model từ mạng xuống...")
        
        # Max length 512 do giới hạn độ dài của chunk_size ~ 1200 chars (Phần 1)
        self.model = CrossEncoder(model_name, max_length=512, device=device)
        print("[Reranker] Mô hình CrossEncoder đã tải thành công và sẵn sàng.")

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Chấm điểm theo kiểu Cross-Attention toàn diện giữa Query và tệp Chunk (Context). 
        Khác với Dual-Encoder (ChromaDB) vốn tính Cosine 2 Vector song song, 
        Cross-Encoder gộp Query và Doc vào cuống của cùng một Neural Network để lấy Context chéo.
        
        Args:
            query (str): Lời hỏi của Y/Bác sĩ.
            candidates (List[Dict]): Kết xuất từ HybridRetriever (RRF) ở bước 3, khoảng 20-40 tài liệu.
            top_n (int): Giới hạn nhặt ra 3-5 tài liệu kim cương nhất làm Input Prompt cho LLM.
            
        Returns:
            List[Dict]: Top N đã sắp xếp lại cực chính xác. Đính kèm score.
        """
        if not candidates:
            return []
            
        # Transform data sang format [ [Query, Doc1], [Query, Doc2],... ]
        # Đây là cấu trúc bắt buộc của package sentence_transformers.CrossEncoder
        cross_inp = [[query, doc.get("chunk_text", "")] for doc in candidates]
        
        # Predict bắn ra tensor array chứa điểm số cho toàn bộ list truyền vào cùng lúc
        scores = self.model.predict(cross_inp)
        
# Ghi đè chỉ số Rerank Score vào lại list candidates gốc
        for idx, score in enumerate(scores):
             # Chuyển kiểu Float32 của Numpy sang Float thường để an toàn serialize ra JSON API
            candidates[idx]["score_cross_encoder"] = float(score)
            
        # Sort desc List các Dict dựa trên value "score_cross_encoder"
        reranked_candidates = sorted(candidates, key=lambda x: x["score_cross_encoder"], reverse=True)
        
        # Cắt gọn trả lại cho LLM
        return reranked_candidates[:top_n]