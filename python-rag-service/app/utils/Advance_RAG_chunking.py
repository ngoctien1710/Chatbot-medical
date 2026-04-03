import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MedicalDocumentChunker:
    """
    Singleton Class cho việc cắt (chunking) tài liệu Y khoa Tiếng Việt.
    Sử dụng Singleton pattern để tối ưu hóa bộ nhớ, tránh khởi tạo lại nhiều lần Splitter.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MedicalDocumentChunker, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, chunk_size: int = 1200, chunk_overlap: int = 250):
        """
        Khởi tạo Langchain's RecursiveCharacterTextSplitter.
        
        Cấu hình cho y khoa (thường là Markdown):
        - chunk_size = 1200: Đủ rộng để bao hàm trọn vẹn một bước trong phác đồ hoặc một triệu chứng chi tiết.
        - chunk_overlap = 250: Tránh mất ngữ cảnh giữa các đoạn (đặc biệt hữu ích khi câu trước đề cập đến thuốc, câu sau nối tiếp liều dùng).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Thứ tự separators được tinh chỉnh để:
        # 1. Bảo toàn cấu trúc Markdown (Header).
        # 2. Cắt theo đoạn văn.
        # 3. Cắt theo danh sách (list) - rất hay gặp trong phác đồ y khoa.
        # 4. Cắt theo dấu câu Tiếng Việt (chấm, hỏi, than) để KHÔNG cắt ngang câu.
        separators = [
            "\n\n# ", "\n\n## ", "\n\n### ", "\n\n#### ",  # Headers (Markdown)
            "\n\n",                                        # Paragraphs
            "\n- ", "\n* ", "\n[0-9]+\. ",                 # Lists (mặc định regex không hỗ trợ tốt ở đây nếu is_separator_regex=False, nhưng LangChain hỗ trợ pattern regex qua tham số nếu bật, tuy nhiên ta ưu tiên text thường)
            "\n",                                          # Line breaks
            ". ", "? ", "! ",                              # End of Sentences
            " ",                                           # Words
            ""                                             # Characters (Fallback cuối)
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )

    def clean_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản: Xóa các khoảng trắng thừa thường xuất hiện khi parse tài liệu.
        """
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def process_markdown(self, text: str, initial_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Cắt nội dung text Markdown thành các chunks.
        
        Args:
            text (str): Văn bản Markdown đầu vào.
            initial_metadata (Dict, optional): Metadata ban đầu (e.g. source_file, title).
            
        Returns:
            List[Dict]: Danh sách các chunk, mỗi dict chứa "content" và "metadata".
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []

        # create_documents là hàm tiện ích của LangChain
        docs = self.splitter.create_documents(
            texts=[cleaned_text], 
            metadatas=[initial_metadata or {}]
        )
        
        result_chunks = []
        for i, doc in enumerate(docs):
            # Cập nhật và tinh chỉnh lại metadata cho từng chunk nhỏ
            meta = doc.metadata.copy()
            meta["chunk_index"] = i
            meta["chunk_length"] = len(doc.page_content)
            
            result_chunks.append({
                "content": doc.page_content,
                "metadata": meta
            })
            
        return result_chunks

# ====== Example Usage ======
# if __name__ == "__main__":
#     chunker1 = MedicalDocumentChunker()
#     chunker2 = MedicalDocumentChunker()
#     print("Is Singleton:", chunker1 is chunker2) # True
#     
#     sample_md = "# Bệnh học\n\n- Triệu chứng 1\n- Triệu chứng 2\n\n## Điều trị\n- Thuốc A: 500mg.\n- Thuốc B: 200mg uống sau ăn."
#     meta = {"source": "Sổ tay nội khoa", "author": "Khoa Hồi Sức"}
#     
#     chunks = chunker1.process_markdown(sample_md, meta)
#     for c in chunks:
#         print("\n--- CHUNK IDX:", c["metadata"]["chunk_index"], "---")
#         print(c["content"])
