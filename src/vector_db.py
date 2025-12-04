# Đây là phần cốt lõi xử lý PDF và Semantic Chunking

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from src.config import Config

class VectorDBManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.splitter = SemanticChunker(
            embeddings=embedding_model,
            buffer_size=Config.CHUNK_BUFFER_SIZE,
            breakpoint_threshold_type=Config.CHUNK_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=Config.CHUNK_BREAKPOINT_AMOUNT,
            min_chunk_size=Config.MIN_CHUNK_SIZE,
            add_start_index=True
        ) 

    """  
         Semantic Chunking giúp chia nhỏ văn bản thành các phần có ý
         nghĩa, giúp tăng tính chính xác khi thực hiện truy vấn. Các tham
         số trong SemanticChunker giúp điều chỉnh cách thức và cách tách
         các đoạn văn bản này:

            • buffer_size=1: Xác định số câu cần gom lại trước khi tách. Ở
            đây, giá trị buffer_size=1 có nghĩa là mỗi nhóm bao gồm một
            câu, giúp mỗi chunk chứa các câu riêng biệt.

            • breakpoint_threshold_type="percentile": Tham số này chỉ định
            cách tính điểm phân đoạn. Trong trường hợp này, percentile
            có nghĩa là dựa trên tỷ lệ phần trăm của sự khác biệt về ngữ
            nghĩa giữa các đoạn văn. Tức là, nếu độ tương đồng của hai
            đoạn văn thấp hơn ngưỡng nhất định, chúng sẽ được tách ra
            làm các chunk riêng biệt.

            • breakpoint_threshold_amount=95: Đây là ngưỡng phần trăm dùng
            để xác định khi nào nên cắt các đoạn văn bản. Giá trị 95 có
            nghĩa là nếu độ tương đồng giữa hai đoạn văn thấp hơn 95%,
            chúng sẽ được tách rời, tạo thành các chunk mới. Điều này
            đảm bảo rằng các đoạn văn bản không bị cắt ngắt một cách
            ngẫu nhiên mà vẫn giữ được sự liên kết ngữ nghĩa.

            • min_chunk_size=500: Thiết lập kích thước tối thiểu của mỗi
            chunk. Khi giá trị này được đặt thành 500, mỗi đoạn văn bản 
    """

    def process_file(self, file_path):
        # 1. Load PDF 
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 2. Semantic Split 
        docs = self.splitter.split_documents(documents)
        print(f"Split into {len(docs)} semantic chunks.")
        
        # 3. Create Vector DB (Chroma) 
        # Sử dụng from_documents để tạo DB in-memory cho session hiện tại
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model
        )
        return vector_db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks