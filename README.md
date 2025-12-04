# 🤖 Local RAG with Semantic Chunking (Expert Edition)

Dự án **RAG (Retrieval-Augmented Generation)** này được thiết kế để chạy cục bộ (Local) sử dụng các kỹ thuật Deep Learning tiên tiến. Điểm khác biệt chính của dự án là việc áp dụng **Semantic Chunking** thay vì cắt file theo kích thước cố định, giúp AI hiểu ngữ cảnh tốt hơn khi truy vấn.

Hệ thống sử dụng mô hình ngôn ngữ lớn (LLM) được lượng tử hóa (Quantized 4-bit) và mô hình Embedding tiếng Việt chuyên dụng.

App Web: 

## 📂 Cấu trúc dự án

```text
My_Unique_RAG/
├── data/                  # Thư mục chứa file PDF đầu vào
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Cấu hình Hyperparameters (Model ID, Chunking thresholds)
│   ├── model_loader.py    # Quản lý load LLM (BitsAndBytes) & Embeddings
│   ├── vector_db.py       # Xử lý PDF & Semantic Chunking logic
│   └── utils.py           # Các tiện ích bổ trợ
├── app.py                 # Giao diện chính (Streamlit)
├── requirements.txt       # Danh sách thư viện cần thiết
└── README.md              # Tài liệu hướng dẫn
````

## 🚀 Tính năng nổi bật

  * **🧠 Semantic Chunking (Phân đoạn theo ngữ nghĩa):**
      * Sử dụng `LangChain Experimental SemanticChunker`.
      * Không cắt văn bản máy móc theo ký tự. Hệ thống phân tích sự thay đổi về ngữ nghĩa (cosine similarity) giữa các câu để quyết định điểm ngắt (breakpoint).
      * Cấu hình: Dựa trên ngưỡng phân vị (Percentile Threshold) để đảm bảo các đoạn văn giữ trọn vẹn ý nghĩa.
  * **⚡ Optimized Local LLM:**
      * Sử dụng model `lmsys/vicuna-7b-v1.5`.
      * Tối ưu hóa bộ nhớ với **4-bit Quantization (NF4)** sử dụng thư viện `bitsandbytes`, cho phép chạy trên GPU có VRAM khiêm tốn (Consumer GPU).
  * **🇻🇳 Vietnamese Embedding:**
      * Tích hợp model `bkai-foundation-models/vietnamese-bi-encoder` để tối ưu hóa khả năng tìm kiếm văn bản tiếng Việt.
  * **💬 Conversational Memory:**
      * Hỗ trợ nhớ ngữ cảnh hội thoại cũ, giúp hỏi đáp tự nhiên hơn.

## 🛠 Yêu cầu hệ thống

  * **OS:** Linux (Ubuntu) hoặc Windows (WSL2 recommended).
  * **Python:** 3.10+
  * **GPU:** NVIDIA GPU (VRAM \>= 6GB recommended) để chạy 4-bit quantization.
  * **CUDA:** Đã cài đặt CUDA Toolkit tương thích với PyTorch.

## ⚙️ Cài đặt

1.  **Clone dự án:**

    ```bash
    git clone https://github.com/lampt224321/rag_chatbot_basic.git
    cd rag_chatbot_basic
    ```

2.  **Tạo môi trường ảo:**

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Cài đặt thư viện:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Lưu ý: Nếu dùng Windows, bạn có thể cần cài `bitsandbytes-windows` nếu gặp lỗi về thư viện bnb).*

## 📖 Hướng dẫn sử dụng

1.  **Khởi chạy ứng dụng:**
    Từ thư mục gốc `My_Unique_RAG`, chạy lệnh:

    ```bash
    streamlit run app.py
    ```

2.  **Sử dụng trên giao diện:**

      * Chờ hệ thống load model (Vicuna & Embedding) lần đầu tiên (sẽ tốn vài phút tùy tốc độ mạng).
      * Tải lên file PDF ở thanh bên trái (Sidebar).
      * Nhấn **"🚀 Xử lý tài liệu"**. Hệ thống sẽ thực hiện *Semantic Chunking* và tạo index vào ChromaDB.
      * Bắt đầu chat với tài liệu của bạn.

## 🔧 Cấu hình nâng cao (Config)

Bạn có thể tinh chỉnh các tham số trong `src/config.py`:

| Tham số | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- |
| `CHUNK_BREAKPOINT_TYPE` | "percentile" | Cách tính điểm ngắt đoạn (theo phần trăm sự khác biệt). |
| `CHUNK_BREAKPOINT_AMOUNT`| 95 | Ngưỡng tương đồng (%). Nếu 2 câu khác nhau \> 5%, sẽ tách đoạn. |
| `MIN_CHUNK_SIZE` | 500 | Kích thước tối thiểu của một đoạn văn bản. |
| `MAX_NEW_TOKENS` | 512 | Độ dài tối đa câu trả lời của AI. |
| `TEMPERATURE` | 0.2 | Độ sáng tạo của AI (thấp để chính xác hơn). |

## 🤝 Đóng góp

Dự án được xây dựng cho mục đích nghiên cứu Deep Learning và RAG. Mọi đóng góp (Pull Request) để cải thiện thuật toán Chunking hoặc thay thế Model đều được hoan nghênh.

-----

*Deep Learning Expert Edition - 2025*

## LICENSE
Distributed under the MIT License. See LICENSE.txt for more information.

Copyright (c) 2025 Pham Tung Lam




