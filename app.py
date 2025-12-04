# app.py
import streamlit as st
import tempfile
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.config import Config
from src.model_loader import ModelManager
from src.vector_db import VectorDBManager

# --- Page Config ---
st.set_page_config(page_title="Personal AI Expert RAG", layout="wide", page_icon="🤖")

# --- CSS Tùy biến ---
st.markdown("""
<style>
    .chat-message {padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex}
    .chat-message.user {background-color: #2b313e}
    .chat-message.bot {background-color: #475063}
    .source-box {font-size: 0.8em; color: #aaa; margin-top: 5px; border-top: 1px solid #555; padding-top: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Trợ lý AI Đọc Hiểu Tài Liệu (Deep Learning Expert Edition)")

# --- Session State Initialization ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("⚙️ Cấu hình & Dữ liệu")
    
    # 1. Load Models (Chỉ load 1 lần)
    if not st.session_state.models_loaded:
        with st.spinner("Đang khởi tạo AI Brain (LLM & Embeddings)..."):
            try:
                embeddings = ModelManager.load_embeddings()
                llm = ModelManager.load_llm()
                st.session_state.embeddings = embeddings
                st.session_state.llm = llm
                st.session_state.models_loaded = True
                st.success("AI đã sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi khởi tạo: {e}")
    else:
        st.success("✅ AI Core Active")

    # 2. Upload File 
    uploaded_file = st.file_uploader("Upload tài liệu PDF", type="pdf")
    
    process_btn = st.button("🚀 Xử lý tài liệu")

# --- Main Logic: Xử lý PDF ---
if process_btn and uploaded_file and st.session_state.models_loaded:
    with st.spinner("Đang phân tích ngữ nghĩa (Semantic Chunking)..."):
        # Lưu file tạm 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Khởi tạo VectorDB Manager
            vector_manager = VectorDBManager(st.session_state.embeddings)
            retriever = vector_manager.process_file(tmp_file_path)
            
            # Dùng ConversationalRetrievalChain thay vì chain đơn giản
            # Giúp bot nhớ được ngữ cảnh (Memory)
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer' # Quan trọng để chain biết đâu là output
            )

            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True, # CẢI TIẾN: Trả về nguồn
                verbose=True
            )
            
            st.success(f"Đã xử lý xong! Sẵn sàng hỏi đáp.")
        except Exception as e:
            st.error(f"Lỗi xử lý: {e}")
        finally:
            os.unlink(tmp_file_path) # Dọn dẹp file tạm

# --- Main Logic: Chat Interface [cite: 673] ---
st.subheader("💬 Hội thoại")

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)
        if "sources" in message and message["sources"]:
            with st.expander("📚 Nguồn tham khảo"):
                for src in message["sources"]:
                    st.markdown(f"- Trang {src['page']}: *{src['content'][:100]}...*")

# Input câu hỏi mới
if user_question := st.chat_input("Đặt câu hỏi về tài liệu của bạn..."):
    if not st.session_state.conversation:
        st.error("Vui lòng upload và xử lý tài liệu trước!")
    else:
        # Hiển thị câu hỏi user
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Xử lý câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("AI đang suy nghĩ..."):
                response = st.session_state.conversation.invoke({"question": user_question})
                answer = response['answer']
                
                # Trích xuất nguồn (Source Documents)
                source_docs = response['source_documents']
                sources_display = []
                for doc in source_docs:
                    sources_display.append({
                        "page": doc.metadata.get('page', 'N/A') + 1, # Page index starts at 0
                        "content": doc.page_content
                    })

                # Hiển thị câu trả lời (làm sạch từ khóa Answer: nếu có) 
                clean_answer = answer.split("Answer:")[-1].strip() if "Answer:" in answer else answer
                st.markdown(clean_answer)
                
                # Hiển thị nguồn
                with st.expander("📚 Nguồn tham khảo (Semantic Chunks)"):
                    for src in sources_display:
                        st.markdown(f"- **Trang {src['page']}**: {src['content'][:150]}...")
                
                # Lưu vào history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": clean_answer,
                    "sources": sources_display
                })