
---

# 🤖 Local RAG with Semantic Chunking (Expert Edition)

This project implements a **local Retrieval-Augmented Generation (RAG)** system powered by advanced **Deep Learning** techniques.
The key differentiation lies in the use of **Semantic Chunking** instead of traditional fixed-size text splitting, enabling the AI system to preserve semantic coherence and achieve significantly better contextual understanding during retrieval.

The system runs entirely **locally**, leveraging a **4-bit quantized Large Language Model (LLM)** and a **Vietnamese-specialized embedding model**.
Linux / Ubuntu is strongly recommended for optimal performance.

**Web App:** *(Streamlit-based interface)*

---

## 📂 Project Structure

```text
RAG/
├── data/                  # Input PDF documents
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Hyperparameter configuration (Model IDs, chunking thresholds)
│   ├── model_loader.py    # LLM & embedding loader (BitsAndBytes integration)
│   ├── vector_db.py       # PDF processing & semantic chunking logic
│   └── utils.py           # Utility functions
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🚀 Key Features

### 🧠 Semantic Chunking (Meaning-Aware Text Segmentation)

* Built on **`LangChain Experimental SemanticChunker`**.
* Text is **not split mechanically** by character or token count.
* The system analyzes **semantic shifts between sentences** using **cosine similarity** to determine chunk boundaries.
* Chunk breakpoints are computed using **percentile-based thresholds**, ensuring each chunk preserves complete semantic intent.

### ⚡ Optimized Local LLM Inference

* Uses **`lmsys/vicuna-7b-v1.5`**.
* Memory-efficient **4-bit quantization (NF4)** via `bitsandbytes`.
* Enables local inference on **consumer-grade GPUs with limited VRAM**.

### 🇻🇳 Vietnamese-Specific Embeddings

* Integrates **`bkai-foundation-models/vietnamese-bi-encoder`**.
* Significantly improves semantic retrieval quality for **Vietnamese-language documents**.

### 💬 Conversational Memory

* Maintains multi-turn conversation history.
* Allows natural, context-aware follow-up questions over retrieved documents.

---

## 🛠 System Requirements

* **Operating System:** Linux (Ubuntu preferred) or Windows (WSL2 recommended)
* **Python:** 3.10+
* **GPU:** NVIDIA GPU (≥ 6GB VRAM recommended)
* **CUDA:** CUDA Toolkit compatible with PyTorch

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lampt224321/rag_chatbot_basic.git
cd rag_chatbot_basic
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Windows, you may need to install `bitsandbytes-windows` if you encounter bnb-related issues.

---

## 📖 Usage Guide

### 1️⃣ Launch the Application

From the project root directory, run:

```bash
streamlit run app.py
```

### 2️⃣ Interact via Web Interface

* Wait for the initial model loading (Vicuna + embeddings). This may take several minutes depending on network speed.
* Upload PDF documents via the left sidebar.
* Click **“🚀 Process Documents”** to:

  * Perform **Semantic Chunking**
  * Index embeddings into **ChromaDB**
* Start chatting with your documents in a conversational manner.

---

## 🔧 Advanced Configuration

Fine-tune system behavior via `src/config.py`:

| Parameter                 | Default Value  | Description                                                |
| ------------------------- | -------------- | ---------------------------------------------------------- |
| `CHUNK_BREAKPOINT_TYPE`   | `"percentile"` | Breakpoint calculation strategy                            |
| `CHUNK_BREAKPOINT_AMOUNT` | `95`           | Similarity percentile threshold (split if difference > 5%) |
| `MIN_CHUNK_SIZE`          | `500`          | Minimum chunk length                                       |
| `MAX_NEW_TOKENS`          | `512`          | Maximum generated response length                          |
| `TEMPERATURE`             | `0.2`          | Controls randomness (lower = more factual)                 |

---

## 🤝 Contributions

This project is designed for **Deep Learning and RAG research purposes**.
Contributions are highly welcome — including:

* Improved **semantic chunking strategies**
* Alternative **LLMs or embedding models**
* Retrieval optimization techniques

Feel free to open a Pull Request 🚀

---

*Deep Learning – Expert Edition (2025)*

---

## 📄 License

Distributed under the **MIT License**.
See `LICENSE.txt` for details.

**Copyright © 2025 Pham Tung Lam**

---


