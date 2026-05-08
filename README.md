# 🧠 AI Content Creator & Auto-Researcher (RAG Agent)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?logo=langchain)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![Qwen](https://img.shields.io/badge/LLM-Qwen2.5-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

This project is an advanced **Retrieval-Augmented Generation (RAG)** application built for the **"Smart Jeevan Shala"** educational initiative. It acts as an intelligent AI assistant that answers user queries using context retrieved from a curated educational knowledge base — instead of relying purely on the LLM's training data.

> 💡 **Why RAG?** Large Language Models hallucinate. RAG grounds responses in your own documents, making answers accurate and traceable.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📄 Multi-format Ingestion | Supports PDF, TXT, and DOCX files |
| 🔍 Semantic Search | FAISS vector store for fast similarity search |
| 🤖 Local LLM | Runs Qwen 2.5 via HuggingFace — no OpenAI key needed |
| 💬 Context-Aware Responses | Answers are grounded in uploaded documents |
| 🏫 Education-Focused | Designed for Smart Jeevan Shala curriculum queries |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Qwen/Qwen2.5-7B (HuggingFace) |
| Framework | LangChain |
| Vector Store | FAISS (faiss-cpu) |
| Embeddings | sentence-transformers |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
AI-Content-Creator-RAG/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── notebooks/
│   └── rag_agent.ipynb          # Main development notebook
├── src/
│   ├── data_loader.py           # PDF/TXT/DOCX ingestion & chunking
│   ├── vectorizer.py            # FAISS index creation & retrieval
│   └── agent.py                 # RAG chain + Streamlit UI
├── data/
│   ├── knowledge_base/          # Your uploaded documents go here
│   └── faiss_index/             # Auto-generated vector index
└── LICENSE
```

---

## ⚙️ Prerequisites

Before you begin, make sure you have:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- A HuggingFace account + API key → [Get it here](https://huggingface.co/settings/tokens)
- Stable internet for first-time model download (~4GB)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/pramodj551-oss/AI-Content-Creator-RAG.git
cd AI-Content-Creator-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain langchain-community langchain-huggingface \
            sentence-transformers faiss-cpu transformers streamlit \
            python-dotenv pypdf docx2txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
HUGGINGFACE_API_KEY=your_hf_api_key_here
MODEL_NAME=Qwen/Qwen2.5-7B
VECTOR_DB_PATH=./data/faiss_index
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 5. Add Your Documents

Place your PDF / TXT / DOCX files inside:

```
data/knowledge_base/
```

### 6. Run the Application

**Option A — Streamlit Chatbot (Recommended):**

```bash
streamlit run src/agent.py
```

**Option B — Jupyter Notebook:**

```bash
jupyter notebook notebooks/rag_agent.ipynb
```

---

## 💡 Example Usage

**User Query:**
```
How does Smart Jeevan Shala help students make better financial decisions?
```

**AI Response:**
```
Based on the uploaded curriculum documents:
Emotional Intelligence is taught alongside financial literacy to help students 
manage stress, avoid impulsive buying, and make long-term financial decisions 
aligned with their goals.

Source: smart_jeevan_shala_module3.pdf — Page 12
```

---

## 🧩 How It Works (Architecture)

```
User Query
    ↓
Embedding Model (sentence-transformers)
    ↓
FAISS Vector Store → Top-K Similar Chunks
    ↓
LangChain RAG Chain
    ↓
Qwen 2.5 LLM
    ↓
Grounded Answer with Source Reference
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `faiss-cpu` install error | Run: `pip install faiss-cpu --no-cache-dir` |
| Model download timeout | Check internet connection — model is ~4GB, cached after first run |
| Out of memory error | Reduce `CHUNK_SIZE` in `.env` or use fewer documents |
| `.env` not loading | Make sure `python-dotenv` is installed and `.env` is in root folder |
| Streamlit port in use | Run: `streamlit run src/agent.py --server.port 8502` |

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Pramod** — IIT Patna Applied AI & ML Program  
GitHub: [@pramodj551-oss](https://github.com/pramodj551-oss)  
For issues, please [open a GitHub Issue](https://github.com/pramodj551-oss/AI-Content-Creator-RAG/issues).

---

> ⭐ If this project helped you, please give it a star!
