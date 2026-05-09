# 🧠 AI Content Creator & Auto-Researcher (RAG Agent)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-orange)
![HuggingFace](https://img.shields.io/badge/LLM-Qwen2.5--0.5B-purple?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built for the **"Smart Jeevan Shala"** educational initiative. The system answers user queries about financial literacy and student development by retrieving context from a curated knowledge base — instead of hallucinating from LLM training data alone.

> 💡 **Why RAG?** LLMs hallucinate. RAG grounds every answer in your own documents, making responses accurate and traceable.

---

## ✨ Features

| Feature | Detail |
|--------|--------|
| 📄 Knowledge Base | Custom text ingestion via LangChain `TextLoader` |
| ✂️ Smart Chunking | `RecursiveCharacterTextSplitter` — chunk size 150, overlap 30 |
| 🔍 Semantic Retrieval | FAISS vector store with `all-MiniLM-L6-v2` embeddings, top-k=2 |
| 🤖 Local LLM | `Qwen/Qwen2.5-0.5B-Instruct` via HuggingFace (runs locally, no API key) |
| 🔗 Modern Pipeline | Built with LCEL (LangChain Expression Language) |
| 🛡️ Grounded Answers | Prompt-engineered to say "I don't know" if answer isn't in context |

---

## 🛠️ Tech Stack

| Layer | Library / Model |
|-------|----------------|
| Framework | LangChain (`langchain-core`, `langchain-community`, `langchain-huggingface`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` (HuggingFace Transformers) |
| Text Splitting | `langchain-text-splitters` |
| Language | Python 3.10+ |

---

## 🧩 RAG Pipeline Architecture

```
Knowledge Base (TXT)
        ↓
TextLoader → RecursiveCharacterTextSplitter
        ↓
   7 Text Chunks (size=150, overlap=30)
        ↓
all-MiniLM-L6-v2 Embeddings → FAISS Vector Store
        ↓
User Query → Retriever (top-k=2 similar chunks)
        ↓
PromptTemplate (context + question)
        ↓
Qwen2.5-0.5B-Instruct (HuggingFacePipeline)
        ↓
StrOutputParser → Grounded Answer
```

---

## 📁 Project Structure

```
AI-Content-Creator-RAG/
├── AI_Content_Creator_RAG.ipynb   # Complete RAG pipeline (single notebook)
├── knowledge_base.txt             # Auto-generated from notebook (Step 1)
├── requirements.txt               # Dependencies
├── .gitignore
└── README.md
```

> 📌 The notebook is self-contained — it creates `knowledge_base.txt` automatically in Step 1.

---

## 📚 Knowledge Base — Smart Jeevan Shala Topics

The current knowledge base covers:

- 🏦 **Financial Literacy** — saving money, basic banking, power of compounding
- 📊 **Budgeting** — the 50-30-20 rule (needs / wants / savings)
- 🧠 **Emotional Intelligence** — managing stress, avoiding impulsive buying
- 💰 **Compounding** — earning interest on interest for long-term wealth

> To add more topics, simply extend the `knowledge_base_content` string in **Cell 1** of the notebook.

---

## ⚙️ Prerequisites

- Python 3.10+
- pip
- ~2GB disk space (for Qwen model download on first run)
- Internet connection (first-time model download only; cached locally after)

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
            langchain-text-splitters langchain-core \
            sentence-transformers faiss-cpu transformers
```

### 4. Run the Notebook

```bash
jupyter notebook AI_Content_Creator_RAG.ipynb
```

Run cells in order:
- **Cell 1** → Creates knowledge base + text chunks
- **Cell 2** → Builds FAISS vector store
- **Cell 3** → Loads Qwen2.5 LLM
- **Cell 4** → Runs RAG pipeline with your question

---

## 💡 Example Q&A

**User Question:**
```
How does Smart Jeevan Shala help students make better financial decisions?
```

**AI Agent Response:**
```
By teaching emotional intelligence. Smart Jeevan Shala teaches Emotional 
Intelligence alongside Financial Literacy to help students manage stress, 
avoid impulsive buying, and make better long-term financial decisions.
```

> ✅ Answer is grounded in `knowledge_base.txt` — not hallucinated.

---

## 🔧 Customization

### Change the Question
In **Cell 4**, update:
```python
question = "Your custom question here"
```

### Extend the Knowledge Base
In **Cell 1**, add more content to:
```python
knowledge_base_content = """
... existing content ...
Your new educational content here.
"""
```

### Tune Retrieval
In **Cell 2**, change `k` for more/fewer retrieved chunks:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # default is 2
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `faiss-cpu` install fails | Run: `pip install faiss-cpu --no-cache-dir` |
| Qwen model download timeout | Stable internet needed; ~2GB download; cached after first run |
| `T5ForConditionalGeneration not supported` warning | Use `text-generation` pipeline, not `text2text-generation` |
| `max_new_tokens` vs `max_length` warning | Set only `max_new_tokens=150`, remove `max_length` |
| Answer includes prompt text | Refine `StrOutputParser` or post-process with `.split("Answer:")[-1]` |
| Out of memory | Switch to smaller model or reduce `max_new_tokens` |

---

## 🚧 Known Limitations & Future Improvements

- [ ] Currently knowledge base is **hardcoded text** — add PDF/DOCX file upload support
- [ ] Add **Streamlit UI** for interactive Q&A without running notebook
- [ ] Replace Qwen 0.5B with larger model (Qwen2.5-7B) for better answers
- [ ] Add **conversation memory** for multi-turn dialogue
- [ ] Persist FAISS index to disk so it doesn't rebuild on every run
- [ ] Add source citation (which chunk answered the question)

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch: `git checkout -b feature/add-pdf-support`
3. Commit: `git commit -m 'Add PDF document ingestion'`
4. Push: `git push origin feature/add-pdf-support`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Suchita** — IIT Patna Applied AI & ML Program  
GitHub: [@pramodj551-oss](https://github.com/pramodj551-oss)  
For issues: [Open a GitHub Issue](https://github.com/pramodj551-oss/AI-Content-Creator-RAG/issues)

---

> ⭐ If this project helped you, please give it a star!
