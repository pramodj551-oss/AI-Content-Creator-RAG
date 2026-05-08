# 🧠 AI Content Creator & Auto-Researcher (RAG Agent)

## 🎯 Project Overview
This project is an advanced **Retrieval-Augmented Generation (RAG)** application designed for the **"Smart Jeevan Shala"** educational initiative. It functions as an intelligent AI assistant that answers user queries using context from a curated educational knowledge base.

## 🛠️ Tech Stack & Concepts
* **Framework:** LangChain (LCEL - LangChain Expression Language)
* **LLM:** Google Flan-T5 / Qwen 2.5 (Open-source models via Hugging Face)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Web Interface:** Streamlit (For Interactive Chatbot)
* **Key Concepts:** RAG, Document Chunking, Semantic Search, Prompt Engineering

## 📋 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum
- Internet connection (for downloading models)

## ⚙️ How It Works
1. **Data Ingestion:** Custom educational content is loaded and split into optimized text chunks.
2. **Vectorization:** These chunks are converted into numerical vectors and stored in the **FAISS** database.
3. **Retrieval:** When a user asks a question, the system searches the database for the most relevant information.
4. **Augmented Generation:** The retrieved context is fed into the LLM along with the user's question to generate a precise answer.

## 🤖 Interactive Chatbot
I have integrated a modern web-based chat interface using **Streamlit**.

**Features:**
* **Context-Aware Responses:** The chatbot only answers based on the "Smart Jeevan Shala" dataset.
* **User-Friendly UI:** A clean, ChatGPT-like chat interface for easy interaction.
* **Real-time Processing:** Shows the agent's "thinking" process while searching the knowledge base.

## 📁 Project Structure
