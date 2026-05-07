# 🧠 AI Content Creator & Auto-Researcher (RAG Agent)

## 🎯 Project Overview
This project is a local **Retrieval-Augmented Generation (RAG)** application built to act as an AI Assistant for **"Smart Jeevan Shala"** (an educational initiative). It accurately answers context-specific questions regarding student development and financial literacy by retrieving information from a custom knowledge base, ensuring zero hallucinations.

## 🛠️ Tech Stack & Concepts Used
* **Framework:** LangChain (LCEL - LangChain Expression Language)
* **LLM:** Qwen/Qwen2.5-0.5B-Instruct (via Hugging Face Pipeline)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Vector Store / Database:** FAISS (Facebook AI Similarity Search)
* **Key Concepts:** Retrieval-Augmented Generation (RAG), Document Chunking, Prompt Engineering, Text-Generation.

## ⚙️ Architecture & Workflow
1. **Knowledge Ingestion:** Loaded custom text about the "Smart Jeevan Shala" curriculum (Financial Literacy, 50-30-20 rule, etc.) and split it into optimized chunks using `RecursiveCharacterTextSplitter`.
2. **Vectorization:** Converted the text chunks into dense vectors and stored them in a **FAISS** vector database for rapid semantic search.
3. **Retrieval:** Configured a retriever to fetch the top 2 most relevant document chunks based on the user's query.
4. **Generation:** Passed the retrieved context and the user query to the **Qwen 2.5** LLM using a strict prompt template to generate an accurate, human-like response.

## 🚀 How to Run
1. Clone the repository: `git clone https://github.com/pramodj551-oss/AI-Content-Creator-RAG.git`
2. Install dependencies: `pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu transformers`
3. Run the Jupyter Notebook to build the vector store and interact with the AI Agent.

## 💡 Example Output
**User Query:** *How does Smart Jeevan Shala help students make better financial decisions?*
**AI Response:** *Emotional Intelligence is taught alongside to help students manage stress, avoid impulsive buying, and make better financial decisions.*
