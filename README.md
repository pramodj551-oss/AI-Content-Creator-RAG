# 🧠 AI Content Creator & Auto-Researcher (RAG Agent)

## 🎯 Project Overview
This project is an advanced **Retrieval-Augmented Generation (RAG)** application designed for the **"Smart Jeevan Shala"** educational initiative. It functions as an intelligent AI assistant that answers specific queries about financial literacy and student development by retrieving facts from a private knowledge base. This ensures that the AI provides accurate information without "hallucinating" or making up facts.

## 🛠️ Tech Stack & Concepts
* **Framework:** LangChain (LCEL - LangChain Expression Language)
* **LLM:** Google Flan-T5 / Qwen 2.5 (Open-source models via Hugging Face)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Web Interface:** Streamlit (For Interactive Chatbot)
* **Key Concepts:** RAG, Document Chunking, Semantic Search, Prompt Engineering.

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

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pramodj551-oss/AI-Content-Creator-RAG.git](https://github.com/pramodj551-oss/AI-Content-Creator-RAG.git)
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
