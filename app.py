import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Smart Jeevan Shala AI Agent", page_icon="🧠")
st.title("🧠 Smart Jeevan Shala: AI RAG Agent")
st.markdown("Ask anything about Financial Literacy and Student Development.")

# --- Load Resources (Cached for performance) ---
@st.cache_resource
def load_rag_system():
    # 1. Load Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if vector store exists, otherwise create or load it
    # Note: Ensure 'knowledge_base.txt' is in the same directory
    return embeddings

# --- UI Logic ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # This is where your actual RAG chain logic will integrate
            # Integration example: response = rag_chain.invoke(prompt)
            # For now, providing a placeholder for the logic we built
            response = "The AI Agent will analyze the knowledge base and provide an answer here." 
            
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
