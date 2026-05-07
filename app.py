import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Configuration ---
st.set_page_config(page_title="Smart Jeevan Shala AI Agent", page_icon="🤖")
st.title("🤖 Smart Jeevan Shala: RAG AI Assistant")
st.markdown("Retrieval-Augmented Generation system for Educational Content.")

# --- Initialize RAG System ---
@st.cache_resource
def initialize_rag():
    # 1. Create Knowledge Base
    content = """
    Smart Jeevan Shala focuses on the holistic development of students.
    One of the core modules is Financial Literacy for teenagers.
    In the Financial Literacy module, students learn about saving money, basic banking, and the power of compounding.
    A good budget follows the 50-30-20 rule: 50% for needs, 30% for wants, and 20% for savings.
    Emotional Intelligence helps students manage stress and make better financial decisions.
    """
    with open("kb.txt", "w") as f:
        f.write(content)

    # 2. Load and Split Documents
    loader = TextLoader("kb.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # 3. Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 4. Load LLM
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=pipe)

    # 5. Build RAG Chain
    template = """Use the following pieces of context to answer the question.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Load System ---
with st.spinner("Initializing AI Agent... Please wait."):
    rag_chain = initialize_rag()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about Smart Jeevan Shala curriculum..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Base..."):
            response = rag_chain.invoke(user_input)
            clean_response = response.split("Answer:")[-1].strip()
            st.markdown(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

# --- Sidebar ---
st.sidebar.title("System Info")
st.sidebar.info("This AI uses RAG to prevent hallucinations by only answering from the provided dataset.")
