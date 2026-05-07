import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Smart Jeevan Shala AI", page_icon="🤖")
st.title("🤖 Smart Jeevan Shala: RAG AI Assistant")

@st.cache_resource
def initialize_rag():
    content = """
    Smart Jeevan Shala focuses on the holistic development of students.
    One of the core modules is Financial Literacy for teenagers.
    In the Financial Literacy module, students learn about saving money, basic banking, and the power of compounding.
    A good budget follows the 50-30-20 rule: 50% for needs, 30% for wants, and 20% for savings.
    Emotional Intelligence helps students manage stress and make better financial decisions.
    """
    with open("kb.txt", "w") as f:
        f.write(content)

    loader = TextLoader("kb.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Updated Model Loading logic
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Task updated to "text-generation" to match your system's capabilities
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    template = """Answer based ONLY on context:
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

try:
    with st.spinner("Initializing System..."):
        rag_chain = initialize_rag()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about the curriculum..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = rag_chain.invoke(user_input)
        final_answer = response.split("Answer:")[-1].strip()
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
