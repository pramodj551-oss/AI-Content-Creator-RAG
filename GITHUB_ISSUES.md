# 📋 GitHub Issues — AI-Content-Creator-RAG

Copy-paste each issue below directly into GitHub → Issues → New Issue

---

## 🐛 Bug #1 — Flan-T5 Pipeline Error in Cell 0

**Title:** `T5ForConditionalGeneration` not supported for `text-generation` pipeline

**Description:**
Cell 0 attempts to load Flan-T5 with `pipeline("text-generation", ...)` but T5 is a Seq2Seq model that requires `text2text-generation` task type. This causes a runtime warning and pipeline fallback.

**Steps to Reproduce:**
1. Run Cell 0 of `AI_Content_Creator_RAG.ipynb`
2. Observe: `T5ForConditionalGeneration is not supported for text-generation`

**Expected:** Clean model load with no warnings
**Actual:** Warning + unexpected model behavior

**Fix:**
```python
# Change from:
pipe = pipeline("text-generation", model="google/flan-t5-base", ...)

# Change to:
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=150)
```

**Labels:** `bug`, `good first issue`

---

## 🐛 Bug #2 — `max_new_tokens` vs `max_length` Conflict

**Title:** Conflicting generation parameters cause warning

**Description:**
Both `max_new_tokens=150` and `max_length=20` are set simultaneously, causing HuggingFace to warn that `max_new_tokens` will take precedence. The `max_length=20` is likely a default being overridden.

**Fix:**
```python
# Set ONLY max_new_tokens, remove max_length
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=150)
```

**Labels:** `bug`, `good first issue`

---

## ✨ Feature #1 — Add PDF and DOCX Document Support

**Title:** Support PDF and DOCX files as knowledge base input

**Description:**
Currently the knowledge base is hardcoded as a Python string in Cell 1. Users should be able to upload real PDF/DOCX files as input.

**Proposed Solution:**
```python
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def load_document(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    else:
        return TextLoader(file_path).load()
```

**Labels:** `enhancement`, `help wanted`

---

## ✨ Feature #2 — Persist FAISS Index to Disk

**Title:** Save and reload FAISS vector store to avoid rebuilding on every run

**Description:**
Currently the FAISS index is rebuilt from scratch every time the notebook runs. For large knowledge bases this is slow.

**Proposed Solution:**
```python
# Save index
vector_store.save_local("faiss_index/")

# Load index (skip rebuilding)
from langchain_community.vectorstores import FAISS
vector_store = FAISS.load_local("faiss_index/", embeddings, 
                                 allow_dangerous_deserialization=True)
```

**Labels:** `enhancement`, `performance`

---

## ✨ Feature #3 — Add Streamlit UI for Interactive Q&A

**Title:** Build a Streamlit web interface for the RAG chatbot

**Description:**
Currently users must run a Jupyter notebook. A Streamlit app would allow non-technical users to interact with the RAG system through a browser.

**Proposed UI:**
- File upload widget for knowledge base documents
- Chat interface for questions
- Display source chunks that were retrieved

**Labels:** `enhancement`, `UI`

---

## ✨ Feature #4 — Add Source Citation in Answers

**Title:** Show which knowledge base chunk answered the question

**Description:**
Answers currently have no traceability. Users should see which part of the document was used to generate the answer.

**Proposed Solution:**
```python
# Return source documents alongside answer
from langchain.chains import RetrievalQAWithSourcesChain
```

**Labels:** `enhancement`, `transparency`
