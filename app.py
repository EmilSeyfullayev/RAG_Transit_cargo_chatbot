import os
import glob
import pickle

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from pypdf import PdfReader
import docx2txt


# =========================
# Config
# =========================
DOCS_DIR = "documents"
INDEX_DIR = ".rag_index"
METADATA_PATH = ".rag_index/meta.pkl"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# =========================
# Utils
# =========================
def list_doc_files(doc_dir):
    pdfs = glob.glob(os.path.join(doc_dir, "**/*.pdf"), recursive=True)
    docxs = glob.glob(os.path.join(doc_dir, "**/*.docx"), recursive=True)
    return sorted(pdfs + docxs)


def load_pdf(path):
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            texts.append(txt)
    return "\n".join(texts)


def load_docx(path):
    text = docx2txt.process(path) or ""
    return text


def ingest_documents(doc_dir):
    files = list_doc_files(doc_dir)
    docs = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == ".pdf":
                text = load_pdf(f)
            elif ext == ".docx":
                text = load_docx(f)
            else:
                continue
        except Exception as e:
            st.warning(f"Skipping {f}: {e}")
            continue

        if not text.strip():
            continue

        docs.append(Document(page_content=text, metadata={"source": f}))
    return docs


def doc_fingerprint(doc_dir):
    files = list_doc_files(doc_dir)
    sig_parts = []
    for f in files:
        try:
            mtime = os.path.getmtime(f)
        except OSError:
            mtime = 0
        sig_parts.append(f"{f}:{mtime}")
    return "|".join(sig_parts)


def save_meta(meta):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(meta, f)


def load_meta():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    return {}


# =========================
# Cache: Build or Load FAISS
# =========================
@st.cache_resource(show_spinner="Loading documents & building index...")
def get_vectorstore(_docs_dir, _index_dir, _fingerprint):
    # Try load from disk
    if os.path.isdir(_index_dir):
        try:
            embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
            vs = FAISS.load_local(_index_dir, embeddings, allow_dangerous_deserialization=True)
            meta = load_meta()
            if meta.get("fingerprint") == _fingerprint:
                return vs
        except Exception:
            pass

    # Build from scratch
    docs = ingest_documents(_docs_dir)
    if not docs:
        raise RuntimeError(f"No PDF/DOCX files found in '{_docs_dir}'")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(_index_dir, exist_ok=True)
    vs.save_local(_index_dir)
    save_meta({"fingerprint": _fingerprint})
    return vs


# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title="RAG Chat", page_icon="💬")
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        st.stop()

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    fingerprint = doc_fingerprint(DOCS_DIR)
    vectorstore = get_vectorstore(DOCS_DIR, INDEX_DIR, fingerprint)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("💬 Azərbaycanda 2023-2024-cü illərə aid tranzit daşımalar haqqında statistik suallarınıza burada cavab ala bilərsiniz")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
                answer = result["answer"]
                st.markdown(answer)

        # Save to state
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append((prompt, answer))


if __name__ == "__main__":

    main()
