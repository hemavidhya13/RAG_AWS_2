import streamlit as st
import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader


# -----------------------------
# 🔧 Setup
# -----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------
# 📄 Document Loading
# -----------------------------
def load_documents_from_sources(pdf_paths=None, excel_paths=None, urls=None):
    all_docs = []

    if pdf_paths:
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                all_docs.extend(pdf_docs)
                st.success(f"✅ Loaded {len(pdf_docs)} pages from {os.path.basename(path)}")
            except Exception as e:
                st.error(f"❌ Error loading PDF {path}: {e}")

    if excel_paths:
        for path in excel_paths:
            try:
                if path.endswith(".csv"):
                    loader = CSVLoader(path)
                    csv_docs = loader.load()
                    all_docs.extend(csv_docs)
                    st.success(f"✅ Loaded {len(csv_docs)} rows from {os.path.basename(path)}")
                else:
                    st.warning(f"⚠️ Skipping {path} (not a CSV file)")
            except Exception as e:
                st.error(f"❌ Error loading CSV {path}: {e}")

    if urls:
        urls = [u.strip() for u in urls if u.strip()]
        if urls:
            try:
                loader = WebBaseLoader(urls)
                web_docs = loader.load()
                all_docs.extend(web_docs)
                st.success(f"✅ Loaded {len(web_docs)} web documents")
            except Exception as e:
                st.error(f"❌ Error loading websites: {e}")

    return all_docs


# -----------------------------
# 🧠 LangGraph State Definition
# -----------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool
    sources: List[str]


def decide_retrieval(state: AgentState) -> AgentState:
    question = state["question"]
    retrieval_keywords = ["what", "how", "explain", "describe", "tell me", "who", "where", "when", "why"]
    needs_retrieval = any(k in question.lower() for k in retrieval_keywords)
    return {**state, "needs_retrieval": needs_retrieval}


def should_retrieve(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


def retrieve_documents(state: AgentState) -> AgentState:
    question = state["question"]
    documents = retriever.invoke(question)
    return {**state, "documents": documents}


def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using retrieved documents and include source info.
    If no relevant information is found, call the LLM directly
    and label the source as 'LLM (no matching docs)'.
    """
    question = state["question"]
    documents = state.get("documents", [])

    # ✅ CASE 1: No docs retrieved at all
    if not documents:
        prompt = f"Answer the following question directly (no context available): {question}"
        response = llm.invoke(prompt)
        answer = response.content
        return {**state, "answer": answer, "sources": ["LLM (no matching docs)"]}

    # ✅ CASE 2: Docs exist but may not be relevant
    # Simple heuristic: check if question keywords appear in the context
    context = "\n\n".join([doc.page_content for doc in documents])
    question_keywords = [w.lower() for w in question.split() if len(w) > 3]
    keyword_hits = sum(context.lower().count(k) for k in question_keywords)

    if keyword_hits < 2:  # very weak match → treat as "no matching docs"
        prompt = f"Answer the following question directly (no relevant documents found): {question}"
        response = llm.invoke(prompt)
        answer = response.content
        return {**state, "answer": answer, "sources": ["LLM (no matching docs)"]}

    # ✅ CASE 3: Docs are relevant — answer using context
    sources = list({
        doc.metadata.get("source")
        or doc.metadata.get("file_path")
        or doc.metadata.get("url")
        or "Unknown Source"
        for doc in documents
    })

    prompt = f"""You are a helpful assistant. Based on the context below, answer the question clearly and concisely.
If the answer is found in the context, base it strictly on that; otherwise, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content
    return {**state, "answer": answer, "sources": sources}


# -----------------------------
# 🧩 LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", should_retrieve, {"retrieve": "retrieve", "generate": "generate"})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()


def ask_question(question: str):
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "sources": [],
    }
    return app.invoke(initial_state)


# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot using LangGraph + OpenAI")

st.sidebar.header("📂 Upload or Link Documents")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
uploaded_csvs = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
urls_input = st.sidebar.text_area("Enter website URLs (comma-separated)")
urls = [u.strip() for u in urls_input.split(",") if u.strip()]

if st.sidebar.button("Load Documents"):
    with st.spinner("Loading documents..."):
        pdf_paths, csv_paths = [], []
        os.makedirs("temp_uploads", exist_ok=True)

        # Save uploaded files
        for f in uploaded_pdfs:
            path = os.path.join("temp_uploads", f.name)
            with open(path, "wb") as file:
                file.write(f.getbuffer())
            pdf_paths.append(path)

        for f in uploaded_csvs:
            path = os.path.join("temp_uploads", f.name)
            with open(path, "wb") as file:
                file.write(f.getbuffer())
            csv_paths.append(path)

        documents = load_documents_from_sources(pdf_paths, csv_paths, urls)

        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            st.session_state["vectorstore"] = FAISS.from_documents(split_docs, embeddings)
            st.session_state["retriever"] = st.session_state["vectorstore"].as_retriever(k=3)
            st.success(f"✅ Loaded {len(split_docs)} text chunks. Ready to chat!")
        else:
            st.error("No documents were loaded.")


if "retriever" in st.session_state:
    retriever = st.session_state["retriever"]
    st.divider()
    st.subheader("💬 Ask a Question")

    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("Thinking..."):
            result = ask_question(user_question)
            st.markdown("### 🧠 Answer:")
            st.write(result["answer"])

            if result.get("sources"):
                st.markdown("### 📚 Sources:")
                for src in result["sources"]:
                    if src.startswith("http"):
                        st.markdown(f"- 🌐 [{src}]({src})")
                    else:
                        st.markdown(f"- 📄 {os.path.basename(src)}")

else:
    st.info("👈 Upload and load documents to start chatting.")