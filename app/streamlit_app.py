import os
import sys
import time
import shutil
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import requests
import streamlit as st

from src.pipeline import run_ingestion_pipeline
from src.rag_chain import generate_answer
from src.feedback_analytics import get_feedback_summary
from src.feedback_db import save_feedback


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide"
)


DOCS_DIR = "data/documents"
VECTORSTORE_DIR = "data/vectorstore"
API_URL = "http://127.0.0.1:8000/ask"


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "last_query": None,
        "last_answer": None,
        "feedback_given": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_source(doc) -> str:
    file_name = doc.metadata.get("file_name", "unknown")
    file_type = doc.metadata.get("file_type", "unknown")
    sheet = doc.metadata.get("sheet_name")
    page = doc.metadata.get("page")
    row = doc.metadata.get("row")

    parts = [f"📄 {file_name}", f"Type: {file_type}"]

    if sheet:
        parts.append(f"Sheet: {sheet}")

    if page:
        parts.append(f"Page: {page}")

    if row:
        parts.append(f"Row: {row}")

    return " | ".join(parts)


def export_chat_as_text() -> str:
    chat_text = ""

    for msg in st.session_state.messages:
        role = msg["role"].upper()
        content = msg["content"]

        chat_text += f"{role}:\n{content}\n\n"

        if msg.get("sources"):
            chat_text += "SOURCES:\n"

            for source in msg["sources"]:
                chat_text += f"- {source}\n"

            chat_text += "\n"

    return chat_text


def reset_uploaded_files() -> None:
    if os.path.exists(DOCS_DIR):
        shutil.rmtree(DOCS_DIR)

    os.makedirs(DOCS_DIR, exist_ok=True)


def reset_vector_store() -> None:
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_query = None
    st.session_state.last_answer = None
    st.session_state.feedback_given = False


def save_uploaded_files(uploaded_files: List) -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, file.name)

        if os.path.exists(file_path):
            st.warning(f"{file.name} already exists. Skipping.")
            continue

        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            st.success(f"{file.name} saved successfully.")

        except Exception as e:
            st.error(f"Failed to save {file.name}")
            st.caption(str(e))


def display_uploaded_documents() -> None:
    if not os.path.exists(DOCS_DIR):
        st.info("No documents folder found.")
        return

    files = os.listdir(DOCS_DIR)

    if not files:
        st.info("No documents uploaded yet.")
        return

    for file in files:
        col1, col2 = st.columns([4, 1])

        with col1:
            st.write(f"📄 {file}")

        with col2:
            if st.button("❌", key=f"delete_{file}"):
                try:
                    os.remove(os.path.join(DOCS_DIR, file))
                    st.success(f"{file} deleted.")
                    st.rerun()

                except Exception as e:
                    st.error("Delete failed.")
                    st.caption(str(e))


def display_feedback_analytics() -> None:
    st.markdown("### 📊 Feedback Analytics")

    try:
        summary = get_feedback_summary()

        st.metric("Total Queries", summary["total_queries"])
        st.metric("Positive Feedback", f'{summary["positive_percentage"]}%')
        st.metric("Negative Feedback", f'{summary["negative_percentage"]}%')

        df = summary["data"]

        if not df.empty:
            st.markdown("#### ❌ Failed Questions")

            failed_df = df[df["feedback"] == "negative"][["query", "created_at"]]

            if not failed_df.empty:
                st.dataframe(failed_df, use_container_width=True)
            else:
                st.success("No failed questions yet.")

    except Exception as e:
        st.warning("Feedback analytics unavailable.")
        st.caption(str(e))


def display_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                if "response_time" in msg:
                    st.caption(f"⚡ Response Time: {msg['response_time']} sec")

                if "retrieved_chunks" in msg:
                    st.caption(f"📊 Retrieved Chunks: {msg['retrieved_chunks']}")

                if msg.get("sources"):
                    st.markdown("#### 📄 Sources")

                    for source in msg["sources"]:
                        st.write(source)


def call_rag(query: str, timeout: int = 180) -> dict:
    try:
        response = requests.post(
            API_URL,
            json={"query": query},
            timeout=timeout
        )

        response.raise_for_status()
        api_result = response.json()

        return {
            "answer": api_result.get("answer", ""),
            "retrieved_chunks": api_result.get("retrieved_chunks", 0),
            "response_time": api_result.get("response_time"),
            "sources": api_result.get("sources", [])
        }

    except requests.exceptions.ConnectionError:
        local_result = generate_answer(query)

        return {
            "answer": local_result["answer"],
            "retrieved_chunks": local_result["retrieved_chunks"],
            "response_time": None,
            "sources": [
                format_source(doc)
                for doc in local_result["sources"]
            ]
        }


def stream_answer(answer: str) -> None:
    placeholder = st.empty()
    streamed_text = ""

    for word in answer.split():
        streamed_text += word + " "
        placeholder.markdown(streamed_text)
        time.sleep(0.01)


initialize_session_state()


with st.sidebar:
    st.title("📚 RAG Panel")
    st.caption(
        "Upload PDF, CSV, TXT, JSON, XLS, XLSX or DOCX files, ingest them, "
        "and ask document-based questions."
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        clear_chat()
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            label="⬇️ Download Chat",
            data=export_chat_as_text(),
            file_name="rag_chat_history.txt",
            mime="text/plain"
        )

    st.markdown("---")
    st.markdown("### 📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF/CSV/TXT/JSON/DOCX/XLS/XLSX files",
        type=["pdf", "csv", "txt", "json", "docx", "xls", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)

    st.markdown("---")

    if st.button("📥 Ingest Documents"):
        try:
            with st.spinner("Ingesting documents..."):
                result = run_ingestion_pipeline()

            st.success("Documents ingested successfully.")
            st.info(f"Total documents loaded: {result['documents']}")
            st.info(f"Total chunks created: {result['chunks']}")

        except Exception as e:
            st.error("Document ingestion failed.")
            st.caption(str(e))

    st.markdown("---")
    st.markdown("### 📄 Uploaded Documents")
    display_uploaded_documents()

    st.markdown("---")

    if st.button("🧹 Reset All Documents"):
        try:
            reset_uploaded_files()
            reset_vector_store()
            clear_chat()

            st.success("Documents and vector store reset successfully.")
            st.rerun()

        except Exception as e:
            st.error("Reset failed.")
            st.caption(str(e))

    st.markdown("---")
    display_feedback_analytics()


st.title("🚀 Enterprise RAG Assistant")
st.caption("Ask questions from your uploaded documents.")

with st.expander("💡 Example Questions"):
    st.write("• What is the laptop policy?")
    st.write("• What is the rank of Somalia in FSI 2023?")
    st.write("• Compare casual leave and sick leave policies.")
    st.write("• What is the salary credit date?")
    st.write("• Which students scored A+?")
    st.write("• What is PCA?")


display_chat_history()


query = st.chat_input("Ask a question from your documents...")

if query:
    start_time = time.time()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Generating answer..."):
                result = call_rag(query)

                answer = result["answer"]
                retrieved_chunks = result.get("retrieved_chunks", 0)
                response_time = result.get("response_time")

                if response_time is None:
                    response_time = round(time.time() - start_time, 2)

                sources = result.get("sources", [])

            stream_answer(answer)

            st.caption(f"⚡ Response Time: {response_time} sec")
            st.caption(f"📊 Retrieved Chunks: {retrieved_chunks}")

            if sources:
                st.markdown("#### 📄 Sources")

                for source in sources:
                    st.write(source)

        except Exception as e:
            answer = "Something went wrong while generating the answer."
            retrieved_chunks = 0
            response_time = round(time.time() - start_time, 2)
            sources = []

            st.error("Answer generation failed.")
            st.caption(str(e))

    st.session_state.last_query = query
    st.session_state.last_answer = answer
    st.session_state.feedback_given = False

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "response_time": response_time,
            "retrieved_chunks": retrieved_chunks,
            "sources": sources
        }
    )


if (
    st.session_state.last_query
    and st.session_state.last_answer
    and not st.session_state.feedback_given
):
    st.markdown("---")
    st.subheader("Was this answer helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful"):
            try:
                save_feedback(
                    st.session_state.last_query,
                    st.session_state.last_answer,
                    "positive"
                )

                st.session_state.feedback_given = True
                st.success("Feedback saved successfully 👍")
                st.rerun()

            except Exception as e:
                st.error("Feedback save failed.")
                st.caption(str(e))

    with col2:
        if st.button("👎 Not Helpful"):
            try:
                save_feedback(
                    st.session_state.last_query,
                    st.session_state.last_answer,
                    "negative"
                )

                st.session_state.feedback_given = True
                st.success("Feedback saved successfully 👎")
                st.rerun()

            except Exception as e:
                st.error("Feedback save failed.")
                st.caption(str(e))