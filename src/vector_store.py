import os
import sys
import hashlib
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import get_embedding_model
from src.config_loader import config
from src.exception import RagException
from src.logger import logger


def clean_for_embedding(text: str) -> str:
    """
    Prepare text before sending it to the embedding model.
    """
    text = str(text)
    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    return text.strip()


def create_chunk_id(chunk: Document) -> str:
    """
    Generate a stable chunk ID using source metadata and chunk content.
    """
    source = chunk.metadata.get("source", "")
    content = chunk.page_content or ""

    raw_text = source + content
    return hashlib.md5(raw_text.encode("utf-8")).hexdigest()


def save_to_vector_store(chunks: List[Document]) -> Chroma:
    """
    Save processed document chunks into a persistent Chroma vector store.
    """
    try:
        logger.info("Started saving chunks to vector store")

        if not chunks:
            raise ValueError("Chunks list is empty. Cannot create vector store.")

        vectorstore_dir = config["paths"]["vectorstore_dir"]
        os.makedirs(vectorstore_dir, exist_ok=True)

        embedding_model = get_embedding_model()

        vector_store = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embedding_model,
        )

        cleaned_chunks = []
        chunk_ids = []

        for chunk in chunks:
            cleaned_text = clean_for_embedding(chunk.page_content)

            if not cleaned_text:
                continue

            chunk.page_content = cleaned_text

            cleaned_chunks.append(chunk)
            chunk_ids.append(create_chunk_id(chunk))

        if not cleaned_chunks:
            raise ValueError("No valid chunks found after cleaning.")

        vector_store.add_documents(
            documents=cleaned_chunks,
            ids=chunk_ids,
        )

        logger.info(
            f"Vector store updated successfully at: {vectorstore_dir}. "
            f"Total chunks saved: {len(cleaned_chunks)}"
        )

        return vector_store

    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
        raise RagException(str(e), sys)


def load_vector_store() -> Chroma:
    """
    Load the persisted Chroma vector store from disk.
    """
    try:
        logger.info("Loading vector store")

        vectorstore_dir = config["paths"]["vectorstore_dir"]
        embedding_model = get_embedding_model()

        vector_store = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embedding_model,
        )

        logger.info("Vector store loaded successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load vector store: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    from src.document_loader import load_documents
    from src.text_splitter import split_text_into_chunks

    documents = load_documents()
    chunks = split_text_into_chunks(documents)

    vector_store = save_to_vector_store(chunks)

    print("Vector store created successfully.")
    print("Total stored chunks:", vector_store._collection.count())