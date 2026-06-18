import os
import shutil
import sys

from src.document_loader import load_documents
from src.text_splitter import split_text_into_chunks
from src.vector_store import save_to_vector_store
from src.config_loader import config
from src.logger import logger
from src.exception import RagException


def clear_existing_vectorstore(vectorstore_dir: str) -> None:
    """
    Remove the existing vector store directory and recreate it.
    """
    if os.path.exists(vectorstore_dir):
        logger.info(f"Deleting old vector store: {vectorstore_dir}")
        shutil.rmtree(vectorstore_dir)

    os.makedirs(vectorstore_dir, exist_ok=True)


def run_ingestion_pipeline() -> dict:
    """
    Run the complete document ingestion pipeline.

    Steps:
    1. Load documents from the configured documents directory.
    2. Split text-based documents into chunks.
    3. Clear the existing vector store.
    4. Save chunks into the persistent vector store.
    """
    try:
        logger.info("Started ingestion pipeline")

        documents = load_documents()

        chunks = split_text_into_chunks(
            documents=documents,
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )

        vectorstore_dir = config["paths"]["vectorstore_dir"]

        clear_existing_vectorstore(vectorstore_dir)

        save_to_vector_store(chunks)

        logger.info("Ingestion pipeline completed successfully")

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectorstore_dir": vectorstore_dir,
        }

    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    result = run_ingestion_pipeline()

    print("\nIngestion completed successfully!")
    print(f"Total documents: {result['documents']}")
    print(f"Total chunks: {result['chunks']}")
    print(f"Vector store: {result['vectorstore_dir']}")