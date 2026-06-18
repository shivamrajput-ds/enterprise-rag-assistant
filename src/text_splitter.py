import sys
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exception import RagException
from src.logger import logger
from src.config_loader import config


DO_NOT_SPLIT_TYPES = {
    "csv",
    "csv_schema",
    "excel",
    "excel_schema",
    "json",
}


def split_text_into_chunks(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split text-based documents into chunks while keeping structured data intact.
    """
    try:
        logger.info("Started text splitting process")

        if not documents:
            raise ValueError("Documents list is empty. Cannot split empty documents.")

        chunk_size = chunk_size or config["chunking"]["chunk_size"]
        chunk_overlap = chunk_overlap or config["chunking"]["chunk_overlap"]

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        documents_to_keep = []
        documents_to_split = []

        for doc in documents:
            file_type = doc.metadata.get("file_type", "")

            if file_type in DO_NOT_SPLIT_TYPES:
                documents_to_keep.append(doc)
            else:
                documents_to_split.append(doc)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        split_chunks = splitter.split_documents(documents_to_split)

        chunks = documents_to_keep + split_chunks

        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        logger.info(
            f"Text splitting completed. "
            f"Input documents: {len(documents)}, "
            f"Kept without split: {len(documents_to_keep)}, "
            f"Split input docs: {len(documents_to_split)}, "
            f"Final chunks: {len(chunks)}"
        )

        return chunks

    except Exception as e:
        logger.error(f"Text splitting failed: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    from src.document_loader import load_documents

    documents = load_documents()
    chunks = split_text_into_chunks(documents=documents)

    print(f"Total Documents: {len(documents)}")
    print(f"Total Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:10], start=1):
        print("\n" + "=" * 80)
        print(f"CHUNK {i}")
        print("=" * 80)
        print(chunk.metadata)
        print(chunk.page_content[:1000])