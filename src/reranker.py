import sys
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config_loader import config
from src.logger import logger
from src.exception import RagException


# CrossEncoder models are expensive to load, so keep one shared instance.
_reranker_model = None


def get_reranker_model() -> CrossEncoder:
    """
    Load and return the configured CrossEncoder reranker model.
    """
    global _reranker_model

    try:
        if _reranker_model is not None:
            return _reranker_model

        model_name = config["reranker"]["model_name"]

        logger.info(f"Loading reranker model: {model_name}")

        _reranker_model = CrossEncoder(model_name)

        logger.info("Reranker model loaded successfully")

        return _reranker_model

    except Exception as e:
        logger.error(f"Reranker model loading failed: {str(e)}")
        raise RagException(str(e), sys)


def _print_reranker_debug(
    scored_docs: List[Tuple[Document, float]],
    top_k: int,
) -> None:
    """
    Print reranker scores and selected chunks during debugging.
    """
    print("\n========== RERANKER RANKING ==========\n")

    for i, (doc, score) in enumerate(scored_docs[:top_k], start=1):
        print(f"\nRerank {i}")
        print("Reranker Score:", score)
        print("Metadata:", doc.metadata)
        print(doc.page_content[:500])
        print("-" * 50)


def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank candidate documents using a CrossEncoder model.

    The reranker receives query-document pairs and assigns a relevance score
    to each pair. This improves final context quality after vector or hybrid
    retrieval.
    """
    try:
        logger.info("Started reranking process")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not docs:
            logger.warning("No documents provided for reranking.")
            return []

        query = query.strip()

        valid_docs = [
            doc
            for doc in docs
            if doc.page_content and doc.page_content.strip()
        ]

        if not valid_docs:
            logger.warning("No valid document content found for reranking.")
            return []

        model = get_reranker_model()

        configured_top_k = config["reranker"].get("top_k", 4)
        top_k = min(configured_top_k, len(valid_docs))

        pairs = [
            [query, doc.page_content]
            for doc in valid_docs
        ]

        scores = model.predict(pairs)

        scored_docs: List[Tuple[Document, float]] = [
            (doc, float(score))
            for doc, score in zip(valid_docs, scores)
        ]

        scored_docs = sorted(
            scored_docs,
            key=lambda item: item[1],
            reverse=True,
        )

        debug_reranker = config.get("debug", {}).get("reranker", False)

        if debug_reranker:
            _print_reranker_debug(
                scored_docs=scored_docs,
                top_k=top_k,
            )

        final_docs = [
            doc
            for doc, _ in scored_docs[:top_k]
        ]

        logger.info(f"Reranked and selected {len(final_docs)} chunks")

        return final_docs

    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        logger.warning("Returning original documents without reranking.")

        return docs


if __name__ == "__main__":
    print("This module provides CrossEncoder-based reranking for retrieved documents.")