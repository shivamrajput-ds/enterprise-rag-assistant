import sys
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.vector_store import load_vector_store
from src.config_loader import config
from src.logger import logger
from src.exception import RagException
from src.query_expansion import expand_query
from src.reranker import rerank_documents


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25 keyword-based retrieval.
    """
    return text.lower().split()


def load_documents_from_chroma(vector_store) -> List[Document]:
    """
    Load stored documents and metadata directly from the Chroma collection.
    """
    collection_data = vector_store._collection.get(
        include=["documents", "metadatas"]
    )

    documents = collection_data.get("documents", [])
    metadatas = collection_data.get("metadatas", [])

    return [
        Document(page_content=doc, metadata=metadata or {})
        for doc, metadata in zip(documents, metadatas)
        if doc and doc.strip()
    ]


def bm25_search(
    query: str,
    documents: List[Document],
    top_k: int,
) -> List[Document]:
    """
    Retrieve relevant documents using BM25 keyword search.
    """
    if not documents:
        return []

    tokenized_corpus = [
        tokenize(doc.page_content)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)
    scored_docs = list(zip(documents, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        doc
        for doc, score in scored_docs[:top_k]
        if score > 0
    ]


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents using source metadata and page content.
    """
    unique_docs = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        content = doc.page_content.strip()
        key = f"{source}-{content}"

        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)

    return unique_docs


def print_debug_chunks(title: str, docs: List[Document]) -> None:
    """
    Print retrieved chunks for local debugging.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        print(f"\n----- {title} {i} -----")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:700])
        print("-" * 50)


def retrieve_relevant_chunks(query: str) -> List[Document]:
    """
    Retrieve relevant chunks using query expansion, vector search,
    optional BM25 hybrid search, deduplication, and reranking.
    """
    try:
        logger.info("Started hybrid retrieval process")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        vector_store = load_vector_store()
        collection_count = vector_store._collection.count()

        if collection_count == 0:
            logger.warning("Vector store is empty.")
            return []

        expanded_queries = expand_query(query)

        per_query_k = config["retriever"].get("per_query_k", 4)
        retriever_top_k = config["retriever"].get("top_k", 8)

        all_vector_results: List[Tuple[Document, float]] = []

        for expanded_query in expanded_queries:
            results = vector_store.similarity_search_with_score(
                query=expanded_query,
                k=per_query_k,
            )

            all_vector_results.extend(results)

        all_vector_results = sorted(
            all_vector_results,
            key=lambda item: item[1],
        )

        vector_docs = [
            doc
            for doc, _ in all_vector_results[:retriever_top_k]
        ]

        bm25_docs = []

        if config.get("hybrid_search", {}).get("enabled", False):
            all_docs = load_documents_from_chroma(vector_store)
            bm25_top_k = config.get("hybrid_search", {}).get("bm25_top_k", 6)

            bm25_docs = bm25_search(
                query=query,
                documents=all_docs,
                top_k=bm25_top_k,
            )

        candidate_docs = vector_docs + bm25_docs
        candidate_docs = deduplicate_documents(candidate_docs)

        if not candidate_docs:
            logger.warning("No candidates found after hybrid retrieval.")
            return []

        try:
            final_docs = rerank_documents(
                query=query,
                docs=candidate_docs,
            )

            if not final_docs:
                logger.warning(
                    "Reranker returned no documents. Falling back to candidates."
                )
                final_docs = candidate_docs[:retriever_top_k]

        except Exception as rerank_error:
            logger.error(f"Reranking failed: {str(rerank_error)}")
            final_docs = candidate_docs[:retriever_top_k]

        final_docs = final_docs[:retriever_top_k]

        logger.info(
            f"Retrieved {len(final_docs)} relevant chunks after hybrid retrieval"
        )

        return final_docs

    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    test_query = input("Enter query: ")

    results = retrieve_relevant_chunks(test_query)

    print_debug_chunks(
        title="Final Retrieved Chunk",
        docs=results,
    )