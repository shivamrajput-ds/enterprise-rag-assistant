import sys
from typing import Any

from src.exception import RagException
from src.llm import get_llm
from src.logger import logger
from src.query_router import is_tabular_analytics_query
from src.retriever import retrieve_relevant_chunks
from src.tabular_query_engine import answer_tabular_query


FALLBACK_ANSWER = "I don't know based on the provided documents."
MAX_CONTEXT_CHARS = 8000


def format_source(doc: Any) -> str:
    """
    Convert document metadata into a readable source reference.
    """
    metadata = getattr(doc, "metadata", {}) or {}

    file_name = metadata.get("file_name", "unknown")
    file_type = metadata.get("file_type", "unknown")
    page = metadata.get("page")
    row = metadata.get("row")
    sheet = metadata.get("sheet_name")

    parts = [f"File: {file_name}", f"Type: {file_type}"]

    if sheet not in [None, ""]:
        parts.append(f"Sheet: {sheet}")

    if page not in [None, ""]:
        parts.append(f"Page: {page}")

    if row not in [None, ""]:
        parts.append(f"Row: {row}")

    return " | ".join(parts)


def _empty_response(query: str, answer: str = FALLBACK_ANSWER) -> dict:
    """
    Create a standard response object when no context is available.
    """
    return {
        "question": query,
        "answer": answer,
        "contexts": [],
        "sources": [],
        "retrieved_chunks": 0,
    }


def _build_context(docs: list[Any]) -> str:
    """
    Build the final context block from retrieved documents.
    """
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source_info = format_source(doc)
        page_content = getattr(doc, "page_content", "")

        context_parts.append(
            f"""
[PASSAGE {i}]
{source_info}

{page_content}
""".strip()
        )

    return "\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]


def _build_prompt(context: str, query: str) -> str:
    """
    Build a strict grounded-answer prompt for the language model.
    """
    return f"""
You are a strict Retrieval-Augmented Generation assistant.

Use ONLY the provided passages to answer the question.

PASSAGES:
{context}

QUESTION:
{query}

RULES:
1. Answer only from the provided passages.
2. Do not use outside knowledge.
3. Do not guess, assume, speculate, or create missing facts.
4. If the answer is present in a table, CSV, Excel, JSON, or structured record, use the exact matching row/record.
5. For structured records, return the relevant fields clearly using the field names shown in the passage.
6. If the user asks about a specific entity, name, ID, code, country, product, employee, student, order, customer, department, category, status, grade, rank, score, percentage, amount, date, or year, prefer the passage that directly contains that entity.
7. If the question asks to list records matching a value such as Grade A, Grade A+, Status Active, Department HR, Country India, Category Electronics, or any other column-value condition, list only the matching records found in the provided passages.
8. If the question asks "which", "show", "list", "give information", or "details", return the matching record details instead of giving a vague summary.
9. If multiple matching structured records are provided, return all matching records from the provided passages.
10. If the question asks for rank, score, marks, percentage, total, year, name, country, ID, status, grade, department, category, or amount, extract the exact value from the matching passage.
11. If the user asks for count, highest, lowest, maximum, minimum, average, mean, top-N, bottom-N, or full-table analytics, answer only if the value is explicitly available in the provided passages. Do not calculate from incomplete retrieved rows unless the passages clearly contain all required rows.
12. If only partial records are retrieved, do not claim the result is complete for the whole dataset.
13. For direct factual questions, answer concisely.
14. For comparison questions, compare only supported facts from the provided passages.
15. For policy or concept questions, give a clear explanation using only supported information.
16. If multiple retrieved passages are present, prefer the passage that directly matches the question.
17. Do not mention passages, context, chunks, embeddings, vector database, or retrieval.
18. Do not reveal chain-of-thought or internal reasoning.
19. Start directly with the answer.
20. If the answer is not supported, respond exactly with:
I don't know based on the provided documents.
21. If the question requires count, average, top, bottom, maximum, minimum, or filtering over a full CSV/Excel table and that result is not already present in the passages, respond exactly with:
I don't know based on the provided documents.

FINAL ANSWER:
""".strip()


def generate_answer(query: str) -> dict:
    """
    Generate an answer using either the tabular analytics engine
    or the document RAG pipeline.
    """
    try:
        logger.info("Started RAG chain")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        if is_tabular_analytics_query(query):
            tabular_answer = answer_tabular_query(query)

            if tabular_answer:
                logger.info("Answered by tabular engine")
                return _empty_response(query, tabular_answer)

            logger.warning(
                "Tabular router matched, but tabular engine returned no answer"
            )
            return _empty_response(query)

        docs = retrieve_relevant_chunks(query)

        if not docs:
            return _empty_response(query)

        context = _build_context(docs)
        prompt = _build_prompt(context, query)

        llm = get_llm()
        response = llm.invoke(prompt)

        answer = response.content.strip()

        logger.info("RAG answer generated successfully")

        return {
            "question": query,
            "answer": answer,
            "contexts": [
                getattr(doc, "page_content", "")
                for doc in docs
            ],
            "sources": docs,
            "retrieved_chunks": len(docs),
        }

    except Exception as e:
        logger.error(f"RAG chain failed: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    user_query = input("Ask question: ")
    result = generate_answer(user_query)

    print("\nANSWER:")
    print(result["answer"])

    if result["answer"].strip() != FALLBACK_ANSWER:
        print("\nSOURCES:")

        seen_sources = set()

        for doc in result["sources"]:
            source_text = format_source(doc)

            if source_text not in seen_sources:
                seen_sources.add(source_text)
                print(f"- {source_text}")