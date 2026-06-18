import re
import sys
from typing import List

from src.llm import get_llm
from src.logger import logger
from src.exception import RagException


def clean_expanded_query(text: str) -> str:
    """
    Clean a single expanded query returned by the language model.
    """
    text = text.strip()
    text = text.lstrip("-•*").strip()
    text = re.sub(r"^\d+[\.\)]\s*", "", text)
    text = text.strip('"').strip("'").strip()

    return text


def is_entity_or_tabular_query(query: str) -> bool:
    """
    Detect queries where expansion may reduce retrieval precision.

    Entity, ID-based, and analytical queries should usually be searched
    exactly as written.
    """
    query_lower = query.lower()

    tabular_patterns = [
        "how many",
        "count",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "average",
        "mean",
        "top",
        "bottom",
        "greater than",
        "less than",
        "more than",
        "below",
        "above",
        "under",
        "between",
        ">",
        "<",
        ">=",
        "<=",
    ]

    if any(pattern in query_lower for pattern in tabular_patterns):
        return True

    if re.search(r"\b(id|student_id|employee_id|order_id)\b", query_lower):
        return True

    return False


def expand_query(query: str) -> List[str]:
    """
    Generate alternative retrieval queries for semantic search.

    Returns the original query plus up to four expanded variants.
    """
    try:
        logger.info("Started query expansion")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        if is_entity_or_tabular_query(query):
            logger.info("Entity or tabular query detected. Skipping expansion.")
            return [query]

        llm = get_llm()

        prompt = f"""
You are an expert query expansion assistant for a Retrieval-Augmented Generation system.

Generate exactly 4 alternative search queries.

Rules:
1. Preserve the original meaning.
2. Do not answer the question.
3. Do not introduce unrelated topics.
4. Do not add facts not implied by the query.
5. Use synonyms and alternative phrasing.
6. Return only search queries.
7. One query per line.
8. No numbering.
9. No bullets.

User Query:
{query}
""".strip()

        response = llm.invoke(prompt)
        raw_lines = response.content.split("\n")

        expanded_queries = []

        for line in raw_lines:
            cleaned = clean_expanded_query(line)

            if cleaned:
                expanded_queries.append(cleaned)

        final_queries = []
        seen = set()

        for item in [query] + expanded_queries:
            normalized = item.lower().strip()

            if normalized not in seen:
                final_queries.append(item)
                seen.add(normalized)

            if len(final_queries) == 5:
                break

        logger.info(f"Expanded queries: {final_queries}")

        return final_queries

    except Exception as e:
        logger.error(f"Query expansion failed: {str(e)}")

        if query and query.strip():
            logger.warning("Falling back to original query only.")
            return [query.strip()]

        raise RagException(str(e), sys)


if __name__ == "__main__":
    user_query = input("Enter query: ")
    queries = expand_query(user_query)

    print("\nExpanded Queries:")

    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")