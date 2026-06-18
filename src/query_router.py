import os
import re
from functools import lru_cache
from typing import Iterable

import pandas as pd

from src.config_loader import config


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


PHRASE_ANALYTICS = {
    "how many",
    "number of",
    "greater than",
    "less than",
    "more than",
    "at least",
    "at most",
    "not in",
}


SINGLE_ANALYTICS = {
    "count",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "most",
    "least",
    "average",
    "mean",
    "top",
    "bottom",
    "above",
    "below",
    "between",
    "records",
    "record",
    "rows",
    "row",
    "where",
    "filter",
    "show",
    "list",
}


STRUCTURED_WORDS = {
    "id",
    "name",
    "student",
    "students",
    "employee",
    "employees",
    "customer",
    "customers",
    "product",
    "products",
    "country",
    "countries",
    "order",
    "orders",
    "department",
    "category",
    "status",
    "grade",
    "section",
    "class",
    "rank",
    "year",
    "date",
}


MEASURE_WORDS = {
    "marks",
    "mark",
    "score",
    "scores",
    "total",
    "percentage",
    "percent",
    "amount",
    "price",
    "sales",
    "revenue",
    "quantity",
    "qty",
    "units",
    "science",
    "math",
    "maths",
    "english",
    "hindi",
    "social",
    "public",
    "services",
    "human",
    "rights",
}


COMPARISON_RE = re.compile(r"(>=|<=|>|<|=)")


def normalize_query(query: str) -> str:
    """
    Normalize common comparison symbols and dash variants in a query.
    """
    return (
        str(query or "")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("–", "-")
        .replace("—", "-")
    )


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase alphanumeric tokens.
    """
    return re.findall(r"[a-zA-Z0-9+]+", str(text).lower())


def clean_column_name(col: str) -> str:
    """
    Normalize column names for query-schema matching.
    """
    text = str(col).lower()
    text = re.sub(r"^[a-z]\d+\s*[:\-]\s*", "", text)
    text = re.sub(r"[^a-zA-Z0-9+]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def _iter_table_files() -> Iterable[tuple[str, str]]:
    """
    Yield supported table files from the configured documents directory.
    """
    docs_dir = config["paths"]["documents_dir"]

    if not os.path.exists(docs_dir):
        return

    for file in os.listdir(docs_dir):
        path = os.path.join(docs_dir, file)
        ext = os.path.splitext(file)[1].lower()

        if os.path.isfile(path) and ext in SUPPORTED_EXTENSIONS:
            yield file, path


@lru_cache(maxsize=1)
def load_table_schema_tokens() -> frozenset[str]:
    """
    Load searchable schema tokens from available CSV and Excel files.

    The cache avoids reading table files repeatedly on every user query.
    """
    tokens = set()

    for file, path in _iter_table_files() or []:
        ext = os.path.splitext(file)[1].lower()

        try:
            tokens.update(tokenize(file))

            if ext == ".csv":
                sheets = {"csv": pd.read_csv(path, nrows=100)}
            else:
                sheets = pd.read_excel(path, sheet_name=None, nrows=100)

            for sheet_name, df in sheets.items():
                tokens.update(tokenize(sheet_name))

                for col in df.columns:
                    tokens.update(tokenize(clean_column_name(col)))

                values = df.astype(str).fillna("").values.flatten().tolist()
                tokens.update(tokenize(" ".join(values)))

        except Exception:
            continue

    return frozenset(tokens)


def has_analytics_intent(query_lower: str) -> bool:
    """
    Detect whether a query asks for analytics or filtering.
    """
    if COMPARISON_RE.search(query_lower):
        return True

    for phrase in PHRASE_ANALYTICS:
        if phrase in query_lower:
            return True

    tokens = set(tokenize(query_lower))

    return bool(tokens & SINGLE_ANALYTICS)


def has_structured_table_language(query_lower: str) -> bool:
    """
    Detect whether a query looks like a structured table lookup.
    """
    tokens = set(tokenize(query_lower))

    if tokens & STRUCTURED_WORDS:
        return True

    if tokens & MEASURE_WORDS:
        return True

    return False


def is_tabular_analytics_query(query: str) -> bool:
    """
    Decide whether a user query should be routed to the tabular engine.
    """
    query = normalize_query(query).strip()

    if not query:
        return False

    query_lower = query.lower()
    query_tokens = set(tokenize(query_lower))

    schema_tokens = set(load_table_schema_tokens())

    if not schema_tokens:
        return False

    matched_schema_tokens = query_tokens & schema_tokens

    if not matched_schema_tokens:
        return False

    if has_analytics_intent(query_lower):
        return True

    if has_structured_table_language(query_lower):
        return True

    return False


def clear_schema_cache() -> None:
    """
    Clear cached schema tokens after files are uploaded or updated.
    """
    load_table_schema_tokens.cache_clear()