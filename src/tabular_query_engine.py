import os
import re
import sys
from typing import Optional

import pandas as pd

from src.config_loader import config
from src.logger import logger
from src.exception import RagException


# Enable this flag when table scoring or column matching needs debugging.
DEBUG_TABULAR = False


# -------------------------------------------------------------------
# Basic text utilities
# -------------------------------------------------------------------
def normalize_query(query: str) -> str:
    """
    Normalize special symbols and whitespace in the user query.
    """
    query = str(query or "")
    query = query.replace("≥", ">=")
    query = query.replace("≤", "<=")
    query = query.replace("–", "-")
    query = query.replace("—", "-")

    return query.strip()


def normalize_implicit_filters(query: str) -> str:
    """
    Convert short natural-language filters into explicit filter format.

    Example:
    "average Science Section A" -> "average Science where Section is A"
    """
    query = normalize_query(query)
    query_lower = query.lower()

    negative_indicators = [
        "not in",
        " not ",
        "!=",
        " but ",
        "except",
        "excluding",
        "other than",
    ]

    if any(phrase in query_lower for phrase in negative_indicators):
        return query

    explicit_filter_indicators = [
        " where ",
        " with ",
        " is ",
        " = ",
        " equals ",
    ]

    if any(phrase in query_lower for phrase in explicit_filter_indicators):
        return query

    patterns = [
        (r"\b(section)\s+([a-zA-Z0-9+_-]+)\b", r"where \1 is \2"),
        (r"\b(grade)\s+([a-zA-Z0-9+_-]+)\b", r"where \1 is \2"),
        (r"\b(class)\s+([a-zA-Z0-9+_-]+)\b", r"where \1 is \2"),
    ]

    for pattern, replacement in patterns:
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

    return query


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase alphanumeric tokens.
    """
    return re.findall(r"[a-zA-Z0-9+]+", str(text).lower())


def clean_column_name(col: str) -> str:
    """
    Normalize a column name for matching against user queries.
    """
    text = str(col).lower()
    text = re.sub(r"^[a-z]\d+\s*[:\-]\s*", "", text)
    text = re.sub(r"[^a-zA-Z0-9+]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_label(text: str) -> str:
    """
    Normalize labels or categorical values for comparison.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------
# Table loading
# -------------------------------------------------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dataframe column names and replace missing values.
    """
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    return df.fillna("")


def load_all_tables() -> list[tuple[str, pd.DataFrame]]:
    """
    Load all CSV and Excel tables from the configured documents directory.
    """
    tables = []
    docs_dir = config["paths"]["documents_dir"]

    if not os.path.exists(docs_dir):
        return tables

    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        ext = os.path.splitext(file)[1].lower()

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
                tables.append((file, clean_df(df)))

            elif ext in [".xlsx", ".xls"]:
                sheets = pd.read_excel(file_path, sheet_name=None)

                for sheet_name, df in sheets.items():
                    table_name = f"{file} | Sheet: {sheet_name}"
                    tables.append((table_name, clean_df(df)))

        except Exception as e:
            logger.warning(f"Failed to read table {file_path}: {str(e)}")

    return tables


def get_column_tokens(df: pd.DataFrame) -> set[str]:
    """
    Extract searchable tokens from dataframe column names.
    """
    tokens = set()

    for col in df.columns:
        tokens.update(tokenize(clean_column_name(col)))
        tokens.update(tokenize(str(col)))

    return tokens


def get_value_tokens(df: pd.DataFrame, max_rows: int = 100) -> set[str]:
    """
    Extract searchable tokens from sample table values.
    """
    sample = df.astype(str).head(max_rows).values.flatten().tolist()

    return set(tokenize(" ".join(sample)))


# -------------------------------------------------------------------
# Table and column selection
# -------------------------------------------------------------------
def table_score(table_name: str, df: pd.DataFrame, query: str) -> int:
    """
    Calculate a relevance score between a user query and a table.
    """
    query_lower = normalize_query(query).lower()
    query_tokens = set(tokenize(query_lower))

    table_tokens = set(tokenize(table_name))
    column_tokens = get_column_tokens(df)
    value_tokens = get_value_tokens(df)

    score = 0
    score += len(query_tokens & column_tokens) * 80
    score += len(query_tokens & table_tokens) * 30
    score += len(query_tokens & value_tokens) * 5

    if any(word in query_tokens for word in ["student", "students"]):
        if "student" in table_tokens or any(
            "student" in clean_column_name(col) for col in df.columns
        ):
            score += 200

        if any("country" in clean_column_name(col) for col in df.columns):
            score -= 120

    if any(word in query_tokens for word in ["country", "countries"]):
        if any("country" in clean_column_name(col) for col in df.columns):
            score += 200

        if "student" in table_tokens or any(
            "student" in clean_column_name(col) for col in df.columns
        ):
            score -= 120

    return score


def find_matching_tables(
    query: str,
    min_score: int = 1,
) -> list[tuple[str, pd.DataFrame, int]]:
    """
    Rank all available tables and return matching candidates.
    """
    scored = []

    for table_name, df in load_all_tables():
        score = table_score(table_name, df, query)

        if score >= min_score:
            scored.append((table_name, df, score))

    scored.sort(key=lambda item: item[2], reverse=True)

    if DEBUG_TABULAR:
        print("\n===== TABLE SCORES =====")
        for name, _, score in scored:
            print(score, name)

    return scored


def find_best_column(
    df: pd.DataFrame,
    query: str,
    numeric_only: bool = False,
) -> Optional[str]:
    """
    Find the dataframe column that best matches the query text.
    """
    query = normalize_query(query).lower()
    query_tokens = set(tokenize(query))

    stop_words = {
        "which", "record", "records", "row", "rows",
        "has", "have", "highest", "lowest", "maximum", "minimum",
        "average", "mean", "top", "bottom", "greater", "than",
        "less", "more", "above", "below", "under", "show", "list",
        "by", "the", "is", "are", "what", "who", "where", "with",
        "and", "not", "in", "between", "from", "to", "but",
        "student", "students", "country", "countries",
        "score", "scores", "marks", "mark",
    }

    useful_tokens = query_tokens - stop_words

    best_col = None
    best_score = 0

    for col in df.columns:
        if numeric_only:
            numeric = pd.to_numeric(df[col], errors="coerce")

            if numeric.notna().sum() == 0:
                continue

        raw_col = str(col).lower()
        clean_col = clean_column_name(col)

        raw_tokens = set(tokenize(raw_col))
        clean_tokens = set(tokenize(clean_col))

        score = 0
        score += len(useful_tokens & clean_tokens) * 80
        score += len(useful_tokens & raw_tokens) * 50

        if clean_col and clean_col in query:
            score += 150

        if raw_col and raw_col in query:
            score += 120

        if len(clean_tokens) == 1 and next(iter(clean_tokens)) in useful_tokens:
            score += 100

        compact_col = re.sub(r"[^a-zA-Z0-9]+", "", raw_col)
        compact_query = re.sub(r"[^a-zA-Z0-9]+", "", query)

        if compact_col and compact_col in compact_query:
            score += 120

        if score > best_score:
            best_score = score
            best_col = col

    if DEBUG_TABULAR:
        print("\n===== COLUMN SELECT =====")
        print("query:", query)
        print("best:", best_col, "score:", best_score)

    return best_col if best_score > 0 else None


# -------------------------------------------------------------------
# Output formatting
# -------------------------------------------------------------------
def extract_n(query: str, default: int = 5) -> int:
    """
    Extract N from top-N or bottom-N queries.
    """
    match = re.search(r"(top|bottom)\s+(\d+)", query.lower())

    return int(match.group(2)) if match else default


def display_columns(df: pd.DataFrame) -> list[str]:
    """
    Select the most useful columns for row-level output.
    """
    priority_words = [
        "id", "name", "student", "employee", "customer", "product",
        "country", "year", "rank", "total", "score", "percentage",
        "grade", "status", "department", "category", "section", "class",
        "science", "math", "maths", "english", "hindi", "human rights",
        "public services",
    ]

    selected = []

    for col in df.columns:
        col_clean = clean_column_name(col)
        raw = str(col).lower()

        if any(word in col_clean or word in raw for word in priority_words):
            selected.append(col)

    return selected if selected else list(df.columns)


def format_value(value):
    """
    Format numeric values for clean display.
    """
    if isinstance(value, float):
        rounded = round(value, 2)

        return int(rounded) if rounded.is_integer() else rounded

    return value


def format_row(row: pd.Series, columns: list[str]) -> str:
    """
    Convert a dataframe row into readable key-value text.
    """
    return " ".join(
        [f"{col}: {format_value(row[col])}" for col in columns]
    )


def rows_to_text(
    df: pd.DataFrame,
    columns: list[str],
    max_rows: int = 10,
) -> str:
    """
    Convert selected dataframe rows into readable text.
    """
    if df.empty:
        return ""

    output_rows = []

    for _, row in df.head(max_rows).iterrows():
        output_rows.append(format_row(row, columns))

    text = "\n\n".join(output_rows)

    if len(df) > max_rows:
        text += f"\n\nShowing first {max_rows} of {len(df)} matching records."

    return text


# -------------------------------------------------------------------
# Query parsing
# -------------------------------------------------------------------
def extract_measure_query(query: str) -> str:
    """
    Extract the target metric from analytical queries.
    """
    q = normalize_query(query).lower()

    by_match = re.search(
        r"\bby\s+(.+?)(?:\bwhere\b|\bwith\b|\bbut\b|\band\b|$)",
        q,
    )

    if by_match:
        return clean_condition_hint(by_match.group(1))

    for separator in [" where ", " with ", " but "]:
        if separator in q:
            q = q.split(separator, 1)[0]
            break

    return clean_condition_hint(q)


def clean_condition_hint(text: str) -> str:
    """
    Remove common query words and keep only the useful condition hint.
    """
    text = normalize_query(str(text).lower())

    for separator in [" where ", " with ", " but ", " and "]:
        if separator in text:
            text = text.rsplit(separator, 1)[-1]

    text = re.sub(
        r"\b(show|list|records|record|rows|row|where|which|students|student|"
        r"countries|country|with|have|has|the|is|are|how|many|but|and|top|"
        r"bottom|highest|lowest|maximum|minimum|average|mean|by|marks|mark|"
        r"score|scores|filter|details|give|me)\b",
        " ",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


def detect_between(query: str) -> Optional[tuple[str, float, float]]:
    """
    Detect range conditions such as 'between 50 and 80'.
    """
    q = normalize_query(query).lower()

    patterns = [
        r"(.+?)\s+between\s+(-?\d+\.?\d*)\s+and\s+(-?\d+\.?\d*)",
        r"(.+?)\s+from\s+(-?\d+\.?\d*)\s+to\s+(-?\d+\.?\d*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)

        if match:
            return (
                clean_condition_hint(match.group(1)),
                float(match.group(2)),
                float(match.group(3)),
            )

    return None


def detect_numeric_conditions(query: str) -> list[tuple[str, str, float]]:
    """
    Extract numeric comparison conditions from the user query.
    """
    q = normalize_query(query).lower()
    conditions = []

    word_ops = [
        ("greater than", ">"),
        ("more than", ">"),
        ("above", ">"),
        ("less than", "<"),
        ("below", "<"),
        ("under", "<"),
        ("at least", ">="),
        ("at most", "<="),
    ]

    delimiter = r"(?:^|\bwhere\b|\bwith\b|\bbut\b|\band\b|,)"

    for phrase, operator in word_ops:
        pattern = (
            rf"{delimiter}\s*([a-zA-Z0-9\s:+_\-/]+?)\s+"
            rf"{re.escape(phrase)}\s+(-?\d+\.?\d*)"
        )

        for match in re.finditer(pattern, q):
            column_hint = clean_condition_hint(match.group(1))

            if column_hint:
                conditions.append(
                    (column_hint, operator, float(match.group(2)))
                )

    symbol_ops = [
        (">=", ">="),
        ("<=", "<="),
        (">", ">"),
        ("<", "<"),
        ("=", "="),
    ]

    for symbol, operator in symbol_ops:
        pattern = (
            rf"{delimiter}\s*([a-zA-Z0-9\s:+_\-/]+?)\s*"
            rf"{re.escape(symbol)}\s*(-?\d+\.?\d*)"
        )

        for match in re.finditer(pattern, q):
            column_hint = clean_condition_hint(match.group(1))

            if column_hint:
                conditions.append(
                    (column_hint, operator, float(match.group(2)))
                )

    unique = []
    seen = set()

    for item in conditions:
        key = (item[0], item[1], item[2])

        if key not in seen:
            unique.append(item)
            seen.add(key)

    return unique


def detect_text_filters(df: pd.DataFrame, query: str) -> list[tuple[str, str, bool]]:
    """
    Detect positive and negative categorical filters from the query.
    """
    q = normalize_query(query).lower()
    filters = []

    blocked_values = {
        "highest", "lowest", "maximum", "minimum", "average", "mean",
        "top", "bottom", "greater", "less", "above", "below", "under",
        "more", "between", "than", "where", "with", "and", "but", "by",
        "is", "are", "have", "has", "score", "scores", "marks", "mark",
    }

    for col in df.columns:
        col_clean = clean_column_name(col)

        if not col_clean:
            continue

        col_pattern = re.escape(col_clean)
        value_pattern = r"([a-zA-Z0-9+_-]+)"

        negative_patterns = [
            rf"\b{col_pattern}\s+is\s+not\s+{value_pattern}",
            rf"\b{col_pattern}\s+not\s+{value_pattern}",
            rf"\bnot\s+{col_pattern}\s+{value_pattern}",
            rf"\bnot\s+in\s+{col_pattern}\s+{value_pattern}",
            rf"\bis\s+not\s+in\s+{col_pattern}\s+{value_pattern}",
            rf"\bbut\s+not\s+in\s+{col_pattern}\s+{value_pattern}",
            rf"\bbut\s+is\s+not\s+in\s+{col_pattern}\s+{value_pattern}",
            rf"\b{col_pattern}\s*!=\s*{value_pattern}",
        ]

        positive_patterns = [
            rf"\b(?:where\s+|with\s+|have\s+|has\s+)?"
            rf"{col_pattern}\s+is\s+{value_pattern}",
            rf"\b(?:where\s+|with\s+|have\s+|has\s+)?"
            rf"{col_pattern}\b\s*(?:=|equals|:)\s*{value_pattern}",
            rf"\b(?:where\s+|with\s+|have\s+|has\s+)?"
            rf"{col_pattern}\s+{value_pattern}",
            rf"\bin\s+{col_pattern}\s+{value_pattern}",
        ]

        for pattern in negative_patterns:
            for match in re.finditer(pattern, q):
                value = str(match.group(1)).lower().strip()

                if value and value not in blocked_values:
                    filters.append((col, value, True))

        for pattern in positive_patterns:
            for match in re.finditer(pattern, q):
                value = str(match.group(1)).lower().strip()

                if not value or value in blocked_values:
                    continue

                start, _ = match.span()
                nearby_prefix = q[max(0, start - 30):start]

                if "not" in nearby_prefix or "!=" in nearby_prefix:
                    continue

                filters.append((col, value, False))

    unique = []
    seen = set()

    for col, value, is_negative in filters:
        key = (col, value, is_negative)

        if key not in seen:
            unique.append((col, value, is_negative))
            seen.add(key)

    return unique


# -------------------------------------------------------------------
# Condition execution
# -------------------------------------------------------------------
def apply_text_filters(
    df: pd.DataFrame,
    filters: list[tuple[str, str, bool]],
) -> pd.DataFrame:
    """
    Apply categorical equality and inequality filters.
    """
    result_df = df.copy()

    for column, value, is_negative in filters:
        if column not in result_df.columns:
            continue

        series = result_df[column].astype(str).str.lower().str.strip()
        value_lower = str(value).lower().strip()

        if is_negative:
            result_df = result_df.loc[series != value_lower]
        else:
            result_df = result_df.loc[series == value_lower]

    return result_df


def apply_between(
    df: pd.DataFrame,
    column: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    """
    Apply an inclusive numeric range filter.
    """
    numeric = pd.to_numeric(df[column], errors="coerce")

    return df[(numeric >= low) & (numeric <= high)]


def apply_comparison(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: float,
) -> pd.DataFrame:
    """
    Apply a numeric comparison filter.
    """
    numeric = pd.to_numeric(df[column], errors="coerce")

    if operator == ">":
        return df[numeric > value]

    if operator == "<":
        return df[numeric < value]

    if operator == ">=":
        return df[numeric >= value]

    if operator == "<=":
        return df[numeric <= value]

    if operator == "=":
        return df[numeric == value]

    return df.iloc[0:0]


def apply_all_conditions(
    df: pd.DataFrame,
    query: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply all extracted filters and return the filtered dataframe.
    """
    result_df = df.copy()
    descriptions = []

    text_filters = detect_text_filters(df, query)

    if text_filters:
        result_df = apply_text_filters(result_df, text_filters)

        for col, value, is_negative in text_filters:
            if is_negative:
                descriptions.append(f"{col} != {value}")
            else:
                descriptions.append(f"{col} = {value}")

    between = detect_between(query)

    if between:
        column_hint, low, high = between

        col = (
            find_best_column(result_df, column_hint, numeric_only=True)
            or find_best_column(df, column_hint, numeric_only=True)
            or find_best_column(df, query, numeric_only=True)
        )

        if col:
            result_df = apply_between(result_df, col, low, high)
            descriptions.append(f"{col} between {low} and {high}")

    numeric_conditions = detect_numeric_conditions(query)

    for column_hint, operator, value in numeric_conditions:
        col = (
            find_best_column(result_df, column_hint, numeric_only=True)
            or find_best_column(df, column_hint, numeric_only=True)
        )

        if not col:
            continue

        result_df = apply_comparison(result_df, col, operator, value)
        descriptions.append(f"{col} {operator} {value}")

    return result_df, descriptions


# -------------------------------------------------------------------
# Answer generation
# -------------------------------------------------------------------
def is_count_query(query_lower: str) -> bool:
    """
    Check whether the query asks for a count.
    """
    return any(
        word in query_lower
        for word in ["how many", "count", "number of"]
    )


def is_list_query(query_lower: str) -> bool:
    """
    Check whether the query asks for row-level records.
    """
    return any(
        word in query_lower
        for word in ["which", "show", "list", "records", "rows", "where"]
    )


def count_answer(
    df: pd.DataFrame,
    table_name: str,
    query: str,
) -> Optional[str]:
    """
    Generate a count-based answer for a table.
    """
    result_df, descriptions = apply_all_conditions(df, query)

    if descriptions:
        desc = " and ".join(descriptions)

        return (
            f"There are {len(result_df)} matching records where "
            f"{desc} in {table_name}."
        )

    query_tokens = set(tokenize(query))
    column_tokens = get_column_tokens(df)

    if query_tokens & column_tokens:
        return f"There are {len(df)} records in {table_name}."

    return None


def answer_single_table_query(
    table_name: str,
    df: pd.DataFrame,
    query: str,
) -> Optional[str]:
    """
    Generate an analytical answer for a single table.

    Supported query types:
    - count
    - list/show records
    - top/bottom N
    - highest/lowest
    - average/mean
    - numeric and categorical filters
    """
    query = normalize_query(query)
    query = normalize_implicit_filters(query)
    query_lower = query.lower()

    filtered_df, descriptions = apply_all_conditions(df, query)
    working_df = filtered_df if descriptions else df

    cols_to_show = display_columns(df)
    measure_query = extract_measure_query(query)

    if "top" in query_lower:
        if working_df.empty:
            return None

        n = extract_n(query)

        col = (
            find_best_column(working_df, measure_query, numeric_only=True)
            or find_best_column(working_df, query, numeric_only=True)
            or find_best_column(df, measure_query, numeric_only=True)
        )

        if not col:
            return None

        numeric = pd.to_numeric(working_df[col], errors="coerce")

        result_df = (
            working_df.assign(_sort=numeric)
            .dropna(subset=["_sort"])
            .sort_values("_sort", ascending=False)
            .head(n)
            .drop(columns=["_sort"])
        )

        if result_df.empty:
            return None

        condition_text = (
            " where " + " and ".join(descriptions)
            if descriptions
            else ""
        )

        return (
            f"Top {n} records by {col}{condition_text} in {table_name}:\n\n"
            + rows_to_text(result_df, cols_to_show, max_rows=n)
        )

    if "bottom" in query_lower:
        if working_df.empty:
            return None

        n = extract_n(query)

        col = (
            find_best_column(working_df, measure_query, numeric_only=True)
            or find_best_column(working_df, query, numeric_only=True)
            or find_best_column(df, measure_query, numeric_only=True)
        )

        if not col:
            return None

        numeric = pd.to_numeric(working_df[col], errors="coerce")

        result_df = (
            working_df.assign(_sort=numeric)
            .dropna(subset=["_sort"])
            .sort_values("_sort", ascending=True)
            .head(n)
            .drop(columns=["_sort"])
        )

        if result_df.empty:
            return None

        condition_text = (
            " where " + " and ".join(descriptions)
            if descriptions
            else ""
        )

        return (
            f"Bottom {n} records by {col}{condition_text} in {table_name}:\n\n"
            + rows_to_text(result_df, cols_to_show, max_rows=n)
        )

    if any(word in query_lower for word in ["highest", "maximum", "most"]):
        if working_df.empty:
            return None

        col = (
            find_best_column(working_df, measure_query, numeric_only=True)
            or find_best_column(working_df, query, numeric_only=True)
            or find_best_column(df, measure_query, numeric_only=True)
        )

        if not col:
            return None

        numeric = pd.to_numeric(working_df[col], errors="coerce").dropna()

        if numeric.empty:
            return None

        idx = numeric.idxmax()
        row = working_df.loc[idx]

        condition_text = (
            " where " + " and ".join(descriptions)
            if descriptions
            else ""
        )

        return (
            f"Highest record by {col}{condition_text} in {table_name}:\n"
            + format_row(row, list(df.columns))
        )

    if any(word in query_lower for word in ["lowest", "minimum", "least"]):
        if working_df.empty:
            return None

        col = (
            find_best_column(working_df, measure_query, numeric_only=True)
            or find_best_column(working_df, query, numeric_only=True)
            or find_best_column(df, measure_query, numeric_only=True)
        )

        if not col:
            return None

        numeric = pd.to_numeric(working_df[col], errors="coerce").dropna()

        if numeric.empty:
            return None

        idx = numeric.idxmin()
        row = working_df.loc[idx]

        condition_text = (
            " where " + " and ".join(descriptions)
            if descriptions
            else ""
        )

        return (
            f"Lowest record by {col}{condition_text} in {table_name}:\n"
            + format_row(row, list(df.columns))
        )

    if any(word in query_lower for word in ["average", "mean"]):
        col = (
            find_best_column(working_df, measure_query, numeric_only=True)
            or find_best_column(working_df, query, numeric_only=True)
            or find_best_column(df, measure_query, numeric_only=True)
        )

        if not col:
            return None

        numeric = pd.to_numeric(working_df[col], errors="coerce").dropna()

        if numeric.empty:
            return None

        avg = round(numeric.mean(), 2)

        condition_text = (
            " where " + " and ".join(descriptions)
            if descriptions
            else ""
        )

        return f"Average {col}{condition_text} is {avg} in {table_name}."

    if descriptions and is_list_query(query_lower):
        if working_df.empty:
            return None

        desc = " and ".join(descriptions)

        return (
            f"Matching records where {desc} in {table_name}:\n\n"
            + rows_to_text(working_df, cols_to_show, max_rows=10)
        )

    if is_count_query(query_lower):
        return count_answer(df, table_name, query)

    if descriptions:
        desc = " and ".join(descriptions)

        return (
            f"There are {len(working_df)} matching records where "
            f"{desc} in {table_name}."
        )

    return None


def answer_tabular_query(query: str) -> Optional[str]:
    """
    Route a tabular analytics query to the most relevant table.
    """
    try:
        query = normalize_query(query)
        matching_tables = find_matching_tables(query)

        if not matching_tables:
            return None

        for table_name, df, _ in matching_tables:
            answer = answer_single_table_query(
                table_name=table_name,
                df=df,
                query=query,
            )

            if answer:
                return answer

        return None

    except Exception as e:
        logger.error(f"Tabular query failed: {str(e)}")
        raise RagException(str(e), sys)