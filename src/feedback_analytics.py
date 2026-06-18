import sys

import pandas as pd

from src.feedback_db import get_connection
from src.logger import logger
from src.exception import RagException


def load_feedback_data() -> pd.DataFrame:
    """
    Load feedback records from the database.

    Returns:
        pd.DataFrame: Feedback records ordered by creation time.
    """
    conn = None

    try:
        logger.info("Loading feedback data")

        conn = get_connection()

        query = """
        SELECT
            id,
            query,
            answer,
            feedback,
            created_at
        FROM feedback
        ORDER BY created_at DESC
        """

        df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} feedback records")

        return df

    except Exception as e:
        logger.error(f"Failed to load feedback data: {str(e)}")
        raise RagException(str(e), sys)

    finally:
        if conn:
            conn.close()


def get_feedback_summary() -> dict:
    """
    Generate summary statistics from stored feedback.

    Returns:
        dict: Feedback metrics and underlying dataframe.
    """
    try:
        df = load_feedback_data()

        if df.empty:
            return {
                "total_queries": 0,
                "positive": 0,
                "negative": 0,
                "positive_percentage": 0,
                "negative_percentage": 0,
                "data": df,
            }

        total_queries = len(df)

        positive_count = len(
            df[df["feedback"] == "positive"]
        )

        negative_count = len(
            df[df["feedback"] == "negative"]
        )

        positive_percentage = (
            round((positive_count / total_queries) * 100, 2)
            if total_queries > 0
            else 0
        )

        negative_percentage = (
            round((negative_count / total_queries) * 100, 2)
            if total_queries > 0
            else 0
        )

        return {
            "total_queries": total_queries,
            "positive": positive_count,
            "negative": negative_count,
            "positive_percentage": positive_percentage,
            "negative_percentage": negative_percentage,
            "data": df,
        }

    except Exception as e:
        logger.error(
            f"Failed to generate feedback summary: {str(e)}"
        )
        raise RagException(str(e), sys)


if __name__ == "__main__":
    summary = get_feedback_summary()

    print("Feedback Summary")
    print("-" * 20)

    print(f"Total Queries: {summary['total_queries']}")
    print(f"Positive: {summary['positive']}")
    print(f"Negative: {summary['negative']}")
    print(f"Positive %: {summary['positive_percentage']}")
    print(f"Negative %: {summary['negative_percentage']}")

    print("\nRecent Feedback:")
    print(summary["data"].head())