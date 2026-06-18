import os
import sys

import psycopg2
import psycopg2.extensions
from dotenv import load_dotenv

from src.logger import logger
from src.exception import RagException


load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """
    Create and return a PostgreSQL database connection.

    Required environment variables:
    - DB_HOST
    - DB_PORT
    - DB_NAME
    - DB_USER
    - DB_PASSWORD
    """
    try:
        required_vars = [
            "DB_HOST",
            "DB_PORT",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
        ]

        missing = [
            var
            for var in required_vars
            if not os.getenv(var)
        ]

        if missing:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing)}"
            )

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require",
        )

        return conn

    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise RagException(str(e), sys)


def save_feedback(
    query: str,
    answer: str,
    feedback: str,
) -> None:
    """
    Save user feedback for a generated answer.

    Args:
        query: User question.
        answer: Generated assistant answer.
        feedback: Either 'positive' or 'negative'.
    """
    conn = None
    cursor = None

    try:
        logger.info("Saving user feedback")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty")

        if feedback not in ["positive", "negative"]:
            raise ValueError(
                "feedback must be either 'positive' or 'negative'"
            )

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO feedback (
                query,
                answer,
                feedback
            )
            VALUES (%s, %s, %s)
            """,
            (
                query.strip(),
                answer.strip(),
                feedback,
            ),
        )

        conn.commit()

        logger.info("Feedback saved successfully")

    except Exception as e:
        if conn:
            conn.rollback()

        logger.error(f"Failed to save feedback: {str(e)}")
        raise RagException(str(e), sys)

    finally:
        if cursor:
            cursor.close()

        if conn:
            conn.close()


if __name__ == "__main__":
    save_feedback(
        query="What is the probation period?",
        answer="The probation period is 6 months.",
        feedback="positive",
    )

    print("Feedback inserted successfully.")