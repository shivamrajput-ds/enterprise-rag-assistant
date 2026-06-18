import os
import sys

import pandas as pd

from src.rag_chain import generate_answer
from src.logger import logger
from src.exception import RagException


def run_e2e_tests(
    test_file: str = "tests/e2e_test_cases.csv",
) -> None:
    """
    Run end-to-end tests for the RAG pipeline.

    Each test case should contain:
    - question
    - expected_keyword
    """
    try:
        logger.info("Started E2E test execution")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"E2E test file not found: {test_file}")

        df = pd.read_csv(test_file)

        required_columns = ["question", "expected_keyword"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        total_tests = len(df)
        passed_tests = 0

        for index, row in df.iterrows():
            question = str(row["question"]).strip()
            expected_keyword = str(row["expected_keyword"]).strip().lower()

            if not question or not expected_keyword:
                continue

            logger.info(f"Running E2E Test {index + 1}/{total_tests}")

            result = generate_answer(question)
            answer = result["answer"].lower()

            assert expected_keyword in answer, (
                f"\nFAILED QUESTION: {question}\n"
                f"EXPECTED KEYWORD: {expected_keyword}\n"
                f"ANSWER: {result['answer']}"
            )

            passed_tests += 1
            print(f"PASSED | Question: {question}")

        print("\n" + "=" * 60)
        print("ALL E2E TESTS PASSED")
        print(f"Passed: {passed_tests}/{total_tests}")
        print("=" * 60)

        logger.info(
            f"E2E tests completed successfully ({passed_tests}/{total_tests})"
        )

    except AssertionError as e:
        logger.error(f"E2E test failed: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"E2E testing failed: {str(e)}")
        raise RagException(str(e), sys)


if __name__ == "__main__":
    run_e2e_tests()