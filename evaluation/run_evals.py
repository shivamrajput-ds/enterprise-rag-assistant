import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from src.rag_chain import generate_answer


def simple_score(answer: str, ground_truth: str) -> float:
    """
    Calculate a simple word-overlap score between
    the generated answer and the expected answer.
    """
    answer_words = set(answer.lower().split())
    truth_words = set(ground_truth.lower().split())

    if not truth_words:
        return 0.0

    common_words = answer_words.intersection(truth_words)

    return round(
        len(common_words) / len(truth_words),
        2,
    )


def run_evaluation() -> None:
    """
    Run evaluation against a predefined test set.
    """
    test_file = "evaluation/test_questions.csv"

    df = pd.read_csv(test_file)

    results = []

    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"\nRunning: {question}")

        result = generate_answer(question)
        answer = result["answer"]

        score = simple_score(
            answer=answer,
            ground_truth=ground_truth,
        )

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "retrieved_chunks": result["retrieved_chunks"],
                "score": score,
            }
        )

    result_df = pd.DataFrame(results)

    os.makedirs("evaluation", exist_ok=True)

    result_df.to_csv(
        "evaluation/evals_results.csv",
        index=False,
    )

    print("\nEvaluation completed!")
    print(result_df[["question", "score"]])

    print("\nSaved to evaluation/evals_results.csv")


if __name__ == "__main__":
    run_evaluation()