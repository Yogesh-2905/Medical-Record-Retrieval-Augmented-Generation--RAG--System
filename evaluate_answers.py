
import csv
import time
from pathlib import Path

from generate_answer import generate_answer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_questions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_answer_results.csv"
)


def evaluate_answers():

    # Read evaluation questions
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        questions = list(reader)

    print(f"Total questions: {len(questions)}")

    total_latency = 0
    successful_answers = 0

    # Prepare output file
    fieldnames = [
        "question_id",
        "question",
        "expected_answer",
        "answerable",
        "question_type",
        "generated_answer",
        "total_latency_seconds",
        "manual_correctness",
        "review_notes"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(questions, start=1):

            question = row["question"]

            print(f"\n[{index}/{len(questions)}] Processing:")
            print(question)

            # Measure complete RAG pipeline latency
            start_time = time.perf_counter()

            try:
                answer = generate_answer(question)
            except Exception as error:
                answer = f"ERROR: {error}"

            end_time = time.perf_counter()

            latency = end_time - start_time

            if not answer.startswith("ERROR:"):
                total_latency += latency
                successful_answers += 1

            print(f"\nTotal response latency: {latency:.2f} seconds")

            # Store evaluation result
            writer.writerow({
                "question_id": row["question_id"],
                "question": question,
                "expected_answer": row["expected_answer"],
                "answerable": row["answerable"],
                "question_type": row["question_type"],
                "generated_answer": answer,
                "total_latency_seconds": round(latency, 2),
                "manual_correctness": "",
                "review_notes": ""
            })

            file.flush()

            print("\nGenerated answer:")
            print(answer)

    # Calculate average latency
    average_latency = (
        total_latency / successful_answers
        if successful_answers > 0
        else 0
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total questions: {len(questions)}")
    print(f"Successful answers: {successful_answers}")
    print(f"Average response latency: {average_latency:.2f} seconds")
    print(f"Total processing time: {total_latency:.2f} seconds")
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_answers()
