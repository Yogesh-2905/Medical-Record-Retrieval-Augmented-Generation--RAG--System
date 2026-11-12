
import csv
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

RESULTS_FILE = (
    DATA_DIR
    / "evaluation"
    / "rag_answer_results.csv"
)

CHROMA_DIR = DATA_DIR / "chroma_db"


def get_folder_size(folder):

    total = 0

    if folder.exists():
        for file in folder.rglob("*"):
            if file.is_file():
                total += file.stat().st_size

    return total


def format_size(size):

    return f"{size / (1024 * 1024):.2f} MB"


def generate_summary():

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    latencies = [
        float(row["total_latency_seconds"])
        for row in rows
        if row["total_latency_seconds"]
    ]

    successful = [
        row for row in rows
        if not row["generated_answer"].startswith("ERROR:")
    ]

    print("\n========== PHASE 6 SUMMARY ==========")

    print(f"Total questions: {len(rows)}")
    print(f"Successful executions: {len(successful)}")

    if latencies:
        print(f"Average latency: {statistics.mean(latencies):.2f} seconds")
        print(f"Minimum latency: {min(latencies):.2f} seconds")
        print(f"Maximum latency: {max(latencies):.2f} seconds")

    chroma_size = get_folder_size(CHROMA_DIR)
    total_data_size = get_folder_size(DATA_DIR)

    print(f"\nChromaDB storage: {format_size(chroma_size)}")
    print(f"Total data storage: {format_size(total_data_size)}")

    print("\nSelected retrieval configuration:")
    print("Chunking: recursive_300")
    print("Top-K: 8")
    print("Reranking: OFF")
    print("Embedding model: BAAI/bge-small-en-v1.5")
    print("Embedding dimension: 384")

    print("\nRetrieval metrics (Top-K = 8):")
    print("Hit Rate: 1.000")
    print("Precision: 0.167")
    print("Recall: 1.000")
    print("MRR: 0.810")
    print("nDCG: 0.834")

    print("\n=====================================")


if __name__ == "__main__":
    generate_summary()
