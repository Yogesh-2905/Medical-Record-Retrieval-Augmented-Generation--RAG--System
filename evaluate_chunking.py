
import csv
import json
import math
import numpy as np

from pathlib import Path
from common.vector_store import client
from common.embeddings import generate_query_embedding


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

EMBEDDING_DIR = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
)

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

TOP_K = 5

CONFIGURATIONS = [
    "fixed_300", "fixed_600", "fixed_900",
    "sentence_300", "sentence_600", "sentence_900",
    "recursive_300", "recursive_600", "recursive_900"
]


# ==========================================
# 2. LOAD EVALUATION DATA
# ==========================================

def load_evaluation_data():

    with open(
        EVALUATION_DIR / "evaluation_questions.csv",
        encoding="utf-8"
    ) as file:
        questions = list(csv.DictReader(file))

    with open(
        EVALUATION_DIR / "relevance_label.csv",
        encoding="utf-8"
    ) as file:
        labels = list(csv.DictReader(file))

    relevance_map = {}

    for label in labels:

        qid = label["question_id"]
        page = str(label["page_number"]).strip()
        grade = int(float(label["relevance_grade"]))

        relevance_map.setdefault(qid, {})

        relevance_map[qid][page] = max(
            grade,
            relevance_map[qid].get(page, 0)
        )

    answerable_questions = [
        q for q in questions
        if q["answerable"].strip().lower()
        in ("true", "yes", "1")
    ]

    for question in answerable_questions:

        if question["question_id"] not in relevance_map:
            raise ValueError(
                f"Missing relevance labels for "
                f"{question['question_id']}"
            )

    return answerable_questions, relevance_map


# ==========================================
# 3. CALCULATE RETRIEVAL METRICS
# ==========================================

def calculate_metrics(retrieved_pages, relevant_pages):

    retrieved_pages = retrieved_pages[:TOP_K]

    relevant_pages = {
        page: grade
        for page, grade in relevant_pages.items()
        if grade > 0
    }

    relevant_count = len(relevant_pages)

    if relevant_count == 0:
        raise ValueError("No relevant pages found.")

    # Hit Rate
    hit = int(
        any(page in relevant_pages for page in retrieved_pages)
    )

    # Precision@K
    relevant_retrieved = sum(
        1 for page in retrieved_pages
        if page in relevant_pages
    )

    precision = relevant_retrieved / TOP_K

    # Recall@K
    recall = relevant_retrieved / relevant_count

    # MRR
    reciprocal_rank = 0

    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant_pages:
            reciprocal_rank = 1 / rank
            break

    # nDCG@K
    dcg = 0

    for rank, page in enumerate(retrieved_pages, start=1):

        grade = relevant_pages.get(page, 0)

        dcg += (2 ** grade - 1) / math.log2(rank + 1)

    ideal_grades = sorted(
        relevant_pages.values(),
        reverse=True
    )[:TOP_K]

    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "hit_rate": hit,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg
    }


# ==========================================
# 4. EVALUATE ONE CONFIGURATION
# ==========================================

def evaluate_configuration(
    config_name,
    questions,
    relevance_map,
    query_vectors
):

    print(f"\nEvaluating: {config_name}")

    # Load document embeddings
    vectors = np.load(
        EMBEDDING_DIR / f"{config_name}.npy"
    )

    # Load chunk metadata
    with open(
        EMBEDDING_DIR / f"{config_name}_metadata.json",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if len(vectors) != len(metadata):
        raise ValueError(
            f"Embedding/metadata mismatch: {config_name}"
        )

    # Create a separate collection for each configuration
    collection_name = f"medical_records_{config_name}"

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Store document embeddings
    collection.upsert(
        ids=[item["chunk_id"] for item in metadata],
        embeddings=vectors.tolist(),
        documents=[item["text"] for item in metadata],
        metadatas=[
            {
                "doc_id": str(item["doc_id"]),
                "page": str(item["page"]),
                "section": str(item["section"])
            }
            for item in metadata
        ]
    )

    print("Stored chunks:", collection.count())

    metric_results = []

    # Evaluate every answerable question
    for question in questions:

        qid = question["question_id"]

        results = collection.query(
        query_embeddings=[
        query_vectors[qid].tolist()],
        n_results=collection.count(),
        include=["metadatas", "distances"]
        )

        # Deduplicate pages because relevance labels
        # identify relevant pages, not individual chunks.
        retrieved_pages = []

        for item in results["metadatas"][0]:

            page = str(item["page"])

            if page not in retrieved_pages:
                retrieved_pages.append(page)

            if len(retrieved_pages) == TOP_K:
                break

        metrics = calculate_metrics(
            retrieved_pages,
            relevance_map[qid]
        )

        metric_results.append(metrics)

    # Average metrics across questions
    summary = {
        metric: float(np.mean([
            result[metric] for result in metric_results
        ]))
        for metric in metric_results[0]
    }

    summary["configuration"] = config_name
    summary["chunks"] = len(metadata)
    summary["questions_evaluated"] = len(metric_results)

    print("Evaluation completed:", config_name)
    print(summary)

    return summary


# ==========================================
# 5. MAIN EVALUATION LOOP
# ==========================================

def main():

    print("Loading evaluation dataset...")

    questions, relevance_map = load_evaluation_data()

    print("Answerable questions:", len(questions))

    # Generate query embeddings once and reuse them
    print("\nGenerating query embeddings...")

    query_vectors = {}

    for question in questions:

        qid = question["question_id"]

        query_vectors[qid] = generate_query_embedding(
            question["question"]
        )

    print("Query embeddings generated.")

    all_results = []

    # Evaluate all 9 configurations
    for config in CONFIGURATIONS:

        result = evaluate_configuration(
            config,
            questions,
            relevance_map,
            query_vectors
        )

        all_results.append(result)

    # Save results
    output_path = (
        EVALUATION_DIR / "chunking_evaluation_results.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "configuration",
                "chunks",
                "questions_evaluated",
                "hit_rate",
                "precision",
                "recall",
                "mrr",
                "ndcg"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n========== FINAL COMPARISON ==========")

    for result in all_results:

        print(
            f"{result['configuration']:15} "
            f"Hit: {result['hit_rate']:.3f} | "
            f"Precision: {result['precision']:.3f} | "
            f"Recall: {result['recall']:.3f} | "
            f"MRR: {result['mrr']:.3f} | "
            f"nDCG: {result['ndcg']:.3f}"
        )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
