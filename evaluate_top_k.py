
import csv
import json
import numpy as np
import chromadb

from pathlib import Path
from common.embeddings import generate_query_embedding

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

CONFIGURATION = "recursive_300"
TOP_K_VALUES = [1, 3, 5, 8, 10, 15]

COLLECTION_NAME = "medical_records_topk"

# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(PROJECT_ROOT / "data" / "chroma_db")
)

print("ChromaDB connected successfully!")

# --------------------------------------------------
# 3. Load evaluation questions
# --------------------------------------------------

questions = {}

with open(
    EVALUATION_DIR / "evaluation_questions.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        if row["answerable"].strip().lower() in ["true", "yes", "1"]:
            questions[row["question_id"]] = row["question"]

print("Answerable questions:", len(questions))

# --------------------------------------------------
# 4. Load relevance labels
# --------------------------------------------------

relevance_map = {}

with open(
    EVALUATION_DIR / "relevance_label.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        qid = row["question_id"]
        page = str(row["page_number"])
        grade = int(row["relevance_grade"])

        relevance_map.setdefault(qid, {})

        relevance_map[qid][page] = max(
            grade,
            relevance_map[qid].get(page, 0)
        )

# --------------------------------------------------
# 5. Load embeddings and metadata
# --------------------------------------------------

embeddings = np.load(
    EMBEDDING_DIR / f"{CONFIGURATION}.npy"
)

with open(
    EMBEDDING_DIR / f"{CONFIGURATION}_metadata.json",
    "r",
    encoding="utf-8"
) as file:

    metadata = json.load(file)

print("Loaded embeddings:", len(embeddings))

# --------------------------------------------------
# 6. Create evaluation collection
# --------------------------------------------------

if COLLECTION_NAME in [
    item.name for item in client.list_collections()
]:
    client.delete_collection(name=COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

collection.upsert(
    ids=[str(item["chunk_id"]) for item in metadata],
    embeddings=embeddings.tolist(),
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

print("Collection created and embeddings stored.")

# --------------------------------------------------
# 7. Generate query embeddings once
# --------------------------------------------------

query_vectors = {}

for qid, question in questions.items():
    query_vectors[qid] = generate_query_embedding(question)

print("Query embeddings generated.")

# --------------------------------------------------
# 8. Calculate retrieval metrics
# --------------------------------------------------

def calculate_metrics(retrieved_pages, relevant_pages):

    relevant_count = len(relevant_pages)

    if relevant_count == 0:
        return {
            "hit_rate": 0,
            "precision": 0,
            "recall": 0,
            "mrr": 0,
            "ndcg": 0
        }

    hits = [
        1 if page in relevant_pages else 0
        for page in retrieved_pages
    ]

    hit_rate = 1 if sum(hits) > 0 else 0

    precision = sum(hits) / len(retrieved_pages)

    recall = sum(hits) / relevant_count

    reciprocal_rank = 0

    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1 / rank
            break

    dcg = sum(
        (
            (2 ** relevant_pages.get(page, 0) - 1)
            / np.log2(rank + 1)
        )
        for rank, page in enumerate(retrieved_pages, start=1)
    )

    ideal_grades = sorted(
        relevant_pages.values(),
        reverse=True
    )[:len(retrieved_pages)]

    idcg = sum(
        (2 ** grade - 1) / np.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg
    }

# --------------------------------------------------
# 9. Evaluate different Top-K values
# --------------------------------------------------

all_results = []

for top_k in TOP_K_VALUES:

    print(f"\nEvaluating Top-K: {top_k}")

    metric_results = []

    for qid in questions:

        results = collection.query(
            query_embeddings=[
                query_vectors[qid].tolist()
            ],
            n_results=collection.count(),
            include=["metadatas", "distances"]
        )

        retrieved_pages = []

        for item in results["metadatas"][0]:

            page = str(item["page"])

            if page not in retrieved_pages:
                retrieved_pages.append(page)

            if len(retrieved_pages) == top_k:
                break

        metrics = calculate_metrics(
            retrieved_pages,
            relevance_map.get(qid, {})
        )

        metric_results.append(metrics)

    average_metrics = {
        metric: np.mean([
            result[metric] for result in metric_results
        ])
        for metric in [
            "hit_rate",
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]
    }

    average_metrics["top_k"] = top_k

    all_results.append(average_metrics)

    print(average_metrics)

# --------------------------------------------------
# 10. Save results
# --------------------------------------------------

output_path = EVALUATION_DIR / "top_k_evaluation_results.csv"

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "top_k",
        "hit_rate",
        "precision",
        "recall",
        "mrr",
        "ndcg"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)

print("\n========== TOP-K COMPARISON ==========")

for result in all_results:

    print(
        f"Top-K: {result['top_k']:2} | "
        f"Hit: {result['hit_rate']:.3f} | "
        f"Precision: {result['precision']:.3f} | "
        f"Recall: {result['recall']:.3f} | "
        f"MRR: {result['mrr']:.3f} | "
        f"nDCG: {result['ndcg']:.3f}"
    )

print("\nResults saved to:", output_path)
