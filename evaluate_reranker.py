import csv
import json
import numpy as np
import chromadb

from pathlib import Path
from fastembed.rerank.cross_encoder import TextCrossEncoder

from common.embeddings import generate_query_embedding

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

CONFIGURATION = "recursive_300"
COLLECTION_NAME = "medical_records_reranker"

TOP_K = 8

RERANKER_MODEL = "jinaai/jina-reranker-v1-tiny-en"

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

        if row["answerable"].strip().lower() in [
            "true", "yes", "1"
        ]:
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

print("Loaded chunks:", len(embeddings))

# --------------------------------------------------
# 6. Create ChromaDB collection
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

print("Chunks stored in ChromaDB.")

# --------------------------------------------------
# 7. Load reranker
# --------------------------------------------------

print("\nLoading reranker model...")

reranker = TextCrossEncoder(
    model_name=RERANKER_MODEL
)

print("Reranker loaded successfully!")

# --------------------------------------------------
# 8. Calculate metrics
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
# 9. Retrieve candidate chunks using BGE
# --------------------------------------------------

def retrieve_candidates(question):

    query_vector = generate_query_embedding(question)

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=collection.count(),
        include=["documents", "metadatas", "distances"]
    )

    candidates = []

    for i, chunk_text in enumerate(results["documents"][0]):

        candidates.append({
            "text": chunk_text,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return candidates

# --------------------------------------------------
# 10. Select Top-K unique pages
# --------------------------------------------------

def select_unique_pages(candidates):

    retrieved_pages = []

    for candidate in candidates:

        page = str(candidate["metadata"]["page"])

        if page not in retrieved_pages:
            retrieved_pages.append(page)

        if len(retrieved_pages) == TOP_K:
            break

    return retrieved_pages

# --------------------------------------------------
# 11. Evaluate reranker OFF and ON
# --------------------------------------------------

evaluation_modes = {
    "reranker_off": [],
    "reranker_on": []
}

for qid, question in questions.items():

    print(f"\nEvaluating question: {qid}")

    # Retrieve the same BGE candidate set for both modes
    candidates = retrieve_candidates(question)

    # ----------------------------------------------
    # Reranker OFF: Original BGE ranking
    # ----------------------------------------------

    baseline_pages = select_unique_pages(candidates)

    baseline_metrics = calculate_metrics(
        baseline_pages,
        relevance_map.get(qid, {})
    )

    evaluation_modes["reranker_off"].append(
        baseline_metrics
    )

    # ----------------------------------------------
    # Reranker ON: Rerank BGE candidates
    # ----------------------------------------------

    reranked_results = list(
        reranker.rerank(
            question,
            [candidate["text"] for candidate in candidates]
        )
    )

    # FastEmbed returns scores in original order.
    # Sort candidates by descending reranker score.

    ranked_candidates = sorted(
        zip(candidates, reranked_results),
        key=lambda item: float(item[1]),
        reverse=True
    )

    reranked_candidates = [
        item[0] for item in ranked_candidates
    ]

    reranked_pages = select_unique_pages(
        reranked_candidates
    )

    reranker_metrics = calculate_metrics(
        reranked_pages,
        relevance_map.get(qid, {})
    )

    evaluation_modes["reranker_on"].append(
        reranker_metrics
    )

    print("Baseline pages:", baseline_pages)
    print("Reranked pages:", reranked_pages)

# --------------------------------------------------
# 12. Calculate average metrics
# --------------------------------------------------

final_results = []

for mode, results in evaluation_modes.items():

    averages = {
        metric: float(np.mean([
            result[metric] for result in results
        ]))
        for metric in [
            "hit_rate",
            "precision",
            "recall",
            "mrr",
            "ndcg"
        ]
    }

    averages["mode"] = mode

    final_results.append(averages)

# --------------------------------------------------
# 13. Display comparison
# --------------------------------------------------

print("\n========== RERANKER COMPARISON ==========")

for result in final_results:

    print(
        f"\n{result['mode'].upper()}\n"
        f"Hit Rate:  {result['hit_rate']:.3f}\n"
        f"Precision: {result['precision']:.3f}\n"
        f"Recall:    {result['recall']:.3f}\n"
        f"MRR:       {result['mrr']:.3f}\n"
        f"nDCG:      {result['ndcg']:.3f}"
    )

# --------------------------------------------------
# 14. Save results
# --------------------------------------------------

output_path = (
    EVALUATION_DIR / "reranker_evaluation_results.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "mode",
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
    writer.writerows(final_results)

print("\nResults saved to:", output_path)
print("\nReranker evaluation completed!")
