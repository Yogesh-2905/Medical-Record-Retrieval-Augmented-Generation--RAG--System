import json
import numpy as np

from common.embeddings import generate_query_embedding


# 1. Load document embeddings
embedding_path = "data/processed/embeddings/fixed_300.npy"

embeddings = np.load(embedding_path)


# 2. Load corresponding metadata
metadata_path = (
    "data/processed/embeddings/fixed_300_metadata.json"
)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# 3. Enter a search question
question = "What medication was prescribed to the patient?"


# 4. Generate query embedding
query_vector = generate_query_embedding(question)


# 5. Calculate cosine similarity
similarity_scores = embeddings @ query_vector


# 6. Retrieve Top-5 chunks
top_k = 5

top_indices = np.argsort(similarity_scores)[::-1][:top_k]


# 7. Display retrieved results
print("\nQuestion:", question)
print("\nTop Retrieved Chunks:\n")

for rank, index in enumerate(top_indices, start=1):

    chunk = metadata[index]

    print(f"Rank: {rank}")
    print("Chunk ID:", chunk["chunk_id"])
    print("Page:", chunk["page"])
    print("Section:", chunk["section"])
    print("Similarity:", round(float(similarity_scores[index]), 4))
    print("Text:", chunk["text"][:300])
    print("-" * 50)