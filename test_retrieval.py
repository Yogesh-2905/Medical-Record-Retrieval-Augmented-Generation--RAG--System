
from common.vector_store import collection
from common.embeddings import generate_query_embedding

# 1. Enter query
query = "What medication was prescribed to the patient?"

# 2. Generate query embedding
query_vector = generate_query_embedding(query)

print("\nQuery:", query)
print("Query vector dimensions:", len(query_vector))
print("Query vector:", query_vector)

# 3. Search ChromaDB
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=5,
    include=[
        "embeddings",
        "documents",
        "metadatas",
        "distances"
    ]
)

# 4. Display retrieved results
print("\n========== RETRIEVAL RESULTS ==========")

for i in range(len(results["ids"][0])):

    print(f"\n{'=' * 50}")
    print(f"RESULT {i + 1}")
    print(f"{'=' * 50}")

    print("Chunk ID:", results["ids"][0][i])
    print("Similarity Distance:", results["distances"][0][i])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    print("\nRetrieved Text:")
    print(results["documents"][0][i])

    # Display stored document vector
    vector = results["embeddings"][0][i]

    if vector is not None:
        print("\nStored Document Vector:")
        print(vector)

        print("Vector Dimensions:", len(vector))
    else:
        print("\nDocument vector was not returned.")

print("\n========== SEARCH COMPLETED ==========")
