
from common.vector_store import retrieve_chunks

query = "What medication was prescribed to the patient?"

results = retrieve_chunks(query, top_k=5)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Chunk ID:", result["chunk_id"])
    print("Cosine Distance:", result["cosine_distance"])
    print("Cosine Similarity:", result["cosine_similarity"])
    print("Metadata:", result["metadata"])
    print("Text:", result["text"][:500])
