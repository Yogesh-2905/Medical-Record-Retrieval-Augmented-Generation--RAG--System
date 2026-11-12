import chromadb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

print("ChromaDB connected successfully!")
print("Database path:", CHROMA_PATH)

COLLECTION_NAME = "medical_records"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print("Collection created successfully!")
print("Collection name:", collection.name)
print("Existing records:", collection.count())


import json
import numpy as np

def load_embeddings(config_name):
    embedding_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"

    vectors = np.load(embedding_dir / f"{config_name}.npy")

    with open(
        embedding_dir / f"{config_name}_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
    )


    collection.upsert(
        ids=[item["chunk_id"] for item in metadata],
        embeddings=vectors.tolist(),
        documents=[item["text"] for item in metadata],
        metadatas=[
            {
                key: str(item[key])
                for key in ["doc_id", "page", "section"]
            }
            for item in metadata
        ]
    )

    print(f"Loaded {len(metadata)} chunks into ChromaDB.")

if __name__ == "__main__":
    load_embeddings("fixed_300")

    print("Total chunks in collection:", collection.count())


def retrieve_chunks(query, top_k=5):

    from common.embeddings import generate_query_embedding

    query_vector = generate_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        retrieved_chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "cosine_distance": distance,
            "cosine_similarity": 1 - distance
        })

    return retrieved_chunks
