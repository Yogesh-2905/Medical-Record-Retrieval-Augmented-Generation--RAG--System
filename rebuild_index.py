
from common.vector_store import client, COLLECTION_NAME, load_embeddings

# 1. Delete existing collection
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("Existing collection deleted.")
except Exception:
    print("No existing collection found.")

# 2. Create a fresh collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print("Fresh collection created.")

# 3. Load selected embeddings
load_embeddings("fixed_300")

# 4. Verify
print("Total records:", collection.count())
print("Index rebuilding completed!")
