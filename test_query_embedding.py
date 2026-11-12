from common.embeddings import generate_query_embedding

question = "What medication was prescribed to the patient?"

query_vector = generate_query_embedding(question)

print("Query:", question)
print("Embedding shape:", query_vector.shape)
print("Vector norm:", sum(query_vector ** 2) ** 0.5)
print("First 5 values:", query_vector[:5])