import json

with open(
    "data/processed/document_chunks.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)

for experiment in data["experiments"]:

    chunks = experiment["chunks"]
    chunk_size = experiment["chunk_size"]

    print(
        f"{experiment['strategy']} | "
        f"{chunk_size} | "
        f"Chunks: {len(chunks)}"
    )

    print(
        "  Missing section:",
        sum(not c.get("section") for c in chunks)
    )

    print(
        "  Missing doc_id:",
        sum(not c.get("doc_id") for c in chunks)
    )

    print(
        "  Missing page:",
        sum(c.get("page") is None for c in chunks)
    )

    print(
        "  Oversized chunks:",
        sum(c["token_count"] > chunk_size for c in chunks)
    )