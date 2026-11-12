import json
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "document_chunks.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32

MAX_MODEL_TOKENS = 512
MAX_CONTENT_TOKENS = 510
WINDOW_OVERLAP = 50

EXPECTED_DIMENSION = 384


# --------------------------------------------------
# 2. Load BGE model and tokenizer
# --------------------------------------------------

print("Loading BGE model using FastEmbed...")

model = TextEmbedding(
    model_name=MODEL_NAME
)

print("Loading BGE tokenizer...")

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

print("Model and tokenizer loaded successfully!")


# --------------------------------------------------
# 3. Split long chunks into token windows
# --------------------------------------------------

def create_token_windows(text):
    """
    Split a chunk into overlapping windows.

    Each window contains at most 510 content tokens.
    This leaves room for special tokens within
    the model's 512-token input limit.
    """

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    ).ids

    if not token_ids:
        return []

    step = MAX_CONTENT_TOKENS - WINDOW_OVERLAP

    windows = []

    for start in range(0, len(token_ids), step):

        window_ids = token_ids[
            start:start + MAX_CONTENT_TOKENS
        ]

        if not window_ids:
            break

        window_text = tokenizer.decode(
            window_ids,
            skip_special_tokens=True
        )

        # Verify the decoded text still fits
        actual_tokens = len(
            tokenizer.encode(
                window_text,
                add_special_tokens=False
            ).ids
        )

        if actual_tokens > MAX_CONTENT_TOKENS:
            raise ValueError(
                "Decoded window exceeds the token limit."
            )

        windows.append(window_text)

        if start + MAX_CONTENT_TOKENS >= len(token_ids):
            break

    return windows


# --------------------------------------------------
# 4. Generate embeddings for original chunks
# --------------------------------------------------

def generate_chunk_embeddings(chunks):

    all_windows = []
    window_mapping = []

    window_counts = []

    for chunk_index, chunk in enumerate(chunks):

        windows = create_token_windows(
            chunk["text"]
        )

        if not windows:
            raise ValueError(
                f"Empty chunk: {chunk['chunk_id']}"
            )

        window_counts.append(len(windows))

        for window in windows:

            all_windows.append(window)

            window_mapping.append(chunk_index)

    print("Total embedding windows:", len(all_windows))

    # Generate embeddings using FastEmbed
    window_embeddings = np.asarray(
        list(
            model.embed(
                all_windows,
                batch_size=BATCH_SIZE
            )
        ),
        dtype=np.float32
    )

    if window_embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {window_embeddings.shape[1]}"
        )

    # Group window embeddings by original chunk
    chunk_vectors = [
        [] for _ in chunks
    ]

    for window_index, chunk_index in enumerate(window_mapping):

        chunk_vectors[chunk_index].append(
            window_embeddings[window_index]
        )

    # Combine window embeddings into one vector per chunk
    final_embeddings = []

    for vectors in chunk_vectors:

        final_vector = np.mean(
            vectors,
            axis=0
        )

        # Normalize the final vector
        norm = np.linalg.norm(final_vector)

        if norm > 0:
            final_vector = final_vector / norm

        final_embeddings.append(final_vector)

    return (
        np.asarray(final_embeddings, dtype=np.float32),
        window_counts
    )

# --------------------------------------------------
# Query embedding with BGE instruction prefix
# --------------------------------------------------

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages:"
)


def generate_query_embedding(question):

    # Add BGE retrieval instruction
    formatted_query = f"{QUERY_PREFIX} {question}"

    # Generate query embedding
    query_vector = np.asarray(
        list(
            model.embed([formatted_query])
        ),
        dtype=np.float32
    )[0]

    # Verify embedding dimension
    if query_vector.shape[0] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {query_vector.shape[0]}"
        )

    # Normalize query vector
    norm = np.linalg.norm(query_vector)

    if norm > 0:
        query_vector = query_vector / norm

    return query_vector

# --------------------------------------------------
# 5. Load all nine chunking experiments
# --------------------------------------------------

def load_chunks():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    experiments = data["experiments"]

    if len(experiments) != 9:
        raise ValueError(
            f"Expected 9 experiments, found {len(experiments)}"
        )

    return experiments


# --------------------------------------------------
# 6. Generate embeddings for all configurations
# --------------------------------------------------

def create_all_embeddings():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    experiments = load_chunks()

    summary = []

    for experiment in experiments:

        strategy = experiment["strategy"]

        chunk_size = experiment["chunk_size"]

        chunks = experiment["chunks"]

        print("\n" + "=" * 55)
        print("Strategy:", strategy)
        print("Chunk size:", chunk_size)
        print("Total chunks:", len(chunks))
        print("=" * 55)

        # Verify original chunk sizes
        for chunk in chunks:

            actual_tokens = len(
                tokenizer.encode(
                    chunk["text"],
                    add_special_tokens=False
                ).ids
            )

            if actual_tokens > chunk_size:
                raise ValueError(
                    f"Chunk exceeds configured size: "
                    f"{chunk['chunk_id']}"
                )

        # Generate embeddings
        embeddings, window_counts = generate_chunk_embeddings(
            chunks
        )

        experiment_name = f"{strategy}_{chunk_size}"

        embedding_path = (
            OUTPUT_DIR / f"{experiment_name}.npy"
        )

        metadata_path = (
            OUTPUT_DIR / f"{experiment_name}_metadata.json"
        )

        # Save embedding vectors
        np.save(
            embedding_path,
            embeddings
        )

        # Preserve original metadata and add window counts
        metadata = []

        for chunk, window_count in zip(chunks, window_counts):

            chunk_metadata = chunk.copy()

            chunk_metadata["embedding_window_count"] = window_count

            metadata.append(chunk_metadata)

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        # Record experiment details
        summary.append({

            "strategy": strategy,

            "chunk_size": chunk_size,

            "total_chunks": len(chunks),

            "total_embedding_windows": sum(window_counts),

            "embedding_dimension": int(embeddings.shape[1]),

            "embedding_shape": list(embeddings.shape),

            "embedding_file": str(embedding_path),

            "metadata_file": str(metadata_path)

        })

        print("Embedding shape:", embeddings.shape)

        print(
            "Total windows:",
            sum(window_counts)
        )

        print("Saved:", embedding_path)

    # --------------------------------------------------
    # 7. Save experiment summary
    # --------------------------------------------------

    summary_path = (
        OUTPUT_DIR / "embedding_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "embedding_model": MODEL_NAME,
                "embedding_library": "FastEmbed",
                "max_model_tokens": MAX_MODEL_TOKENS,
                "max_content_tokens": MAX_CONTENT_TOKENS,
                "window_overlap": WINDOW_OVERLAP,
                "experiments": summary
            },
            file,
            indent=4
        )

    print("\n" + "=" * 55)

    print("ALL EMBEDDINGS GENERATED SUCCESSFULLY!")

    print("Total configurations:", len(summary))

    print("Embedding dimension:", EXPECTED_DIMENSION)

    print("Output directory:", OUTPUT_DIR)

    print("=" * 55)


# --------------------------------------------------
# 8. Execute
# --------------------------------------------------

if __name__ == "__main__":
    create_all_embeddings()
