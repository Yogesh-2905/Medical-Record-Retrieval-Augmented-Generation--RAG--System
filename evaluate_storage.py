
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CHROMA_DIR = DATA_DIR / "chroma_db"
EMBEDDING_DIR = DATA_DIR / "processed" / "embeddings"


def get_folder_size(folder):

    total_size = 0

    if folder.exists():
        for file in folder.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

    return total_size


def format_size(size_bytes):

    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"

    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"

    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"

    return f"{size_bytes} bytes"


def evaluate_storage():

    print("========== STORAGE EVALUATION ==========")

    # Verify embedding directory
    print(f"\nEmbedding directory exists: {EMBEDDING_DIR.exists()}")

    if not EMBEDDING_DIR.exists():
        print("ERROR: Embedding directory not found!")
        return

    # ChromaDB storage
    chroma_size = get_folder_size(CHROMA_DIR)

    print("\n1. ChromaDB Storage:")
    print(format_size(chroma_size))

    # Embedding files
    npy_files = list(EMBEDDING_DIR.glob("*.npy"))
    metadata_files = list(EMBEDDING_DIR.glob("*_metadata.json"))

    npy_size = sum(file.stat().st_size for file in npy_files)
    metadata_size = sum(file.stat().st_size for file in metadata_files)

    print("\n2. Embedding Storage:")
    print(f"NumPy files: {len(npy_files)}")
    print(f"NumPy storage: {format_size(npy_size)}")

    print(f"Metadata files: {len(metadata_files)}")
    print(f"Metadata storage: {format_size(metadata_size)}")

    print(
        f"Total embedding storage: "
        f"{format_size(npy_size + metadata_size)}"
    )

    # Embedding summary
    summary_file = EMBEDDING_DIR / "embedding_summary.json"

    if summary_file.exists():
        print(
            f"Embedding summary: "
            f"{format_size(summary_file.stat().st_size)}"
        )

    # Total data storage
    total_data_size = get_folder_size(DATA_DIR)

    print("\n3. Total Data Directory Storage:")
    print(format_size(total_data_size))

    print("\n=========================================")


if __name__ == "__main__":
    evaluate_storage()
