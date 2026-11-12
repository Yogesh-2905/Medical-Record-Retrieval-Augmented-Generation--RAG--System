import json
import re
from pathlib import Path

from tokenizers import Tokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

INPUT_PATH = Path("data/processed/cleaned_pages.json")

OUTPUT_PATH = Path("data/processed/document_chunks.json")

CHUNK_SIZES = [300, 600, 900]

OVERLAP_TOKENS = 50

STRATEGIES = ["fixed", "sentence", "recursive"]


# ============================================================
# 2. SECTION MAPPING
# ============================================================

# Page-level document section labels.
# These describe the document type on each PDF page.

SECTION_MAPPING = {
    1: "Prehospital Care Report",
    2: "Emergency Department Triage & Nursing Record",
    3: "Emergency Physician Note",
    4: "Diagnostic Imaging Reports",
    5: "ED Discharge Summary & Prescription Record",
    6: "Family Medicine Record",
    7: "Outpatient Physical Therapy Evaluation",
    8: "Advanced Diagnostic Imaging Report",
    9: "Spine Surgery Consultation",
    10: "Outpatient Physical Therapy Progress Note",
    11: "Ambulatory Procedure Record",
    12: "Outpatient Physical Therapy Draft Note",
    13: "Handwritten Clinical Note",
    14: "Electromyography & Nerve Conduction Study",
    15: "Department of Surgery Record",
    16: "Outpatient Physical Therapy Record - Different Patient",
    17: "Post-Operative Follow-Up",
    18: "Urgent Care Prior Record",
    19: "Patient Medication History",
    20: "Patient Financial Services Record",
    21: "Occupational / Employment Record",
    22: "Records Custodian Certification",
}


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

print("Loading BGE tokenizer...")

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

print("BGE tokenizer loaded successfully!")


# ============================================================
# 4. TOKENIZER HELPER FUNCTIONS
# ============================================================

def encode_text(text):
    """
    Convert text into token IDs.

    Special tokens are not added.
    """

    return tokenizer.encode(
        text,
        add_special_tokens=False
    ).ids


def decode_tokens(token_ids):
    """
    Convert token IDs back into readable text.
    """

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True
    )


def count_tokens(text):
    """
    Return the actual token count of text.
    """

    return len(encode_text(text))


# ============================================================
# 5. SAFE TOKEN WINDOW GENERATION
# ============================================================

def safe_token_windows(text, chunk_size, overlap_tokens):
    """
    Split text into chunks.

    Important:
    Decoding token IDs and encoding the decoded text again
    can sometimes produce a different token count.

    Therefore, every decoded chunk is rechecked before saving.
    """

    token_ids = encode_text(text)

    if not token_ids:
        return []

    chunks = []

    start = 0
    total_tokens = len(token_ids)

    while start < total_tokens:

        end = min(start + chunk_size, total_tokens)

        while end > start:

            candidate_ids = token_ids[start:end]

            candidate_text = decode_tokens(candidate_ids)

            actual_count = count_tokens(candidate_text)

            if candidate_text.strip() and actual_count <= chunk_size:
                break

            end -= 1

        if end <= start:
            raise ValueError(
                "Unable to create a valid chunk within the token limit."
            )

        chunks.append(candidate_text)

        if end >= total_tokens:
            break

        next_start = end - overlap_tokens

        # Guarantee forward progress.
        start = max(next_start, start + 1)

    return chunks


# ============================================================
# 6. FIXED-SIZE CHUNKING
# ============================================================

def fixed_chunking(text, chunk_size):
    """
    Divide text into fixed token-size chunks.

    Overlap is applied between consecutive chunks.
    """

    return safe_token_windows(
        text=text,
        chunk_size=chunk_size,
        overlap_tokens=OVERLAP_TOKENS
    )


# ============================================================
# 7. SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split text into sentence-level units.

    Sentence boundaries are identified using punctuation
    and newline characters.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# 8. RECURSIVE TEXT SPLITTING
# ============================================================

def recursive_split(text, chunk_size, separators=None):
    """
    Recursively split text using increasingly smaller separators.

    Priority:
    1. Paragraphs
    2. Newlines
    3. Sentence boundaries
    4. Semicolons
    5. Commas
    6. Spaces
    7. Token windows as final fallback
    """

    if separators is None:
        separators = [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " "
        ]

    text = text.strip()

    if not text:
        return []

    # Stop when the text fits the requested size.
    if count_tokens(text) <= chunk_size:
        return [text]

    if not separators:
        return safe_token_windows(
            text,
            chunk_size,
            OVERLAP_TOKENS
        )

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator not in text:
        return recursive_split(
            text,
            chunk_size,
            remaining_separators
        )

    # Preserve separators when splitting.
    parts = re.split(
        f"(?<={re.escape(separator)})",
        text
    )

    # If splitting did not make progress, try the next separator.
    if len(parts) <= 1:
        return recursive_split(
            text,
            chunk_size,
            remaining_separators
        )

    result = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if count_tokens(part) <= chunk_size:
            result.append(part)

        else:
            result.extend(
                recursive_split(
                    part,
                    chunk_size,
                    remaining_separators
                )
            )

    return result


# ============================================================
# 9. PACK UNITS INTO CHUNKS
# ============================================================

def pack_units(units, chunk_size):
    """
    Combine sentences or recursively split units into chunks.

    Each final chunk is checked using the actual tokenizer.
    """

    chunks = []

    current_text = ""

    for unit in units:

        unit = unit.strip()

        if not unit:
            continue

        # Handle a unit larger than the maximum chunk size.
        if count_tokens(unit) > chunk_size:

            if current_text:
                chunks.append(current_text)
                current_text = ""

            chunks.extend(
                safe_token_windows(
                    unit,
                    chunk_size,
                    OVERLAP_TOKENS
                )
            )

            continue

        if not current_text:

            current_text = unit

        else:

            candidate = current_text + " " + unit

            if count_tokens(candidate) <= chunk_size:

                current_text = candidate

            else:

                chunks.append(current_text)

                # Retain the trailing overlap from the previous chunk.
                previous_ids = encode_text(current_text)

                overlap_ids = previous_ids[-OVERLAP_TOKENS:]

                overlap_text = decode_tokens(overlap_ids)

                candidate = overlap_text + " " + unit

                # Recheck after decoding the overlap.
                if count_tokens(candidate) <= chunk_size:
                    current_text = candidate

                else:
                    current_text = unit

    if current_text:
        chunks.append(current_text)

    return chunks


# ============================================================
# 10. SENTENCE-BASED CHUNKING
# ============================================================

def sentence_chunking(text, chunk_size):
    """
    Split text into sentences and pack them into chunks.
    """

    sentences = split_sentences(text)

    return pack_units(
        units=sentences,
        chunk_size=chunk_size
    )


# ============================================================
# 11. RECURSIVE CHUNKING
# ============================================================

def recursive_chunking(text, chunk_size):
    """
    Split text recursively and pack the resulting units.
    """

    units = recursive_split(
        text=text,
        chunk_size=chunk_size
    )

    return pack_units(
        units=units,
        chunk_size=chunk_size
    )


# ============================================================
# 12. GENERATE CHUNKS FOR ONE PAGE
# ============================================================

def create_page_chunks(
    page,
    strategy,
    chunk_size
):
    """
    Generate chunks for one PDF page.

    Attach document metadata to every chunk.
    """

    page_number = page["page_number"]

    record_id = page["record_id"]

    text = page["text"]

    section = SECTION_MAPPING.get(
        page_number,
        "Unknown"
    )

    if not text.strip():
        return []

    # Select chunking strategy.
    if strategy == "fixed":

        chunk_texts = fixed_chunking(
            text,
            chunk_size
        )

    elif strategy == "sentence":

        chunk_texts = sentence_chunking(
            text,
            chunk_size
        )

    elif strategy == "recursive":

        chunk_texts = recursive_chunking(
            text,
            chunk_size
        )

    else:

        raise ValueError(
            f"Unsupported chunking strategy: {strategy}"
        )

    page_chunks = []

    for index, chunk_text in enumerate(
        chunk_texts,
        start=1
    ):

        actual_token_count = count_tokens(chunk_text)

        # Final validation.
        if actual_token_count > chunk_size:

            raise ValueError(
                f"Chunk exceeds configured size: "
                f"{strategy}, page {page_number}, "
                f"{actual_token_count} tokens"
            )

        chunk_id = (
            f"{strategy}_{chunk_size}_"
            f"p{page_number:03d}_c{index:03d}"
        )

        chunk = {

            # Required source metadata.
            "chunk_id": chunk_id,

            "doc_id": record_id,
            "page": page_number,
            "section": section,

            # Preserve previous metadata field names.
            "record_id": record_id,
            "page_number": page_number,

            "chunk_index": index,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "token_count": actual_token_count,

            # Actual clinical text.
            "text": chunk_text
        }

        page_chunks.append(chunk)

    return page_chunks


# ============================================================
# 13. RUN ALL CHUNKING EXPERIMENTS
# ============================================================

def run_chunking_experiments():

    print("\nLoading cleaned medical records...")

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        pages = json.load(file)

    if not isinstance(pages, list):
        raise ValueError(
            "Expected cleaned_pages.json to contain a list."
        )

    experiments = []

    # Three strategies × three chunk sizes = nine configurations.
    for strategy in STRATEGIES:

        for chunk_size in CHUNK_SIZES:

            print("\n" + "=" * 60)

            print(f"Strategy: {strategy}")
            print(f"Chunk size: {chunk_size}")

            all_chunks = []

            for page in pages:

                page_chunks = create_page_chunks(
                    page=page,
                    strategy=strategy,
                    chunk_size=chunk_size
                )

                all_chunks.extend(page_chunks)

            experiment = {

                "strategy": strategy,

                "chunk_size": chunk_size,

                "overlap_tokens": OVERLAP_TOKENS,

                "total_chunks": len(all_chunks),

                "chunks": all_chunks
            }

            experiments.append(experiment)

            print(
                "Total chunks:",
                len(all_chunks)
            )

            print(
                "All chunks within token limit:",
                all(
                    chunk["token_count"] <= chunk_size
                    for chunk in all_chunks
                )
            )

            print(
                "All chunks have section metadata:",
                all(
                    chunk["section"] != "Unknown"
                    for chunk in all_chunks
                )
            )

    # ========================================================
    # 14. SAVE RESULTS
    # ========================================================

    output_data = {
        "model_name": MODEL_NAME,
        "chunk_sizes": CHUNK_SIZES,
        "strategies": STRATEGIES,
        "experiments": experiments
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)

    print("CHUNKING EXPERIMENTS COMPLETED!")

    print("Total configurations:", len(experiments))

    print("Total pages processed:", len(pages))

    print("Output saved to:", OUTPUT_PATH.resolve())


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":

    run_chunking_experiments()
