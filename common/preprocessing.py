
import json
import re

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted_pages.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_pages.json"
)


def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Normalize Windows and other line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # Remove unnecessary spaces at line endings
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )

    # Reduce repeated spaces and tabs within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce three or more consecutive newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess_pages(input_path):

    with open(input_path, "r", encoding="utf-8") as file:
        pages = json.load(file)

    cleaned_pages = []

    for page in pages:

        original_text = page.get("text", "")

        cleaned_text = clean_text(original_text)

        processed_page = {
            "page_number": page["page_number"],
            "record_id": page.get("record_id"),
            "text": cleaned_text,
            "raw_character_count": len(original_text),
            "cleaned_character_count": len(cleaned_text),
            "is_empty": not cleaned_text.strip()
        }

        cleaned_pages.append(processed_page)

    return cleaned_pages


def save_cleaned_pages(pages, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            pages,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    cleaned_pages = preprocess_pages(INPUT_PATH)

    save_cleaned_pages(cleaned_pages, OUTPUT_PATH)

    empty_pages = [
        page["page_number"]
        for page in cleaned_pages
        if page["is_empty"]
    ]

    print("Text preprocessing completed!")
    print(f"Total pages processed: {len(cleaned_pages)}")
    print(f"Empty pages: {empty_pages}")

    print(
        "Total raw characters:",
        sum(p["raw_character_count"] for p in cleaned_pages)
    )

    print(
        "Total cleaned characters:",
        sum(p["cleaned_character_count"] for p in cleaned_pages)
    )

    print(f"Output saved to: {OUTPUT_PATH}")
