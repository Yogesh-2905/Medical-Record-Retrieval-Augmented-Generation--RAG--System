import pymupdf
import json
import re

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Synthetic_Medical_Record_Exercise_Whitfield.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "extracted_pages.json"
)


def extract_pdf_pages(pdf_path):

    extracted_pages = []

    # Open the PDF
    pdf_document = pymupdf.open(pdf_path)

    for page_index, page in enumerate(pdf_document):

        # Extract text from the current page
        page_text = page.get_text("text")

        # Identify record ID if present
        record_match = re.search(
            r"LTM-PI-\d{5}",
            page_text
        )

        record_id = (
            record_match.group()
            if record_match
            else None
        )

        page_data = {
            "page_number": page_index + 1,
            "record_id": record_id,
            "text": page_text,
            "character_count": len(page_text),
            "is_empty": not page_text.strip(),
        }

        extracted_pages.append(page_data)

    pdf_document.close()

    return extracted_pages


def save_extracted_pages(pages, output_path):

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

    pages = extract_pdf_pages(PDF_PATH)

    save_extracted_pages(pages, OUTPUT_PATH)

    print("PDF extraction completed!")
    print(f"Total pages: {len(pages)}")

    empty_pages = [
        page["page_number"]
        for page in pages
        if page["is_empty"]
    ]

    print(f"Empty pages: {empty_pages}")
    print(f"Output saved to: {OUTPUT_PATH}")
