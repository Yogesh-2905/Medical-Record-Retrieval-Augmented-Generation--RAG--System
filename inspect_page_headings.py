import json

with open(
    "data/processed/cleaned_pages.json",
    "r",
    encoding="utf-8"
) as file:
    pages = json.load(file)

for page in pages:
    if page["page_number"] >= 6:
        preview = page["text"][:250].replace("\n", " ")

        print(
            f"\nPage {page['page_number']} "
            f"| {page['record_id']}"
        )
        print(preview)