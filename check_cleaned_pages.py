import json

with open(
    "data/processed/cleaned_pages.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)

print("Top-level type:", type(data).__name__)

if isinstance(data, list):
    print("First page keys:", list(data[0].keys()))
    print("First page metadata:", {
        k: v for k, v in data[0].items() if k != "text"
    })

elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))