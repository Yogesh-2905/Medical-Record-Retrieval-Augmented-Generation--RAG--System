import fitz
import os

doc = fitz.open("Synthetic_Medical_Record_Exercise_Whitfield 1.pdf")

# Create output folder
os.makedirs("Output", exist_ok=True)

for page_number, page in enumerate(doc):

    text = page.get_text()

    file_name = f"Output/text{page_number + 1}.txt"

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"Page {page_number + 1} saved successfully")

doc.close()