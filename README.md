# Medical Record Retrieval-Augmented Generation (RAG) System

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) application for asking questions about a collection of medical-record documents.

The system does not ask the language model to answer from its general knowledge alone. It first retrieves relevant passages from the indexed medical records, prepares those passages as context, and then asks an OpenAI model to generate an answer grounded in that context. The response is instructed to cite the source document and page and to return **`I don't know.`** when the retrieved context does not contain enough information.

The project includes:
- PDF text extraction and preprocessing.
- Fixed-size, sentence-based, and recursive chunking experiments.
- BGE-small embeddings generated with FastEmbed.
- Retrieval evaluation using relevance-labelled questions.
- ChromaDB vector storage and cosine-similarity retrieval.
- Query rewriting and grounded answer generation using the OpenAI API.
- A Streamlit demonstration interface.
- Evaluation scripts for chunking, Top-K, reranking, answer latency, and storage.

## 2. Technology Stack and Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` | Extract text from PDF documents. The project imports it as `pymupdf` in the main extraction pipeline and as `fitz` in the legacy extraction helper. |
| `fastembed` | Generate BGE document/query embeddings and run the optional cross-encoder reranker experiment. |
| `tokenizers` | Tokenize text using the BGE tokenizer and enforce token-based chunk/window limits. |
| `numpy` | Store and process embedding vectors and calculate evaluation metrics. |
| `chromadb` | Persist document embeddings and metadata and perform similarity search. |
| `openai` | Rewrite user queries and generate context-grounded answers through the OpenAI Responses API. |
| `python-dotenv` | Load API configuration from the `.env` file. |
| `streamlit` | Provide the interactive web interface. |

The project also uses Python standard-library modules such as `json`, `csv`, `re`, `os`, `time`, `math`, `statistics`, and `pathlib`. These are included with Python and do not need to be listed in `requirements.txt`.

**Dependency-version note:** The uploaded ZIP did not include a `pip freeze` output or an existing dependency lock file. Therefore, `requirements.txt` lists the direct third-party dependencies without claiming unverified package versions. After confirming the project works in the target virtual environment, exact versions can be pinned for a reproducible submission.

## 3. Project File Guide

The following table explains the responsibility of each supplied Python source file.

### Main application files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit user interface. Accepts a question, calls `generate_answer()`, and displays the generated answer, fallback response, or error. |
| `generate_answer.py` | Implements the online RAG flow: loads environment variables, rewrites the question, retrieves relevant chunks, prepares source-labelled context, calls the OpenAI model, and returns the answer. |
| `rebuild_index.py` | Deletes and recreates the main ChromaDB collection, then loads the configured embedding set into it. Check the selected configuration before running; see the important configuration note below. |
| `.gitignore` | Excludes local secrets, the virtual environment, Python cache files, and IDE settings from Git. |

### `common/` — document preparation, embeddings, and storage

| File | Responsibility |
|---|---|
| `common/pdf_extraction.py` | Extracts text page by page from the source PDF, detects record IDs where available, and writes `data/processed/extracted_pages.json`. |
| `common/preprocessing.py` | Normalizes line endings, whitespace, tabs, and blank lines; writes cleaned page text to `data/processed/cleaned_pages.json`. |
| `common/chunking.py` | Creates the nine chunking experiment configurations: fixed, sentence, and recursive strategies at 300, 600, and 900 tokens, using the configured overlap. It writes `document_chunks.json`. |
| `common/embeddings.py` | Loads all nine chunk configurations, creates BGE-small 384-dimensional embeddings, handles long chunks using overlapping token windows, normalizes vectors, and saves `.npy` vectors and JSON metadata. It also creates query embeddings with the BGE retrieval instruction prefix. |
| `common/vector_store.py` | Connects to persistent ChromaDB, creates/opens the `medical_records` collection, loads embeddings and metadata, and exposes `retrieve_chunks()` for similarity retrieval. |
| `common/evaluation_dataset.py` | Defines the 28 answer-evaluation questions and relevance labels, then writes the evaluation CSV files. |

### Evaluation scripts

| File | Responsibility |
|---|---|
| `evaluate_chunking.py` | Compares the nine chunking configurations using the answerable retrieval questions and relevance labels. Reports Hit Rate, Precision, Recall, MRR, and nDCG. |
| `evaluate_top_k.py` | Compares the configured Top-K values and writes the Top-K evaluation results. |
| `evaluate_reranker.py` | Compares retrieval with reranking disabled and enabled, using the FastEmbed cross-encoder reranker. |
| `evaluate_answers.py` | Runs the RAG pipeline against the 28 answer-evaluation questions, records generated answers and latency, and leaves manual correctness/review fields for assessment. This script makes OpenAI API calls. |
| `evaluate_storage.py` | Measures the storage used by ChromaDB and the generated embedding/metadata files. |
| `phase6_summary.py` | Reads answer-evaluation output and summarizes execution, latency, and storage information. |

### Inspection and test helpers

| File | Responsibility |
|---|---|
| `check_cleaned_pages.py` | Inspects the structure and metadata of `cleaned_pages.json`. |
| `check_chunk_metadata.py` | Checks chunk metadata completeness and whether chunk token counts exceed configured limits. |
| `inspect_page_headings.py` | Prints previews of selected cleaned document pages to help inspect section mapping. |
| `test_query_embedding.py` | Small helper to test query-embedding generation. |
| `test_retrieval.py` | Tests direct retrieval from the ChromaDB collection using a generated query embedding. |
| `test_retrieval_function.py` | Tests the `retrieve_chunks()` helper. |
| `test_similarity_search.py` | Tests similarity-search behavior against saved embeddings/metadata. |
| `extraction.py` | Earlier standalone PDF extraction helper that writes one text file per page into `Output/`. The main pipeline uses `common/pdf_extraction.py` instead. |

## 4. Project Structure

The ZIP supplied for analysis contains the Python source files and `.gitignore`, but does not contain the source PDF or generated `data/` artifacts. The following is the expected working layout after adding the permitted source document and running the preparation steps:

```text
Rag task/
│
├── app.py
├── generate_answer.py
├── rebuild_index.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── common/
│   ├── __init__.py                 # Optional; add if you want an explicit package marker
│   ├── pdf_extraction.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── evaluation_dataset.py
│
├── data/
│   ├── raw/
│   │   └── Synthetic_Medical_Record_Exercise_Whitfield.pdf
│   ├── processed/
│   │   ├── extracted_pages.json
│   │   ├── cleaned_pages.json
│   │   ├── document_chunks.json
│   │   └── embeddings/
│   │       ├── fixed_300.npy
│   │       ├── fixed_300_metadata.json
│   │       └── ... other configurations ...
│   ├── evaluation/
│   │   ├── evaluation_questions.csv
│   │   ├── relevance_label.csv
│   │   └── ... evaluation result CSV files ...
│   └── chroma_db/
│
└── evaluation and inspection scripts
```

Generated data files and the ChromaDB directory are created during execution; they are not source-code files. The original ZIP did not include these data artifacts.

## 5. Installation

### Prerequisites

- Python installed (the development environment used for this project was Python 3.13).
- Internet access for installing packages and downloading embedding/tokenizer/reranker model files on first use.
- An OpenAI API key for query rewriting and answer generation.
- The permitted medical-record PDF used by the project.

### Step 1 — Open the project directory

Run commands from the project root, the folder containing `app.py` and `generate_answer.py`.

```powershell
cd "path\to\Rag task"
```

### Step 2 — Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Environment Configuration

Create a local `.env` file in the project root. Do not commit the real API key.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

`OPENAI_MODEL` is optional in the current code; if it is not set, `generate_answer.py` defaults to `gpt-4.1-mini`.

The existing `.gitignore` excludes `.env` and `.env.*` while allowing `.env.example`. For the repository, provide an `.env.example` containing placeholders only, never a real key.

## 7. Data Preparation and Offline Experiments

Place the permitted source PDF at the path expected by `common/pdf_extraction.py`:

```text
data/raw/Synthetic_Medical_Record_Exercise_Whitfield.pdf
```

Run the following commands from the project root, in order:

```powershell
python -m common.pdf_extraction
python -m common.preprocessing
python -m common.chunking
python -m common.evaluation_dataset
python -m common.embeddings
```

These steps:
1. Extract text from the PDF page by page.
2. Clean and normalize extracted text.
3. Create fixed, sentence, and recursive chunks at 300, 600, and 900 tokens, with the overlap configured in `common/chunking.py`.
4. Create the question and relevance-label CSV files.
5. Generate BGE-small embeddings for all nine chunk configurations and save vectors plus metadata.

The first embedding run may download the BGE model and tokenizer. The optional reranker evaluation may download its own model.

### Run the retrieval experiments

```powershell
python evaluate_chunking.py
python evaluate_top_k.py
python evaluate_reranker.py
```

The evaluation scripts create or replace their own named ChromaDB collections and write result CSV files under `data/evaluation/`.

### Important: selected index configuration must match the experiment decision

The project experiment results identify **`recursive_300`** as the selected chunk configuration. However, in the supplied source code:
- `rebuild_index.py` currently calls `load_embeddings("fixed_300")`.
- The standalone execution block in `common/vector_store.py` also calls `load_embeddings("fixed_300")`.

Before rebuilding the final production collection for the selected configuration, change those calls to:

```python
load_embeddings("recursive_300")
```

Then run:

```powershell
python rebuild_index.py
```

This check is important: otherwise the final `medical_records` collection can be populated with `fixed_300` even though the experiment decision was `recursive_300`. The online application reads from the `medical_records` collection and does not itself rebuild or populate it.

## 8. Run the RAG Application

After the final ChromaDB collection has been populated and the `.env` file has a valid API key, start Streamlit from the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal and open the application in a browser. Enter a question about the indexed records and submit it.

The app displays:
- The generated answer when the retrieved context supports one.
- `I don't know.` when the model is instructed that the context is insufficient.
- An error message if processing fails.

The sidebar displays the configured embedding model, vector database, similarity metric, Top-K, and reranking setting.

## 9. Run Answer and Storage Evaluation

The answer-evaluation script processes the 28-question evaluation set and calls the online RAG pipeline for each question:

```powershell
python evaluate_answers.py
```

It writes `data/evaluation/rag_answer_results.csv`. This script makes repeated OpenAI API calls and may incur API charges. The `manual_correctness` and `review_notes` columns are intended for human review; a successful execution is not by itself proof that an answer is factually correct.

Run storage measurement:

```powershell
python evaluate_storage.py
```

Generate the consolidated execution/latency/storage summary:

```powershell
python phase6_summary.py
```

## 10. Selected System Configuration

The experiment decisions recorded for this project are:

| Component | Selected configuration |
|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Chunking strategy | Recursive |
| Chunk size | 300 tokens |
| Chunk overlap | 50 tokens |
| Top-K | 8 |
| Similarity metric | Cosine similarity |
| Reranking | OFF |
| Vector database | ChromaDB |
| LLM default | `gpt-4.1-mini` (overridable using `OPENAI_MODEL`) |
| User interface | Streamlit |

The Jina cross-encoder reranker is included in the experiment code, but the selected runtime configuration is reranking OFF.

## 11. Security and Data Handling

- Never commit `.env` or expose an OpenAI API key.
- Do not publish real patient records, extracted text, embeddings, ChromaDB files, or evaluation outputs containing sensitive information unless you have explicit authorization.
- The supplied `.gitignore` does not currently exclude the `data/` directory. Review the data-sharing permissions and add appropriate data paths to `.gitignore` before pushing the repository.
- Use only synthetic, de-identified, or otherwise authorized records in a public demonstration.
- Treat generated answers as retrieval-assisted summaries, not as a substitute for professional medical judgment. Verify important details against the original record.

## 12. Known Implementation Notes

1. **Index configuration mismatch:** The supplied `rebuild_index.py` and the `__main__` block in `common/vector_store.py` load `fixed_300`, while the recorded experiment decision is `recursive_300`. Correct the configuration before rebuilding the final collection.
2. **Source data not included in the ZIP:** The uploaded archive contains source code but not the PDF or generated `data/` files. A new user must provide the permitted PDF and run the preparation workflow before using the application.
3. **Dependency versions:** No package-version lock or `pip freeze` file was included in the archive. The dependency list therefore names the direct imports without claiming exact tested versions.
4. **Legacy extraction helper:** `extraction.py` uses a hard-coded PDF filename and writes page text files to `Output/`; it is not the main JSON-based preparation pipeline.
5. **Answer quality requires review:** `evaluate_answers.py` records outputs and leaves manual correctness fields blank. Human review is needed before making claims about answer accuracy.

## 13. Limitations

- Retrieval quality depends on the completeness and quality of the supplied records and relevance labels.
- The system can only answer from the information present in the retrieved context.
- PDF text extraction may not preserve every layout feature, handwritten content, or scanned image content.
- The selected embedding model and retrieval settings were evaluated on a limited project-specific question set; results may not generalize to other medical-record collections.
- OpenAI API access and model availability are required for query rewriting and answer generation.
- The application is a project demonstration and is not a clinically validated decision-support system.

## 14. Conclusion

This project implements an end-to-end medical-record RAG workflow: document extraction, preprocessing, chunking experiments, embedding generation, retrieval evaluation, persistent vector storage, query rewriting, grounded answer generation, citations/fallback behavior, and a Streamlit interface.

The experiment-selected configuration is recursive 300-token chunking, Top-K 8, BGE-small 384-dimensional embeddings, cosine similarity, and reranking OFF. Before rebuilding the final index, ensure the configuration in the supplied index-building code is changed from `fixed_300` to `recursive_300`.
