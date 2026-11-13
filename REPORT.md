# Medical Record Retrieval-Augmented Generation (RAG) System

## 1. Project Overview

The Medical Record RAG System is designed to answer questions about a
collection of medical-record documents by retrieving relevant passages
and using a Large Language Model (LLM) to generate answers grounded in
those passages.

Instead of asking an LLM to answer from its general training knowledge,
the system first searches the indexed medical records. The retrieved
text is then supplied as context to the LLM. The generated response
includes source references where available, and the system is instructed
to return **"I don't know."** when the supplied context is insufficient.

### Project objectives

-   Extract and prepare text from medical-record PDF documents.
-   Compare chunking strategies and chunk sizes using a
    relevance-labelled retrieval evaluation set.
-   Generate vector embeddings for document chunks and user queries.
-   Select retrieval settings based on measured retrieval performance.
-   Store the selected document vectors and metadata in ChromaDB.
-   Build a RAG query pipeline with query rewriting, retrieval, context
    preparation, grounded answer generation, citations and fallback
    handling.
-   Evaluate retrieval quality, answer behaviour, storage and response
    latency.
-   Provide a Streamlit interface for demonstrating the system.

## 2. System Architecture

The project has two connected workflows:

1.  **Offline document preparation and retrieval experiments:**
    documents are extracted, cleaned, chunked, embedded and evaluated.
    The best-performing configuration is selected before the final
    vectors are stored in ChromaDB.
2.  **Online question-answering workflow:** a user question is
    rewritten, embedded, used to retrieve relevant chunks, and passed
    with the retrieved context to the LLM for grounded answer
    generation.

### 2.1 End-to-end architecture

``` mermaid
flowchart TD
    A[Medical Record PDF Documents] --> B[PDF Text Extraction]
    B --> C[Text Preprocessing]
    C --> D[Candidate Chunking Strategies]

    D --> D1[Fixed: 300, 600, 900]
    D --> D2[Sentence: 300, 600, 900]
    D --> D3[Recursive: 300, 600, 900]

    D1 --> E[BGE Embedding Generation]
    D2 --> E
    D3 --> E

    E --> F[Chunking Evaluation]
    F --> G[Select recursive_300]

    G --> H[Top-K Evaluation]
    H --> I[Select Top-K = 8]

    I --> J[Reranker Evaluation]
    J --> K[Select Reranking OFF]

    K --> L[Store Final Selected Embeddings and Metadata]
    L --> M[(ChromaDB: medical_records)]

    U[User Question] --> V[Query Rewriting]
    V --> W[Query Embedding]
    W --> X[ChromaDB Similarity Search]
    M --> X
    X --> Y[Retrieve Top-K = 8 Chunks]
    Y --> Z[Context Preparation]
    Z --> AA[OpenAI LLM]
    AA --> AB[Grounded Answer Generation]
    AB --> AC[Answer with Citations]
    AB --> AD[Fallback: I don't know.]

    AC --> AE[Streamlit Demo UI]
    AD --> AE
```

**Important ordering:** embeddings are generated for the candidate
chunking configurations before chunking evaluation. The `recursive_300`
configuration is selected from the evaluation results. Top-K and
reranker experiments are then performed. Only after those decisions are
made are the final selected chunks, embeddings and metadata stored in
ChromaDB.

## 3. Document Preparation and Indexing

### 3.1 PDF text extraction

Medical-record PDFs are processed page by page using PyMuPDF. Text and
relevant page/document identifiers are extracted so that retrieved
passages can later be associated with their source.

The extracted page-level content is saved as structured data for
subsequent preprocessing and chunking.

**Purpose:** preserve the text and source information needed for
retrieval and citations.

### 3.2 Text preprocessing

Extracted text is normalized before chunking. The preprocessing includes
handling line endings, non-breaking spaces, tabs, repeated whitespace
and excessive blank lines.

**Purpose:** reduce formatting noise while retaining the underlying
medical-record content.

### 3.3 Candidate chunking strategies

The project compared three chunking strategies at three chunk sizes:

  Strategy    Chunk sizes evaluated
  ----------- -----------------------
  Fixed       300, 600, 900
  Sentence    300, 600, 900
  Recursive   300, 600, 900

The overlap used was 50. Each candidate configuration was prepared for
embedding and retrieval evaluation.

### 3.4 Embedding generation

The embedding model used was **`BAAI/bge-small-en-v1.5`**, through
FastEmbed.

  ------------------------------------------------------------------------------------------------
  Setting                             Value
  ----------------------------------- ------------------------------------------------------------
  Embedding model                     `BAAI/bge-small-en-v1.5`

  Embedding dimension                 384

  Chunking overlap                    50

  Query instruction                   `Represent this sentence for searching relevant passages:`

  Vector normalization                L2 normalization
  ------------------------------------------------------------------------------------------------

The same embedding model and compatible query-embedding process are used
for document chunks and user queries. Long text is handled using
token-aware windows within the model's input limit, with window
embeddings combined for the chunk representation.

## 4. Retrieval Experiments and Configuration Selection

A relevance-labelled set of **24 retrieval evaluation questions** was
used to compare the candidate chunking configurations and retrieval
settings. The evaluation used Hit Rate, Precision, Recall, Mean
Reciprocal Rank (MRR) and normalized Discounted Cumulative Gain (nDCG).

### 4.1 Chunking strategy and size evaluation

All nine candidate configurations were evaluated after embedding
generation.

  ------------------- ------------ ------------ ------------ ------------ ------------
  Configuration           Hit Rate    Precision       Recall          MRR         nDCG
  ------------------- ------------ ------------ ------------ ------------ ------------
  fixed_300                  1.000        0.242        0.951        0.774        0.815

  fixed_600                  0.958        0.250        0.938        0.800        0.793

  fixed_900                  0.917        0.242        0.903        0.819        0.798

  sentence_300               1.000        0.250        0.965        0.772        0.815

  sentence_600               0.917        0.242        0.896        0.833        0.806

  sentence_900               0.917        0.242        0.903        0.819        0.798

  **recursive_300**      **1.000**    **0.250**    **0.965**    **0.810**    **0.827**

  recursive_600              0.917        0.242        0.896        0.875        0.822

  recursive_900              0.917        0.242        0.903        0.819        0.798


**Selected configuration: `recursive_300`.**

The recursive 300 configuration achieved the highest nDCG (0.827) while
also achieving Hit Rate 1.000, Recall 0.965 and Precision 0.250.
Although `recursive_600` achieved the highest MRR (0.875), its Hit Rate,
Recall and nDCG were lower. `recursive_300` was therefore selected as
the overall ranking-quality and retrieval-coverage balance for the next
experiments.

### 4.2 Top-K retrieval evaluation

After selecting `recursive_300`, different retrieval counts were
compared to determine how many chunks should be passed forward from the
vector search.

    Top-K    Hit Rate   Precision      Recall         MRR        nDCG
  ------- ----------- ----------- ----------- ----------- -----------
        1       0.708       0.708       0.611       0.708       0.673
        3       0.917       0.333       0.826       0.792       0.764
        5       1.000       0.250       0.965       0.810       0.827
      **8       1.000       0.167       1.000       0.810       0.834**
       10       1.000       0.133       1.000       0.810       0.834
       15       1.000       0.089       1.000       0.810       0.834

**Selected Top-K: 8.**

K = 8 achieved full Recall (1.000), Hit Rate (1.000) and the highest
observed nDCG (0.834). Increasing K to 10 or 15 did not improve Recall,
MRR or nDCG, while Precision decreased. K = 8 was selected as the
retrieval setting.

### 4.3 Reranker evaluation

The reranker experiment compared the retrieval pipeline with reranking
disabled and enabled, using the selected chunking configuration and
Top-K setting.

  Metric          Reranker OFF       Reranker ON
  -----------     --------------     -------------
  Hit Rate             1.000           1.000
  Precision            0.167           0.167
  Recall               1.000           1.000
  MRR                **0.810**         0.743
  nDCG               **0.834**         0.783

**Selected configuration: reranking OFF.**

Enabling the reranker did not improve Hit Rate, Precision or Recall. It
reduced MRR from 0.810 to 0.743 and nDCG from 0.834 to 0.783. Reranking
was therefore disabled in the final retrieval configuration. The Jina
reranker was evaluated experimentally but is not active in the selected
runtime configuration.

### 4.4 Final selected retrieval configuration

  Component                    Selected setting
  ---------------------------- --------------------------
  Chunking strategy and size   `recursive_300`
  Chunk overlap                 50
  Embedding model              `BAAI/bge-small-en-v1.5`
  Embedding dimension          384
  Similarity metric            Cosine similarity
  Top-K                        8
  Reranking                    OFF
  Vector database              ChromaDB

## 5. Vector Storage with ChromaDB

After selecting the chunking, embedding and retrieval settings, the
final selected chunk embeddings and associated metadata are stored in a
persistent ChromaDB collection.

  -----------------------------------------------------------------------
  Setting                             Value
  -----------------------------------------------------------------------
  Vector database                     ChromaDB

  Persistence location                `data/chroma_db`

  Collection                          `medical_records`

  Distance space                      Cosine

  Stored information                  Chunk text, embeddings, identifiers
                                      and metadata
  -----------------------------------------------------------------------

Metadata such as document ID, page number and section information
supports source traceability and citation generation.

The application startup log observed during the Streamlit launch
reported that the `medical_records` collection was connected and
contained **56 existing records** at that time. This is a runtime
observation; the final indexed record count should be confirmed against
the current collection before submission.

## 6. Online RAG Question-Answering Workflow

Once the vector collection is available, the system processes user
questions through the following sequence.

### 6.1 User question

The user submits a natural-language question about the medical records.

### 6.2 Query rewriting

The question is sent through the query-rewriting step using the OpenAI
API. The rewritten query is intended to improve retrieval wording while
preserving the original question's intent.

### 6.3 Query embedding

The rewritten query is embedded using the same BGE embedding model and
compatible query instruction used by the retrieval pipeline. This
produces a normalized 384-dimensional query vector.

### 6.4 Similarity retrieval

The query vector is compared with vectors stored in ChromaDB using
cosine distance. The system retrieves the selected Top-K of 8 relevant
chunks, along with their text and metadata.

### 6.5 Context preparation

The retrieved passages are assembled into context for the LLM. Document
identifiers and page information are retained so the response can
reference its sources.

### 6.6 Grounded answer generation

The original user question and retrieved context are passed to the
OpenAI LLM. The prompt instructs the model to answer using the supplied
context, preserve important medical details and negations, and avoid
unsupported claims based on outside knowledge.

### 6.7 Citation generation

The answer includes source references using the retrieved document
identifiers and page numbers where available.

### 6.8 Fallback handling

If no chunks are retrieved, the application-level fallback is used. If
the retrieved context is insufficient, the LLM is instructed to respond:

**"I don't know."**

This fallback instruction is evaluated as part of answer testing; it
should not be interpreted as a guarantee that every generated answer
will always be correct.

## 7. Answer Evaluation

Answer evaluation used a separate set of **28 questions**, distinct from
the 24 retrieval evaluation questions.

  Question category          Count
  ------------------------ -------
  Answerable questions          24
  Unanswerable questions         4
  Total                         28

The evaluation involved running the questions through the RAG pipeline
and reviewing the generated responses. Unanswerable questions were
intended to test whether the system avoids inventing information and
returns the fallback response when the available records do not contain
sufficient evidence.

### Execution and latency summary

  Measurement                        Result
  -------------------------- --------------
  Total questions executed               28
  Successful executions                  28
  Average latency              4.40 seconds
  Minimum latency              3.07 seconds
  Maximum latency              7.74 seconds

Successful execution means the pipeline completed for the questions; it
does not by itself mean every answer was factually correct. Answer
correctness and citation support should be assessed from the
answer-review results.

## 8. Storage and Performance Evaluation

  Measurement                     Result
  ----------------------- --------------
  ChromaDB storage              17.53 MB
  Total data storage            19.45 MB
  Average query latency     4.40 seconds
  Minimum query latency     3.07 seconds
  Maximum query latency     7.74 seconds

These figures are the recorded project evaluation results. Latency can
vary with hardware, network conditions, API response time and whether
model or database initialization is included in a measurement.

## 9. Application Demonstration

The project includes a Streamlit demonstration interface in `app.py`.
The UI calls the existing RAG function rather than implementing a
separate retrieval pipeline.

The demonstration is intended to allow a user to:

-   Enter a medical-record question.
-   Submit it to the existing RAG pipeline.
-   View the generated answer.
-   View citations included in the returned answer.
-   See the fallback response when the available context is
    insufficient.
-   Inspect retrieval details if that option is enabled in the
    application.

Run the application from the project root using:

``` bash
python -m streamlit run app.py
```

The project requires the environment variables used by the existing
application, including the OpenAI API key. A sample environment file
should document the required variable names without containing real
credentials.

## 10. Key Findings

1.  **Chunking selection:** `recursive_300` achieved the highest nDCG
    among the nine tested chunking configurations (0.827), with Hit Rate
    1.000 and Recall 0.965.
2.  **Retrieval count:** Top-K = 8 achieved Recall 1.000 and nDCG 0.834.
    Larger K values did not improve these metrics and reduced Precision.
3.  **Reranking decision:** The tested Jina reranker did not improve the
    retrieval metrics. The reranking-enabled results had lower MRR and
    nDCG, so reranking was disabled.
4.  **Final retrieval quality:** The selected configuration achieved Hit
    Rate 1.000, Recall 1.000, MRR 0.810 and nDCG 0.834 at K = 8.
5.  **Answer pipeline:** The system combines query rewriting, vector
    retrieval, context-grounded generation, citations and fallback
    handling.
6.  **Measured performance:** The recorded average latency was 4.40
    seconds across 28 question executions, with total data storage of
    19.45 MB.

## 11. Limitations and Considerations

-   **Dataset scope:** Evaluation results reflect the medical-record
    documents and evaluation questions used in this project. They do not
    establish performance on all medical records or clinical scenarios.
-   **Evaluation-set size:** The retrieval comparison used 24 questions
    and answer evaluation used 28 questions. A larger and more diverse
    evaluation set would provide stronger evidence of generalization.
-   **Answer correctness:** Successful execution is not equivalent to
    factual correctness. Answers and citations require review against
    the original records.
-   **LLM dependence:** Query rewriting and answer generation depend on
    the configured OpenAI model and API availability. Output quality and
    latency may vary.
-   **Context limitations:** If relevant evidence is not present in the
    retrieved chunks, the system may not be able to answer. The fallback
    instruction helps reduce unsupported responses but does not
    guarantee their complete elimination.
-   **Reranking:** The tested reranker did not improve the measured
    results for this dataset and configuration. It may behave
    differently with other datasets or retrieval settings.
-   **Privacy and security:** Medical records can contain sensitive
    information. Real deployments require appropriate access controls,
    secure storage, secrets management, data-handling policies and
    compliance review. The demo should use data that is authorized for
    the intended environment.
-   **Runtime record count:** The observed ChromaDB collection contained
    56 records during the Streamlit startup log. Confirm the final
    collection contents and count before publishing results.

## 12. Conclusion

This project implements an end-to-end Medical Record RAG workflow, from
PDF extraction and preprocessing to embedding-based retrieval and
context-grounded answer generation.

The retrieval configuration was selected through measured experiments
rather than assumed in advance. The final configuration uses recursive
chunking with a chunk size of 300, BGE small 384-dimensional embeddings,
cosine similarity, Top-K = 8 and reranking disabled. The selected
vectors and metadata are persisted in ChromaDB and accessed by the
online RAG pipeline.

The resulting application demonstrates how retrieval-augmented
generation can provide answers grounded in a defined collection of
medical records while retaining source references and a fallback
response for insufficient context.
