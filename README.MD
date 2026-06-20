# RFP Intelligence System

## Overview

The RFP Intelligence System is a Retrieval-Augmented Generation (RAG) platform designed to help proposal teams search organizational knowledge, retrieve supporting evidence, and generate grounded RFP responses.

The system uses Beam Data and WeCloudData knowledge assets, proposal documents, tender documents, and past project information as its knowledge base.

---

# System Architecture

```text
Source Documents
↓
Parsing
↓
Cleaning
↓
PII Masking
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Retrieval
↓
LLM Generation
↓
Cited Response
```

---

# Project Structure

```text
RFP/

├── _source_data/
│   ├── Beam_WeCloud_RAG_Documents/
│   └── Tenders - Raw and Annotated/
│
├── data/
│   └── chroma_db/
│
├── data_langchain/
│   └── kb_documents.json
│
├── frontend/
│   └── index.html
│
└── src/
    ├── api.py
    ├── build_kb.py
    ├── chunking_langchain.py
    ├── pii_masking.py
    ├── vectorstores.py
    ├── config.py
    │
    ├── chains/
    │   ├── qa_chain.py
    │   └── draft_chain.py
    │
    ├── prompts/
    │   ├── qa.py
    │   └── draft.py
    │
    ├── retrievers/
    │   ├── semantic.py
    │   ├── hybrid.py
    │   └── reranked.py
    │
    └── evaluation/
        ├── generate_eval_questions.py
        ├── evaluate_without_reference.py
        └── wandb_logger.py
```

---

# Environment Setup

## Install UV

```powershell
pip install uv
```

or

```powershell
winget install astral-sh.uv
```

---

## Create Virtual Environment

```powershell
uv venv
```

---

## Activate Environment

```powershell
.venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```powershell
uv pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
```

---

# Configuration

Main configuration file:

```text
src/config.py
```

Key parameters:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVER_K = 20
RERANK_TOP_N = 5

LLM_MODEL = "gpt-4o-mini"
```

---

# Source Documents

Place source files under:

```text
_source_data/
```

Current folders:

```text
Beam_WeCloud_RAG_Documents
Tenders - Raw and Annotated
```

---

# Build Knowledge Base

Build or rebuild the KB:

```powershell
python -m src.build_kb
```

Rebuild whenever:

* New files are added.
* Chunking changes.
* PII masking changes.
* Metadata changes.
* Embedding model changes.

Generated outputs:

```text
data_langchain/kb_documents.json
data/chroma_db/
```

---

# Run API

```powershell
uvicorn src.api:app --reload
```

Default URL:

```text
http://127.0.0.1:8000
```

---

# Frontend

Frontend file:

```text
frontend/index.html
```

Make sure the API is running before opening the frontend.

---

# Retrieval Methods

## Semantic Retrieval

Vector similarity search using ChromaDB.

---

## Hybrid Retrieval

Combines:

```text
Semantic Similarity
+
BM25 Keyword Search
```

Default weighting:

```text
0.6 Semantic
0.4 BM25
```

---

## Reranked Retrieval

Uses Hybrid Retrieval followed by CrossEncoder reranking.

Typical reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# Example Questions

```text
What AI capabilities does the company provide?

What experience does the company have in higher education?

What technologies have been used in previous projects?

What products support Retrieval-Augmented Generation (RAG)?

What examples demonstrate workflow automation capabilities?

What AI platform deployment experience does the company have?
```

---

# Generate Evaluation Questions

```powershell
python -m src.evaluation.generate_eval_questions
```

Output:

```text
data/eval_questions_auto.json
```

Review the generated questions manually.

Save the reviewed version as:

```text
data/eval_questions_v1.json
```

---

# Run Evaluation

```powershell
python -m src.evaluation.evaluate_without_reference
```

Evaluated Retrieval Setups:

* Semantic
* Hybrid
* Hybrid + Reranker

Generated files:

```text
data/rag_eval_without_reference.csv
data/rag_eval_summary.csv
```

---

# Evaluation Metrics

### Context Precision

Measures retrieval quality.

### Faithfulness

Measures grounding and hallucination risk.

### Answer Relevancy

Measures whether the answer addresses the question.

### Citation Accuracy

Custom metric that verifies whether cited chunks support the claims in the answer.

### Latency

Average response time.

---

# Optional: Weights & Biases

Install:

```powershell
uv pip install wandb
```

Login:

```powershell
wandb login
```

Used for:

* Experiment tracking
* Retriever comparison
* Hyperparameter tracking
* Evaluation dashboards

---

# Recommended Workflow

### 1. Add Documents

```text
_source_data/
```

### 2. Build Knowledge Base

```powershell
python -m src.build_kb
```

### 3. Run API

```powershell
uvicorn src.api:app --reload
```

### 4. Test Retrieval Manually

Verify that answers and citations are correct.

### 5. Generate Evaluation Questions

```powershell
python -m src.evaluation.generate_eval_questions
```

### 6. Review Questions

Save the final benchmark as:

```text
data/eval_questions_v1.json
```

### 7. Run Evaluation

```powershell
python -m src.evaluation.evaluate_without_reference
```

### 8. Compare Results

Open:

```text
data/rag_eval_summary.csv
```

Compare:

* Semantic
* Hybrid
* Reranked

and select the best-performing retrieval strategy.

---

# Notes

* Always review generated benchmark questions before evaluation.
* Rebuild the KB after modifying parsing, chunking, masking, or metadata logic.
* GPT-4o is used as the evaluation judge.
* GPT-4o-mini is used for normal answer generation.
* Citation Accuracy is a custom grounding metric.
* Verify CSV evaluation outputs before enabling W&B tracking.
