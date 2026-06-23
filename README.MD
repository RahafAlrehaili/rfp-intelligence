# RFP Intelligence System

## Overview

The RFP Intelligence System is a Retrieval-Augmented Generation (RAG) platform designed to help proposal teams search organizational knowledge, retrieve supporting evidence, and generate grounded RFP responses.

The system uses BeamData and WeCloudData knowledge assets, proposal documents, tender documents, and past project information as its knowledge base.

---

## System Architecture

```text
Source Documents
↓
Multi-Format Parsing
↓
Cleaning & Metadata Extraction
↓
PII Masking for Confidential Documents
↓
Recursive Chunking
↓
Embeddings
↓
ChromaDB Vector Store
↓
Hybrid Retrieval
↓
Cross-Encoder Reranking
↓
GPT-4o-mini Generation
↓
Citation-Backed Response
```

---

## Project Structure

```text
RFP/
├── _source_data/
├── data/
│   ├── chroma_db/
│   ├── eval_runs/
│   └── evaluation outputs
├── data_langchain/
│   ├── kb_documents.json
│   └── build_log_v2.txt
├── frontend/
│   └── index.html
└── src/
    ├── api.py
    ├── build_kb.py
    ├── config.py
    ├── chains/
    ├── retrievers/
    ├── prompts/
    └── evaluation/
```

---

## Source Data

Main source collections:

- BeamData and WeCloudData company knowledge
- Past project documents
- Proposal documents
- Tender and procurement documents
- Capability and marketing materials

---

## Configuration

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVER_K = 40
RERANK_TOP_N = 5

LLM_MODEL = "gpt-4o-mini"
```

---

## Knowledge Base Construction

### PDF Pipeline
- pdfplumber
- Page-aware extraction
- Citation support

### DOCX Pipeline
- python-docx
- Section-aware parsing
- Proposal structure preservation

### Excel Pipeline
- openpyxl
- Tender requirement extraction
- Table-aware processing

### Parallel Processing

Documents are processed concurrently to improve scalability and reduce indexing time.

---

## Chunking Strategy

The project uses RecursiveCharacterTextSplitter with:

- Chunk Size: 800
- Chunk Overlap: 150
- Metadata preservation
- Duplicate chunk removal
- Section-aware chunking

---

## Retrieval Methods

### Semantic Retrieval
Vector similarity search using ChromaDB.

### Hybrid Retrieval
Semantic retrieval + BM25 keyword search.

### Reranked Retrieval
Hybrid retrieval followed by Cross-Encoder reranking.

Reranker model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Final Retrieval Configuration

```python
RETRIEVER_K = 40
RERANK_TOP_N = 5
```

This configuration produced the strongest overall balance between retrieval quality, answer quality, and citation performance.

---

## Evaluation

The project uses a ground-truth evaluation framework based on:

- RAGAS
- LLM-as-a-Judge
- Ground-truth answers
- Ground-truth contexts

### Metrics

Retrieval:
- Context Precision
- Context Recall

Generation:
- Faithfulness
- Answer Relevancy
- Semantic Similarity

Citation:
- Citation Accuracy

---

## Reranker Experiments

Multiple reranking configurations were evaluated:

- K20 Top3
- K30 Top3
- K40 Top3
- K50 Top3
- K30 Top5
- K40 Top5
- K30 Top7

Results are stored in:

```text
data/eval_runs/
```

---

## Run API

```powershell
uvicorn src.api:app --reload
```

Default URL:

```text
http://127.0.0.1:8000
```

---

## Recommended Workflow

1. Add documents to `_source_data/`
2. Build the knowledge base
3. Run the API
4. Test QA and proposal drafting
5. Run evaluation
6. Review evaluation results

---

## Key Findings

- Hybrid retrieval improved retrieval coverage.
- Reranking improved retrieval precision and citation accuracy.
- Retrieval quality had a larger impact on answer quality than prompt engineering.
- Information fragmentation across chunks was a major source of retrieval failure.

---

## Future Improvements

- Incremental indexing
- Metadata-based retrieval filtering
- Entity-aware chunking
- Expanded document coverage
- Additional reranker optimization
