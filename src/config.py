"""
Central configuration for the RFP LangChain project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DOCS_DIR = DATA_DIR / "raw_docs"

CHROMA_DIR = DATA_DIR / "chroma_db"

CHUNKS_JSON = DATA_DIR / "chunks.json"

RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Chunking
# ============================================================

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150

MIN_CHUNK_LENGTH = 80

# ============================================================
# Embeddings
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# Chroma
# ============================================================

COLLECTION_NAME = "rfp_kb_langchain"

# ============================================================
# Retrieval
# ============================================================

RETRIEVER_K = 20

RERANK_TOP_N = 5

# ============================================================
# OpenAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.1")
)

LLM_MAX_TOKENS = int(
    os.getenv("LLM_MAX_TOKENS", "1000")
)

# ============================================================
# API
# ============================================================

API_HOST = os.getenv(
    "API_HOST",
    "0.0.0.0",
)

API_PORT = int(
    os.getenv("API_PORT", "8000"),
)

# ============================================================
# Validation
# ============================================================

def validate_config() -> list[str]:
    issues = []

    if not OPENAI_API_KEY:
        issues.append(
            "OPENAI_API_KEY is missing."
        )

    return issues


def print_config():
    print("=" * 60)
    print("RFP LangChain Configuration")
    print("=" * 60)

    print(f"ROOT_DIR:         {ROOT_DIR}")
    print(f"RAW_DOCS_DIR:     {RAW_DOCS_DIR}")
    print(f"CHROMA_DIR:       {CHROMA_DIR}")

    print(f"CHUNK_SIZE:       {CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP:    {CHUNK_OVERLAP}")

    print(f"EMBEDDING_MODEL:  {EMBEDDING_MODEL}")

    print(f"COLLECTION_NAME:  {COLLECTION_NAME}")

    print(f"RETRIEVER_K:      {RETRIEVER_K}")
    print(f"RERANK_TOP_N:     {RERANK_TOP_N}")

    print(f"LLM_MODEL:        {LLM_MODEL}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()