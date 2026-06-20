from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class ChromaVectorStore:
"""
ChromaDB wrapper used by the RAG system.

```
Responsibilities:
- Generate embeddings using a HuggingFace embedding model.
- Build and persist the vector database.
- Load an existing collection.
- Expose the collection as a LangChain retriever.

This component does NOT perform:
- document parsing
- chunking
- metadata extraction

It is only responsible for storing and retrieving vectorized chunks.
"""

def __init__(
    self,
    persist_dir: Path,
    collection_name: str,
    embedding_model: str,
):
    """
    Initialize vector store configuration.

    Args:
        persist_dir:
            Directory where Chroma persists the vector database.

        collection_name:
            Chroma collection name.

        embedding_model:
            HuggingFace embedding model used during indexing
            and query retrieval.
    """

    self.persist_dir = Path(persist_dir)
    self.collection_name = collection_name

    # Embedding model used to convert:
    # Text Chunk -> Vector
    #
    # Example:
    # "Beam Data provides AI services"
    # ->
    # [0.12, -0.44, 0.91, ...]
    #
    # normalize_embeddings=True allows cosine similarity
    # to behave consistently during retrieval.
    self.embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={
            "normalize_embeddings": True,

            # Number of chunks processed per embedding batch.
            # Larger batches are generally faster but require more memory.
            "batch_size": 32,
        },
    )

    # Configure Chroma to use cosine similarity
    # when comparing embedding vectors.
    self.collection_metadata = {
        "hnsw:space": "cosine"
    }

def build(
    self,
    documents: list[Document],
    reset: bool = False,
) -> Chroma:
    """
    Build or rebuild the Chroma collection.

    Each LangChain Document should contain:
    - page_content
    - metadata["chunk_id"]

    Process:
    Documents
    -> Embeddings
    -> Chroma Collection
    -> Persist to Disk

    Args:
        documents:
            Chunks to index.

        reset:
            Delete existing collection before rebuilding.

    Returns:
        Chroma vector store.
    """

    # Remove existing vector database when rebuilding
    # from scratch.
    if reset and self.persist_dir.exists():
        shutil.rmtree(self.persist_dir)

    self.persist_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma.from_documents(
        documents=documents,

        # Embedding model used during indexing.
        embedding=self.embeddings,

        # Stable unique ID for each chunk.
        ids=[
            str(doc.metadata["chunk_id"])
            for doc in documents
        ],

        collection_name=self.collection_name,
        persist_directory=str(self.persist_dir),
        collection_metadata=self.collection_metadata,
    )

def load(self) -> Chroma:
    """
    Load an existing persisted collection.

    Used during retrieval time so the system can search
    the knowledge base without rebuilding embeddings.
    """

    return Chroma(
        collection_name=self.collection_name,
        persist_directory=str(self.persist_dir),

        # Embedding model used to convert user queries
        # into vectors before similarity search.
        embedding_function=self.embeddings,

        collection_metadata=self.collection_metadata,
    )

def as_retriever(
    self,
    k: int = 5,
):
    """
    Create a LangChain retriever.

    Retrieval Flow:
    User Question
    -> Query Embedding
    -> Chroma Similarity Search
    -> Top-K Chunks

    Args:
        k:
            Number of chunks returned by vector search.

    Returns:
        LangChain retriever object.
    """

    return self.load().as_retriever(
        search_kwargs={
            "k": k
        }
    )
```
