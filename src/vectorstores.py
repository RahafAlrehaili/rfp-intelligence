from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class ChromaVectorStore:
    """
    ChromaDB wrapper used by the RAG system.

    Responsibilities:
    - Generate embeddings using a HuggingFace embedding model.
    - Build and persist the vector database.
    - Load an existing collection.
    - Expose the collection as a LangChain retriever.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

        self.collection_metadata = {
            "hnsw:space": "cosine"
        }

    def build(
        self,
        documents: list[Document],
        reset: bool = False,
    ) -> Chroma:
        if reset and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)

        self.persist_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            ids=[
                str(doc.metadata["chunk_id"])
                for doc in documents
            ],
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            collection_metadata=self.collection_metadata,
        )

    def load(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_metadata=self.collection_metadata,
        )

    def as_retriever(
        self,
        k: int = 5,
    ):
        return self.load().as_retriever(
            search_kwargs={
                "k": k
            }
        )