from langchain_core.documents import Document

from src import config
from src.vectorstores import ChromaVectorStore

class SemanticRetriever:
    """
    Pure semantic retrieval using Chroma vector search.

    Retrieval Flow:
    User Query
    -> Query Embedding
    -> Chroma Similarity Search
    -> Top-K Chunks

    This retriever uses only vector similarity.

    No BM25.
    No hybrid scoring.
    No reranking.

    Primarily used for:
    - Baseline retrieval experiments
    - Comparing semantic search against hybrid retrieval
    """

    def __init__(self):
        """
        Initialize the semantic retriever.

        Loads the existing Chroma collection and exposes it
        through LangChain's retriever interface.
        """

        store = ChromaVectorStore(
            persist_dir=config.CHROMA_DIR,
            collection_name=config.COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
        )

        # Create a similarity-based retriever.
        #
        # During retrieval:
        # Query
        # -> Embedding Model
        # -> Vector Search in Chroma
        # -> Top-K Results
        self.retriever = store.as_retriever(
            k=config.RETRIEVER_K,
        )

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve the most semantically similar chunks.

        Args:
            query:
                User question.

        Returns:
            List of retrieved LangChain Documents.
        """

        return self.retriever.invoke(query)


if __name__ == "__main__":
    query = "What AI projects has Beamdata delivered?"

    print("=" * 80)
    print("SEMANTIC RETRIEVAL")
    print("=" * 80)

    retriever = SemanticRetriever()
    docs = retriever.retrieve(query)

    print(f"\nResults: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print("\n" + "-" * 80)
        print(f"Result {i}")

        # Metadata stored during KB construction.
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Section:", doc.metadata.get("section"))
        print("Doc Type:", doc.metadata.get("doc_type"))

        # Preview first part of retrieved chunk.
        print(doc.page_content[:300])
