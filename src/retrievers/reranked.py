from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src import config
from src.retrievers.hybrid import HybridRetriever


class RerankedRetriever:
    """
    Final retrieval layer for the RAG system.

    This retriever improves the ranking quality after hybrid retrieval.

    Pipeline:

    User Query
        ↓
    HybridRetriever
        ↓
    Candidate Chunks
        ↓
    CrossEncoder Reranker
        ↓
    Reordered Chunks
        ↓
    Top-N Final Context

    Why reranking is needed:
    - HybridRetriever is good for candidate selection.
    - CrossEncoder is better at judging query-to-chunk relevance.
    - The reranker reads the query and chunk together, then assigns a relevance score.
    """

    def __init__(
        self,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the reranked retriever.

        Args:
            reranker_model:
                CrossEncoder model used to score query-chunk pairs.
        """

        # First stage:
        # Retrieve a candidate set using semantic search + BM25.
        self.hybrid_retriever = HybridRetriever(
            alpha=0.6,
        )

        # Second stage:
        # Load CrossEncoder model for reranking.
        #
        # Unlike embeddings, CrossEncoder receives:
        # (query, chunk_text)
        # together as one pair.
        print(f"[reranker] Loading {reranker_model}...")

        self.reranker = CrossEncoder(
            reranker_model
        )

        print("[reranker] Ready.")

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        """
        Retrieve and rerank the best chunks for a user query.

        Args:
            query:
                User question.

        Returns:
            Final top-N LangChain Documents after reranking.
        """

        # ------------------------------------------------------------------
        # STEP 1: Retrieve candidate chunks using HybridRetriever.
        #
        # HybridRetriever returns top candidates based on:
        # - semantic score
        # - BM25 score
        # - hybrid score
        #
        # These are not final yet; they are only candidates.
        # ------------------------------------------------------------------
        docs = self.hybrid_retriever.retrieve(
            query=query,
            top_k=config.RETRIEVER_K,
            semantic_k=config.RETRIEVER_K,
            bm25_k=config.RETRIEVER_K,
        )

        if not docs:
            return []

        # ------------------------------------------------------------------
        # STEP 2: Build query-document pairs for the CrossEncoder.
        #
        # Each pair looks like:
        #
        # (
        #   user_query,
        #   retrieved_chunk_text
        # )
        #
        # The CrossEncoder scores each pair independently.
        # ------------------------------------------------------------------
        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        # ------------------------------------------------------------------
        # STEP 3: Predict rerank scores.
        #
        # Higher rerank_score = the chunk is more relevant to the query.
        # ------------------------------------------------------------------
        scores = self.reranker.predict(
            pairs
        )

        # ------------------------------------------------------------------
        # STEP 4: Attach rerank scores to document metadata.
        #
        # This helps with:
        # - debugging
        # - evaluation
        # - comparing hybrid score vs rerank score
        # ------------------------------------------------------------------
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = round(
                float(score),
                4,
            )

        # ------------------------------------------------------------------
        # STEP 5: Sort documents by rerank score.
        #
        # This is the final ranking step before sending context to the LLM.
        # ------------------------------------------------------------------
        docs.sort(
            key=lambda d: d.metadata["rerank_score"],
            reverse=True,
        )

        # Return only the best N chunks after reranking.
        return docs[: config.RERANK_TOP_N]


if __name__ == "__main__":
    """
    Quick local test for the RerankedRetriever.

    Run:
        python src/retrievers/reranked.py

    This block does not run when the class is imported by the API.
    """

    query = "What AI projects has Beamdata delivered?"

    retriever = RerankedRetriever()

    docs = retriever.retrieve(
        query
    )

    print(f"\nResults: {len(docs)}\n")

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Section:", doc.metadata.get("section"))
        print("Doc Type:", doc.metadata.get("doc_type"))
        print("Hybrid:", doc.metadata.get("hybrid_score"))
        print("Rerank:", doc.metadata.get("rerank_score"))
        print()
        print(doc.page_content[:500])