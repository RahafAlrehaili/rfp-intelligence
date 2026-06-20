import json
import re
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src import config
from src.vectorstores import ChromaVectorStore


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[\u0600-\u06FF]+|[a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores

    mn = float(scores.min())
    mx = float(scores.max())

    if mx - mn < 1e-9:
        return np.zeros_like(scores)

    return (scores - mn) / (mx - mn)


def load_documents_from_json(path: Path) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    return [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in raw_docs
    ]


class HybridRetriever:
    def __init__(
        self,
        alpha: float = 0.6,
        documents_path: Path = Path("data_langchain/kb_documents.json"),
    ):
        self.alpha = alpha
        self.documents = load_documents_from_json(documents_path)

        self.chunk_id_to_doc = {
            doc.metadata.get("chunk_id"): doc
            for doc in self.documents
        }

        self.chunk_id_to_index = {
            doc.metadata.get("chunk_id"): i
            for i, doc in enumerate(self.documents)
        }

        tokenized_docs = [
            tokenize(doc.page_content)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

        store = ChromaVectorStore(
            persist_dir=config.CHROMA_DIR,
            collection_name=config.COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
        )

        self.vectorstore = store.load()

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVER_K,
        semantic_k: int = config.RETRIEVER_K,
        bm25_k: int = config.RETRIEVER_K,
        preferred_doc_type: str | None = None,
        doc_type_boost: float = 0.15,
    ) -> list[Document]:

        semantic_results = self.vectorstore.similarity_search_with_score(
            query,
            k=semantic_k,
        )

        semantic_scores_by_id = {}

        for doc, distance in semantic_results:
            chunk_id = doc.metadata.get("chunk_id")
            semantic_scores_by_id[chunk_id] = 1.0 - float(distance)

        query_tokens = tokenize(query)
        bm25_scores_all = self.bm25.get_scores(query_tokens)

        bm25_top_indices = np.argsort(bm25_scores_all)[-bm25_k:][::-1]

        candidate_ids = set(semantic_scores_by_id.keys())

        for idx in bm25_top_indices:
            chunk_id = self.documents[int(idx)].metadata.get("chunk_id")
            candidate_ids.add(chunk_id)

        candidate_ids = [
            cid for cid in candidate_ids
            if cid in self.chunk_id_to_index
        ]

        if not candidate_ids:
            return []

        semantic_arr = np.array([
            semantic_scores_by_id.get(cid, 0.0)
            for cid in candidate_ids
        ])

        bm25_arr = np.array([
            bm25_scores_all[self.chunk_id_to_index[cid]]
            for cid in candidate_ids
        ])

        semantic_norm = normalize_scores(semantic_arr)
        bm25_norm = normalize_scores(bm25_arr)

        hybrid_scores = (
            self.alpha * semantic_norm
            + (1.0 - self.alpha) * bm25_norm
        )

        if preferred_doc_type:
            for i, cid in enumerate(candidate_ids):
                doc = self.chunk_id_to_doc[cid]
                if doc.metadata.get("doc_type") == preferred_doc_type:
                    hybrid_scores[i] += doc_type_boost

        ranked_positions = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []

        for pos in ranked_positions:
            chunk_id = candidate_ids[pos]
            original_doc = self.chunk_id_to_doc[chunk_id]

            doc = Document(
                page_content=original_doc.page_content,
                metadata=dict(original_doc.metadata),
            )

            doc.metadata["semantic_score"] = round(float(semantic_arr[pos]), 4)
            doc.metadata["bm25_score"] = round(float(bm25_arr[pos]), 4)
            doc.metadata["hybrid_score"] = round(float(hybrid_scores[pos]), 4)

            results.append(doc)

        return results


if __name__ == "__main__":
    query = "What AI projects has Beamdata delivered?"

    retriever = HybridRetriever(alpha=0.6)

    docs = retriever.retrieve(
        query,
        preferred_doc_type="past_projects",
    )

    print(f"Results: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Section:", doc.metadata.get("section"))
        print("Doc Type:", doc.metadata.get("doc_type"))
        print("Semantic:", doc.metadata.get("semantic_score"))
        print("BM25:", doc.metadata.get("bm25_score"))
        print("Hybrid:", doc.metadata.get("hybrid_score"))
        print(doc.page_content[:300])