from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.chains.qa_chain import build_qa_chain
from src.chains.draft_chain import build_draft_chain
from src.retrievers.hybrid import HybridRetriever


app = FastAPI(
    title="RFP Intelligence API - LCEL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"


class QARequest(BaseModel):
    question: str = Field(..., min_length=3)


class DraftRequest(BaseModel):
    section_prompt: str = Field(..., min_length=5)
    extra_instructions: Optional[str] = ""


class SimilarRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=8, ge=1, le=20)
    doc_type: Optional[str] = None


_qa_chain = None
_draft_chain = None


def get_qa_chain():
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = build_qa_chain()
    return _qa_chain


def get_draft_chain():
    global _draft_chain
    if _draft_chain is None:
        _draft_chain = build_draft_chain()
    return _draft_chain


@app.get("/", include_in_schema=False)
def root():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return JSONResponse({"message": "Frontend not found"})


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "api": "lcel",
        "config": {
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "retriever_k": config.RETRIEVER_K,
        },
    }


@app.post("/api/qa")
def qa(req: QARequest):
    try:
        result = get_qa_chain().invoke(req.question)
        return {
            "question": req.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "method": "lcel_hybrid",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/draft")
def draft(req: DraftRequest):
    try:
        SEMANTIC_THRESHOLD = 0.35

        # Quick relevance check before spending LLM tokens
        probe = HybridRetriever(alpha=0.6).retrieve(
            query=req.section_prompt,
            top_k=3,
            semantic_k=3,
            bm25_k=3,
        )
        top_semantic = max(
            (d.metadata.get("semantic_score") or 0.0 for d in probe),
            default=0.0,
        )
        if top_semantic < SEMANTIC_THRESHOLD:
            return {
                "section_prompt": req.section_prompt,
                "draft_text": "The knowledge base does not contain relevant information to draft this section.",
                "sources": [],
                "word_count": 0,
                "method": "lcel_hybrid_draft",
            }

        result = get_draft_chain().invoke(
            {
                "section_prompt": req.section_prompt,
                "extra_instructions": req.extra_instructions or "",
            }
        )

        draft_text = result["draft"]

        return {
            "section_prompt": req.section_prompt,
            "draft_text": draft_text,
            "sources": result["sources"],
            "word_count": len(draft_text.split()),
            "method": "lcel_hybrid_draft",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/similar")
def similar(req: SimilarRequest):
    try:
        retriever = HybridRetriever(alpha=0.6)

        # Fetch a larger pool when filtering so we have enough after narrowing
        fetch_k = req.top_k * 4 if req.doc_type else req.top_k

        docs = retriever.retrieve(
            query=req.query,
            top_k=fetch_k,
            semantic_k=fetch_k,
            bm25_k=fetch_k,
            preferred_doc_type=req.doc_type,
        )

        # Drop results with no real semantic similarity to the query
        SEMANTIC_THRESHOLD = 0.35
        docs = [d for d in docs if (d.metadata.get("semantic_score") or 0.0) >= SEMANTIC_THRESHOLD]

        # Hard filter by doc_type then trim to requested top_k
        if req.doc_type:
            docs = [d for d in docs if d.metadata.get("doc_type") == req.doc_type]
        docs = docs[: req.top_k]

        items = []

        for i, doc in enumerate(docs, start=1):
            items.append(
                {
                    "rank": i,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                    "section": doc.metadata.get("section", ""),
                    "doc_type": doc.metadata.get("doc_type", ""),
                    "score": doc.metadata.get("hybrid_score", 0.0),
                    "hybrid_score": doc.metadata.get("hybrid_score"),
                    "semantic_score": doc.metadata.get("semantic_score"),
                    "bm25_score": doc.metadata.get("bm25_score"),
                }
            )

        return {
            "query": req.query,
            "doc_type_filter": req.doc_type,
            "n_results": len(items),
            "items": items,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reload")
def reload():
    global _qa_chain, _draft_chain
    _qa_chain = None
    _draft_chain = None
    return {"status": "ok", "message": "LCEL chains cleared"}