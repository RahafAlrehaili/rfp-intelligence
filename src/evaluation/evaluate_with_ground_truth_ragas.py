import asyncio
import json
import re
import sys
from pathlib import Path
import time
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from openai import OpenAI

from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
)

# These metric names are available in newer RAGAS versions.
# The fallback keeps the script usable if the installed RAGAS version differs.
try:
    from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecisionMetric
except ImportError:
    from ragas.metrics import LLMContextPrecisionWithoutReference as ContextPrecisionMetric

try:
    from ragas.metrics import SemanticSimilarity
except ImportError:
    SemanticSimilarity = None

try:
    from ragas.metrics import ContextRecall
except ImportError:
    ContextRecall = None

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import config
from src.prompts.qa import QA_PROMPT
from src.chains.qa_chain import call_openai, format_docs

from src.retrievers.semantic import SemanticRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.reranked import RerankedRetriever


# =============================================================================
# Evaluation settings
# =============================================================================

# This dataset should include at least:
# question, chunk_text, expected_answer
# Optional but useful: category, difficulty, source, doc_type, page, section
EVAL_FILE = "data/eval_questions_v1_with_answers.xlsm"

OUTPUT_FILE = "data/rag_eval_with_ground_truth.csv"
SUMMARY_FILE = "data/rag_eval_with_ground_truth_summary.csv"

JUDGE_MODEL = "gpt-4o"


RETRIEVAL_SETUPS = [
    {
        "name": "no_rag",
        "retriever": None,
    },
    {
        "name": "semantic",
        "retriever": SemanticRetriever(),
    },
    {
        "name": "hybrid",
        "retriever": HybridRetriever(alpha=0.6),
    },
    {
        "name": "reranked",
        "retriever": RerankedRetriever(),
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
]


# =============================================================================
# Dataset loading
# =============================================================================

def _clean_text(value: Any) -> str:
    """Normalize empty cells and whitespace without changing meaning."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_eval_dataset(path: str) -> list[dict[str, Any]]:
    """Load a ground-truth evaluation dataset from Excel, CSV, or JSON."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")

    if file_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("questions", data)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported evaluation file type: {file_path.suffix}")

    required_columns = ["question", "chunk_text", "expected_answer"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Evaluation dataset is missing required columns: "
            + ", ".join(missing)
        )

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        question = _clean_text(row.get("question"))
        expected_answer = _clean_text(row.get("expected_answer"))
        ground_truth_context = _clean_text(row.get("chunk_text"))

        if not question or not expected_answer or not ground_truth_context:
            continue

        rows.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "ground_truth_context": ground_truth_context,
                "category": _clean_text(row.get("category")),
                "difficulty": _clean_text(row.get("difficulty")),
                "source": _clean_text(row.get("source")),
                "doc_type": _clean_text(row.get("doc_type")),
                "section": _clean_text(row.get("section")),
                "page": _clean_text(row.get("page")),
                # Keep old citation for audit only. Do not rely on chunk_id after rebuilds.
                "expected_citation": _clean_text(row.get("expected_citation")),
            }
        )

    return rows


# =============================================================================
# RAG answer generation
# =============================================================================

def generate_answer(question: str, docs) -> str:
    """Generate an answer using the same QA prompt used by the API."""
    context = format_docs(docs)

    prompt_value = QA_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return call_openai(prompt_value)


def generate_answer_no_rag(question: str) -> str:
    """Baseline: call the LLM directly with no retrieved context."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. "
                    "Answer the user's question using only your own knowledge."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content.strip()


# =============================================================================
# Citation accuracy helper
# =============================================================================

def extract_sentences_with_citations(answer: str):
    """Extract answer claims that include citations like [1], [2]."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    rows = []

    for sentence in sentences:
        citations = re.findall(r"\[(\d+)\]", sentence)

        if not citations:
            continue

        clean_sentence = re.sub(r"\[\d+\]", "", sentence).strip()

        rows.append(
            {
                "claim": clean_sentence,
                "citations": [int(n) for n in citations],
            }
        )

    return rows


def call_judge(prompt: str) -> str:
    """Small strict LLM judge used only for citation accuracy."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=20,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. Answer only YES or NO.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()


def citation_accuracy(claims, docs) -> float | None:
    """Check whether each cited claim is supported by the cited retrieved chunk."""
    if not claims:
        return None

    scores = []

    for item in claims:
        cited_contexts = []

        for n in item["citations"]:
            if 1 <= n <= len(docs):
                cited_contexts.append(docs[n - 1].page_content)

        if not cited_contexts:
            scores.append(0.0)
            continue

        cited_context = "\n\n".join(cited_contexts)

        prompt = f"""
Claim:
{item["claim"]}

Cited context:
{cited_context}

Question:
Is the claim directly supported by the cited context?

Answer only YES or NO.
""".strip()

        judge_answer = call_judge(prompt)

        if judge_answer.upper().startswith("YES"):
            scores.append(1.0)
        elif judge_answer.upper().startswith("NO"):
            scores.append(0.0)
        else:
            scores.append(0.5)

    return sum(scores) / len(scores)


# =============================================================================
# RAGAS scoring helper
# =============================================================================

async def safe_score(metric, sample: SingleTurnSample) -> float | None:
    """Return None instead of failing the whole run if one metric errors."""
    if metric is None:
        return None

    try:
        return await metric.single_turn_ascore(sample)
    except Exception as e:
        print(f"  ⚠️ Metric failed: {metric.__class__.__name__}: {e}")
        return None


# =============================================================================
# Main evaluation loop
# =============================================================================

async def main():
    eval_rows = load_eval_dataset(EVAL_FILE)

    print(f"Loaded {len(eval_rows)} ground-truth evaluation rows")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
        )
    )

    # Metrics that do not require a reference answer.
    faithfulness_metric = Faithfulness(llm=evaluator_llm)

    answer_relevancy_metric = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # This uses reference context if the installed RAGAS supports it.
    context_precision_metric = ContextPrecisionMetric(llm=evaluator_llm)

    semantic_similarity_metric = (
        SemanticSimilarity(embeddings=evaluator_embeddings)
        if SemanticSimilarity is not None
        else None
    )

    context_recall_metric = (
        ContextRecall(llm=evaluator_llm)
        if ContextRecall is not None
        else None
    )

    rows = []

    for setup in RETRIEVAL_SETUPS:
        print("\n" + "=" * 100)
        print(f"RUNNING SETUP: {setup['name']}")
        print("=" * 100)

        retriever = setup["retriever"]

        for i, item in enumerate(eval_rows, start=1):
            question = item["question"]
            expected_answer = item["expected_answer"]
            ground_truth_context = item["ground_truth_context"]

            print("=" * 80)
            print(f"[{i}/{len(eval_rows)}] {question}")

            start_time = time.time()
            is_no_rag = setup["retriever"] is None

            if is_no_rag:
                docs = []
                answer = generate_answer_no_rag(question)
            else:
                docs = retriever.retrieve(question)
                answer = generate_answer(question=question, docs=docs)

            latency = time.time() - start_time

            retrieved_contexts = [doc.page_content for doc in docs]

            # RAGAS ground-truth sample.
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts if retrieved_contexts else [""],
                reference=expected_answer,
                reference_contexts=[ground_truth_context],
            )

            context_precision = await safe_score(context_precision_metric, sample)
            faithfulness = await safe_score(faithfulness_metric, sample)
            answer_relevancy = await safe_score(answer_relevancy_metric, sample)
            semantic_similarity = await safe_score(semantic_similarity_metric, sample)
            context_recall = await safe_score(context_recall_metric, sample)

            claims = extract_sentences_with_citations(answer)
            cite_acc = citation_accuracy(claims, docs)

            row = {
                "setup": setup["name"],
                "embedding_model": config.EMBEDDING_MODEL,
                "generation_model": config.LLM_MODEL,
                "judge_model": JUDGE_MODEL,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "retriever_k": config.RETRIEVER_K,
                "rerank_top_n": config.RERANK_TOP_N,
                "question": question,
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", ""),
                "source": item.get("source", ""),
                "doc_type": item.get("doc_type", ""),
                "section": item.get("section", ""),
                "page": item.get("page", ""),
                "expected_citation": item.get("expected_citation", ""),
                "expected_answer": expected_answer,
                "ground_truth_context": ground_truth_context,
                "answer": answer,
                "retrieved_contexts": "\n\n---\n\n".join(retrieved_contexts),
                "latency_seconds": latency,
                "n_contexts": len(docs),
                "n_claims": len(claims),
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "semantic_similarity": semantic_similarity,
                "citation_accuracy": cite_acc,
            }
            rows.append(row)

            print(
                {
                    "setup": setup["name"],
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "semantic_similarity": semantic_similarity,
                    "citation_accuracy": cite_acc,
                }
            )

    df = pd.DataFrame(rows)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    metric_columns = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "semantic_similarity",
        "citation_accuracy",
        "latency_seconds",
        "n_claims",
    ]

    available_metrics = [col for col in metric_columns if col in df.columns]

    summary_df = (
        df.groupby("setup")[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )

    summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    # Optional breakdown by category. Useful because Stan mentioned category consistency.
    if "category" in df.columns and df["category"].notna().any():
        category_summary = (
            df.groupby(["setup", "category"])[available_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        category_summary.to_csv(
            "data/rag_eval_with_ground_truth_by_category.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\nSUMMARY")
    print(summary_df)

    print("\nCONFIGURATION")
    print("=" * 80)
    print(
        {
            "evaluation_file": EVAL_FILE,
            "embedding_model": config.EMBEDDING_MODEL,
            "generation_model": config.LLM_MODEL,
            "judge_model": JUDGE_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retriever_k": config.RETRIEVER_K,
            "rerank_top_n": config.RERANK_TOP_N,
        }
    )

    print(f"\nSaved -> {OUTPUT_FILE}")
    print(f"Saved -> {SUMMARY_FILE}")
    print("Saved -> data/rag_eval_with_ground_truth_by_category.csv")


if __name__ == "__main__":
    asyncio.run(main())
