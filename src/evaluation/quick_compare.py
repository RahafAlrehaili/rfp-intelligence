"""
Quick 5-question comparison across 4 setups with RAGAS metrics.
Output: data/quick_comparison_4setups.csv
"""

import asyncio
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

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

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from src import config
from src.prompts.qa import QA_PROMPT
from src.chains.qa_chain import call_openai, format_docs
from src.retrievers.semantic import SemanticRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.reranked import RerankedRetriever

JUDGE_MODEL = "gpt-4o"
OUTPUT_FILE = "data/quick_comparison_4setups.csv"

TEST_QUESTIONS = [
    {
        "question": "What AI capabilities and services does Beamdata offer?",
        "expected_answer": "Beamdata offers AI-powered solutions including data engineering, machine learning, NLP, computer vision, and intelligent automation services.",
        "ground_truth_context": "Beamdata provides end-to-end AI services including data engineering pipelines, machine learning model development, NLP solutions, computer vision applications, and intelligent process automation.",
        "category": "AI capabilities",
    },
    {
        "question": "What past projects has Beamdata successfully delivered?",
        "expected_answer": "Beamdata has delivered multiple AI and data projects for clients in government and private sectors including analytics dashboards, chatbots, and automation systems.",
        "ground_truth_context": "Beamdata has successfully delivered projects including government analytics platforms, enterprise chatbots, data warehousing solutions, and AI-powered automation systems.",
        "category": "past projects",
    },
    {
        "question": "What is the WeCloud Data platform and its main features?",
        "expected_answer": "WeCloud Data is a cloud-based data platform offering data integration, storage, analytics, and AI capabilities for enterprises.",
        "ground_truth_context": "WeCloud Data is Beamdata's cloud platform providing data integration, scalable storage, advanced analytics, and built-in AI/ML capabilities for enterprise customers.",
        "category": "product",
    },
    {
        "question": "What is the capital of France?",
        "expected_answer": "The capital of France is Paris.",
        "ground_truth_context": "This question is outside the scope of the knowledge base.",
        "category": "out of scope",
    },
    {
        "question": "What experience does Beamdata have in the higher education sector?",
        "expected_answer": "Beamdata has experience working with universities and higher education institutions on data and AI projects.",
        "ground_truth_context": "Beamdata has collaborated with higher education institutions to deliver AI and analytics solutions supporting academic research and administrative efficiency.",
        "category": "higher education",
    },
]


def generate_answer_rag(question, docs):
    ctx = format_docs(docs)
    prompt = QA_PROMPT.invoke({"context": ctx, "question": question})
    return call_openai(prompt)


def generate_answer_no_rag(question):
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    r = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are an AI assistant. Answer using only your own knowledge."},
            {"role": "user", "content": question},
        ],
    )
    return r.choices[0].message.content.strip()


async def safe_score(metric, sample):
    if metric is None:
        return None
    try:
        return await metric.single_turn_ascore(sample)
    except Exception as e:
        print(f"  [warn] {metric.__class__.__name__}: {e}")
        return None


async def main():
    print("Loading models...")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=JUDGE_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    )

    faithfulness_metric     = Faithfulness(llm=evaluator_llm)
    relevancy_metric        = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    precision_metric        = ContextPrecisionMetric(llm=evaluator_llm)
    similarity_metric       = SemanticSimilarity(embeddings=evaluator_embeddings) if SemanticSimilarity else None
    recall_metric           = ContextRecall(llm=evaluator_llm) if ContextRecall else None

    print("Loading retrievers...")
    setups = [
        {"name": "no_rag",   "retriever": None},
        {"name": "semantic",  "retriever": SemanticRetriever()},
        {"name": "hybrid",    "retriever": HybridRetriever(alpha=0.6)},
        {"name": "reranked",  "retriever": RerankedRetriever()},
    ]

    rows = []
    total = len(TEST_QUESTIONS) * len(setups)
    done  = 0

    for setup in setups:
        print(f"\n{'='*70}")
        print(f"SETUP: {setup['name'].upper()}")
        print(f"{'='*70}")

        for item in TEST_QUESTIONS:
            done += 1
            q = item["question"]
            print(f"\n[{done}/{total}] ({setup['name']}) {q[:60]}")

            t0 = time.time()
            if setup["retriever"] is None:
                docs = []
                answer = generate_answer_no_rag(q)
            else:
                docs = setup["retriever"].retrieve(q)
                answer = generate_answer_rag(q, docs)
            latency = round(time.time() - t0, 2)

            retrieved_contexts = [d.page_content for d in docs]

            sample = SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=retrieved_contexts if retrieved_contexts else [""],
                reference=item["expected_answer"],
                reference_contexts=[item["ground_truth_context"]],
            )

            similarity = await safe_score(similarity_metric, sample)
            recall     = await safe_score(recall_metric,     sample)
            faith      = await safe_score(faithfulness_metric, sample)
            relevancy  = await safe_score(relevancy_metric,  sample)
            precision  = await safe_score(precision_metric,  sample)

            def fmt(v): return f"{v:.2f}" if v is not None else "N/A"
            print(f"  Recall={fmt(recall)}  Faith={fmt(faith)}  Precision={fmt(precision)}  Latency={latency}s")

            rows.append({
                "setup":               setup["name"],
                "category":            item["category"],
                "question":            q,
                "answer":              answer,
                "latency_seconds":     latency,
                "n_docs":              len(docs),
                "semantic_similarity": similarity,
                "context_recall":      recall,
                "faithfulness":        faith,
                "answer_relevancy":    relevancy,
                "context_precision":   precision,
            })

    df = pd.DataFrame(rows)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    metric_cols = ["semantic_similarity", "context_recall",
                   "faithfulness", "answer_relevancy", "context_precision", "latency_seconds"]

    summary = df.groupby("setup")[metric_cols].mean(numeric_only=True).round(3)
    summary = summary.reindex(["no_rag", "semantic", "hybrid", "reranked"])

    print("\n" + "="*80)
    print("COMPARISON SUMMARY (5 questions average)")
    print("="*80)
    print(summary.to_string())
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
