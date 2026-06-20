
import asyncio
import json
import re
from pathlib import Path
import time

import pandas as pd
from openai import OpenAI

from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
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


QUESTIONS_FILE = "data/eval_questions_v1.json"
OUTPUT_FILE = "data/rag_eval_without_reference.csv"

JUDGE_MODEL = "gpt-4o"


RETRIEVAL_SETUPS = [
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
def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_questions = data.get("questions", data)
    elif isinstance(data, list):
        raw_questions = data
    else:
        raise ValueError("Unsupported questions file format")

    questions = []

    for item in raw_questions:
        if isinstance(item, str):
            questions.append(item.strip())

        elif isinstance(item, dict):
            question = item.get("question", "").strip()

            if question:
                questions.append(question)

    return questions

# def load_questions(path: str) -> list[str]:
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     raw_questions = data.get("questions", data)

#     questions = []

#     for item in raw_questions:
#         if isinstance(item, str):
#             questions.append(item)

#         elif isinstance(item, dict):
#             question = item.get("question", "").strip()

#             if question:
#                 questions.append(question)

#     return questions


def generate_answer(question: str, docs) -> str:
    context = format_docs(docs)

    prompt_value = QA_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )
   
    return call_openai(prompt_value)


def extract_sentences_with_citations(answer: str):
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    rows = []

    for sentence in sentences:

        citations = re.findall(
            r"\[(\d+)\]",
            sentence,
        )

        if not citations:
            continue

        clean_sentence = re.sub(
            r"\[\d+\]",
            "",
            sentence,
        ).strip()

        rows.append(
            {
                "claim": clean_sentence,
                "citations": [int(n) for n in citations],
            }
        )

    return rows


def call_judge(prompt: str) -> str:
    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=20,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. "
                    "Answer only YES or NO."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()



def citation_accuracy(claims, docs) -> float | None:

    if not claims:
        return None

    scores = []

    for item in claims:

        cited_contexts = []

        for n in item["citations"]:

            if 1 <= n <= len(docs):
                cited_contexts.append(
                    docs[n - 1].page_content
                )

        if not cited_contexts:
            scores.append(0.0)
            continue

        cited_context = "\n\n".join(
            cited_contexts
        )

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
async def main():

    questions = load_questions(
        QUESTIONS_FILE
    )

    print(
        f"Loaded {len(questions)} questions"
    )

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=JUDGE_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
        )
    )

    evaluator_embeddings = (
        LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
            )
        )
    )

    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy_metric = (
        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    )

    context_precision_metric = (
        LLMContextPrecisionWithoutReference(
            llm=evaluator_llm,
        )
    )

    rows = []

    for setup in RETRIEVAL_SETUPS:

        print("\n" + "=" * 100)
        print(
            f"RUNNING SETUP: {setup['name']}"
        )
        print("=" * 100)

        retriever = setup["retriever"]

        for i, question in enumerate(
            questions,
            start=1,
        ):

            print("=" * 80)
            print(
                f"[{i}/{len(questions)}] "
                f"{question}"
            )
            start_time = time.time()
            docs = retriever.retrieve(
                question
            )

            answer = generate_answer(
                question=question,
                docs=docs,
            )
            latency = time.time() - start_time
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=[
                    doc.page_content
                    for doc in docs
                ],
            )

            context_precision = await (
                context_precision_metric
                .single_turn_ascore(sample)
            )

            faithfulness = await (
                faithfulness_metric
                .single_turn_ascore(sample)
            )

            answer_relevancy = await (
                answer_relevancy_metric
                .single_turn_ascore(sample)
            )
            claims = extract_sentences_with_citations(
                answer
            )
            cite_acc = citation_accuracy(
                claims,
                docs,
            )

            row = {
                "setup": setup["name"],

                "embedding_model":
                    config.EMBEDDING_MODEL,

                "generation_model":
                    config.LLM_MODEL,

                "judge_model":
                    JUDGE_MODEL,

                "chunk_size":
                    config.CHUNK_SIZE,

                "chunk_overlap":
                    config.CHUNK_OVERLAP,

                "retriever_k":
                    config.RETRIEVER_K,

                "rerank_top_n":
                    config.RERANK_TOP_N,

                "question":
                    question,

                "answer":
                    answer,

                "latency_seconds":
                    latency,

                "n_contexts":
                    len(docs),

                "n_claims":
                    len(claims),

                "context_precision":
                    context_precision,

                "faithfulness":
                    faithfulness,

                "answer_relevancy":
                    answer_relevancy,

                "citation_accuracy":
                    cite_acc,
            }
            rows.append(row)

            print(
                {
                    "setup":
                        setup["name"],

                    "context_precision":
                        context_precision,

                    "faithfulness":
                        faithfulness,

                    "answer_relevancy":
                        answer_relevancy,

                    "citation_accuracy":
                        cite_acc,
                }
            )

    df = pd.DataFrame(rows)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    summary_df = (
        df.groupby("setup")
        .agg(
            {
                "context_precision": "mean",
                "faithfulness": "mean",
                "answer_relevancy": "mean",
                "citation_accuracy": "mean",
                "latency_seconds": "mean",
                "n_claims": "mean",
            }
        )
        .reset_index()
    )

    summary_df.to_csv(
        "data/rag_eval_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSUMMARY")
    print(summary_df)
    print("\nCONFIGURATION")
    print("=" * 80)

    print(
        {
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
    print("\nSaved -> data/rag_eval_summary.csv")


if __name__ == "__main__":
    asyncio.run(main())

