import re

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import OpenAI

from src import config
from src.prompts.qa import QA_PROMPT
from src.retrievers.hybrid import HybridRetriever


def format_docs(docs):
    """
    Convert retrieved LangChain Documents into a formatted context string.

    This context is passed to the QA prompt.

    Each document is numbered:
        [1], [2], [3] ...

    These numbers are later used as citations in the LLM answer.
    """

    formatted = []

    print("\n" + "=" * 80)
    print("RETRIEVED DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        section = doc.metadata.get("section", "")
        hybrid_score = doc.metadata.get("hybrid_score", "")

        # Debug output to inspect which chunks were retrieved.
        print(f"\n[{i}]")
        print("SOURCE :", source)
        print("PAGE   :", page)
        print("SECTION:", section)
        print("HYBRID :", hybrid_score)
        print("TEXT   :", doc.page_content[:200].replace("\n", " "))

        # This exact numbering format is important because the prompt
        # asks the LLM to cite sources using [1], [2], etc.
        formatted.append(
            f"""
[{i}]
Source: {source}
Page: {page}
Section: {section}

{doc.page_content}
""".strip()
        )

    print("\n" + "=" * 80)
    print(f"TOTAL DOCS: {len(docs)}")
    print("=" * 80)

    return "\n\n".join(formatted)


def extract_used_citation_numbers(answer: str) -> list[int]:
    """
    Extract citation numbers from the generated answer.

    Example:
        "Beam Data delivered AI solutions [1][3]."

    Returns:
        [1, 3]
    """

    matches = re.findall(r"\[(\d+)\]", answer)

    return sorted(
        set(int(match) for match in matches)
    )


def build_used_sources(docs, answer: str):
    """
    Map citation numbers in the answer back to the retrieved documents.

    If the answer uses citation [2], this function returns metadata
    from docs[1].

    This is used by the API/UI to show the sources actually cited
    by the generated answer.
    """

    used_numbers = extract_used_citation_numbers(answer)
    sources = []

    for n in used_numbers:
        if 1 <= n <= len(docs):
            doc = docs[n - 1]

            sources.append(
                {
                    "citation_number": n,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", ""),
                    "section": doc.metadata.get("section", ""),
                    "doc_type": doc.metadata.get("doc_type", ""),
                    "text": doc.page_content,
                }
            )

    return sources


def call_openai(prompt_value):
    """
    Send the formatted LangChain prompt to OpenAI.

    LangChain prompts return message objects.
    This function converts them into the message format expected by
    the OpenAI chat completion API.
    """

    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }

    messages = prompt_value.to_messages()

    openai_messages = [
        {
            "role": role_map.get(message.type, message.type),
            "content": message.content,
        }
        for message in messages
    ]

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=openai_messages,
    )

    return response.choices[0].message.content


def build_qa_chain():
    """
    Build the QA chain used for question answering.

    Pipeline:

    User Question
        ↓
    HybridRetriever
        ↓
    Retrieved Documents
        ↓
    format_docs()
        ↓
    QA_PROMPT
        ↓
    OpenAI LLM
        ↓
    Answer + Used Sources
    """

    retriever = HybridRetriever(
        alpha=0.6
    )

    def retrieve_payload(question: str):
        """
        Retrieve relevant chunks and prepare payload for generation.
        """

        docs = retriever.retrieve(question)

        return {
            "question": question,
            "docs": docs,
            "context": format_docs(docs),
        }

    def generate(payload: dict):
        """
        Generate an answer using retrieved context and return cited sources.
        """

        prompt_value = QA_PROMPT.invoke(
            {
                "context": payload["context"],
                "question": payload["question"],
            }
        )

        answer = call_openai(
            prompt_value
        )

        used_sources = build_used_sources(
            docs=payload["docs"],
            answer=answer,
        )

        print("\n" + "=" * 80)
        print("GENERATED ANSWER")
        print("=" * 80)
        print(answer)

        print("\n" + "=" * 80)
        print("USED SOURCES FROM ANSWER")
        print("=" * 80)

        for source in used_sources:
            print(source)

        print("=" * 80)

        return {
            "answer": answer,
            "sources": used_sources,
        }

    # LangChain runnable pipeline:
    #
    # input question
    # -> retrieve_payload()
    # -> generate()
    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_payload)
        | RunnableLambda(generate)
    )

    return chain


if __name__ == "__main__":
    """
    Local test for the QA chain.

    Run:
        python src/chains/qa_chain.py
    """

    chain = build_qa_chain()

    result = chain.invoke(
        "What AI projects has Beamdata delivered?"
    )

    print("\nANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\nUSED SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(
            f"[{source['citation_number']}] "
            f"{source['source']} | "
            f"page {source['page']} | "
            f"{source['doc_type']}"
        )