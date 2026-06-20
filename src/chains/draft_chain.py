import re

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai import OpenAI

from src import config
from src.prompts.draft import DRAFT_PROMPT
from src.retrievers.hybrid import HybridRetriever


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        formatted.append(
            f"""
[{i}]
Source: {doc.metadata.get("source", "")}
Page: {doc.metadata.get("page", "")}
Section: {doc.metadata.get("section", "")}

{doc.page_content}
""".strip()
        )

    return "\n\n".join(formatted)


def extract_used_citation_numbers(answer: str) -> list[int]:
    matches = re.findall(r"\[(\d+)\]", answer)
    return sorted(set(int(m) for m in matches))


def build_used_sources(docs, answer: str):
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
    client = OpenAI(api_key=config.OPENAI_API_KEY)

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
        temperature=0.4,
        max_tokens=700,
        messages=openai_messages,
    )

    return response.choices[0].message.content


def build_draft_chain():
    retriever = HybridRetriever(alpha=0.6)

    def retrieve_payload(payload: dict):
        section_prompt = payload["section_prompt"]
        extra_instructions = payload.get("extra_instructions", "")

        docs = retriever.retrieve(section_prompt)

        return {
            "section_prompt": section_prompt,
            "extra_instructions": extra_instructions,
            "docs": docs,
            "context": format_docs(docs),
        }

    def generate(payload: dict):
        prompt_value = DRAFT_PROMPT.invoke(
            {
                "context": payload["context"],
                "section_prompt": payload["section_prompt"],
                "extra_instructions": payload["extra_instructions"] or "None",
            }
        )

        draft = call_openai(prompt_value)

        return {
            "draft": draft,
            "sources": build_used_sources(
                docs=payload["docs"],
                answer=draft,
            ),
        }

    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_payload)
        | RunnableLambda(generate)
    )

    return chain


if __name__ == "__main__":
    chain = build_draft_chain()

    result = chain.invoke(
        {
            "section_prompt": "Describe Beamdata's AI consulting and project delivery experience.",
            "extra_instructions": "Keep it formal and suitable for an RFP response.",
        }
    )

    print("\nDRAFT")
    print("=" * 80)
    print(result["draft"])

    print("\nUSED SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(
            f"[{source['citation_number']}] "
            f"{source['source']} | page {source['page']} | {source['doc_type']}"
        )