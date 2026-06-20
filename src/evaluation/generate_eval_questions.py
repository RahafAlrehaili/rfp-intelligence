import json
from pathlib import Path

from openai import OpenAI

from src import config


INPUT_FILE = "data_langchain/kb_documents.json"
OUTPUT_FILE = "data/eval_questions_auto.json"

GENERATION_MODEL = "gpt-4o"

BATCH_SIZE = 80
QUESTIONS_PER_BATCH = 10


QUESTION_PROMPT = """
You are generating benchmark questions for evaluating a RAG system for RFP intelligence.

Generate evaluation questions WITH the supporting chunk/citation.

The questions should test:
- company overview
- product capabilities
- past projects
- higher education experience
- AI / data capabilities
- proposal-style questions
- tender scope capability summaries
- industry-specific capabilities from tender scopes
- in-scope questions
- reasoning questions
- multi-document questions
- out-of-scope questions

Rules:
- Do not generate answers.
- Avoid duplicate questions.
- Make the questions realistic for an RFP assistant.
- Cover as many different topics as possible.
- Out-of-scope questions may have empty citation fields.
- Return valid JSON only.

Format:
{
  "questions": [
    {
      "question": "...",
      "category": "...",
      "difficulty": "easy|medium|hard",
      "source": "...",
      "doc_type": "...",
      "section": "...",
      "page": "",
      "chunk_id": "...",
      "chunk_text": "...",
      "expected_citation": {
        "source": "...",
        "doc_type": "...",
        "section": "...",
        "page": "",
        "chunk_id": "..."
      }
    }
  ]
}
""".strip()


def load_kb_documents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["documents", "chunks", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unsupported kb_documents.json format")


def get_text(item):
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in [
            "page_content",
            "text",
            "content",
            "chunk_text",
        ]:
            if key in item and item[key]:
                return item[key]

    return ""


def get_metadata(item):
    if isinstance(item, dict):
        return item.get("metadata", item)

    return {}


def prepare_chunks(documents):
    usable = []

    for item in documents:
        text = get_text(item).strip()

        if len(text) < 150:
            continue

        metadata = get_metadata(item)

        usable.append({
            "text": text[:1500],
            "source": metadata.get("source", ""),
            "doc_type": metadata.get("doc_type", ""),
            "content_type": metadata.get("content_type", ""),
            "section": metadata.get("section", ""),
            "page": metadata.get("page", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "industry": metadata.get("industry", ""),
            "capabilities_text": metadata.get("capabilities_text", ""),
            "tags_text": metadata.get("tags_text", ""),
        })

    return usable


def chunk_batches(chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def build_context(chunks):
    blocks = []

    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"""
[Chunk {i}]
Source: {chunk["source"]}
Doc Type: {chunk["doc_type"]}
Content Type: {chunk["content_type"]}
Section: {chunk["section"]}
Page: {chunk["page"]}
Chunk ID: {chunk["chunk_id"]}
Industry: {chunk["industry"]}
Capabilities: {chunk["capabilities_text"]}
Tags: {chunk["tags_text"]}

Chunk Text:
{chunk["text"]}
""".strip())

    return "\n\n".join(blocks)


def generate_questions(
    context: str,
    n_questions: int,
):
    client = OpenAI(
        api_key=config.OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=0.3,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": QUESTION_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Generate {n_questions} benchmark questions from this knowledge base context.

Context:

{context}
""".strip(),
            },
        ],
    )

    return json.loads(
        response.choices[0]
        .message.content
    )


def merge_questions(results):
    seen = set()
    merged = []

    for result in results:
        for q in result.get("questions", []):
            text = q.get("question", "").strip()

            if not text:
                continue

            normalized = text.lower().strip()

            if normalized in seen:
                continue

            seen.add(normalized)
            merged.append(q)

    return {
        "questions": merged
    }


def add_manual_oos_questions(data):
    manual_questions = [
        {
            "question": "What is the capital of France?",
            "category": "out_of_scope",
            "difficulty": "easy",
            "source": "",
            "doc_type": "",
            "section": "",
            "page": "",
            "chunk_id": "",
            "chunk_text": "",
            "expected_citation": {
                "source": "",
                "doc_type": "",
                "section": "",
                "page": "",
                "chunk_id": "",
            },
        },
        {
            "question": "Who founded Google?",
            "category": "out_of_scope",
            "difficulty": "easy",
            "source": "",
            "doc_type": "",
            "section": "",
            "page": "",
            "chunk_id": "",
            "chunk_text": "",
            "expected_citation": {
                "source": "",
                "doc_type": "",
                "section": "",
                "page": "",
                "chunk_id": "",
            },
        },
        {
            "question": "What is Tesla's market capitalization?",
            "category": "out_of_scope",
            "difficulty": "medium",
            "source": "",
            "doc_type": "",
            "section": "",
            "page": "",
            "chunk_id": "",
            "chunk_text": "",
            "expected_citation": {
                "source": "",
                "doc_type": "",
                "section": "",
                "page": "",
                "chunk_id": "",
            },
        },
    ]

    existing = {
        q["question"].lower()
        for q in data["questions"]
    }

    for q in manual_questions:
        if q["question"].lower() not in existing:
            data["questions"].append(q)

    return data


def save_output(
    data,
    output_file,
):
    Path(output_file).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    print("Loading KB documents...")

    docs = load_kb_documents(INPUT_FILE)

    print(f"Loaded {len(docs)} records")

    chunks = prepare_chunks(docs)

    print(f"Prepared {len(chunks)} chunks")

    batches = list(chunk_batches(chunks, BATCH_SIZE))

    print(f"Total batches: {len(batches)}")

    all_results = []

    for i, batch in enumerate(batches, start=1):
        print(f"\nGenerating from batch {i}/{len(batches)}")

        context = build_context(batch)

        result = generate_questions(
            context=context,
            n_questions=QUESTIONS_PER_BATCH,
        )

        all_results.append(result)

    merged = merge_questions(all_results)

    merged = add_manual_oos_questions(merged)

    merged["review_required"] = True

    merged["note"] = (
        "Auto-generated benchmark "
        "questions from all KB chunks. "
        "Review manually before "
        "running evaluation."
    )

    save_output(
        merged,
        OUTPUT_FILE,
    )

    print(
        f"\nGenerated "
        f"{len(merged['questions'])} "
        f"unique questions"
    )

    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
