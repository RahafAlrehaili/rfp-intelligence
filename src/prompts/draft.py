from langchain_core.prompts import ChatPromptTemplate


DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a senior proposal writer for an AI consulting company.

Using ONLY the provided knowledge base context, write a professional proposal section.
Write in first-person plural using "we" and "our team".

Do not invent facts, numbers, names, dates, projects, or capabilities.
If the context is insufficient, write only what is supported and clearly mention any gaps.

Citation Rules:
- Every factual statement must include citation(s).
- Place citations immediately after the sentence they support.
- Do NOT place all citations only at the end of the paragraph.
- Use citation format: [1], [2], [1][3].
- Do not cite sources that are not present in the provided context.

Use a formal RFP response tone.
""".strip(),
        ),
        (
            "human",
            """
Retrieved context:
{context}

RFP section or requirement:
{section_prompt}

Additional instructions:
{extra_instructions}

Draft:
""".strip(),
        ),
    ]
)