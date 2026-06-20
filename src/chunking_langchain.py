import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


class SmartChunker:
    """
    Text chunking component used during KB construction.

    Responsibilities:
    - Split large documents into manageable chunks.
    - Remove very small chunks.
    - Remove duplicate chunks.

    Output:
        Clean chunks ready for embedding and indexing.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_length: int,
    ):
        """
        Configure chunking behavior.

        Args:
            chunk_size:
                Maximum chunk size before splitting.

            chunk_overlap:
                Number of characters shared between neighboring chunks.
                Helps preserve context across chunk boundaries.

            min_length:
                Minimum chunk length required to keep a chunk.
                Very short chunks are usually low quality retrieval candidates.
        """

        self.min_length = min_length

        # RecursiveCharacterTextSplitter attempts to split text
        # using progressively smaller separators.
        #
        # Priority:
        # 1. Paragraphs
        # 2. Lines
        # 3. Sentences
        # 4. Words
        # 5. Character-level fallback
        #
        # This helps preserve semantic meaning better than
        # fixed-length splitting.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into clean chunks.

        Process:

        Raw Text
            ↓
        Recursive Split
            ↓
        Minimum Length Filter
            ↓
        Deduplication
            ↓
        Final Chunks
        """

        # Initial chunk generation.
        basic_chunks = self.splitter.split_text(text)

        seen = set()
        final_chunks = []

        for chunk in basic_chunks:

            # Remove leading/trailing whitespace.
            chunk = chunk.strip()

            # Ignore very small chunks.
            #
            # These are often:
            # - isolated headings
            # - page artifacts
            # - low-information fragments
            if len(chunk) < self.min_length:
                continue

            # Normalize text before duplicate detection.
            #
            # Example:
            #
            # "Hello   World"
            # "hello world"
            #
            # become the same normalized string.
            normalized = re.sub(
                r"\s+",
                " ",
                chunk.lower(),
            )

            # Skip duplicate chunks.
            #
            # Duplicate content can appear because of:
            # - repeated headers/footers
            # - duplicated slides
            # - repeated sections across documents
            if normalized in seen:
                continue

            seen.add(normalized)
            final_chunks.append(chunk)

        return final_chunks