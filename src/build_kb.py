"""
LangChain Knowledge Base Builder for RFP Intelligence.

This version keeps your original business logic:
- ZIP-based source reading
- PDF page skip rules
- DOCX section skip/stop rules
- Cleaning
- Optional LLM-based sensitive data obfuscation before chunking
- Tags + metadata
- Global deduplication

But changes the RAG object model to LangChain:
- Chunks become langchain_core.documents.Document
- Chunking is delegated to src/chunking_langchain.py
- Chroma indexing is delegated to src/vector_store.py

Run:
    python src/build_kb_langchain.py --reset
"""

from __future__ import annotations
import openpyxl
import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI
from vectorstores import ChromaVectorStore

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None

import config
from chunking_langchain import SmartChunker

load_dotenv()

# =============================================================================
# 1) PATHS / SETTINGS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DATA_DIR = ROOT_DIR / "_source_data"
COMPANY_ZIP = SOURCE_DATA_DIR / "Beam_WeCloud_RAG_Documents.zip"
TENDERS_ZIP = SOURCE_DATA_DIR / "Tenders - Raw and Annotated.zip"
DATA_DIR = ROOT_DIR / "data_langchain"
DOCS_JSON = DATA_DIR / "kb_documents.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
USASK_RFP_ID = "USask-GenAI-730126"

SCRIPT_VERSION = "v1.0.0-langchain-documents"
PROCESSING_TIMESTAMP = datetime.now(timezone.utc).isoformat()

CHUNKER = SmartChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    min_length=config.MIN_CHUNK_LENGTH,
)

# =============================================================================
# 2) FILE RULES
# =============================================================================

COMPANY_PDF_FILES: Dict[str, Dict[str, object]] = {
    "Beamdata Past Project Descriptions.pdf": {
        "doc_type": "past_projects",
        "document_tags": ["past-project", "weclouddata", "beamdata", "company"],
        "pipeline_name": "company_pdf_past_projects_pipeline",
        "prefix": "Note: WeCloudData is a sister company of Beamdata. Projects appear under both brands.\n\n",
    },
    "Beam Data AI Hub Intro.pptx (1).pdf": {
        "doc_type": "company_intro",
        "document_tags": ["beamdata", "company", "genai", "product", "ai-hub"],
        "pipeline_name": "company_pdf_ai_hub_intro_pipeline",
        "prefix": "",
    },
}

PDF_PAGE_OVERRIDES: Dict[str, Dict[str, List[int]]] = {
    "Beam Data AI Hub Intro.pptx (1).pdf": {
        "skip_pages": [1, 2, 6, 7, 13, 24, 25, 26, 27, 28],
    },
}

DOCX_SECTION_OVERRIDES: Dict[str, Dict[str, object]] = {
    "Copy of CP-730126 - Generative Artificial Intelligence (AI) Software.docx": {
        "stop_at_section": "Tender Summary",
        "skip_sections": [
            "table of contents",
            "submission checklist",
            "questions to tender",
            "appendix",
            "pricing",
            "software as a service vendor form",
            "section planning guide",
            "point system",
            "point weighting",
            "key dates",
            "document to summit",
            "things need to mention",
            "reseach",
            "research",
            "remark",
        ],
    },
}

DEFAULT_DOCX_SKIP_SECTION_PATTERNS = {
    "table of contents",
    "tender summary",
    "submission checklist",
    "questions to tender",
    "appendix",
    "pricing",
    "software as a service vendor form",
    "section planning guide",
    "point system",
    "point weighting",
    "key dates",
    "document to summit",
    "things need to mention",
    "reseach",
    "research",
    "remark",
}
# Target Excel tender files to transform into reusable Beam Data capability summaries.
# Use normalized file stems instead of exact filenames to avoid missing files because
# of capitalization, extra spaces, or the .xlsx extension.
TENDER_EXCEL_TARGET_FILES = {
    "usask - genai software",
    "ttc - implementation of a high availability sql database solution",
    "scaleai - digital intelligence training grants for canadian businesses",
    "nrfp 5549 - learning design and authoring tool",
    "icbc bid planner - learning design and authoring tool",
    "glenbow-alberta institute - website development",
    "design and implementation of an ai-powered data platform for pakistan customs",
    
}


def normalize_file_stem(name: str) -> str:
    stem = Path(name).stem.lower().strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def slugify_tag(value: Any) -> str:
    tag = str(value or "").strip().lower()
    tag = tag.replace("/", "-").replace("&", "and")
    tag = re.sub(r"[^a-z0-9]+", "-", tag)
    tag = re.sub(r"-+", "-", tag).strip("-")
    return tag
# =============================================================================
# 3) TAGS
# =============================================================================

_TAG_MAP: Dict[str, List[str]] = {
    "genai": ["generative ai", "large language model", "llm", "gpt", "chatgpt", "genai", "foundation model"],
    "ml": ["machine learning", "deep learning", "neural network", "model training", "ai model"],
    "rag": ["retrieval augmented", "vector search", "embedding", "semantic search", "knowledge base"],
    "cloud": ["aws", "amazon web services", "azure", "google cloud", "gcp", "cloud infrastructure"],
    "microsoft": ["microsoft", "microsoft 365", "m365", "sharepoint", "teams", "azure ad", "copilot"],
    "data-pipeline": ["data pipeline", "etl", "data ingestion", "apache spark", "airflow", "data engineering"],
    "nlp": ["natural language processing", "nlp", "text analysis", "sentiment", "text classification"],
    "automation": ["automation", "automated", "rpa", "workflow automation", "process automation"],
    "government": ["government", "public sector", "municipal", "federal", "crown corporation", "procurement"],
    "education": ["university", "college", "student", "academic", "learning", "curriculum", "campus"],
    "utility": ["utility", "bcuc", "bc utilities", "energy", "hydro", "grid", "regulation"],
    "healthcare": ["health", "medical", "patient", "clinical", "hospital", "pharmacy"],
    "finance": ["financial", "banking", "payment", "billing", "invoice", "accounting"],
    "past-project": ["project description", "past project", "case study", "client engagement", "delivered"],
    "proposal": ["proposal", "we propose", "our approach", "beamdata will", "our team will"],
    "lms": ["learning management", "lms", "course", "training platform", "e-learning"],
    "compliance": ["compliance", "governance", "regulation", "audit", "policy", "gdpr", "privacy"],
    "migration": ["migration", "migrate", "transition", "modernization", "upgrade"],
    "implementation": ["implementation", "deploy", "rollout", "integration", "onboarding"],
    "data-governance": ["data governance", "data quality", "master data", "data catalog", "lineage"],
}


def auto_tag(text: str, base_tags: Optional[List[str]] = None) -> List[str]:
    tags = set(base_tags or [])
    t = text.lower()
    for tag, keywords in _TAG_MAP.items():
        if any(keyword in t for keyword in keywords):
            tags.add(tag)
    return sorted(tags)


def llm_tag(text: str, client: OpenAI, base_tags: Optional[List[str]] = None) -> List[str]:
    base = auto_tag(text, base_tags)
    tag_list = ", ".join(sorted(_TAG_MAP.keys()))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    f"Choose up to 5 relevant tags from this list for the text below.\n"
                    f"Tags: {tag_list}\n"
                    f"Return ONLY a comma-separated list of tag names.\n\n"
                    f"Text: {text[:600]}"
                ),
            }],
        )
        raw = resp.choices[0].message.content.strip().lower()
        llm_tags = [t.strip() for t in raw.split(",") if t.strip() in _TAG_MAP]
        return sorted(set(base) | set(llm_tags))
    except Exception:
        return base

# =============================================================================
# 3.5) OPTIONAL LLM-BASED SENSITIVE DATA OBFUSCATION
# =============================================================================

_MASKING_WHITELIST_STR = (
    "Beamdata, WeCloudData, Beam Data, We Cloud Data, BeamData AI, "
    "Sportradar, WCC, Globe and Mail, TrustABC, XYZ Robotics, "
    "Saudi Digital Academy, MCIT, HRDF, AI71, Ministry of Defense, "
    "Singapore Management University, SIT, StackFuel"
)

def _regex_validate_and_fix(text: str) -> str:

    patterns = [
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "[EMAIL]"
        ),
        (
            r"(?:\+?\d[\d\-\s()]{7,}\d)",
            "[PHONE]"
        ),
        (
            r"https?://\S+|www\.\S+",
            "[URL]"
        ),
    ]

    for pattern, replacement in patterns:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text
def _mask_via_llm(text: str, client: OpenAI) -> str:
    """
    Use the LLM to obfuscate sensitive information while preserving technical content.
    If the LLM call fails, return the original text so the pipeline does not crash.
    """
    prompt = (
        "You are a sensitive-data obfuscation tool.\n\n"
        "Obfuscate any sensitive information in the text below using these placeholders:\n"
        "- Personal names                      -> [PERSON_NAME]\n"
        "- Client or company names              -> [CLIENT_NAME]\n"
        "- Email addresses                      -> [EMAIL]\n"
        "- Phone numbers                        -> [PHONE]\n"
        "- Physical addresses                   -> [ADDRESS]\n"
        "- URLs                                  -> [URL]\n"
        "- Identification numbers or IDs         -> [ID]\n"
        "- Postal codes                          -> [POSTAL_CODE]\n\n"
        f"Whitelist: never obfuscate these names: {_MASKING_WHITELIST_STR}\n\n"
        "Rules:\n"
        "- Return only the obfuscated text. Do not add explanations.\n"
        "- Do not change technical or business context unless it is sensitive.\n"
        "- Preserve the original structure and formatting as much as possible.\n\n"
        f"Text:\n{text}"
    )

    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  LLM masking failed: {e} — keeping original text")
        return text


# =============================================================================
# 4) HELPERS
# =============================================================================

FOOTER_PATTERNS = [
    r"(?m)^Page \d+\s+of\s+\d+\s*$",
    r"(?m)^\s*\d{1,3}\s*$",
    r"(?m)^Confidential\s*$",
    r"(?m)^CONFIDENTIAL\s*$",
]


def clean(text: str) -> str:
    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"[-_]{8,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_zip_path(preferred_path: Path, base_name: str) -> Optional[Path]:
    if preferred_path.exists():
        return preferred_path
    candidates = list(SOURCE_DATA_DIR.glob(f"{base_name}*.zip"))
    candidates += [p for p in SOURCE_DATA_DIR.glob(f"{base_name}*") if p.is_file() and p.suffix.lower() == ".zip"]
    return candidates[0] if candidates else None


def find_zip_entry(zf: zipfile.ZipFile, target_file_name: str) -> Optional[str]:
    target_lower = target_file_name.lower()
    for entry in zf.infolist():
        if entry.is_dir():
            continue
        if Path(entry.filename).name.lower() == target_lower:
            return entry.filename
    return None


def safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma accepts only str/int/float/bool/None metadata, so serialize lists/dicts."""
    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False)
    return safe


def build_document_metadata(
    *,
    source: str,
    source_folder: str,
    doc_type: str,
    pipeline_name: str,
    document_tags: List[str],
    file_hash_value: Optional[str],
    rfp_id: Optional[str] = None,
    is_proposal: bool = False,
    content_type: Optional[str] = None,
    file_ext: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "source_folder": source_folder,
        "doc_type": doc_type,
        "content_type": content_type or "",
        "pipeline_name": pipeline_name,
        "file_ext": file_ext or "",
        "file_hash": file_hash_value or "",
        "rfp_id": rfp_id or "",
        "is_proposal": is_proposal,
        "document_tags": sorted(set(document_tags)),
        "processed_at": PROCESSING_TIMESTAMP,
        "script_version": SCRIPT_VERSION,
    }


def make_chunk_id(document_metadata: Dict[str, Any], chunk_index: int, page: Optional[int] = None, section: str = "") -> str:
    raw = "|".join([
        str(document_metadata.get("source", "")),
        str(document_metadata.get("file_hash", "")),
        str(page or ""),
        section or "",
        str(chunk_index),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class KBDocumentBuilder:
    """Builds LangChain Document objects while preserving your old metadata logic."""

    def build(
        self,
        *,
        content: str,
        document_metadata: Dict[str, Any],
        chunk_index: int,
        chunk_tags: List[str],
        page: Optional[int] = None,
        section: str = "",
    ) -> Document:
        document_tags = document_metadata.get("document_tags", []) or []
        merged_tags = sorted(set(document_tags) | set(chunk_tags))
        chunk_id = make_chunk_id(document_metadata, chunk_index, page=page, section=section)

        metadata = {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "source": document_metadata["source"],
            "source_folder": document_metadata["source_folder"],
            "doc_type": document_metadata["doc_type"],
            "content_type": document_metadata.get("content_type", ""),
            "is_proposal": document_metadata.get("is_proposal", False),
            "page": page if page is not None else "",
            "section": section or "",
            "rfp_id": document_metadata.get("rfp_id", ""),
            "tags": merged_tags,
            "tags_text": ", ".join(merged_tags),
            "sensitive_data_obfuscated": document_metadata.get("sensitive_data_obfuscated", False),
            "language": "en",
            "file_hash": document_metadata.get("file_hash", ""),
            "pipeline_name": document_metadata.get("pipeline_name", ""),
            "processed_at": document_metadata.get("processed_at", ""),
            "script_version": SCRIPT_VERSION,
            "chunking_strategy": "langchain_recursive_character_splitter",
            "chunk_size_setting": config.CHUNK_SIZE,
            "chunk_overlap_setting": config.CHUNK_OVERLAP,
            "chunk_size_chars": len(content),
            "document_metadata": document_metadata,
        }

        return Document(page_content=content, metadata=safe_metadata(metadata))


DOC_BUILDER = KBDocumentBuilder()

# =============================================================================
# 5) PDF READER
# =============================================================================


def read_pdf(path: Path, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    rules = PDF_PAGE_OVERRIDES.get(source_name or path.name, {})
    skip_pages = set(rules.get("skip_pages", []))

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if page_num in skip_pages:
                continue
            text = clean(page.extract_text() or "")
            if len(text) >= config.MIN_CHUNK_LENGTH:
                results.append({"page": page_num, "text": text})
    return results

# =============================================================================
# 6) DOCX READER
# =============================================================================


def normalize_markdown_heading(text: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text).strip()
    text = re.sub(r"<!\-\-.*?\-\->", "", text)
    text = text.replace("**", "")
    text = unescape(text)
    text = re.sub(r"\\_", "_", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" |\t")


def extract_markdown_heading(line: str) -> Optional[Tuple[int, str]]:
    match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
    if not match:
        return None
    level = len(match.group(1))
    heading = normalize_markdown_heading(match.group(2))
    if not heading:
        return None
    return level, heading


def extract_table_section_heading(line: str) -> Optional[Tuple[int, str]]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    if re.fullmatch(r"\|[\s:\-\|]+\|", stripped):
        return None

    cells = [normalize_markdown_heading(cell) for cell in stripped.strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    if len(cells) < 2:
        return None

    first_cell = cells[0]
    second_cell = cells[1]

    if re.fullmatch(r"\d{1,2}", first_cell):
        title = re.sub(r"^#+\s*", "", second_cell).strip()
        if len(title) >= 4:
            return 1, f"{first_cell} {title}"

    joined = " ".join(cells)
    if "disclosure" in joined.lower() and "judgment" in joined.lower():
        return 1, joined

    return None


def section_should_skip(heading_path: List[str], skip_patterns: set) -> bool:
    path_text = " > ".join(heading_path).lower()
    return any(pattern in path_text for pattern in skip_patterns)


def read_docx_proposal(path: Path, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
    if DocumentConverter is None:
        raise ImportError("Docling is required. Install it with: pip install docling")

    converter = DocumentConverter()
    file_name = source_name or path.name
    result = converter.convert(str(path))
    markdown_text = result.document.export_to_markdown()

    file_rules = DOCX_SECTION_OVERRIDES.get(file_name, {})
    extra_skip = {str(s).lower() for s in file_rules.get("skip_sections", [])}
    skip_patterns = DEFAULT_DOCX_SKIP_SECTION_PATTERNS | extra_skip

    stop_at_section = file_rules.get("stop_at_section")
    stop_at_section = str(stop_at_section).lower() if stop_at_section else None

    sections: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    current_heading_path: List[str] = []
    current_lines: List[str] = []
    current_keep = False
    section_index = 0

    def flush_current_section() -> None:
        nonlocal section_index
        if not current_keep:
            return
        body = clean("\n".join(current_lines))
        if len(body) < config.MIN_CHUNK_LENGTH:
            return
        heading = " > ".join(current_heading_path).strip()
        if not heading:
            return
        sections.append({"heading": heading, "text": body, "section_index": section_index})
        section_index += 1

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if current_keep:
                current_lines.append("")
            continue

        heading_info = extract_markdown_heading(stripped) or extract_table_section_heading(stripped)

        if heading_info:
            level, heading = heading_info
            flush_current_section()

            if stop_at_section and stop_at_section in heading.lower():
                current_heading_path = []
                current_lines = []
                current_keep = False
                break

            level = max(level, 1)
            if len(heading_stack) < level:
                heading_stack.extend([""] * (level - len(heading_stack)))
            heading_stack[level - 1] = heading
            heading_stack = heading_stack[:level]

            current_heading_path = [h for h in heading_stack if h]
            current_lines = []
            current_keep = not section_should_skip(current_heading_path, skip_patterns)
            continue

        if current_keep:
            current_lines.append(line)

    flush_current_section()

    print("\n===== DOCX SECTIONS =====")
    for s in sections:
        print(s["heading"])
    print("TOTAL SECTIONS:", len(sections))

    return sections

# =============================================================================
# 7) PII TRACKING
# =============================================================================


def track_pii_on_file(text: str, source: str, use_ner: bool = True):
    """
    Development-only text tracking.

    This function does not depend on any external masking module or regex detection.
    It only logs basic text length so you can confirm which files produced text
    without storing the raw scope in Chroma metadata.
    """
    try:
        logs_dir = ROOT_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)

        safe_source = re.sub(r"[^a-zA-Z0-9_]", "_", source)[:40]
        filename = f"text_track_{safe_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        payload = {
            "source": source,
            "text_length_chars": len(text or ""),
            "tracked_at": datetime.now(timezone.utc).isoformat(),
            "note": "PII masking module is not used. This is a length-only development log.",
        }

        with open(logs_dir / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n🔍 Text Tracking: {source}")
        print(f"   📄 Text length: {len(text or ''):,} characters")
        print(f"   📝 Saved tracking log: {filename}")

        return payload

    except Exception as e:
        print(f"   ⚠️  Text tracking failed: {e}")
        return None


# =============================================================================
# 8) PIPELINES RETURN LANGCHAIN DOCUMENTS
# =============================================================================


def pipeline_a(oai_client: Optional[OpenAI] = None) -> List[Document]:
    documents: List[Document] = []

    company_zip = resolve_zip_path(COMPANY_ZIP, "Beam_WeCloud_RAG_Documents")
    if company_zip is None:
        print(f"  [skip] Company ZIP not found in: {SOURCE_DATA_DIR}")
        return documents

    zip_hash = file_hash(company_zip)

    with zipfile.ZipFile(company_zip) as zf:
        for pdf_name, meta in COMPANY_PDF_FILES.items():
            entry_path = find_zip_entry(zf, pdf_name)
            if entry_path is None:
                print(f"  [skip] Not found inside ZIP: {pdf_name}")
                continue

            entry_bytes = zf.read(entry_path)
            entry_hash = sha256_bytes(entry_bytes)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(entry_bytes)
                tmp_path = tmp.name

            try:
                doc_meta = build_document_metadata(
                    source=pdf_name,
                    source_folder="Beam_WeCloud_RAG_Documents.zip",
                    doc_type=str(meta["doc_type"]),
                    content_type="company_knowledge",
                    pipeline_name=str(meta["pipeline_name"]),
                    document_tags=list(meta["document_tags"]),
                    file_hash_value=entry_hash,
                    is_proposal=False,
                    file_ext=".pdf",
                )
                doc_meta["container_zip_hash"] = zip_hash
                doc_meta["zip_entry_path"] = entry_path
                doc_meta["sensitive_data_obfuscated"] = False

                pages = read_pdf(Path(tmp_path), source_name=pdf_name)
                full_text = "\n\n".join(p["text"] for p in pages)
                track_pii_on_file(full_text, pdf_name, use_ner=True)

                for page_item in pages:
                    page_num = page_item["page"]
                    page_text = page_item["text"]
                    
                    masked_page_text = page_text

                    for local_chunk_index, piece in enumerate(CHUNKER.split(masked_page_text)):
                        chunk_index = (page_num * 1000) + local_chunk_index
                        chunk_tags = auto_tag(piece, list(meta["document_tags"]))
                        documents.append(DOC_BUILDER.build(
                            content=piece,
                            document_metadata=doc_meta,
                            chunk_index=chunk_index,
                            chunk_tags=chunk_tags,
                            page=page_num,
                            section="",
                        ))
            finally:
                os.unlink(tmp_path)

    return documents


def pipeline_b(oai_client: Optional[OpenAI] = None) -> List[Document]:
    documents: List[Document] = []

    tenders_zip = resolve_zip_path(TENDERS_ZIP, "Tenders - Raw and Annotated")
    if tenders_zip is None:
        print(f"  [skip] Tenders ZIP not found in: {SOURCE_DATA_DIR}")
        return documents

    zip_hash = file_hash(tenders_zip)

    with zipfile.ZipFile(tenders_zip) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue

            fname = Path(entry.filename).name
            ext = Path(fname).suffix.lower()
            folder_name = Path(entry.filename).parent.name or "Tenders"

            if ext == ".docx" and "copy of cp-730126" in fname.lower():
                entry_bytes = zf.read(entry.filename)
                entry_hash = sha256_bytes(entry_bytes)

                is_usask = "usask" in folder_name.lower() or "730126" in folder_name.lower() or "730126" in fname.lower()
                rfp_id = USASK_RFP_ID if is_usask else re.sub(r"[^\w]", "_", folder_name)[:50]

                

                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    tmp.write(entry_bytes)
                    tmp_path = tmp.name

                try:
                    doc_meta = build_document_metadata(
                        source=fname,
                        source_folder="Tenders - Raw and Annotated.zip",
                        doc_type="proposal",
                        content_type="proposal_response",
                        pipeline_name="tender_usask_docx_proposal_pipeline",
                        document_tags=["proposal", "usask", "genai", "education", "government"],
                        file_hash_value=entry_hash,
                        rfp_id=rfp_id,
                        is_proposal=True,
                        file_ext=ext,
                    )
                    doc_meta["container_zip_hash"] = zip_hash
                    doc_meta["zip_entry_path"] = entry.filename
                    doc_meta["sensitive_data_obfuscated"] = bool(oai_client)

                    sections = read_docx_proposal(Path(tmp_path), source_name=fname)
                    full_text = "\n\n".join( s["text"] for s in sections)
                    track_pii_on_file(full_text, fname, use_ner=True)

                    for section_index, section in enumerate(sections):
                        section_text = section["text"]

                        if oai_client:
                            print(f"    [mask] Running LLM masking on section: {section['heading']}")

                            llm_masked = _mask_via_llm(
                                section_text,
                                oai_client
                            )

                            masked_section_text = _regex_validate_and_fix(
                                llm_masked
                            )
                        else:
                            masked_section_text = section_text

                        # 2. Tagging on the final text.
                        if oai_client:
                            section_tags = llm_tag(masked_section_text, oai_client, base_tags=doc_meta["document_tags"])
                        else:
                            section_tags = auto_tag(masked_section_text, doc_meta["document_tags"])

                        for local_chunk_index, piece in enumerate(CHUNKER.split(masked_section_text)):
                            chunk_index = (section.get("section_index", section_index) * 1000) + local_chunk_index
                            chunk_tags = auto_tag(piece, section_tags)
                            documents.append(DOC_BUILDER.build(
                                content=piece,
                                document_metadata=doc_meta,
                                chunk_index=chunk_index,
                                chunk_tags=chunk_tags,
                                page=None,
                                section=section["heading"],
                            ))
                finally:
                    os.unlink(tmp_path)

    return documents
# Pipeline C 
SCOPE_HEADER_KEYWORDS = [
    "scope of services",
    "scope of service",
    "scope of work",
    "project scope",
    "requirements",
]

SECTION_STOP_KEYWORDS = [
    "deliverables",
    "timeline",
    "timelines",
    "pricing",
    "submission",
    "evaluation",
    "appendix",
    "references",
]


def normalize_excel_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def clean_scope_cell_text(text: Any) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()

    if "|" in cleaned:
        cleaned = cleaned.split("|")[0].strip()

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def find_scope_header(ws) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for row in range(1, ws.max_row + 1):
        for col in range(1, min(ws.max_column, 6) + 1):
            value = ws.cell(row=row, column=col).value
            text = normalize_excel_text(value)

            if any(k in text for k in SCOPE_HEADER_KEYWORDS):
                return row, col, str(value)

    return None, None, None


def extract_scope_openpyxl(path: Path) -> Tuple[str, Optional[Dict[str, Any]]]:
    wb = openpyxl.load_workbook(path, data_only=True)

    for ws in wb.worksheets:
        header_row, header_col, header_value = find_scope_header(ws)

        if header_row is None:
            continue

        candidate_cells = []

        for r in range(header_row + 1, min(ws.max_row, header_row + 15) + 1):
            candidate_cells.append((r, header_col))
            if header_col != 1:
                candidate_cells.append((r, 1))

        for r, c in candidate_cells:
            raw_value = ws.cell(row=r, column=c).value
            cleaned = clean_scope_cell_text(raw_value)
            norm = normalize_excel_text(cleaned)

            if not cleaned:
                continue

            if len(norm) < 100 and any(k in norm for k in SECTION_STOP_KEYWORDS):
                continue

            if len(cleaned) > 120 or re.search(r"\n\s*1\.|1\.", cleaned):
                return cleaned, {
                    "sheet": ws.title,
                    "scope_header_row": header_row,
                    "scope_header_col": header_col,
                    "scope_header_value": header_value,
                    "scope_cell_row": r,
                    "scope_cell_col": c,
                    "scope_cell_address": ws.cell(row=r, column=c).coordinate,
                }

    return "", None


def build_scope_capability_prompt(scope_text: str) -> str:
    return f"""
You are preparing clean structured data for a Beam Data RAG knowledge base.

You will receive ONLY the Scope text extracted from an Excel tender/project file.

Your task:
1. Extract reusable capabilities from the Scope only.
2. Infer the industry only if it is clearly supported by the Scope text.
3. Remove or obfuscate sensitive information.
4. Produce a detailed capability summary that preserves all relevant technical capabilities found in the Scope.

Important rules:
Only extract capabilities that represent reusable technical services,
platforms, architectures, integrations, infrastructure, AI methods,
or consulting offerings.

Many scope items may be written as project phases or implementation activities.
When this occurs, identify and extract the underlying reusable technical capabilities,
platforms, architectures, AI methods, data capabilities, and infrastructure implied by those activities.
Focus on WHAT can be delivered, not HOW the project is organized.

Do not extract:
- project phases
- business outcomes
- planning activities
- management activities
- operational tasks
- generic benefits
- reporting outputs

Do not create capabilities that are not explicitly supported by the scope.

Return valid JSON only with this exact structure:

{{
  "capability_text": "",
  "metadata": {{
    "source_type": "tender_scope_capabilities",
    "evidence_type": "inferred_from_tender_scope",
    "confidence": "medium",
    "industry": "",
    "capabilities": []
  }},
  "capability_details": [
    {{
      "capability": "",
      "description": "",
      "generic_evidence": ""
    }}
  ]
}}

The capability_text should provide a comprehensive capability summary rather than a brief paragraph.
Multiple paragraphs are allowed if needed to preserve all important technical details.

The summary should begin with:
"Beam Data Capability Areas in <Industry>:"

Then describe all relevant capabilities supported by the scope in a clear, reusable format suitable for a RAG knowledge base.

Do not use the word "chunk" in the output text.
Do not summarize too aggressively.
Do not omit specific technical terms if they are explicitly mentioned in the scope.

Scope text:
{scope_text}
"""


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Parse JSON returned by the LLM, even if it wraps the JSON in markdown."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_capabilities_with_llm(scope_text: str, client: OpenAI) -> Dict[str, Any]:
    prompt = build_scope_capability_prompt(scope_text)

    resp = client.responses.create(
        model=config.LLM_MODEL,
        input=prompt,
    )

    return parse_llm_json(resp.output_text)


def pipeline_c_excel_scopes(oai_client: Optional[OpenAI] = None) -> List[Document]:
    """Extract Excel tender scopes and store LLM-generated capability summaries as RAG documents."""
    documents: List[Document] = []

    target_count = len(TENDER_EXCEL_TARGET_FILES)
    matched_files: List[str] = []
    processed_files: List[str] = []
    skipped_files: List[str] = []
    error_files: List[str] = []

    if oai_client is None:
        print("  [skip] Excel capability pipeline requires OpenAI client")
        return documents

    tenders_zip = resolve_zip_path(TENDERS_ZIP, "Tenders - Raw and Annotated")
    if tenders_zip is None:
        print(f"  [skip] Tenders ZIP not found in: {SOURCE_DATA_DIR}")
        return documents

    zip_hash = file_hash(tenders_zip)

    with zipfile.ZipFile(tenders_zip) as zf:
        for entry in zf.infolist():
            if entry.is_dir():
                continue

            fname = Path(entry.filename).name.strip()
            ext = Path(fname).suffix.lower()

            if ext != ".xlsx":
                continue

            file_stem = normalize_file_stem(fname)
            if file_stem not in TENDER_EXCEL_TARGET_FILES:
                continue

            matched_files.append(fname)
            print(f"\n  [excel] Processing: {entry.filename}")

            entry_bytes = zf.read(entry.filename)
            entry_hash = sha256_bytes(entry_bytes)

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(entry_bytes)
                tmp_path = tmp.name

            try:
                scope_text, extraction_info = extract_scope_openpyxl(Path(tmp_path))

                if not scope_text.strip():
                    print(f"  [skip] No scope found: {fname}")
                    skipped_files.append(f"{fname} — no scope found")
                    continue

                result = extract_capabilities_with_llm(scope_text, oai_client)

                capability_text = result.get("capability_text", "").strip()
                result_meta = result.get("metadata", {}) or {}
                capabilities = result_meta.get("capabilities", []) or []
                industry = result_meta.get("industry", "") or ""

                if not capability_text:
                    print(f"  [skip] Empty capability_text: {fname}")
                    skipped_files.append(f"{fname} — empty capability_text")
                    continue

                # Development visibility: track the generated capability text length.
                track_pii_on_file(capability_text, fname, use_ner=True)

                # Excel output is stored as returned by the capability-extraction prompt.
                # No masking utility and no regex cleanup are applied to Excel capability text.
                masked_text = capability_text

                base_tags = [
                    "tender",
                    "tender-scope",
                    "capabilities",
                    "beamdata",
                ]

                if industry:
                    industry_tag = slugify_tag(industry)
                    if industry_tag:
                        base_tags.append(industry_tag)

                for cap in capabilities:
                    cap_tag = slugify_tag(cap)
                    if cap_tag and len(cap_tag) <= 50:
                        base_tags.append(cap_tag)

                base_tags = sorted(set(base_tags))

                doc_meta = build_document_metadata(
                    source=fname,
                    source_folder="Tenders - Raw and Annotated.zip",
                    doc_type="tender_scope_capabilities",
                    content_type="capability_summary",
                    pipeline_name="tender_excel_scope_capabilities_pipeline",
                    document_tags=base_tags,
                    file_hash_value=entry_hash,
                    rfp_id=re.sub(r"[^\w]+", "_", Path(entry.filename).parent.name)[:80],
                    is_proposal=False,
                    file_ext=".xlsx",
                )

                doc_meta["container_zip_hash"] = zip_hash
                doc_meta["zip_entry_path"] = entry.filename
                doc_meta["sensitive_data_obfuscated"] = False
                doc_meta["industry"] = industry
                doc_meta["capabilities"] = capabilities
                doc_meta["capabilities_text"] = ", ".join(map(str, capabilities))
                doc_meta["source_type"] = result_meta.get("source_type", "tender_scope_capabilities")
                doc_meta["evidence_type"] = result_meta.get("evidence_type", "inferred_from_tender_scope")
                doc_meta["confidence"] = result_meta.get("confidence", "medium")
                doc_meta["scope_length_chars"] = len(scope_text)
                doc_meta["capability_text_length_chars"] = len(capability_text)

                if extraction_info:
                    doc_meta["scope_sheet"] = extraction_info.get("sheet", "")
                    doc_meta["scope_cell"] = extraction_info.get("scope_cell_address", "")
                    doc_meta["scope_header_value"] = extraction_info.get("scope_header_value", "")
                    doc_meta["scope_header_row"] = extraction_info.get("scope_header_row", "")
                    doc_meta["scope_header_col"] = extraction_info.get("scope_header_col", "")

                chunk_tags = auto_tag(masked_text, base_tags)
                for cap in capabilities:
                    cap_tag = slugify_tag(cap)
                    if cap_tag and cap_tag not in chunk_tags and len(cap_tag) <= 50:
                        chunk_tags.append(cap_tag)
                chunk_tags = sorted(set(chunk_tags))

                file_doc_count_before = len(documents)

                for local_chunk_index, piece in enumerate(CHUNKER.split(masked_text)):
                    documents.append(DOC_BUILDER.build(
                        content=piece,
                        document_metadata=doc_meta,
                        chunk_index=local_chunk_index,
                        chunk_tags=auto_tag(piece, chunk_tags),
                        page=None,
                        section="Tender Scope Capabilities",
                    ))

                file_doc_count = len(documents) - file_doc_count_before
                processed_files.append(fname)
                print(f"  [excel] Generated {file_doc_count} capability document(s) from: {fname}")

            except Exception as e:
                print(f"  [error] Excel pipeline failed for {fname}: {e}")
                error_files.append(f"{fname} — {e}")

            finally:
                os.unlink(tmp_path)

    unmatched_targets = sorted(TENDER_EXCEL_TARGET_FILES - {normalize_file_stem(f) for f in matched_files})

    print("\n" + "=" * 80)
    print("EXCEL PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Target Excel files       : {target_count}")
    print(f"Matched target files     : {len(matched_files)}")
    print(f"Processed successfully   : {len(processed_files)}")
    print(f"Skipped after matching   : {len(skipped_files)}")
    print(f"Errors                   : {len(error_files)}")
    print(f"Documents generated      : {len(documents)}")

    if processed_files:
        print("\nProcessed files:")
        for name in processed_files:
            print(f"  - {name}")

    if skipped_files:
        print("\nSkipped files:")
        for name in skipped_files:
            print(f"  - {name}")

    if error_files:
        print("\nFiles with errors:")
        for name in error_files:
            print(f"  - {name}")

    if unmatched_targets:
        print("\nTarget files not found in ZIP:")
        for name in unmatched_targets:
            print(f"  - {name}")

    print("=" * 80)

    return documents

# =============================================================================
# 9) SAVE / INDEX
# =============================================================================


def dedupe_documents(documents: List[Document]) -> List[Document]:
    seen = set()
    deduped: List[Document] = []
    for doc in documents:
        norm = re.sub(r"\s+", " ", doc.page_content.lower().strip())
        if norm not in seen:
            seen.add(norm)
            deduped.append(doc)
    removed = len(documents) - len(deduped)
    if removed:
        print(f"\n  Deduplication: removed {removed} duplicate chunks")
    return deduped


def save_documents_json(documents: List[Document]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    with open(DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved documents JSON -> {DOCS_JSON}")


def print_stats(documents: List[Document]) -> None:
    print(f"\nTotal LangChain Documents: {len(documents)}")

    by_type: Dict[str, int] = {}
    for doc in documents:
        doc_type = str(doc.metadata.get("doc_type", "unknown"))
        by_type[doc_type] = by_type.get(doc_type, 0) + 1

    print("\nDocuments by doc_type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type:<25} {count}")

    all_tags: Dict[str, int] = {}
    for doc in documents:
        tags_text = str(doc.metadata.get("tags_text", ""))
        for tag in [t.strip() for t in tags_text.split(",") if t.strip()]:
            all_tags[tag] = all_tags.get(tag, 0) + 1

    top_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:10]
    print("\nTop tags:")
    print("  " + ", ".join(f"{tag}({count})" for tag, count in top_tags))


# =============================================================================
# 10) MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the LangChain Chroma DB",
    )

    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Only write kb_documents.json, do not build Chroma",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"  Building LangChain Knowledge Base — {SCRIPT_VERSION}")
    print("=" * 70)

    oai_client = None

    if config.OPENAI_API_KEY:
        oai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        print("  LLM tagging: ENABLED")
    else:
        print("  LLM tagging: DISABLED — keyword tags only")

    print("\n[1/3] Pipeline A — Company PDFs")
    docs_a = pipeline_a(oai_client)
    print(f"      {len(docs_a)} documents")

    print("\n[2/3] Pipeline B — USask DOCX Proposal")
    docs_b = pipeline_b(oai_client)
    print(f"      {len(docs_b)} documents")

    print("\n[3/3] Pipeline C — Excel Tender Scope Capabilities")
    docs_c = pipeline_c_excel_scopes(oai_client)
    print(f"      {len(docs_c)} documents")

    documents = dedupe_documents(docs_a + docs_b + docs_c)

    proposals = sum(
        1 for d in docs_b if d.metadata.get("is_proposal") is True
    )
    print(f"      {proposals} proposal documents")

    print_stats(documents)

    save_documents_json(documents)

    if not args.no_index:

        print("\nBuilding Chroma index with LangChain...")

        store = ChromaVectorStore(
            persist_dir=config.CHROMA_DIR,
            collection_name=config.COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
        )

        vectorstore = store.build(
            documents=documents,
            reset=args.reset,
        )

        print(f"Chroma ready -> {config.CHROMA_DIR}")
        print(f"Collection: {config.COLLECTION_NAME}")
        print(f"Count: {vectorstore._collection.count()}")

    print("\nDone.")
    print(
        "Next: run the API with: "
        "uvicorn src.api_langchain:app --reload"
    )


if __name__ == "__main__":
    main()