"""Ingestion pipeline: load documents, chunk, embed, upload to Qdrant.

Usage:
    python -m hse_prom_prog.rag.ingest          # default: knowledge_base/
    python -m hse_prom_prog.rag.ingest /path     # custom directory
"""

import logging
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)

# Default knowledge base path (project root / knowledge_base)
_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"

# Table detection: line with 3+ whitespace-separated columns
_TABLE_LINE_MULTI_COL = re.compile(r"\S+\s{2,}\S+\s{2,}\S+")
_MIN_TABLE_LINES = 3
_MIN_PIPE_COUNT = 2

# Markdown heading levels to split on
_MD_HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def _load_documents(kb_dir: Path) -> list:
    """Load .md and .pdf documents from *kb_dir* with metadata."""
    docs = []

    # Markdown files
    md_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    md_docs = md_loader.load()
    for doc in md_docs:
        _enrich_metadata(doc, kb_dir)
    docs.extend(md_docs)
    logger.info("[Ingest] Loaded %d markdown documents", len(md_docs))

    # PDF files
    pdf_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        _enrich_metadata(doc, kb_dir)
    docs.extend(pdf_docs)
    logger.info("[Ingest] Loaded %d PDF documents", len(pdf_docs))

    logger.info("[Ingest] Total documents loaded: %d", len(docs))
    return docs


def _enrich_metadata(doc, kb_dir: Path) -> None:
    """Add category (sub-folder name), filename, and ingestion timestamp."""
    source = doc.metadata.get("source", "")
    try:
        rel = Path(source).relative_to(kb_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
    except ValueError:
        category = "general"
    doc.metadata["category"] = category
    doc.metadata["ingested_at"] = datetime.now(tz=UTC).isoformat()
    doc.metadata["filename"] = Path(source).name if source else "unknown"


def _extract_section(text: str) -> str:
    """Extract the first markdown heading from text as the section name.

    Returns the heading text, or "\u2014" (em dash) if no heading found.
    """
    match = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "\u2014"


# ------------------------------------------------------------------
# Table detection for PDF content
# ------------------------------------------------------------------


def _detect_table_blocks(text: str) -> list[tuple[int, int]]:
    """Detect table-like regions in text via regex heuristics.

    A "table region" is 3+ consecutive lines where each line has either:
      - 3+ whitespace-separated columns (2+ spaces between tokens), OR
      - pipe '|' delimiters (2+ pipes per line)

    Returns:
        List of (start_line, end_line) tuples (0-indexed, inclusive).
    """
    lines = text.split("\n")
    table_regions: list[tuple[int, int]] = []
    run_start: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = bool(
            _TABLE_LINE_MULTI_COL.search(stripped) or stripped.count("|") >= _MIN_PIPE_COUNT
        )
        if is_table_line:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= _MIN_TABLE_LINES:
                table_regions.append((run_start, i - 1))
            run_start = None

    # Handle run ending at last line
    if run_start is not None and (len(lines) - run_start) >= _MIN_TABLE_LINES:
        table_regions.append((run_start, len(lines) - 1))

    return table_regions


def _split_text_preserving_tables(
    text: str,
    parent_splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, bool]]:
    """Split text into segments, preserving table regions as atomic blocks.

    Returns:
        List of (text_segment, is_table) tuples in document order.
    """
    table_regions = _detect_table_blocks(text)
    if not table_regions:
        chunks = parent_splitter.split_text(text)
        return [(chunk, False) for chunk in chunks]

    lines = text.split("\n")
    segments: list[tuple[str, bool]] = []
    prev_end = 0

    for start, end in table_regions:
        # Non-table text before this table
        if prev_end < start:
            non_table_text = "\n".join(lines[prev_end:start]).strip()
            if non_table_text:
                for chunk in parent_splitter.split_text(non_table_text):
                    segments.append((chunk, False))
        # Table text (atomic)
        table_text = "\n".join(lines[start : end + 1]).strip()
        if table_text:
            segments.append((table_text, True))
        prev_end = end + 1

    # Non-table text after last table
    if prev_end < len(lines):
        trailing_text = "\n".join(lines[prev_end:]).strip()
        if trailing_text:
            for chunk in parent_splitter.split_text(trailing_text):
                segments.append((chunk, False))

    return segments


# ------------------------------------------------------------------
# Format-aware parent splitting
# ------------------------------------------------------------------


def _split_markdown_semantically(
    docs: list,
    parent_splitter: RecursiveCharacterTextSplitter,
) -> list:
    """Split Markdown documents by headings, falling back to size-based splitting.

    Uses MarkdownHeaderTextSplitter to split on #, ##, ### headings.
    Sections exceeding parent_chunk_size are further split by parent_splitter.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS_TO_SPLIT,
        strip_headers=False,
    )

    parents: list = []
    for doc in docs:
        sections = md_splitter.split_text(doc.page_content)
        for sec in sections:
            section_name = (
                sec.metadata.get("Header 3")
                or sec.metadata.get("Header 2")
                or sec.metadata.get("Header 1")
                or _extract_section(sec.page_content)
            )

            base_meta = {**doc.metadata, "section": section_name}

            if len(sec.page_content) > settings.parent_chunk_size:
                sub_chunks = parent_splitter.split_text(sec.page_content)
                for chunk_text in sub_chunks:
                    parents.append(Document(page_content=chunk_text, metadata={**base_meta}))
            else:
                parents.append(Document(page_content=sec.page_content, metadata={**base_meta}))
    return parents


def _split_pdf_with_tables(
    docs: list,
    parent_splitter: RecursiveCharacterTextSplitter,
) -> list:
    """Split PDF documents respecting paragraph boundaries and table regions.

    Table regions are detected and kept atomic (never split).
    Non-table text is split with paragraph-aware separators.
    """
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    parents: list = []
    for doc in docs:
        segments = _split_text_preserving_tables(doc.page_content, pdf_splitter)
        for text, is_table in segments:
            meta = {**doc.metadata, "is_table": is_table}
            if is_table:
                meta["section"] = "Table"
            else:
                meta["section"] = _extract_section(text)
            parents.append(Document(page_content=text, metadata=meta))
    return parents


# ------------------------------------------------------------------
# Parent-child splitting orchestrator
# ------------------------------------------------------------------


def _split_parent_child(docs: list) -> list:
    """Two-level parent-child splitting with format-aware parent splitting.

    1. Separate documents into Markdown vs PDF groups by source extension.
    2. Split Markdown docs by headings (MarkdownHeaderTextSplitter + fallback).
    3. Split PDF docs with table protection and paragraph-aware splitting.
    4. For each parent, split into child chunks (small, for indexing).
    5. Table parents (is_table=True) skip child splitting — indexed as-is.
    6. Each child stores: parent_id, parent_content, section, chunk_index.

    Returns:
        List of child Document objects ready for Qdrant upload.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    # Step 1: Separate by file type
    md_docs = [d for d in docs if d.metadata.get("source", "").endswith(".md")]
    pdf_docs = [d for d in docs if d.metadata.get("source", "").endswith(".pdf")]
    other_docs = [d for d in docs if not d.metadata.get("source", "").endswith((".md", ".pdf"))]

    # Step 2: Format-aware parent splitting
    parent_chunks: list = []
    if md_docs:
        parent_chunks.extend(_split_markdown_semantically(md_docs, parent_splitter))
    if pdf_docs:
        parent_chunks.extend(_split_pdf_with_tables(pdf_docs, parent_splitter))
    if other_docs:
        parent_chunks.extend(parent_splitter.split_documents(other_docs))

    logger.info("[Ingest] Split into %d parent chunks", len(parent_chunks))

    # Step 3: Child splitting
    all_children: list = []
    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        section = parent.metadata.get("section") or _extract_section(parent.page_content)
        is_table = parent.metadata.get("is_table", False)

        if is_table:
            # Table chunks are atomic: index as-is, no child splitting
            parent.metadata["parent_id"] = parent_id
            parent.metadata["parent_content"] = parent.page_content
            parent.metadata["section"] = section
            parent.metadata["chunk_index"] = 0
            all_children.append(parent)
            continue

        children = child_splitter.split_documents([parent])

        if not children:
            # Parent is too small to split further; use as its own child
            parent.metadata["parent_id"] = parent_id
            parent.metadata["parent_content"] = parent.page_content
            parent.metadata["section"] = section
            parent.metadata["chunk_index"] = 0
            all_children.append(parent)
            continue

        for idx, child in enumerate(children):
            child.metadata["parent_id"] = parent_id
            child.metadata["parent_content"] = parent.page_content
            child.metadata["section"] = section
            child.metadata["chunk_index"] = idx

        all_children.extend(children)

    logger.info(
        "[Ingest] Split into %d child chunks from %d parents",
        len(all_children),
        len(parent_chunks),
    )
    return all_children


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def run_ingestion(kb_dir: Path | None = None) -> int:
    """Run the full ingestion pipeline.

    Returns:
        Number of chunks uploaded to Qdrant.
    """
    kb_dir = kb_dir or _DEFAULT_KB_DIR
    logger.info("[Ingest] Starting ingestion from %s", kb_dir)

    # 1. Load
    docs = _load_documents(kb_dir)
    if not docs:
        logger.warning("[Ingest] No documents found in %s", kb_dir)
        return 0

    # 2. Chunk (parent-child: children indexed, parents stored in metadata)
    chunks = _split_parent_child(docs)

    # 3. Embeddings
    embeddings = _get_embeddings()

    # 4. Recreate Qdrant collection (idempotent on re-run)
    client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection_name

    # Get embedding dimension from a test embedding
    test_vec = embeddings.embed_query("test")
    dim = len(test_vec)

    if client.collection_exists(collection):
        logger.info("[Ingest] Dropping existing collection '%s'", collection)
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("[Ingest] Created collection '%s' (dim=%d)", collection, dim)

    # 5. Upload
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        collection_name=collection,
    )
    logger.info("[Ingest] Uploaded %d chunks to Qdrant", len(chunks))

    return len(chunks)


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    n = run_ingestion(kb)
    print(f"Done. {n} chunks ingested.")
