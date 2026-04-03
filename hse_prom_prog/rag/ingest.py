"""Ingestion pipeline: load documents, chunk, embed, upload to Qdrant.

Creates a collection with two vector types:
- **dense** (``intfloat/multilingual-e5-base``, 768-d, cosine) — semantic search
- **bm25** (Qdrant/bm25 via fastembed, sparse, IDF) — keyword search

PDF loading uses pdfplumber for two-mode extraction:
- **Text mode** — regular paragraphs, chunked via RecursiveCharacterTextSplitter
- **Table mode** — tables are denormalized into self-contained statements
  (one Document per row) and skip chunking entirely

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

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from hse_prom_prog.config import settings
from hse_prom_prog.rag.embeddings import (
    get_embeddings,
    get_target_dim,
    truncate_vectors,
)
from hse_prom_prog.rag.sparse import embed_sparse_batch

logger = logging.getLogger(__name__)

# Default knowledge base path (project root / knowledge_base)
_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

# Vector names used in Qdrant collection
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


# ── PDF loading with pdfplumber ──────────────────────────────


def _denormalize_table(
    headers: list[str | None],
    rows: list[list[str | None]],
    doc_title: str,
    source: str,
    page: int,
) -> list[Document]:
    """Convert a table into self-contained text Documents (one per column).

    Collects all non-empty values in each column into a single Document:
    ``{doc_title}. {header}: value1, value2, value3, ...``

    For the jira_status_mapping table this produces documents like:
    ``Jira Status Mapping. TO DO: Open, To Do, Planned, Queued, ...``
    """
    docs: list[Document] = []
    clean_headers = [(h or "").replace("\n", " ").strip() for h in headers]

    for col_idx, header in enumerate(clean_headers):
        if not header:
            continue
        values = []
        for row in rows:
            cell = row[col_idx] if col_idx < len(row) else None
            if not cell or not cell.strip():
                continue
            value = cell.replace("\n", " ").strip()
            if value != header:
                values.append(value)
        if not values:
            continue
        text = f"{doc_title}. {header}: {', '.join(values)}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page,
                    "element_type": "table_denormalized",
                    "table_header": header,
                },
            )
        )
    return docs


def _load_pdf_pdfplumber(pdf_path: Path, kb_dir: Path) -> tuple[list[Document], list[Document]]:
    """Load a single PDF via pdfplumber.

    Returns:
        (text_docs, table_docs) — text docs will be chunked later;
        table docs are already self-contained and skip chunking.
    """
    text_docs: list[Document] = []
    table_docs: list[Document] = []
    source = str(pdf_path)
    doc_title = _doc_title_from_source(source)

    pdf = pdfplumber.open(str(pdf_path))
    for page_num, page in enumerate(pdf.pages):
        tables = page.extract_tables()

        # Collect bounding boxes of all tables to exclude from text
        table_bboxes = [t.bbox for t in page.find_tables()]

        # Extract text outside tables
        text_page = page
        for bbox in table_bboxes:
            text_page = text_page.outside_bbox(bbox)
        page_text = (text_page.extract_text() or "").strip()

        if page_text:
            text_docs.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "element_type": "text",
                    },
                )
            )

        # Denormalize tables (need at least header + 1 data row)
        _min_table_rows = 2
        for table in tables:
            if not table or len(table) < _min_table_rows:
                continue
            headers = table[0]
            rows = table[1:]
            table_docs.extend(_denormalize_table(headers, rows, doc_title, source, page_num))

    pdf.close()

    # Enrich metadata
    for doc in [*text_docs, *table_docs]:
        _enrich_metadata(doc, kb_dir)

    return text_docs, table_docs


def _load_documents(kb_dir: Path) -> tuple[list[Document], list[Document]]:
    """Load .md and .pdf documents from *kb_dir*.

    Returns:
        (text_docs, table_docs) — text_docs go through chunking,
        table_docs are already self-contained.
    """
    text_docs: list[Document] = []
    table_docs: list[Document] = []

    # Markdown files → text only
    md_loader = DirectoryLoader(
        str(kb_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    md_docs = md_loader.load()
    for doc in md_docs:
        doc.metadata["element_type"] = "text"
        _enrich_metadata(doc, kb_dir)
    text_docs.extend(md_docs)
    logger.info("[Ingest] Loaded %d markdown documents", len(md_docs))

    # PDF files → pdfplumber (two-mode: text + tables)
    pdf_paths = sorted(kb_dir.rglob("*.pdf"))
    pdf_text_count = 0
    pdf_table_count = 0
    for pdf_path in pdf_paths:
        t_docs, tb_docs = _load_pdf_pdfplumber(pdf_path, kb_dir)
        text_docs.extend(t_docs)
        table_docs.extend(tb_docs)
        pdf_text_count += len(t_docs)
        pdf_table_count += len(tb_docs)
    logger.info(
        "[Ingest] Loaded %d PDF pages as text, %d table rows denormalized",
        pdf_text_count,
        pdf_table_count,
    )

    logger.info(
        "[Ingest] Total: %d text docs (→ chunking), %d table docs (self-contained)",
        len(text_docs),
        len(table_docs),
    )
    return text_docs, table_docs


# ── metadata helpers ─────────────────────────────────────────


def _enrich_metadata(doc: Document, kb_dir: Path) -> None:
    """Add category (sub-folder name) and ingestion timestamp."""
    source = doc.metadata.get("source", "")
    try:
        rel = Path(source).relative_to(kb_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
    except ValueError:
        category = "general"
    doc.metadata["category"] = category
    doc.metadata["ingested_at"] = datetime.now(tz=UTC).isoformat()


def _doc_title_from_source(source: str) -> str:
    """Extract human-readable title from file path.

    ``done_total.pdf`` → ``Done Total``
    ``team_lead_time.pdf`` → ``Team Lead Time``
    """
    stem = Path(source).stem
    return stem.replace("_", " ").title()


# Regex: lines that look like section headers in extracted PDF/MD text.
# Matches "# Heading", "## Heading", or short question-style headings
# (e.g. "Что это?", "Как считается?").
_SECTION_RE = re.compile(
    r"^(?:#{1,3}\s+(.+)|([А-ЯЁA-Z][А-ЯЁа-яёA-Za-z\s\-]{2,40}\?))\s*$",
    re.MULTILINE,
)


def _find_last_section(text: str) -> str | None:
    """Find the last section-like heading that appears in *text*."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return (last.group(1) or last.group(2)).strip()


def _prepend_metadata(chunks: list[Document]) -> list[Document]:
    """Add document title (and section if found) to the start of each chunk.

    Skips table_denormalized docs — they already have the title prepended
    during denormalization.
    """
    for chunk in chunks:
        if chunk.metadata.get("element_type") == "table_denormalized":
            continue

        source = chunk.metadata.get("source", "")
        doc_title = _doc_title_from_source(source)
        chunk.metadata["doc_title"] = doc_title

        section = _find_last_section(chunk.page_content)
        chunk.metadata["section"] = section or ""

        prefix = f"{doc_title}. {section}. " if section else f"{doc_title}. "
        chunk.page_content = prefix + chunk.page_content

    if chunks:
        preview = chunks[0].page_content[:120].replace("\n", " ")
        logger.info("[Ingest] Prepend example: %s...", preview)

    return chunks


# ── chunking ─────────────────────────────────────────────────


def _split_documents(docs: list[Document]) -> list[Document]:
    """Split text documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("[Ingest] Split into %d chunks", len(chunks))
    return chunks


# ── main pipeline ────────────────────────────────────────────


def run_ingestion(kb_dir: Path | None = None) -> int:
    """Run the full ingestion pipeline.

    Returns:
        Number of chunks uploaded to Qdrant.
    """
    kb_dir = kb_dir or _DEFAULT_KB_DIR
    logger.info("[Ingest] Starting ingestion from %s", kb_dir)

    # 1. Load (two-mode: text + table)
    text_docs, table_docs = _load_documents(kb_dir)
    if not text_docs and not table_docs:
        logger.warning("[Ingest] No documents found in %s", kb_dir)
        return 0

    # 2. Chunk text docs only; table docs are already self-contained
    text_chunks = _split_documents(text_docs)
    all_chunks = text_chunks + table_docs

    # 3. Prepend metadata (skips table_denormalized)
    all_chunks = _prepend_metadata(all_chunks)
    logger.info(
        "[Ingest] Final: %d text chunks + %d table chunks = %d total",
        len(text_chunks),
        len(table_docs),
        len(all_chunks),
    )

    if table_docs:
        preview = table_docs[0].page_content[:120].replace("\n", " ")
        logger.info("[Ingest] Table doc example: %s", preview)

    texts = [chunk.page_content for chunk in all_chunks]

    # 4. Dense embeddings (with optional Matryoshka truncation)
    embeddings = get_embeddings()
    test_vec = embeddings.embed_query("test")
    full_dim = len(test_vec)
    dim = get_target_dim(full_dim)
    logger.info(
        "[Ingest] model=%s, full_dim=%d, target_dim=%d%s",
        settings.embedding_model,
        full_dim,
        dim,
        " (truncated)" if dim < full_dim else "",
    )
    dense_vectors = embeddings.embed_documents(texts)
    dense_vectors = truncate_vectors(dense_vectors, dim, full_dim)

    # 5. Sparse BM25 embeddings via fastembed
    logger.info("[Ingest] Generating BM25 sparse embeddings ...")
    sparse_vectors = embed_sparse_batch(texts)
    logger.info("[Ingest] Generated %d sparse vectors", len(sparse_vectors))

    # 6. Recreate Qdrant collection (idempotent on re-run)
    client = QdrantClient(url=settings.qdrant_url, timeout=60)
    collection = settings.qdrant_collection_name

    if client.collection_exists(collection):
        logger.info("[Ingest] Dropping existing collection '%s'", collection)
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    logger.info(
        "[Ingest] Created collection '%s' (dense=%d-d, sparse=bm25+IDF)",
        collection,
        dim,
    )

    # 7. Build points with both vector types
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[i],
                    SPARSE_VECTOR_NAME: sparse_vectors[i],
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
        )

    # 8. Upload (batch by 64 to avoid large payloads)
    batch_size = 64
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch)
    logger.info("[Ingest] Uploaded %d chunks (dense + sparse) to Qdrant", len(all_chunks))

    return len(all_chunks)


# ── CLI entry point ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kb = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    n = run_ingestion(kb)
    print(f"Done. {n} chunks ingested.")
