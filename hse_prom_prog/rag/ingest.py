"""Ingestion pipeline: load documents, chunk, embed, upload to Qdrant.

Usage:
    python -m hse_prom_prog.rag.ingest          # default: knowledge_base/
    python -m hse_prom_prog.rag.ingest /path     # custom directory
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from hse_prom_prog.config import settings

logger = logging.getLogger(__name__)

# Default knowledge base path (project root / knowledge_base)
_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


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
    """Add category (sub-folder name) and ingestion timestamp."""
    source = doc.metadata.get("source", "")
    try:
        rel = Path(source).relative_to(kb_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
    except ValueError:
        category = "general"
    doc.metadata["category"] = category
    doc.metadata["ingested_at"] = datetime.now(tz=UTC).isoformat()


def _split_documents(docs: list) -> list:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("[Ingest] Split into %d chunks", len(chunks))
    return chunks


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

    # 2. Chunk
    chunks = _split_documents(docs)

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
