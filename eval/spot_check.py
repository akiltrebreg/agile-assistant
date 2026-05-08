"""Spot-check: per-question retrieval analysis for specific questions.

Runs a subset of questions through retrieval (no LLM generation, no RAGAS)
and prints retrieved chunks so you can visually inspect recall quality.

Supports multiple search modes via SEARCH_TYPE env var.

Usage::

    # Dense (default)
    docker compose run --rm -e RERANKER_ENABLED=false \
        app python -m eval.spot_check

    # Sparse
    docker compose run --rm -e SEARCH_TYPE=sparse -e RERANKER_ENABLED=false \
        app python -m eval.spot_check

    # Hybrid
    docker compose run --rm -e SEARCH_TYPE=hybrid -e RERANKER_ENABLED=false \
        app python -m eval.spot_check
"""

import json
import logging
from pathlib import Path

from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Questions to spot-check (with baseline recall for reference)
SPOT_QUESTIONS = [
    ("Какой целевой порог метрики Done Total?", 0.0),
    ("Какой целевой уровень метрики Sprint Tasks Grooming?", 0.0),
    ("Какие целевые пороги основных метрик?", 0.0),
    ("Какие четыре макро-категории статусов?", 0.5),
]

_GOLDEN = Path(__file__).resolve().parent / "golden_dataset.json"
_CHUNK_PREVIEW_LEN = 120


def _load_ground_truths() -> dict[str, str]:
    """Load ground truths from golden dataset keyed by question."""
    with _GOLDEN.open() as f:
        dataset = json.load(f)
    return {item["question"]: item["ground_truth"] for item in dataset}


def main() -> None:
    from agile_assistant.config import settings
    from agile_assistant.rag.retriever import get_retriever

    max_context = settings.max_context_chars

    reranker = None
    if settings.reranker_enabled:
        from agile_assistant.rag.reranker import get_reranker

        reranker = get_reranker()

    retriever = get_retriever()
    ground_truths = _load_ground_truths()

    k = settings.retriever_initial_k if settings.reranker_enabled else settings.retriever_top_k

    print(f"\n{'=' * 70}")
    print(
        f"Spot-check: search_type={settings.search_type}, k={k}, "
        f"reranker={settings.reranker_enabled}"
    )
    print(f"{'=' * 70}\n")

    rows = []
    for question, baseline_recall in SPOT_QUESTIONS:
        gt = ground_truths.get(question, "")

        # Retrieve
        docs = retriever.invoke(question)

        # Rerank if enabled
        if reranker:
            docs = reranker.rerank(question, docs)

        # Build context (same logic as run_eval)
        parts = []
        total = 0
        for doc in docs:
            chunk = doc.page_content
            if total + len(chunk) > max_context:
                remaining = max_context - total
                if remaining > 0:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            total += len(chunk) + 2

        context = "\n\n".join(parts)

        # Display
        q_short = question if len(question) <= 55 else question[:52] + "..."
        print(f"Q: {question}")
        print(f"  Ground truth: {gt[:100]}{'...' if len(gt) > 100 else ''}")
        print(f"  Retrieved {len(docs)} chunks, context={len(context)} chars:")
        for i, doc in enumerate(docs):
            preview = doc.page_content[:_CHUNK_PREVIEW_LEN].replace("\n", " ")
            source = doc.metadata.get("source", "?")
            print(f"    [{i + 1}] ({source}) {preview}...")
        print()

        # Check if ground truth keywords appear in context
        gt_words = set(gt.lower().split())
        ctx_words = set(context.lower().split())
        overlap = len(gt_words & ctx_words)
        coverage = overlap / len(gt_words) if gt_words else 0

        rows.append([q_short, baseline_recall, f"{coverage:.2f}", len(docs), len(context)])

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(
        tabulate(
            rows,
            headers=["Question", "BL recall", "Word overlap", "Chunks", "Ctx chars"],
            tablefmt="simple",
        )
    )
    print()


if __name__ == "__main__":
    main()
