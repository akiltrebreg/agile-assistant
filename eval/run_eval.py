"""RAG evaluation runner.

Loads golden_dataset.json, runs each question through the RAG pipeline,
evaluates with RAGAS metrics, and saves results.

The eval pipeline mirrors the production RAG pipeline as closely as possible:
same retriever, same context truncation (4000 chars), and the same
AGILE_COACH_PROMPT as system prompt.

Usage::

    python -m eval.run_eval
    python -m eval.run_eval --experiment semantic_v2
"""

import argparse
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from tabulate import tabulate

from eval.metrics import RAGSample, evaluate_rag

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_DATASET = _EVAL_DIR / "golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

_MAX_CONTEXT_CHARS: int | None = None  # lazy: set from settings in _build_pipeline()

AGILE_COACH_PROMPT = """Ты — Agile-коуч компании. Отвечай ТОЛЬКО на русском языке.

### ПРАВИЛА ###
1. Отвечай СТРОГО на основе контекста ниже.\
 Каждое утверждение в ответе должно следовать из контекста.
2. Перескажи факты близко к тексту контекста. Числа, названия, формулы, списки — приводи точно.
3. НЕ добавляй выводов, интерпретаций и рекомендаций от себя. НЕ объясняй, почему что-то важно.
4. Если в контексте НЕТ ответа — скажи только: "В моей базе знаний нет информации по этому вопросу."
5. Если вопрос содержит приветствие — начни с короткого приветствия.
6. Приведи факты из контекста, отвечающие на вопрос. Отвечай кратко.
   Каждое предложение должно следовать из контекста.\
 НЕ ДОБАВЛЯЙ пояснений, выводов и рекомендаций от себя.
   После ответа ОСТАНОВИСЬ.

### КОНТЕКСТ ###
{context}

### ВОПРОС ###
{question}

### ОТВЕТ ###
"""

# Short aliases for wide per-question table
_SHORT = {
    "llm_context_precision_with_reference": "c_prec",
    "context_precision": "c_prec",
    "context_recall": "c_rec",
    "faithfulness": "faith",
    "answer_relevancy": "a_rel",
    "response_relevancy": "a_rel",
    "answer_correctness": "a_cor",
}


# ── helpers ──────────────────────────────────────────────────


def _load_golden_dataset() -> list[dict]:
    with _GOLDEN_DATASET.open() as f:
        return json.load(f)


def _pipeline_config() -> dict:
    """Snapshot of current pipeline parameters."""
    from agile_assistant.config import settings
    from agile_assistant.rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE

    return {
        "vllm_model": settings.vllm_model,
        "vllm_base_url": settings.vllm_base_url,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "embedding_sparse_model": settings.embedding_sparse_model,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "search_type": settings.search_type,
        "rrf_k": settings.rrf_k if settings.search_type == "hybrid" else None,
        "retriever_initial_k": settings.retriever_initial_k,
        "retriever_top_k": settings.retriever_top_k,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_model": settings.reranker_model if settings.reranker_enabled else None,
        "reranker_threshold": settings.reranker_threshold if settings.reranker_enabled else None,
        "reranker_top_n": settings.reranker_top_n if settings.reranker_enabled else None,
    }


def _chunk_stats() -> dict:
    """Collect chunk-level statistics from Qdrant collection."""
    from qdrant_client import QdrantClient

    from agile_assistant.config import settings

    client = QdrantClient(url=settings.qdrant_url)
    collection = settings.qdrant_collection_name

    info = client.get_collection(collection)
    total_chunks = info.points_count

    # Scroll through all points to compute length stats
    lengths: list[int] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            text = (pt.payload or {}).get("page_content", "")
            lengths.append(len(text))
        if offset is None:
            break

    if not lengths:
        return {"total_chunks": total_chunks}

    return {
        "total_chunks": total_chunks,
        "avg_chunk_chars": round(statistics.mean(lengths), 1),
        "median_chunk_chars": round(statistics.median(lengths), 1),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
    }


def _build_pipeline():
    from agile_assistant.config import settings
    from agile_assistant.llm.client import LLMClient
    from agile_assistant.rag.retriever import get_retriever

    global _MAX_CONTEXT_CHARS  # noqa: PLW0603
    _MAX_CONTEXT_CHARS = settings.max_context_chars

    llm_client = LLMClient()
    retriever = get_retriever()
    reranker = None
    if settings.reranker_enabled:
        from agile_assistant.rag.reranker import get_reranker

        reranker = get_reranker()
    return retriever, reranker, llm_client


def _truncate_context(docs) -> tuple[str, list[str]]:
    """Build context string from docs with 4000 char limit (mirrors rag_agent.py)."""
    parts: list[str] = []
    total = 0
    for doc in docs:
        chunk = doc.page_content
        if total + len(chunk) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total
            if remaining > 0:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk) + 2  # account for "\n\n" separator

    context = "\n\n".join(parts)

    sources: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        category = doc.metadata.get("category", "")
        label = f"{category}/{source}" if category else source
        if label not in sources:
            sources.append(label)

    return context, sources


# ── pipeline execution ───────────────────────────────────────


def _run_pipeline(
    retriever, reranker, llm_client, questions: list[dict]
) -> tuple[list[dict], list[float], list[float]]:
    """Run RAG on every question.

    Returns:
        results: per-question dicts with answer, contexts, etc.
        retrieval_times: per-query retrieval latency (seconds).
        e2e_times: per-query end-to-end latency (seconds).
    """
    results = []
    retrieval_times: list[float] = []
    e2e_times: list[float] = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        question = item["question"]
        logger.info("[%d/%d] %s", i, total, question[:60])
        t_start = time.perf_counter()

        # Step 1: Retrieve initial candidates
        try:
            t0 = time.perf_counter()
            docs = retriever.invoke(question)
            retrieval_times.append(time.perf_counter() - t0)
        except Exception:
            logger.exception("Retrieval failed for: %s", question[:60])
            docs = []

        # Step 2: Rerank with cross-encoder (if enabled)
        if reranker:
            docs = reranker.rerank(question, docs)

        contexts = [doc.page_content for doc in docs]

        # Step 3: Truncate context (same logic as rag_agent.py)
        context, sources = _truncate_context(docs)

        # Step 4: Generate with prompt containing context + question
        prompt = AGILE_COACH_PROMPT.format(context=context, question=question)

        try:
            response = llm_client.client.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            logger.info("[raw %d chars] %s", len(raw), raw[:120].replace("\n", " "))
            answer = raw.strip()
        except Exception:
            logger.exception("LLM generation failed for: %s", question[:60])
            answer = ""

        e2e_times.append(time.perf_counter() - t_start)

        results.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "category": item["category"],
                "sources": sources,
            }
        )
    return results, retrieval_times, e2e_times


# ── display ──────────────────────────────────────────────────


_Q_MAX_LEN = 42


def _print_summary(ragas_out: dict, pipeline_results: list[dict], ctx: dict) -> None:
    """Print human-readable evaluation summary to stdout."""
    aggregate = ragas_out["scores"]
    per_sample = ragas_out["per_sample"]

    header = f'RAGAS Evaluation — "{ctx["experiment"]}"  {ctx["timestamp"]}'
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")

    # Aggregate scores
    agg_table = [[m, f"{v:.4f}"] for m, v in aggregate.items()]
    print(tabulate(agg_table, headers=["Metric", "Score"], tablefmt="simple"))
    print()

    # Per-question compact table
    if not per_sample:
        return

    metric_keys = list(per_sample[0].keys())
    short_keys = [_SHORT.get(k, k[:6]) for k in metric_keys]

    rows = []
    pairs = zip(pipeline_results, per_sample, strict=True)
    for i, (pr, ps) in enumerate(pairs, 1):
        q = pr["question"]
        if len(q) > _Q_MAX_LEN:
            q = q[: _Q_MAX_LEN - 3] + "..."
        row = [i, pr["category"], q]
        row.extend(f"{ps[k]:.2f}" for k in metric_keys)
        rows.append(row)

    col_headers = ["#", "cat", "question", *short_keys]
    n_neg = ctx["n_negative"]
    print(f"Per-question ({len(per_sample)} evaluated, {n_neg} negative skipped):")
    print(tabulate(rows, headers=col_headers, tablefmt="simple"))


# ── main ─────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--experiment",
        default="baseline",
        help="Experiment name (used in output filename)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Load dataset
    dataset = _load_golden_dataset()
    eval_qs = [q for q in dataset if q["ground_truth"]]
    neg_qs = [q for q in dataset if not q["ground_truth"]]
    logger.info(
        "Golden dataset: %d total, %d evaluable, %d negative",
        len(dataset),
        len(eval_qs),
        len(neg_qs),
    )

    # 2. Chunk statistics from Qdrant
    logger.info("Collecting chunk statistics from Qdrant ...")
    chunk_info = _chunk_stats()
    logger.info("Chunk stats: %s", chunk_info)

    # 3. Build pipeline
    retriever, reranker, llm_client = _build_pipeline()

    # 4. Run pipeline on all questions
    eval_results, eval_ret_times, eval_e2e = _run_pipeline(retriever, reranker, llm_client, eval_qs)
    neg_results, neg_ret_times, neg_e2e = _run_pipeline(retriever, reranker, llm_client, neg_qs)
    all_ret_times = eval_ret_times + neg_ret_times
    all_e2e = eval_e2e + neg_e2e
    avg_retrieval_s = round(statistics.mean(all_ret_times), 4) if all_ret_times else 0
    latency_stats = {}
    if all_e2e:
        sorted_e2e = sorted(all_e2e)
        p95_idx = int(len(sorted_e2e) * 0.95)
        latency_stats = {
            "latency_avg_s": round(statistics.mean(all_e2e), 3),
            "latency_median_s": round(statistics.median(all_e2e), 3),
            "latency_p95_s": round(sorted_e2e[min(p95_idx, len(sorted_e2e) - 1)], 3),
            "latency_min_s": round(min(all_e2e), 3),
            "latency_max_s": round(max(all_e2e), 3),
        }

    # 6. RAGAS evaluation (non-negative only)
    samples = [
        RAGSample(
            question=r["question"],
            answer=r["answer"],
            contexts=r["contexts"],
            ground_truth=r["ground_truth"],
        )
        for r in eval_results
    ]
    logger.info("Starting RAGAS evaluation …")
    ragas = evaluate_rag(samples)

    # 7. Compose output
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    per_question = []
    for pr, ps in zip(eval_results, ragas["per_sample"], strict=True):
        per_question.append(
            {
                "question": pr["question"],
                "category": pr["category"],
                "answer": pr["answer"],
                "ground_truth": pr["ground_truth"],
                "contexts": pr["contexts"],
                "sources": pr["sources"],
                "metrics": ps,
            }
        )
    for nr in neg_results:
        per_question.append(
            {
                "question": nr["question"],
                "category": nr["category"],
                "answer": nr["answer"],
                "ground_truth": "",
                "contexts": nr["contexts"],
                "sources": nr["sources"],
                "metrics": None,
            }
        )

    config = _pipeline_config()
    config.update(chunk_info)
    config["avg_retrieval_time_s"] = avg_retrieval_s

    aggregate = {**ragas["scores"], **latency_stats}

    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "config": config,
        "aggregate": aggregate,
        "per_question": per_question,
    }

    # 8. Save
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved → %s", out_path)

    # 9. Print summary
    ctx = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "n_negative": len(neg_qs),
    }
    _print_summary(ragas, eval_results, ctx)


if __name__ == "__main__":
    main()
