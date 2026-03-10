"""RAG evaluation runner.

Loads golden_dataset.json, runs each question through the RAG pipeline,
evaluates with RAGAS metrics, and saves results.

Usage::

    python -m eval.run_eval
    python -m eval.run_eval --experiment semantic_v2
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from tabulate import tabulate

from eval.metrics import RAGSample, evaluate_rag

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_DATASET = _EVAL_DIR / "golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

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
    from hse_prom_prog.config import settings
    from hse_prom_prog.rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE

    return {
        "vllm_model": settings.vllm_model,
        "vllm_base_url": settings.vllm_base_url,
        "embedding_model": settings.embedding_model,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retriever_top_k": 4,
    }


def _build_pipeline():
    from hse_prom_prog.agents.rag_agent import RAGAgent
    from hse_prom_prog.llm.client import LLMClient
    from hse_prom_prog.rag.retriever import get_retriever

    llm_client = LLMClient()
    retriever = get_retriever()
    rag_agent = RAGAgent(llm_client=llm_client, retriever=retriever)
    return retriever, rag_agent


# ── pipeline execution ───────────────────────────────────────


def _run_pipeline(retriever, rag_agent, questions: list[dict]) -> list[dict]:
    """Run RAG on every question; return list of dicts with answers + contexts."""
    results = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        question = item["question"]
        logger.info("[%d/%d] %s", i, total, question[:60])

        # Retrieve
        try:
            docs = retriever.invoke(question)
        except Exception:
            logger.exception("Retrieval failed for: %s", question[:60])
            docs = []
        contexts = [doc.page_content for doc in docs]

        # Generate
        state = {"original_query": question}
        result = rag_agent.process(state)

        results.append(
            {
                "question": question,
                "answer": result.get("rag_response") or "",
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "category": item["category"],
                "sources": result.get("rag_sources", []),
            }
        )
    return results


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

    # 2. Build pipeline
    retriever, rag_agent = _build_pipeline()

    # 3. Run pipeline on all questions
    eval_results = _run_pipeline(retriever, rag_agent, eval_qs)
    neg_results = _run_pipeline(retriever, rag_agent, neg_qs)

    # 4. RAGAS evaluation (non-negative only)
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

    # 5. Compose output
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

    output = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "config": _pipeline_config(),
        "aggregate": ragas["scores"],
        "per_question": per_question,
    }

    # 6. Save
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{args.experiment}_{timestamp}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Results saved → %s", out_path)

    # 7. Print summary
    ctx = {
        "experiment": args.experiment,
        "timestamp": timestamp,
        "n_negative": len(neg_qs),
    }
    _print_summary(ragas, eval_results, ctx)


if __name__ == "__main__":
    main()
