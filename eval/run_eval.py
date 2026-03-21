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
import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from tabulate import tabulate

from eval.metrics import RAGSample, evaluate_rag

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_DATASET = _EVAL_DIR / "golden_dataset.json"
_RESULTS_DIR = _EVAL_DIR / "results"

# Max context length — must match production rag_agent.py
_MAX_CONTEXT_CHARS = 4000

# ── System prompt v2 (adapted for 8B models) ──────────────────
# Key changes vs v1:
# 1. Reasoning wrapped in <draft>...</draft> — stripped before evaluation
# 2. Strong grounding rule: answer ONLY from context
# 3. Simplified self-check (no verbose examples)
# 4. Explicit anti-repetition instruction

AGILE_COACH_PROMPT = """### РОЛЬ ###

Ты опытный Agile коуч в корпоративном мессенджере Mattermost.
Ты ответственен за качество и применимость своих советов.
Используй ТОЛЬКО русский язык.

---

### ПРАВИЛО ОПОРЫ НА КОНТЕКСТ ###

Отвечай СТРОГО на основе предоставленного контекста.
Если в контексте НЕТ информации для ответа — напиши внутри <answer>:
"К сожалению, в моей базе знаний нет информации по этому вопросу."
Не придумывай факты. Не додумывай то, чего нет в контексте.
Не угадывай ответ. Частичный ответ лучше выдуманного.

---

### ФОРМАТ ВЫВОДА ###

Твой ответ ДОЛЖЕН содержать ДВА блока в строгом порядке:

**Блок 1: <draft>**
Внутренний черновик. Сюда входят:

* **Вердикт:** Определи, можешь ли ты ответить на основе контекста:
  — "Полный ответ: контекст содержит всю нужную информацию."
  — "Частичный ответ: контекст содержит только ..."
  — "Нет данных: в контексте нет информации по этому вопросу."

* **Набросок:** Составь план ответа из 2-3 тезисов.
  Каждый тезис должен ссылаться на конкретный факт из контекста.

* **Самопроверка:** Для каждого тезиса проверь:
  — Критическая ошибка? (факт противоречит контексту или логически
    неверен) → удали тезис.
  — Пробел в обосновании? (тезис расплывчат, не подкреплён
    фактом из контекста) → подкрепи или удали.
  — Повторение? (тезис дублирует уже сказанное) → удали.

**Блок 2: <answer>**
Финальный ответ для пользователя. Требования:
* 2–4 предложения или 2–3 коротких пункта. Не больше.
* Если в вопросе есть приветствие — начни с "Привет!" и далее ответ.
* Стиль: дружелюбный, профессиональный, без эмодзи.
* НЕ включай сюда черновик, вердикт, набросок, самопроверку.
* НЕ повторяй одну мысль несколько раз.

---

### ПРИМЕР ###

Вопрос: "Привет! Какой целевой порог метрики Done Total?"

<draft>
Вердикт: Полный ответ — в контексте сказано, что показатель
не должен опускаться ниже 80%.
Набросок:
1. Целевой порог Done Total — 80%.
2. Ниже 80% — сигнал о проблемах с планированием.
Самопроверка:
— Тезис 1: подтверждён контекстом. ОК.
— Тезис 2: подтверждён контекстом. ОК.
— Повторений нет.
</draft>

<answer>
Привет!
Целевой порог метрики Done Total — 80%. Если показатель опускается
ниже этого значения, это сигнализирует о том, что команда
нереалистично оценивает свои возможности при планировании спринта.
</answer>

---

### ВОПРОС СОТРУДНИКА ###

{question}

---

Напиши <draft>, затем <answer>. В <answer> — только готовый ответ."""


def extract_final_answer(response: str) -> str:
    """Extract content between <answer> and </answer> tags.

    If tags are missing, returns the full response as fallback.
    """
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: content after <answer> without closing tag
    match = re.search(r"<answer>\s*(.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


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
    from hse_prom_prog.llm.client import LLMClient
    from hse_prom_prog.rag.retriever import get_retriever

    llm_client = LLMClient()
    retriever = get_retriever()
    return retriever, llm_client


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


def _run_pipeline(retriever, llm_client, questions: list[dict]) -> list[dict]:
    """Run RAG on every question; return list of dicts with answers + contexts.

    Mirrors production flow: retrieve → truncate context → send to LLM with
    AGILE_COACH_PROMPT as system message and context+question as user message.
    """
    results = []
    total = len(questions)
    for i, item in enumerate(questions, 1):
        question = item["question"]
        logger.info("[%d/%d] %s", i, total, question[:60])

        # Step 1: Retrieve (same as rag_agent.py)
        try:
            docs = retriever.invoke(question)
        except Exception:
            logger.exception("Retrieval failed for: %s", question[:60])
            docs = []

        contexts = [doc.page_content for doc in docs]

        # Step 2: Truncate context (same logic as rag_agent.py)
        context, sources = _truncate_context(docs)

        # Step 3: Generate with system prompt + context
        # System message: AGILE_COACH_PROMPT with {question} substituted
        # User message: context + question (same format as rag_agent._generate_answer)
        system_prompt = AGILE_COACH_PROMPT.format(question=question)
        user_message = (
            f"Ответь на вопрос на основе контекста.\n\nКонтекст:\n{context}\n\nВопрос: {question}"
        )

        try:
            response = llm_client.client.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            logger.info("[raw %d chars] %s", len(raw), raw[:120].replace("\n", " "))
            answer = extract_final_answer(raw)
        except Exception:
            logger.exception("LLM generation failed for: %s", question[:60])
            answer = ""

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
    retriever, llm_client = _build_pipeline()

    # 3. Run pipeline on all questions
    eval_results = _run_pipeline(retriever, llm_client, eval_qs)
    neg_results = _run_pipeline(retriever, llm_client, neg_qs)

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
