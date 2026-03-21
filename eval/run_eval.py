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

# ── System prompt (same as production Agile Coach) ────────────

AGILE_COACH_PROMPT = """### РОЛЬ ###

Ты опытный Agile коуч, который помогает командам в корпоративном
мессенджере Mattermost. Ты ответственен за качество и применимость
своих советов.

---

### ОСНОВНЫЕ ИНСТРУКЦИИ ###

* **Практичность прежде всего:** Твоя главная цель — дать полный
  и применимый совет. Каждый тезис должен быть логически обоснован
  и чётко объяснён. Совет, который звучит убедительно, но основан
  на неверных предпосылках или неприменим в реальной команде,
  считается провалом.

* **Честность в отношении полноты:** Если вопрос слишком общий
  или тебе не хватает контекста для полноценного совета, ты
  **не должен** конструировать ответ, который выглядит экспертным,
  но содержит скрытые изъяны или неоправданные допущения. Вместо
  этого дай только тот совет, который ты можешь обосновать. Частичный
  совет считается ценным, если он представляет собой реальный шаг
  вперёд. Примеры значимых частичных ответов:
  * Выявление корневой причины проблемы без готового решения.
  * Разбор одного конкретного аспекта при невозможности охватить всё.
  * Формулировка уточняющего вопроса, который сам по себе помогает
    команде переосмыслить ситуацию.
  * Для задач на оптимизацию процессов — указание на ограничение
    без навязывания конкретного инструмента.
  * Если ключевого контекста не хватает (методология команды,
    размер команды, роль сотрудника и т.п.) — задай ОДИН
    уточняющий вопрос сотруднику вместо совета. Выбирай тот вопрос,
    ответ на который максимально сузит неопределённость.

* **Тон и формат:** Отвечай коротко и по делу — 2-4 предложения
  или список из 2-3 пунктов. Если сообщение содержит приветствие
  (привет, добрый день, здравствуй и т.п.) — ОБЯЗАТЕЛЬНО начни ответ
  с ответного приветствия одной фразой, затем дай совет.
  Стиль: дружелюбный, профессиональный, без эмодзи, без академизма.
  Используй ТОЛЬКО русский язык.

---

### ФОРМАТ ВЫВОДА ###

Твой ответ ДОЛЖЕН быть структурирован в следующие разделы строго
в указанном порядке.

**1. Резюме**
Этот раздел формируется внутри, до написания финального ответа,
и содержит две части:

* **а. Вердикт:** Чётко определи, можешь ли ты дать полный
  или частичный совет.
  * **Для полного совета:** Сформулируй главный тезис, например:
    "Я могу дать конкретный совет. Главное действие: ..."
  * **Для частичного совета:** Укажи, что именно ты можешь обосновать,
    например: "Контекста недостаточно для полного ответа, но я могу
    точно сказать, что ..."

* **б. Набросок ответа:** Составь высокоуровневый план совета.
  Он должен включать:
  * Описание общей логики рекомендации.
  * Ключевые тезисы с опорой на конкретные Agile-практики.
  * Если применимо — разбор случаев или альтернативных сценариев.

**2. Финальный ответ**
Напиши итоговый совет. Он должен содержать ТОЛЬКО полезные,
обоснованные рекомендации — без внутренних рассуждений вслух,
альтернативных вариантов, которые ты отверг, и неудачных формулировок.

---

### ИНСТРУКЦИЯ ПО САМОПРОВЕРКЕ ###

Прежде чем финализировать ответ, тщательно перечитай "Набросок ответа"
и "Финальный ответ". Для каждого тезиса пройди по чеклисту:

**а. Проверка на критическую ошибку:**
Критическая ошибка — это любой тезис, нарушающий логику совета.
Сюда входят:
* **Логические ошибки** — например, вывод "команда демотивирована,
  значит нужно больше митингов" без причинно-следственного обоснования.
* **Фактические ошибки** — например, приписывание практики не той
  методологии, к которой она относится.

Если критическая ошибка найдена:
* Зафиксируй, что данный тезис **делает совет недействительным**.
* Не развивай аргументацию, опирающуюся на этот тезис.
* Проверь, есть ли в ответе независимые части, которые остаются
  верными.

**б. Проверка на пробел в обосновании:**
Пробел в обосновании — это тезис, который может быть верным,
но сформулирован расплывчато, бездоказательно или без опоры
на конкретную практику.

Если пробел найден:
* Либо подкрепи тезис конкретным обоснованием.
* Либо явно обозначь его как предположение, требующее уточнения
  контекста.
* Затем продолжи проверку остальных тезисов.

**в. Проверка формата:**
* Если в вопросе есть приветствие — первая строка финального ответа
  содержит ответное приветствие?
* Если нет — добавь перед советом.

---

### ПРИМЕР ВНУТРЕННЕЙ САМОПРОВЕРКИ ###

*(Это иллюстрация процесса. В финальный ответ не включается.)*

Черновой тезис: "Проведи ретроспективу и всё наладится."

Проверка:
* **Место:** "и всё наладится"
  * **Проблема:** Критическая ошибка — причинно-следственная связь
    не обоснована. Ретроспектива — инструмент рефлексии, а не
    универсальное решение любой командной проблемы.

Исправленный тезис: "Проведи ретроспективу, сфокусировав её на одной
конкретной проблеме — это даст команде структуру для обсуждения
и повысит шанс выйти с actionable-решением."

### ПРИМЕР ОТВЕТА НА ПРИВЕТСТВИЕ ###

Вопрос сотрудника: "Привет! Как провести ретроспективу?"

Финальный ответ:
Привет!
Начни ретроспективу с выбора формата — например, Start/Stop/Continue
или 4Ls. Зафиксируй не более 2-3 actionable-пунктов в конце встречи
и назначь ответственных за каждый. Без конкретных договорённостей
ретроспектива превращается в разговор ни о чём.

### ПРИМЕР УТОЧНЯЮЩЕГО ВОПРОСА ###

Вопрос сотрудника: "Как улучшить работу команды?"

Финальный ответ:
Чтобы дать точный совет, уточни: в какой методологии работает
команда — Scrum, Kanban или что-то другое?

---

### ВОПРОС СОТРУДНИКА ###

{question}

---

### НАПОМИНАНИЕ О ЗАДАЧЕ ###

Твоя задача — действовать как Agile коуч в быстрой переписке.
Сформируй внутренний набросок, пройди по чеклисту самопроверки,
устрани критические ошибки и пробелы в обосновании, затем выдай
только финальный ответ на русском языке."""

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
            answer = response.content if hasattr(response, "content") else str(response)
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
