"""LLM-as-a-Judge — asynchronous response quality scoring (Phase 4).

After every successful workflow finalises its response, ``workflow_task``
fires this Celery task off into the dedicated ``judge`` queue. The task
calls GPT-5.2 (via vsellm) to grade the answer on six binary criteria,
records the per-criterion scores plus a weighted total to Prometheus,
and attaches the same scores to the originating Langfuse trace.

Why a dedicated queue:
    The judge LLM call can stall for tens of seconds (rate limits,
    network blips, 5xx). Sharing a queue with workflow tasks would let
    a slow judge starve user-facing requests. ``celery-judge`` is a
    separate worker process listening on ``--queues=judge`` only.

Why no @observe wrapper:
    The judge runs *after* the workflow's root trace is finalised and
    flushed. langfuse_context (contextvars) is empty by then, so an
    @observe span would land orphaned in Langfuse. Scores are attached
    imperatively via ``langfuse_client.score(trace_id=...)`` instead.

Failure model:
    * ``parse_error`` (invalid JSON): not retried — same prompt usually
      gives the same broken output. Logged + counted, returned to caller.
    * ``api_error`` (connection / 5xx / rate limit): retried up to
      ``max_retries`` with a 15s backoff. After that, Celery marks the
      task FAILED. Judge is best-effort, so a permanently dead vsellm
      must not bring the workflow path down.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from hse_prom_prog.config import settings
from hse_prom_prog.metrics import (
    JUDGE_CRITERION_SCORES,
    JUDGE_EVALUATIONS_TOTAL,
    JUDGE_WEIGHTED_TOTAL,
)
from hse_prom_prog.tasks.celery_app import celery_app
from hse_prom_prog.tracing import langfuse_client

logger = logging.getLogger(__name__)


# Weights chosen so practicality dominates (it's the metric users feel
# day-to-day) and language/cleanliness together match it (a polished but
# useless answer must not score above an actionable rough one).
CRITERIA_WEIGHTS: dict[str, float] = {
    "practicality": 0.30,
    "language_quality": 0.20,
    "text_cleanliness": 0.20,
    "agile_correctness": 0.10,
    "completeness": 0.10,
    "politeness": 0.10,
}


JUDGE_SYSTEM_PROMPT = """\
Ты строгий оценщик качества ответов AI-ассистента, который анализирует
Jira-задачи и Agile-метрики. На вход тебе даётся запрос пользователя
(возможно с типом запроса) и ответ ассистента. Твоя задача — выставить
по каждому из 6 критериев оценку 1 (критерий выполнен) или 0 (не выполнен).

Критерии:

1. practicality (вес 0.30) — практичность.
   1: ответ содержит конкретные данные (числа, статусы, имена), actionable
      рекомендации и/или чёткие шаги действий.
   0: ответ абстрактный, общий, без конкретики, бесполезен на практике.

2. language_quality (вес 0.20) — качество языка.
   1: текст грамотный, предложения структурированы, падежи и согласования
      правильные.
   0: грамматические ошибки, несогласованные предложения, нечитаемый текст.

3. text_cleanliness (вес 0.20) — чистота текста.
   1: чистый текст без артефактов генерации — нет raw SQL, markdown-разметки
      (```, ###), английских технических вставок, traceback, внутренних
      идентификаторов (имена таблиц, имена колонок, connection strings).
   0: видны SQL-запросы, markdown-мусор, Python traceback, имена внутренних
      таблиц, connection strings.

4. agile_correctness (вес 0.10) — Agile-корректность.
   1: Agile-термины (velocity, scope drop, story points, sprint goal)
      использованы корректно, метрики интерпретируются верно.
   0: путаница в терминах, неверная интерпретация метрик, фактические ошибки
      в Agile-контексте.
   Если в ответе нет Agile-терминов (например, приветствие) — выставляй 1.

5. completeness (вес 0.10) — полнота.
   1: все части запроса адресованы, нет значимых пропусков.
   0: часть вопроса проигнорирована, важная информация пропущена.

6. politeness (вес 0.10) — вежливость.
   1: вежливый, профессиональный тон, без грубости или снисходительности.
   0: грубость, снисходительность, чрезмерная холодность.

Формат ответа: ТОЛЬКО JSON без пояснений и без markdown-обёрток.
Пример:
{"practicality": 0, "language_quality": 1, "text_cleanliness": 1, \
"agile_correctness": 1, "completeness": 0, "politeness": 1}
"""


JUDGE_USER_TEMPLATE = """\
## Запрос пользователя
{query}

## Тип запроса
{query_type}

## Ответ ассистента
{response}
"""


# Match the judge model used by RAGAS eval (eval/metrics.py) so quality
# numbers between offline and online evaluation stay comparable.
_JUDGE_MODEL = "openai/gpt-5.2"
_JUDGE_TEMPERATURE = 0.0
_JUDGE_MAX_TOKENS = 200
_JUDGE_RETRY_COUNTDOWN_SECONDS = 15


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` fences if the LLM wrapped its JSON."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence (```json or just ```), keep the body.
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        text = text.removesuffix("```")
    return text.strip()


def _validate_scores(parsed: Any) -> dict[str, int] | None:
    """Return the scores dict if it has all 6 keys with int 0/1 values; else None."""
    if not isinstance(parsed, dict):
        return None
    scores: dict[str, int] = {}
    for criterion in CRITERIA_WEIGHTS:
        value = parsed.get(criterion)
        # Reject bools — bool is a subclass of int in Python and would
        # otherwise sneak through as 0/1.
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        if value not in (0, 1):
            return None
        scores[criterion] = value
    return scores


def _record_prometheus(scores: dict[str, int], weighted_total: float) -> None:
    """Push the per-criterion gauges, the weighted total, and a success counter."""
    for criterion, value in scores.items():
        JUDGE_CRITERION_SCORES.labels(criterion=criterion).set(value)
    JUDGE_WEIGHTED_TOTAL.set(weighted_total)
    JUDGE_EVALUATIONS_TOTAL.labels(status="success").inc()


def _record_langfuse_scores(
    trace_id: str,
    scores: dict[str, int],
    weighted_total: float,
) -> None:
    """Attach per-criterion + weighted total scores to the originating trace.

    Uses the imperative client (not @observe / contextvars) because the
    workflow's root trace is already finalised by the time the judge task
    runs. ``trace_id`` is required — pass an empty string to skip Langfuse.
    """
    if not trace_id or langfuse_client is None:
        return
    try:
        for criterion, value in scores.items():
            langfuse_client.score(
                trace_id=trace_id,
                name=f"judge_{criterion}",
                value=value,
                comment=f"weight={CRITERIA_WEIGHTS[criterion]}",
            )
        langfuse_client.score(
            trace_id=trace_id,
            name="judge_weighted_total",
            value=weighted_total,
        )
        langfuse_client.flush()
    except Exception as exc:
        # Score persistence is best-effort. Prometheus already has the
        # numbers; do not let a Langfuse hiccup retry the LLM call.
        logger.warning("[Judge] Langfuse score write failed (trace=%s): %s", trace_id, exc)


def _call_judge_llm(query: str, response: str, query_type: str) -> str:
    """Single GPT-5.2 chat completion. Returns raw assistant text.

    Lazy import of ``openai`` so the module loads cleanly when the SDK
    is absent (judge disabled, dev environment). Caller decides whether
    to retry on the OpenAI-shaped exceptions raised here.
    """
    from openai import OpenAI  # noqa: PLC0415 — lazy import, see docstring

    client = OpenAI(
        api_key=settings.vsellm_api_key,
        base_url=settings.vsellm_base_url,
    )
    completion = client.chat.completions.create(
        model=_JUDGE_MODEL,
        temperature=_JUDGE_TEMPERATURE,
        max_tokens=_JUDGE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    query=query, query_type=query_type, response=response
                ),
            },
        ],
    )
    return completion.choices[0].message.content or ""


@celery_app.task(
    bind=True,
    name="evaluate_response",
    queue="judge",
    max_retries=2,
    soft_time_limit=30,
    time_limit=60,
    acks_late=True,
)
def evaluate_response_async(
    self,
    trace_id: str,
    query: str,
    response: str,
    query_type: str,
) -> dict[str, Any]:
    """Score *response* against *query* on six criteria via GPT-5.2.

    Records results into Prometheus (always, when judging happens) and
    Langfuse (only when ``trace_id`` is non-empty and the SDK is live).
    Returns a small status dict for Celery logging — judge results are
    *not* a result-backend concern, the dict is purely diagnostic.

    Args:
        trace_id: Langfuse trace id from workflow_task. Empty string when
            Langfuse is disabled — judge still runs and writes Prometheus.
        query: Original user question.
        response: Final assistant response (post-guardrails).
        query_type: Workflow classification (sql/rag/hybrid/simple). Sent
            to GPT-5.2 so it knows e.g. completeness expectations differ
            for ``simple`` (greeting) vs ``hybrid``.
    """
    # Step A — kill switch. Settings re-read on each call so a docker
    # env flip is picked up without restarting the worker.
    if not settings.judge_enabled:
        JUDGE_EVALUATIONS_TOTAL.labels(status="skipped").inc()
        logger.debug("[Judge] Skipped (JUDGE_ENABLED=false), trace=%s", trace_id or "-")
        return {"status": "skipped", "trace_id": trace_id}

    # Step B — call GPT-5.2.
    try:
        raw = _call_judge_llm(query=query, response=response, query_type=query_type)
    except SoftTimeLimitExceeded:
        # 30s soft limit hit — log and exit, do not retry. The vsellm
        # call already burned the budget; another attempt would just
        # double the wasted time.
        JUDGE_EVALUATIONS_TOTAL.labels(status="api_error").inc()
        logger.error("[Judge] Soft time limit exceeded, trace=%s", trace_id or "-")
        return {"status": "api_error", "trace_id": trace_id, "reason": "timeout"}
    except Exception as exc:
        # Treat every API-side failure (connection, rate limit, 5xx) as
        # retryable. After max_retries, Celery raises and the task is
        # marked failed by the standard signal handler in celery_app.py.
        JUDGE_EVALUATIONS_TOTAL.labels(status="api_error").inc()
        logger.error("[Judge] API error (trace=%s): %s", trace_id or "-", exc)
        try:
            raise self.retry(exc=exc, countdown=_JUDGE_RETRY_COUNTDOWN_SECONDS)
        except MaxRetriesExceededError:
            return {"status": "api_error", "trace_id": trace_id, "reason": str(exc)}

    # Step C+D — parse + validate.
    cleaned = _strip_markdown_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        JUDGE_EVALUATIONS_TOTAL.labels(status="parse_error").inc()
        logger.warning(
            "[Judge] JSON decode failed (trace=%s) raw=%r",
            trace_id or "-",
            cleaned[:200],
        )
        return {"status": "parse_error", "trace_id": trace_id, "raw": cleaned[:500]}

    scores = _validate_scores(parsed)
    if scores is None:
        JUDGE_EVALUATIONS_TOTAL.labels(status="parse_error").inc()
        logger.warning(
            "[Judge] Score validation failed (trace=%s) parsed=%r",
            trace_id or "-",
            parsed,
        )
        return {"status": "parse_error", "trace_id": trace_id, "raw": cleaned[:500]}

    # Step E — weighted total.
    weighted_total = sum(
        scores[criterion] * weight for criterion, weight in CRITERIA_WEIGHTS.items()
    )

    # Step F — Prometheus.
    _record_prometheus(scores, weighted_total)

    # Step G — Langfuse.
    _record_langfuse_scores(trace_id, scores, weighted_total)

    logger.info(
        "[Judge] trace=%s weighted=%.2f scores=%s",
        trace_id or "-",
        weighted_total,
        scores,
    )
    return {
        "status": "success",
        "trace_id": trace_id,
        "scores": scores,
        "weighted_total": weighted_total,
    }
