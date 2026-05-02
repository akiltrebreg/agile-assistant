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


# The current judge prompt does not assign weights — all six criteria
# count equally. weighted_total = mean(scores), so the Prometheus/Grafana
# 0-1 scale (alerts at 0.7, dashboard thresholds 0.6/0.8) keeps working
# unchanged. The dict is still indexed because the rest of the module
# uses its keys as the canonical criterion list.
CRITERIA_WEIGHTS: dict[str, float] = {
    "agile_correctness": 1 / 6,
    "practicality": 1 / 6,
    "context_handling": 1 / 6,
    "politeness": 1 / 6,
    "text_cleanliness": 1 / 6,
    "language_quality": 1 / 6,
}


# System prompt — a strict, step-by-step verifier rubric. Curly braces
# in the embedded JSON example are doubled because this string is fed
# verbatim to the LLM and must not be touched by str.format(); the
# user template is the only string we .format(), and it has its own
# placeholders (``question``, ``answer``).
JUDGE_SYSTEM_PROMPT = """\
Ты — строгий проверяющий. Твоя задача — оценить ответ Agile коуча
по 6 критериям. Каждый критерий бинарный: 0 или 1.

### ОСНОВНЫЕ ИНСТРУКЦИИ ###

* Ты действуешь исключительно как **верификатор**, а не как советчик.
* Проверяй ответ **шаг за шагом** по каждому критерию.
* Для каждого нарушения **цитируй конкретный фрагмент** ответа.
* Не допускай промежуточных значений: только 0 или 1.
* У каждого критерия свой порог — читай внимательно.
* Оценивай только финальный ответ модели, не черновик.
* Помни: это ответ в корпоративном мессенджере.

---

### КРИТЕРИИ ###

**1. agile_correctness — СТРОГИЙ порог. При сомнении — ставь 0.**

Задай себе два вопроса:
1. "Есть ли в ответе конкретная Agile-практика?"
2. "Объясняет ли ответ, зачем эта практика нужна именно здесь?"

1 — оба ответа "да":
  - упомянута конкретная практика (ретроспектива, дейли, бэклог,
    спринт, планирование, демо, воркшоп, канбан-доска)
  - есть хотя бы минимальное объяснение связи с проблемой

0 — при любом из:
  - совет является общим менеджерским без Agile-практики:
    "поговорите с командой", "обсудите ожидания",
    "выстраивайте доверие", "назначьте ответственного",
    "улучшайте процессы", "повышайте прозрачность"
  - практика упомянута, но как дежурное слово без связи
    с проблемой ("проведите ретроспективу" в ответ на вопрос
    про найм — без объяснения зачем)
  - практика приписана неверной методологии
  - совет противоречит Agile Manifesto или Scrum Guide

---

**2. practicality — ДВУХСТУПЕНЧАТАЯ проверка.**

Ступень 1. Есть ли конкретный элемент?
  Найди в ответе хотя бы одно из:
  - встреча или церемония (ретроспектива, дейли, планирование, демо)
  - артефакт (бэклог, DoD, story points, канбан-доска)
  - инструмент (Jira, Confluence, воркшоп)
  - конкретный глагол + предмет ("добавь в бэклог",
    "обсуди на ретро", "зафиксируй в DoD")
  Если ничего нет → 0, дальше не проверяй.

Ступень 2. Это конкретный совет или дежурное слово?
  Если конкретный элемент найден — проверь:
  есть ли в ответе хотя бы одно предложение, которое
  объясняет, КАК или ЗАЧЕМ этот элемент применить
  к данному вопросу?

  1 — элемент присутствует И есть объяснение его применения:
    пример хорошего ответа: "Вынеси этот вопрос на ретроспективу
    и попроси команду назвать конкретные шаги для улучшения"

  0 — элемент упомянут как дежурное слово без объяснения:
    примеры нарушений:
    "Рекомендую использовать ретроспективу и бэклог"
    "Попробуйте применить практики Scrum в вашей команде"
    "Проведите дейли для синхронизации команды" — если вопрос
    не про синхронизацию, а про что-то другое, и объяснения нет

---

**3. context_handling — МЯГКИЙ порог. При сомнении — ставь 1.**

Определи: вопрос конкретный или общий?

Конкретный (есть методология, роль, конкретная проблема):
  1 — дан совет по существу
  0 — модель уклонилась без причины

Общий (нет деталей, нет методологии):
  1 — выполнено хотя бы одно:
    * задан уточняющий вопрос
    * названо допущение ("если работаете по Scrum...")
    * совет применим при любом контексте
  0 — специфичный совет без оговорок при явно общем вопросе

---

**4. politeness — МЯГКИЙ порог. При сомнении — ставь 1.**

1 — дружелюбный профессиональный тон, ответное приветствие
  если сотрудник поздоровался.
0 — только явное нарушение: игнорировано приветствие,
  менторский ("Вы должны понимать, что...") или
  фамильярный тон.

---

**5. text_cleanliness — МЯГКИЙ порог. При сомнении — ставь 1.**

1 — нет эмодзи, висящих символов (* ** :) в начале,
  артефактов генерации.
0 — только явное: эмодзи, текст начинается с ** или :,
  очевидный артефакт генерации.

---

**6. language_quality — УМЕРЕННЫЙ порог.**

Допустимые Agile-термины на английском (не нарушение):
спринт, бэклог, скрам, ретроспектива, стендап, дейли,
канбан, эпик, velocity, roadmap, actionable, фреймворк,
фасилитация, воркшоп, Agile, Scrum, Kanban, SAFe, LeSS,
Jira, Confluence, демо, ревью.

1 — текст написан преимущественно на русском языке,
  английские слова только из списка допустимых или
  используются единично для точности.

0 — при любом из:
  - орфографические ошибки, затрудняющие понимание
  - замена стандартных русских слов английскими без нужды:
    "имплементировать" → "внедрить",
    "коммитить" → "брать в работу",
    "перформанс" → "производительность",
    "менеджить" → "управлять",
    "митинг" → "встреча" (НО: стендап, дейли — допустимо)
  - в ответе 3 и более английских слов НЕ из списка
    допустимых терминов
  - текст написан преимущественно на английском

---

### ИНСТРУКЦИЯ ПО ВЕРИФИКАЦИИ ###

Шаг 1. Прочитай вопрос и ответ полностью. Запомни суть вопроса.

Шаг 2. agile_correctness: есть ли практика И объяснена ли связь
  с проблемой? Оба "да" → 1, иначе → 0.

Шаг 3. practicality — двухступенчато:
  Сначала: есть ли конкретный элемент (встреча/артефакт)?
  Нет → 0. Есть → проверяй дальше.
  Затем: объяснено ли, как/зачем применить к данному вопросу?
  Да → 1. Нет, просто упомянуто как дежурное слово → 0.

Шаг 4. language_quality: есть ли слова с очевидным русским
  эквивалентом или 3+ недопустимых англицизма? Да → 0, иначе → 1.

Шаг 5. Остальные критерии: ищи явное нарушение.
  Нет явного нарушения — ставь 1.

Шаг 6. Согласованность: agile_correctness=0 из-за отсутствия
  Agile-практики → practicality скорее всего тоже 0.

Шаг 7. Сформируй JSON.

---

### ФОРМАТ ВЫВОДА ###

Верни результат строго в формате JSON без пояснений вне блока:
```json
{
  "agile_correctness": <0 или 1>,
  "practicality": <0 или 1>,
  "context_handling": <0 или 1>,
  "politeness": <0 или 1>,
  "text_cleanliness": <0 или 1>,
  "language_quality": <0 или 1>,
  "total": <сумма 0-6>,
  "violations": [
    {"criterion": "<критерий>", "quote": "<цитата>", "reason": "<причина>"}
  ]
}
```
"""


# User template — only ``question`` and ``answer`` placeholders. The
# system prompt already specifies the rubric; we deliberately do not
# echo query_type into the prompt because the new rubric does not
# distinguish on it. ``query_type`` stays in the task signature for
# logging / future routing rules.
JUDGE_USER_TEMPLATE = """\
### ДАННЫЕ ДЛЯ ОЦЕНКИ ###
Вопрос сотрудника: {question}
Финальный ответ модели: {answer}
"""


# Match the judge model used by RAGAS eval (eval/metrics.py) so quality
# numbers between offline and online evaluation stay comparable.
_JUDGE_MODEL = "openai/gpt-5.2"
_JUDGE_TEMPERATURE = 0.0
# Bumped from 200: the rubric asks the judge to emit a ``violations``
# array with quoted fragments and reasons in addition to the six 0/1
# scores. 600 covers a worst-case 6-violation answer with comfortable
# margin without inflating cost (typical responses are <200 tokens).
_JUDGE_MAX_TOKENS = 600
_JUDGE_RETRY_COUNTDOWN_SECONDS = 15


def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` fences if the LLM wrapped its JSON.

    Args:
        raw: Raw assistant text returned by the judge LLM.

    Returns:
        The body without surrounding triple-backtick fences.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence (```json or just ```), keep the body.
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        text = text.removesuffix("```")
    return text.strip()


def _validate_scores(parsed: Any) -> dict[str, int] | None:
    """Validate that ``parsed`` carries six binary criterion scores.

    Args:
        parsed: Decoded JSON payload returned by the judge.

    Returns:
        Scores dict when every criterion has an ``int`` value in ``{0, 1}``,
        otherwise ``None``.
    """
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
    """Push per-criterion gauges, the weighted total and a success counter.

    Args:
        scores: Validated per-criterion 0/1 scores.
        weighted_total: Equal-weighted aggregate score in ``[0, 1]``.
    """
    for criterion, value in scores.items():
        JUDGE_CRITERION_SCORES.labels(criterion=criterion).set(value)
    JUDGE_WEIGHTED_TOTAL.set(weighted_total)
    JUDGE_EVALUATIONS_TOTAL.labels(status="success").inc()


def _record_langfuse_scores(
    trace_id: str,
    scores: dict[str, int],
    weighted_total: float,
) -> None:
    """Attach per-criterion and weighted scores to the originating trace.

    Uses the imperative client (not ``@observe`` / contextvars) because the
    workflow's root trace is already finalised by the time the judge task
    runs. Pass an empty ``trace_id`` to skip Langfuse entirely.

    Args:
        trace_id: Langfuse trace id from the workflow, or ``""`` to skip.
        scores: Validated per-criterion 0/1 scores.
        weighted_total: Equal-weighted aggregate score in ``[0, 1]``.
    """
    if not trace_id or langfuse_client is None:
        return
    try:
        for criterion, value in scores.items():
            langfuse_client.score(
                trace_id=trace_id,
                name=f"judge_{criterion}",
                value=value,
                comment="binary criterion (0/1), equal weight",
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


def _call_judge_llm(query: str, response: str) -> str:
    """Run a single GPT-5.2 chat completion and return raw assistant text.

    Lazy import of ``openai`` so the module loads cleanly when the SDK
    is absent (judge disabled, dev environment). Caller decides whether
    to retry on the OpenAI-shaped exceptions raised here.

    Args:
        query: Original user question.
        response: Final assistant response to evaluate.

    Returns:
        Raw text returned by the judge LLM (may include markdown fences).
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
                "content": JUDGE_USER_TEMPLATE.format(question=query, answer=response),
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
    query_type: str = "",
) -> dict[str, Any]:
    """Score ``response`` against ``query`` on six criteria via GPT-5.2.

    Records results into Prometheus (always, when judging happens) and
    Langfuse (only when ``trace_id`` is non-empty and the SDK is live).
    Returns a small status dict for Celery logging — judge results are
    *not* a result-backend concern, the dict is purely diagnostic.

    Retry semantics: API-level errors retry up to ``max_retries`` with a
    fixed 15s countdown; parse errors and soft-time-limit hits do not
    retry (the same prompt typically produces the same broken output).

    Args:
        trace_id: Langfuse trace id from ``workflow_task``. Empty string
            when Langfuse is disabled — judge still runs and writes
            Prometheus metrics.
        query: Original user question.
        response: Final assistant response (post-guardrails).
        query_type: Workflow classification (``sql``/``rag``/``hybrid``/
            ``simple``). Currently unused by the judge prompt itself but
            kept in the signature so old in-flight Celery messages don't
            crash after deploy and the field is available for future
            routing rules.

    Returns:
        Diagnostic status dict with ``status`` (one of ``success``,
        ``skipped``, ``parse_error``, ``api_error``) plus ``trace_id``
        and, on success, ``scores`` and ``weighted_total``.
    """
    # Step A — kill switch. Settings re-read on each call so a docker
    # env flip is picked up without restarting the worker.
    if not settings.judge_enabled:
        JUDGE_EVALUATIONS_TOTAL.labels(status="skipped").inc()
        logger.debug("[Judge] Skipped (JUDGE_ENABLED=false), trace=%s", trace_id or "-")
        return {"status": "skipped", "trace_id": trace_id}

    _ = query_type  # accepted for compatibility, see docstring
    # Step B — call GPT-5.2.
    try:
        raw = _call_judge_llm(query=query, response=response)
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
