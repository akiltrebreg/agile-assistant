"""Supervisor agent for classifying user queries and extracting entities.

This agent classifies the user's intent (task lookup, filtered search,
metric query, or general question) and extracts structured entities
(issue_key, team_name, sprint_name, etc.) using regex + LLM.

It also determines *query_type* for the workflow router:
  sql     -- data lives in PostgreSQL
  rag     -- question about practices, theory, internal docs
  hybrid  -- needs DB data + knowledge-base context
  simple  -- greeting / chitchat, no external data needed
"""

import json
import logging
import re
from typing import Any

from hse_prom_prog.agents.entity_sanitizer import sanitize_entities
from hse_prom_prog.agents.schema_description import (
    SCHEMA_DESCRIPTION,
    SUPERVISOR_FEW_SHOT_EXAMPLES,
)
from hse_prom_prog.llm.client import LLMClient
from hse_prom_prog.memory.formatter import format_history
from hse_prom_prog.models.memory import ConversationContext

logger = logging.getLogger(__name__)

_ISSUE_KEY_RE = re.compile(r"\b([A-Za-z]{2,}-\d+)\b")

# Mapping from (intent) -> default query_type
_INTENT_TO_QUERY_TYPE: dict[str, str] = {
    "task": "sql",
    "tasks_filter": "sql",
    "metric": "sql",
    "general": "simple",
}

# Valid query types
_VALID_QUERY_TYPES = frozenset({"sql", "rag", "hybrid", "simple", "off_topic"})

# Deterministic post-processing markers
_RAG_MARKERS = (
    "целевой порог",
    "бейзлайн",
    "как рассчитывается",
    "что такое",
    "как снизить",
    "как повысить",
)
_HYBRID_MARKERS = (
    "это нормально",
    "можно улучшить",
    "что можно улучшить",
    "дай совет",
    "как с этим бороться",
    "как улучшить",
)
_DANGEROUS_PREFIXES = (
    "drop ",
    "delete ",
    "alter ",
    "truncate ",
    "insert ",
    "update ",
)


# JSON schema for structured output — vLLM enforces this via guided decoding.
# Entity fields are OPTIONAL (no "required" list) so the model can OMIT
# a field when it doesn't apply, instead of being forced to invent a value.
# strict=False because OpenAI strict mode requires all properties in required.
_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "supervisor_classification",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["sql", "rag", "hybrid", "simple", "off_topic"],
                },
                "intent": {
                    "type": "string",
                    "enum": ["task", "tasks_filter", "metric", "general"],
                },
                "entities": {
                    "type": "object",
                    "properties": {
                        "issue_key": {"type": ["string", "null"]},
                        "team_name": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ],
                        },
                        "sprint_name": {"type": ["string", "null"]},
                        "metric_name": {"type": ["string", "null"]},
                        "issue_type": {"type": ["string", "null"]},
                        "status": {"type": ["string", "null"]},
                        "assignee": {"type": ["string", "null"]},
                        "cluster": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["query_type", "intent", "entities"],
            "additionalProperties": False,
        },
    },
}


class SupervisorAgent:
    """Agent that classifies queries and extracts structured entities.

    The Supervisor uses regex for fast issue-key detection and LLM
    for intent classification + entity extraction from natural language.

    Attributes:
        llm_client: LLM client for intent classification.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_engine: Any | None = None,
    ) -> None:
        """Initialize the supervisor.

        Args:
            llm_client: LLM client for slow-path classification.
            db_engine: Optional SQLAlchemy engine. When provided, the
                entity sanitizer validates enum values against real
                DB values (issue_type, status, metric_name columns).
                Without engine, falls back to synonym-only normalization.
        """
        self.llm_client = llm_client
        self.db_engine = db_engine

    def _extract_issue_key_regex(self, text: str) -> str | None:
        """Extract Jira issue key using regex (fast path).

        Normalizes to uppercase so downstream agents see a canonical form
        regardless of whether the user typed 'al-38787' or 'AL-38787'.
        """
        match = _ISSUE_KEY_RE.search(text)
        return match.group(1).upper() if match else None

    @staticmethod
    def _format_profile_block(profile: dict[str, Any] | None) -> str:
        """Render the long-term profile block for the classifier prompt.

        Adds two pieces:
          * ``default_team`` — Supervisor should use it when the query has
            no explicit team mention (tiny bias, ~20 tokens).
          * ``context_summary`` — rolling roll-up of prior sessions, helps
            disambiguate intent for repeat users (~50-100 tokens).

        Returns an empty string when the profile is missing or empty so
        anonymous / first-time users keep the pre-memory prompt verbatim.
        """
        if not profile:
            return ""
        preferences = profile.get("preferences") or {}
        lines: list[str] = []

        default_team = preferences.get("default_team")
        if default_team:
            lines.append(
                f"Команда пользователя по умолчанию: {default_team}. "
                "Если пользователь спрашивает без указания команды — "
                "используй эту команду в entities.team_name."
            )

        context_summary = profile.get("context_summary")
        if context_summary:
            lines.append(f"Контекст пользователя: {context_summary}")

        if not lines:
            return ""
        body = "\n".join(lines)
        return f"<user_profile>\n{body}\n</user_profile>\n\n"

    @staticmethod
    def _format_history_block(ctx: ConversationContext | None) -> str:
        """Render the ``<conversation_history>`` block for the classifier prompt.

        Returns an empty string when there's no context — keeps the prompt
        identical to the pre-memory behaviour so old eval cases stay stable.
        """
        block = format_history(ctx)
        if not block:
            return ""
        return (
            f"{block}\n"
            "Используй историю для разрешения местоимений и ссылок. "
            "Если пользователь говорит «эта команда», «тот спринт», «покажи "
            "ещё» — найди конкретные значения (team_name, sprint_name и т.д.) "
            "в истории и заполни ими entities.\n\n"
        )

    @staticmethod
    def _prev_entities_from_context(
        ctx: ConversationContext | None,
    ) -> dict[str, Any] | None:
        """Pick the most-recent user-turn entities for carry-forward.

        We look at user turns because that's where Supervisor's own
        extraction was stored — assistant turns carry the response payload,
        not the entity dict.
        """
        if ctx is None:
            return None
        for turn in reversed(ctx.get("recent_turns") or []):
            if turn.get("role") != "user":
                continue
            meta = turn.get("metadata") or {}
            entities = meta.get("entities")
            if isinstance(entities, dict) and entities:
                return entities
        return None

    def _classify_with_llm(
        self,
        user_query: str,
        conversation_context: ConversationContext | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classify query intent, entities, and query_type via LLM.

        Returns:
            Dict with 'intent', 'entities', and 'query_type' keys.
        """
        prompt = (
            "Ты — классификатор запросов к базе данных Jira и базе знаний.\n\n"
            f"{SCHEMA_DESCRIPTION}\n\n"
            "Верни ТОЛЬКО JSON (без пояснений, без markdown):\n"
            "{\n"
            '  "intent": "task" | "tasks_filter" | "metric" | "general",\n'
            '  "query_type": "sql" | "rag" | "hybrid" | "simple" | "off_topic",\n'
            '  "entities": {\n'
            '    "issue_key": "ABC-123" | null,\n'
            '    "team_name": "..." | ["...", "..."] | null,\n'
            '    "sprint_name": "..." | null,\n'
            '    "metric_name": "done_total" | null,\n'
            '    "issue_type": "Bug" | null,\n'
            '    "status": "In Progress" | null,\n'
            '    "assignee": "..." | null,\n'
            '    "cluster": "..." | null\n'
            "  }\n"
            "}\n\n"
            "## КРИТИЧНО: нормализация enum-полей\n"
            "Для issue_type, status, metric_name ВСЕГДА переводи русские/\n"
            "разговорные формы к точным enum-значениям:\n"
            "\n"
            "issue_type:\n"
            '  "баг", "багов", "баги", "бага" -> "Bug"\n'
            '  "сторис", "стори", "story" -> "Story"\n'
            '  "улучшение", "улучшения" -> "Improvement"\n'
            '  "эпик", "эпики" -> "Epic"\n'
            '  "задача" (без issue_key, как тип) -> "Task"\n'
            '  "саб-таска", "подзадача" -> "Sub-task"\n'
            "\n"
            "status:\n"
            '  "открыта", "открытые" -> "Open"\n'
            '  "в работе", "в прогрессе" -> "In Progress"\n'
            '  "сделана", "готово", "done" -> "Done"\n'
            '  "закрыта", "закрытые", "closed" -> "Closed"\n'
            '  "отменена", "отменённые", "cancelled" -> "Cancelled"\n'
            "\n"
            "metric_name:\n"
            '  "скорость", "velocity" -> "velocity"\n'
            '  "процент выполнения", "done total" -> "done_total"\n'
            '  "сброс скоупа", "scope drop" -> "scope_drop"\n'
            '  "цель спринта", "sprint goal" -> "sprint_goal"\n'
            '  "доля отменённого", "cancel rate" -> "cancel_rate"\n'
            "\n"
            "НЕ оставляй пустую строку в этих полях — или валидное\n"
            "enum-значение, или null.\n"
            "\n"
            "## Множественные команды\n"
            "Если в запросе упомянуто 2+ команд — верни team_name как массив:\n"
            '  "задачи cthulhu и linehaul" -> team_name=["cthulhu", "linehaul"]\n'
            'Одна команда -> team_name="cthulhu" (строка, не массив).\n'
            "Ни одной -> team_name=null.\n"
            "Сохраняй порядок упоминания.\n\n"
            "## Off-topic\n"
            "query_type=off_topic — запрос НЕ связан с задачами, спринтами,\n"
            "командами, метриками или Agile-практиками разработки.\n"
            "Триггеры off_topic: просьба написать стихи/анекдот/песню/код,\n"
            "вопросы о погоде, еде/рецептах, здоровье, финансах, курсах валют,\n"
            "политике, развлечениях (фильмы, игры), перевод текста.\n"
            "Правило: если запрос связан с работой команды или разработкой —\n"
            "НЕ off_topic. Иначе — off_topic.\n\n"
            "## Алгоритм классификации (следуй строго по шагам)\n"
            "\n"
            "Шаг 0. Off-topic фильтр.\n"
            "  Исключение: приветствия ('Привет', 'Как дела', 'Спасибо'),\n"
            "  мета-вопросы ('Что ты умеешь'), пустая строка — это НЕ\n"
            "  off_topic, переходи к Шагу 1 (обработается в Шаге 2 как simple).\n"
            "  Запрос попадает под триггеры off_topic (стихи, анекдоты,\n"
            "  погода, еда, здоровье, финансы, развлечения, перевод)?\n"
            "  ДА → query_type=off_topic, intent=general, entities={}. СТОП.\n"
            "  Запрос связан с работой команды, задачами, метриками,\n"
            "  спринтами или практиками разработки → к шагу 1.\n"
            "\n"
            "Шаг 1. Есть ли в запросе issue_key в формате [A-Z]+-число "
            "(например, AL-123)?\n"
            "  ДА → query_type=sql, intent=task, entities.issue_key=<ключ>. СТОП.\n"
            "  НЕТ → к шагу 2.\n"
            "\n"
            "Шаг 2. Запрос — приветствие, благодарность, или мета-вопрос "
            "о возможностях?\n"
            "  ('Привет', 'Спасибо', 'Как дела', 'Что ты умеешь', 'Hi')\n"
            "  ДА → query_type=simple, intent=general, entities={}. СТОП.\n"
            "  НЕТ → к шагу 3.\n"
            "\n"
            "Шаг 3. Запрос содержит ОБА признака:\n"
            "  (a) конкретная КОМАНДА или конкретный СПРИНТ или issue_key.\n"
            "      Название метрики БЕЗ команды/спринта — НЕ считается\n"
            "      признаком (a); такой запрос идёт на Шаг 4.\n"
            "  (b) просьба о рекомендациях/совете/объяснении — слова:\n"
            "      'улучшить', 'повысить', 'снизить', 'дай совет',\n"
            "      'что делать', 'это нормально', 'объясни',\n"
            "      'как с этим бороться'.\n"
            "  ДА → query_type=hybrid. Определи intent по шагу 5. Извлеки entities.\n"
            "  НЕТ → к шагу 4.\n"
            "\n"
            "Шаг 4. Запрос — теоретический вопрос?\n"
            "  Критерий: запрос НЕ упоминает ни конкретную команду, ни\n"
            "  конкретный спринт, ни issue_key. Если есть ЛЮБОЙ из этих\n"
            "  конкретных идентификаторов — это НЕ rag, переходи к Шагу 5.\n"
            "\n"
            "  Маркеры теоретического вопроса: 'что такое', 'как\n"
            "  рассчитывается', 'как снизить', 'как идентифицируется',\n"
            "  'какой целевой порог', 'какие бейзлайновые', 'что делать\n"
            "  если', 'расскажи про метрику'.\n"
            "  Также: одно слово-термин без контекста (например, 'Scope drop').\n"
            "\n"
            "  ДА (теоретический И нет команды/спринта/issue_key) →\n"
            "      query_type=rag, intent=general, entities={}. СТОП.\n"
            "  НЕТ → к шагу 5.\n"
            "\n"
            "Шаг 5. Определи intent (применимо для query_type=sql или hybrid).\n"
            "  ПРИОРИТЕТ: проверяй условия ПО ПОРЯДКУ, первое сработавшее\n"
            "  определяет intent.\n"
            "\n"
            "  1) Есть название метрики (velocity, done_total, scope_drop,\n"
            "     sprint_goal, cancel_rate, complete_sp и т.д.) ИЛИ слово\n"
            "     'метрика/метрики' → intent=metric. Это условие перевешивает\n"
            "     любые другие слова ('задачи', 'баги' и т.п.).\n"
            "  2) Есть 'задачи/баги/сторис/эпики/story/задача' БЕЗ issue_key\n"
            "     и БЕЗ названия метрики → intent=tasks_filter.\n"
            "  3) Fallback → intent=tasks_filter, entities={}.\n"
            "\n"
            "  ВАЖНО: intent=task ТОЛЬКО если Шаг 1 нашёл issue_key в формате\n"
            "  [A-Z]+-число. НИКОГДА не выдумывай issue_key и не копируй его\n"
            "  из примеров в промпте (AL-38787, ABC-123 и т.д.).\n"
            "\n"
            "  query_type=sql если шаг 3 не сработал, hybrid если сработал.\n"
            "\n"
            f"{SUPERVISOR_FEW_SHOT_EXAMPLES}\n"
            "\n"
            "## Частые ошибки (избегай!)\n"
            "\n"
            'BAD:  "Расскажи про метрику scope drop" -> query_type=sql/metric\n'
            "GOOD: query_type=rag, intent=general (нет команды, это вопрос\n"
            "      о сути метрики — шаг 4).\n"
            "\n"
            'BAD:  "Задачи по Done Total у команды cthulhu" -> intent=tasks_filter\n'
            "GOOD: intent=metric, query_type=sql, metric_name='done_total'\n"
            "      ЖЁСТКОЕ ПРАВИЛО: слово 'задачи/задач' + название метрики\n"
            "      (done_total, scope_drop, velocity, sprint_goal,\n"
            "      cancel_rate) -> ВСЕГДА intent=metric, НИКОГДА\n"
            "      intent=tasks_filter. Слово 'задачи' здесь обманчиво —\n"
            "      пользователь хочет значение метрики, а не список задач.\n"
            "      Название метрики ПЕРЕВЕШИВАЕТ слово 'задачи'.\n"
            "\n"
            'BAD:  "Scope drop" (одно слово) -> query_type=sql/metric\n'
            "GOOD: query_type=rag, intent=general (без контекста —\n"
            "      вопрос о сути метрики).\n"
            "\n"
            'BAD:  "Как снизить Scope Drop?" -> query_type=hybrid\n'
            "GOOD: query_type=rag, intent=general ('как снизить' — запрос\n"
            "      рекомендации, НО команда не указана -> чистый rag,\n"
            "      не hybrid).\n"
            "\n"
            'BAD:  "Done total команды lpop и что можно улучшить" -> query_type=sql\n'
            "GOOD: query_type=hybrid, intent=metric ('можно улучшить' —\n"
            "      запрос совета + есть команда -> hybrid, шаг 3).\n"
            "\n"
            'BAD:  "Метрики спринта #1 Q1-26" -> intent=tasks_filter\n'
            "GOOD: intent=metric (слово 'метрики' прямо указывает).\n"
            "\n"
            'BAD:  "Метрики спринта #1 Q1\'26" -> query_type=rag, intent=general\n'
            'GOOD: query_type=sql, intent=metric, sprint_name="#1 Q1\'26"\n'
            "      Причина: упомянут конкретный спринт -> это запрос данных,\n"
            "      НЕ теоретический вопрос. Шаг 4 отбраковывает rag, если есть\n"
            "      конкретные команда/спринт/issue_key.\n"
            "\n"
            'BAD:  "Привет! Done Total в спринте 26Q1.1 Конь не валялся" ->\n'
            "      query_type=simple\n"
            "GOOD: query_type=sql, intent=metric (игнорируй приветствие,\n"
            "      классифицируй по СУТИ вопроса).\n"
            "\n"
            'BAD:  "расскажи как считается Done Total и покажи значение у cthulhu"\n'
            "      -> query_type=sql, intent=metric\n"
            "GOOD: query_type=hybrid, intent=metric (комбинация 'расскажи как'\n"
            "      + данные конкретной команды — нужны и теория, и цифры).\n"
            "\n"
            'BAD:  "Покажи все баги" -> intent=task, issue_key="ALL-BAIGS"\n'
            "      (выдуманное значение!)\n"
            "GOOD: intent=tasks_filter, issue_type='Bug', остальные entities=null\n"
            "      Причина: множественное число + 'баги/задачи/сторис' без\n"
            "      issue_key -> tasks_filter. НЕ выдумывай issue_key.\n"
            "\n"
            'BAD:  "Закрытые баги команды lpop" -> intent=task\n'
            "GOOD: intent=tasks_filter, team_name='lpop', issue_type='Bug',\n"
            "      status='Closed'.\n"
            "      ПРАВИЛО: множественное число ('баги', 'задачи', 'сторис',\n"
            "      'эпики', 'сабтаски') БЕЗ issue_key в формате [A-Z]+-число\n"
            "      -> ВСЕГДА intent=tasks_filter, НИКОГДА intent=task.\n"
            "      intent=task — только когда Шаг 1 нашёл явный issue_key.\n"
            "\n"
            'BAD:  "Покажи задачи" -> intent=task, issue_key="ABC-123"\n'
            "      (скопировано из примера в промпте!)\n"
            "GOOD: intent=tasks_filter, все entities=null\n"
            "      Причина: 'задачи' без фильтра -> tasks_filter с пустыми\n"
            "      entities. Примеры issue_key (AL-38787, ABC-123) —\n"
            "      это иллюстрация формата, а не значение для подстановки.\n"
            "\n"
            f"{self._format_profile_block(user_profile)}"
            f"{self._format_history_block(conversation_context)}"
            f'Запрос пользователя: "{user_query}"\n'
            "JSON:"
        )

        # JSON response is ~30-60 tokens; even a maximal hybrid answer stays
        # under ~120 tokens. 256 gives ~4x headroom while returning ~256
        # tokens of prompt budget vs. the old max_tokens=512.
        llm_response = self.llm_client.invoke(
            prompt,
            response_format=_RESPONSE_FORMAT,
            max_tokens=256,
        )
        logger.info("[Supervisor] Raw LLM response: %s", llm_response[:300])
        return self._parse_llm_json(llm_response)

    def _parse_llm_json(self, raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences.

        With structured output enabled, the first json.loads should succeed.
        The regex fallback is kept only for legacy/degraded responses.
        On total parse failure, returns explicit error intent — not a silent
        'general' fallback — so the caller can distinguish parsing errors
        from legitimate general queries.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()

        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None

        if parsed is None:
            logger.warning("[Supervisor] Failed to parse LLM JSON: %s", raw[:200])
            return {"intent": "error", "entities": {}, "query_type": "error"}

        intent = parsed.get("intent", "general")
        if intent not in ("task", "tasks_filter", "metric", "general"):
            intent = "general"

        entities = parsed.get("entities", {})
        entities = {k: v for k, v in entities.items() if v is not None}

        query_type = parsed.get("query_type")
        if query_type not in _VALID_QUERY_TYPES:
            query_type = _INTENT_TO_QUERY_TYPE.get(intent, "simple")

        return {"intent": intent, "entities": entities, "query_type": query_type}

    def _post_process_classification(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        """Deterministic corrections applied after LLM classification.

        Fixes residual misclassifications that the LLM keeps making despite
        anti-examples. Rules:
          1. Dangerous SQL prefix (DROP/DELETE/…) -> simple/general (safety).
          2. rag-marker words without team/sprint -> rag/general.
          3. hybrid-marker words WITH team/sprint -> hybrid (keep intent).
        """
        query_lower = query.lower()

        # Rule 1: SQL-injection attempt -> block early
        if any(query_lower.startswith(d) for d in _DANGEROUS_PREFIXES):
            logger.info("[Supervisor] Post-process: dangerous prefix -> simple")
            return {
                "query_type": "simple",
                "intent": "general",
                "entities": {},
            }

        entities = result.get("entities", {}) or {}
        qt = result.get("query_type")
        has_team = bool(entities.get("team_name"))
        has_sprint = bool(entities.get("sprint_name"))
        has_ctx = has_team or has_sprint

        # Rule 2: rag-marker without team/sprint -> rag
        if qt == "sql" and not has_ctx and any(m in query_lower for m in _RAG_MARKERS):
            logger.info("[Supervisor] Post-process: rag-marker, no team -> rag")
            result = dict(result)
            result["query_type"] = "rag"
            result["intent"] = "general"
            result["entities"] = {}
            return result

        # Rule 3: hybrid-marker + team/sprint -> hybrid
        if qt in ("sql", "rag") and has_ctx and any(m in query_lower for m in _HYBRID_MARKERS):
            logger.info("[Supervisor] Post-process: hybrid-marker + team -> hybrid")
            result = dict(result)
            result["query_type"] = "hybrid"
            if not result.get("intent") or result["intent"] == "general":
                result["intent"] = "metric"
            return result

        return result

    def process(
        self,
        user_query: str,
        conversation_context: ConversationContext | None = None,
        user_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classify the user query and extract structured entities.

        Fast path: if regex finds an issue key and the query is simple
        (just the key or "задача KEY"), skip the LLM call entirely.

        ``conversation_context`` is optional — when provided, it is injected
        into the classifier prompt for anaphora resolution and used to
        carry-forward entities from prior turns via ``sanitize_entities``.

        ``user_profile`` is optional — when provided, ``preferences``
        (notably ``default_team``) and ``context_summary`` are surfaced in
        the classifier prompt so the LLM can default team fields and bias
        intent toward the user's usual queries.

        Returns state update with: original_query, intent, entities,
        query_type, route.
        """
        logger.info("[Supervisor] Processing query: %s", user_query)

        # Fast path: regex finds an issue key
        issue_key = self._extract_issue_key_regex(user_query)

        if issue_key:
            logger.info("[Supervisor] Fast path — issue key found: %s", issue_key)
            return {
                "original_query": user_query,
                "intent": "task",
                "entities": {"issue_key": issue_key},
                "query_type": "sql",
                "route": "db_query",
            }

        # Slow path: LLM classification
        logger.info("[Supervisor] No issue key in regex, calling LLM")
        try:
            classification = self._classify_with_llm(
                user_query,
                conversation_context,
                user_profile,
            )
        except Exception as e:
            logger.error("[Supervisor] LLM classification failed: %s", e)
            return {
                "original_query": user_query,
                "intent": "error",
                "entities": {},
                "query_type": "error",
                "route": "direct_response",
                "error": f"Classifier unavailable: {type(e).__name__}: {e}",
            }

        # 1) Sanitize entities FIRST — drop hallucinated team_name etc.
        #    Post-processing makes routing decisions based on entities
        #    (has_team/has_sprint), so it must see clean data.
        classification["entities"] = sanitize_entities(
            classification.get("entities", {}),
            user_query,
            engine=self.db_engine,
            prev_entities=self._prev_entities_from_context(conversation_context),
        )

        # 1b) Profile-driven fallback: if the sanitizer ended up with no
        #     team (either the LLM didn't pick it up, or the team wasn't
        #     mentioned in the query and got dropped as a "hallucination"),
        #     substitute the user's default team. Runs AFTER sanitization
        #     on purpose — the profile value is a trusted signal, not LLM
        #     output, and does not need hallucination filtering.
        default_team = ((user_profile or {}).get("preferences") or {}).get("default_team")
        if default_team and not classification["entities"].get("team_name"):
            classification["entities"]["team_name"] = default_team
            logger.info(
                "[Supervisor] Applied profile default_team=%s to empty entities.team_name",
                default_team,
            )

        # 2) Post-process classification — routing based on CLEAN entities
        classification = self._post_process_classification(user_query, classification)

        intent = classification["intent"]
        entities = classification["entities"]
        query_type = classification["query_type"]

        route = "direct_response" if query_type == "simple" else "db_query"

        logger.info(
            "[Supervisor] intent=%s, query_type=%s, entities=%s, route=%s",
            intent,
            query_type,
            entities,
            route,
        )

        return {
            "original_query": user_query,
            "intent": intent,
            "entities": entities,
            "query_type": query_type,
            "route": route,
        }
