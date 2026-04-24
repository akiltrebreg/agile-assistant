"""Response agent for generating natural language responses using LLM.

Takes validated outputs from SQL Agent and/or RAG Agent and generates
a natural language response. Supports four query_types:
  sql     -- DB data only
  rag     -- knowledge-base context + sources
  hybrid  -- DB data + knowledge-base context
  simple  -- direct LLM answer (no external data)
"""

import json
import logging
from datetime import datetime
from typing import Any

from hse_prom_prog.llm.client import LLMClient
from hse_prom_prog.memory.formatter import format_history
from hse_prom_prog.memory.token_estimator import estimate_tokens
from hse_prom_prog.models.memory import ConversationContext

logger = logging.getLogger(__name__)

# Maximum rows to include in the LLM prompt to avoid token overflow
_MAX_ROWS_IN_PROMPT = 20

_SYSTEM_ROLE = (
    "Ты — ассистент для анализа данных о Jira-задачах и Agile-метриках. "
    "Отвечай на русском языке, по делу, без воды и без выдумывания данных."
)

# Per-branch history budget, in tokens, for the short-term memory block
# injected into the user prompt. The hybrid branch is the tightest because
# it already carries both SQL payload and RAG context.
_BRANCH_HISTORY_BUDGET: dict[str, int] = {
    "sql": 1500,
    "rag": 1000,
    "hybrid": 800,
    "simple": 1500,
}
_MIN_HISTORY_BUDGET_TOKENS = 300
_HISTORY_INSTRUCTION = (
    "Не повторяй то, что уже было сказано в истории. "
    "На уточняющий вопрос отвечай кратко, без повторения контекста."
)


class ResponseAgent:
    """Agent that generates natural language responses using LLM.

    Attributes:
        llm_client: LLM client for generating responses.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        logger.info("[Response Agent] Initialized with LLM client")

    # ------------------------------------------------------------------
    # Value formatting
    # ------------------------------------------------------------------

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "не указано"
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        if isinstance(value, bool):
            return "да" if value else "нет"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    # ------------------------------------------------------------------
    # Data preparation for LLM
    # ------------------------------------------------------------------

    def _prepare_single_task(self, row: dict[str, Any]) -> str:
        """Format a single task row with Russian labels."""
        labels = {
            "issue_key": "Ключ задачи",
            "issue_project": "Проект",
            "issue_type": "Тип задачи",
            "summary": "Описание",
            "issue_status_act": "Текущий статус",
            "issue_status_end_of_sprint": "Статус в конце спринта",
            "create_time": "Дата создания",
            "resolution": "Резолюция",
            "resolution_time": "Время резолюции",
            "sprint_name": "Спринт",
            "sprint_state": "Состояние спринта",
            "feature_teams": "Команда",
            "reporter": "Репортер",
            "assignee_name": "Исполнитель",
            "cluster": "Кластер",
            "unit": "Подразделение",
            "storypoints_act": "Story Points",
            "time_h_in_progress": "Время в работе (ч)",
            "merged_pr_count": "Количество PR",
            "dev_approach": "Подход разработки",
            "epic_issue_key": "Эпик",
            "labels": "Метки",
        }
        formatted = {}
        for col, label in labels.items():
            val = row.get(col)
            if val is not None:
                formatted[label] = self._format_value(val)
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    def _prepare_task_list(self, rows: list[dict[str, Any]], include_assignee: bool = False) -> str:
        """Format a list of tasks as a compact summary.

        If `include_assignee` is True, adds assignee_name column — useful
        when the user filter is by assignee, so the model can confirm.
        """
        total = len(rows)
        shown = rows[:_MAX_ROWS_IN_PROMPT]
        lines = []
        for i, row in enumerate(shown, 1):
            key = row.get("issue_key", "?")
            summary = row.get("summary", "")[:80]
            status = row.get("issue_status_act", "?")
            team = row.get("feature_teams", "?")
            sp = row.get("storypoints_act", "?")
            base = f"{i}. {key} | {status} | {team} | SP:{sp}"
            if include_assignee:
                assignee = row.get("assignee_name", "?")
                base += f" | @{assignee}"
            lines.append(f"{base} | {summary}")
        result = "\n".join(lines)
        if total > _MAX_ROWS_IN_PROMPT:
            extra = total - _MAX_ROWS_IN_PROMPT
            result += f"\n\n... и ещё {extra} задач (показаны первые {_MAX_ROWS_IN_PROMPT})"
        return result

    def _prepare_metrics(self, rows: list[dict[str, Any]]) -> str:
        """Format metric rows as JSON."""
        shown = rows[:_MAX_ROWS_IN_PROMPT]
        formatted = []
        for row in shown:
            entry = {}
            for k, v in row.items():
                if v is not None:
                    entry[k] = self._format_value(v)
            formatted.append(entry)
        return json.dumps(formatted, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Conversation history injection
    # ------------------------------------------------------------------

    @staticmethod
    def _get_history_budget(
        query_type: str,
        intent: str,
        data_length: int,
    ) -> int:
        """Return available history tokens for a branch given the data size.

        ``data_length`` is chars of the DB/RAG payload already in the prompt.
        We subtract its rough token cost from the branch's base budget and
        floor at ``_MIN_HISTORY_BUDGET_TOKENS`` so some history always fits.
        """
        del intent  # kept in signature for future per-intent tuning
        base = _BRANCH_HISTORY_BUDGET.get(query_type, 1000)
        # estimate_tokens is char_len // 3, matching what the context
        # builder used when accounting for the payload upstream.
        return max(_MIN_HISTORY_BUDGET_TOKENS, base - estimate_tokens(" " * data_length))

    @staticmethod
    def _history_prefix(ctx: ConversationContext | None, budget_tokens: int) -> str:
        """Format a history prefix that fits within ``budget_tokens``.

        Drops oldest turns from a *copy* of the context until the rendered
        block is within budget. Returns ``""`` if no turns survive — caller
        gets a clean prompt in that case.
        """
        if ctx is None:
            return ""
        turns = list(ctx.get("recent_turns") or [])
        if not turns:
            return ""

        while turns:
            trimmed: ConversationContext = {
                "summary": ctx.get("summary") or "",
                "recent_turns": turns,
                "history_token_count": 0,
                "needs_summarization": ctx.get("needs_summarization", False),
            }
            block = format_history(trimmed)
            if estimate_tokens(block) <= budget_tokens:
                return f"{block}\n{_HISTORY_INSTRUCTION}\n\n" if block else ""
            turns = turns[1:]  # drop the oldest turn and retry
        return ""

    # ------------------------------------------------------------------
    # LLM response generators
    # ------------------------------------------------------------------

    def _generate_direct_response(self, original_query: str, history: str = "") -> str:
        """Generate a direct LLM response without external context."""
        prompt = (
            f"{_SYSTEM_ROLE}\n\n"
            f"{history}"
            "Пользователь задал общий вопрос или поздоровался.\n\n"
            f"Вопрос: {original_query}\n\n"
            "Ответь кратко и дружелюбно. Если это приветствие — представься "
            "и предложи, чем можешь помочь (анализ задач, метрики спринтов, "
            "рекомендации по Agile-практикам). Не выдумывай данные."
        )
        response = self.llm_client.invoke(prompt)
        logger.info("[Response Agent] Generated direct response")
        return response

    def _generate_task_response(
        self,
        original_query: str,
        data: dict[str, Any],
        history: str = "",
    ) -> str:
        """Generate response for a single task lookup."""
        structured = self._prepare_single_task(data)
        prompt = (
            f"{_SYSTEM_ROLE}\n\n"
            f"{history}"
            "Пользователь спросил о конкретной задаче. Вот её данные:\n\n"
            f"{structured}\n\n"
            f"Вопрос: {original_query}\n\n"
            "Сформируй краткое описание задачи. Включи:\n"
            "- Ключ, тип и текущий статус\n"
            "- Команду и исполнителя (если есть в данных)\n"
            "- Story Points и спринт\n"
            "- Другие поля, если они релевантны вопросу\n\n"
            "Правила:\n"
            "- Не перечисляй все поля подряд — выбери важное.\n"
            "- Пропускай поля, значения которых отсутствуют в данных. "
            "НЕ пиши «не указано», «отсутствует», «нет данных» — "
            "просто не упоминай эти поля.\n"
            "- Если в вопросе пользователя названы команда / исполнитель / "
            "спринт — обязательно упомяни их в ответе."
        )
        return self.llm_client.invoke(prompt)

    def _generate_tasks_filter_response(
        self,
        original_query: str,
        rows: list[dict[str, Any]],
        entities: dict[str, Any] | None = None,
        history: str = "",
    ) -> str:
        """Generate response for a filtered task list."""
        entities = entities or {}
        include_assignee = bool(entities.get("assignee"))
        task_list = self._prepare_task_list(rows, include_assignee=include_assignee)
        total = len(rows)
        prompt = (
            f"{_SYSTEM_ROLE}\n\n"
            f"{history}"
            f"Пользователь запросил список задач. Найдено: {total}.\n\n"
            f"{task_list}\n\n"
            f"Вопрос: {original_query}\n\n"
            "Сформируй ответ:\n"
            "- Начни с общего количества найденных задач.\n"
            "- Перечисли задачи (ключ, статус, SP, краткое описание).\n"
            f"- Если задач больше {_MAX_ROWS_IN_PROMPT}, "
            f"укажи что показаны первые {_MAX_ROWS_IN_PROMPT}.\n"
            "- Если в вопросе названы команда / исполнитель / спринт — "
            "обязательно упомяни их в ответе.\n"
            "- Если полезно — сгруппируй по статусу или типу."
        )
        return self.llm_client.invoke(prompt)

    def _generate_metric_response(
        self,
        original_query: str,
        rows: list[dict[str, Any]],
        history: str = "",
    ) -> str:
        """Generate response for metric queries."""
        metrics_data = self._prepare_metrics(rows)
        prompt = (
            f"{_SYSTEM_ROLE}\n\n"
            f"{history}"
            "Пользователь запросил метрики. Вот данные:\n\n"
            f"{metrics_data}\n\n"
            f"Вопрос: {original_query}\n\n"
            "Сформируй ответ:\n"
            "- Назови конкретные значения метрик.\n"
            "- Если данные за несколько спринтов — покажи динамику "
            "(рост/падение/стабильность).\n"
            "- Если значение одно — просто назови его.\n"
            "- Не добавляй рекомендации — только факты из данных."
        )
        return self.llm_client.invoke(prompt)

    def _generate_rag_response(
        self,
        original_query: str,
        rag_response: str,
        rag_sources: list[str],
    ) -> str:
        """Format RAG-agent answer with source citations."""
        sources_text = "\n".join(f"- {s}" for s in rag_sources) if rag_sources else ""
        suffix = f"\n\n---\n**Источники:**\n{sources_text}" if sources_text else ""
        return f"{rag_response}{suffix}"

    def _generate_hybrid_response(  # noqa: PLR0913
        self,
        original_query: str,
        sql_result: list[dict[str, Any]],
        intent: str,
        rag_response: str,
        rag_sources: list[str],
        history: str = "",
    ) -> str:
        """Combine DB data + RAG context into one answer."""
        # Prepare SQL data section
        if intent == "task" and sql_result:
            sql_section = self._prepare_single_task(sql_result[0])
        elif intent == "tasks_filter" and sql_result:
            sql_section = self._prepare_task_list(sql_result)
        elif intent == "metric" and sql_result:
            sql_section = self._prepare_metrics(sql_result)
        else:
            sql_section = ""

        prompt = (
            f"{_SYSTEM_ROLE}\n\n"
            f"{history}"
            "Пользователь задал вопрос, требующий и данных из БД, "
            "и контекста из базы знаний.\n\n"
            f"Данные из БД:\n{sql_section}\n\n"
            f"Контекст из базы знаний:\n{rag_response}\n\n"
            f"Вопрос: {original_query}\n\n"
            "Сформируй ответ в два блока:\n"
            "1. ДАННЫЕ: приведи конкретные числа/факты из БД. "
            "Обязательно назови команду, спринт и конкретные числа из БД, "
            "если они есть в данных.\n"
            "2. АНАЛИЗ: дай рекомендации на основе базы знаний. "
            "Указывай ТОЛЬКО источники из раздела «Контекст из базы знаний». "
            "НЕ выдумывай ссылки, названия документов или URL. "
            "Если контекст из базы знаний пуст — напиши "
            "«Рекомендации из базы знаний недоступны»."
        )
        answer = self.llm_client.invoke(prompt)

        sources_text = "\n".join(f"- {s}" for s in rag_sources) if rag_sources else ""
        suffix = f"\n\n---\n**Источники:**\n{sources_text}" if sources_text else ""
        return f"{answer}{suffix}"

    # ------------------------------------------------------------------
    # SQL-only response
    # ------------------------------------------------------------------

    def _process_sql_response(self, state: dict[str, Any]) -> dict[str, Any]:
        """Handle SQL-only response generation."""
        error = state.get("error")
        sql_result = state.get("sql_result")
        intent = state.get("intent", "general")
        original_query = state.get("original_query", "")

        if error:
            logger.warning("[Response Agent] SQL error: %s", error)
            return {
                "final_response": (
                    f"Ошибка при выполнении запроса:\n\n{error}\n\n"
                    "---\n*Попробуйте переформулировать запрос*"
                ),
            }

        if not sql_result:
            logger.info("[Response Agent] No results from SQL Agent")
            entities = state.get("entities", {})
            issue_key = entities.get("issue_key")
            if issue_key:
                msg = (
                    f"Задача с ключом `{issue_key}` не найдена в базе данных.\n\n"
                    "---\n*Проверьте правильность ключа задачи*"
                )
            else:
                msg = "По вашему запросу ничего не найдено.\n\n---\n*Попробуйте уточнить фильтры*"
            return {"final_response": msg}

        entities = state.get("entities", {})
        ctx = state.get("conversation_context")
        data_chars = sum(len(str(row)) for row in sql_result[:_MAX_ROWS_IN_PROMPT])
        budget = self._get_history_budget("sql", intent, data_chars)
        history = self._history_prefix(ctx, budget)
        try:
            if intent == "task":
                response = self._generate_task_response(
                    original_query, sql_result[0], history=history
                )
            elif intent == "tasks_filter":
                response = self._generate_tasks_filter_response(
                    original_query, sql_result, entities, history=history
                )
            elif intent == "metric":
                response = self._generate_metric_response(
                    original_query, sql_result, history=history
                )
            else:
                response = self._generate_task_response(
                    original_query, sql_result[0], history=history
                )

            logger.info("[Response Agent] Generated SQL response for intent=%s", intent)

        except Exception as e:
            logger.error("[Response Agent] Error generating response: %s", e)
            response = (
                f"Не удалось сгенерировать ответ: {e}\n\n---\n*Попробуйте переформулировать запрос*"
            )

        return {"final_response": response}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate natural language response based on query_type and data.

        Args:
            state: Workflow state with intent, entities, sql_result,
                   rag_response, validation_result, error, etc.

        Returns:
            State update with final_response.
        """
        logger.info("[Response Agent] Generating response...")

        query_type = state.get("query_type", "sql")
        route = state.get("route", "db_query")
        intent = state.get("intent", "general")
        original_query = state.get("original_query", "")

        # --- Classifier error (explicit error intent from Supervisor) ---
        if query_type == "error" or intent == "error":
            err_msg = state.get("error", "Classifier unavailable")
            logger.warning("[Response Agent] Error intent: %s", err_msg)
            return {
                "final_response": (
                    "Извините, классификатор запросов временно недоступен. "
                    "Попробуйте повторить запрос через несколько секунд или "
                    "переформулировать его."
                ),
            }

        # --- Direct response (simple) ---
        if route == "direct_response" or query_type == "simple":
            logger.info("[Response Agent] Generating direct response")
            ctx = state.get("conversation_context")
            history = self._history_prefix(ctx, self._get_history_budget("simple", intent, 0))
            try:
                response = self._generate_direct_response(original_query, history=history)
            except Exception as e:
                logger.error("[Response Agent] Direct response error: %s", e)
                response = (
                    f"Не удалось сгенерировать ответ: {e}\n\n"
                    "---\n*Попробуйте переформулировать запрос*"
                )
            return {"final_response": response}

        # Read validation result
        validation = state.get("validation_result", {})
        use_sql = validation.get("use_sql", True)
        use_rag = validation.get("use_rag", False)
        note = validation.get("note")

        sql_result = state.get("sql_result")
        rag_response = state.get("rag_response")
        rag_sources = state.get("rag_sources", [])

        # --- Both failed (only hybrid/rag — sql-only handles empty in _process_sql_response) ---
        if query_type != "sql" and note and not use_sql and not use_rag:
            logger.warning("[Response Agent] Validation note: %s", note)
            return {
                "final_response": (
                    "К сожалению, в моей базе знаний нет ответа на этот вопрос.\n\n"
                    "Я могу помочь с:\n"
                    "• Agile-практиками (Sprint Goal, Velocity, Backlog Grooming)\n"
                    "• Метриками команд (Done Total, Scope Drop, Cancel Rate)\n"
                    "• Деталях о задачах Jira (текущий статус, описание)\n\n"
                    "Попробуйте переформулировать запрос ближе к этим темам."
                ),
            }

        # --- RAG only ---
        if query_type == "rag" and use_rag:
            logger.info("[Response Agent] RAG-only response")
            try:
                response = self._generate_rag_response(original_query, rag_response, rag_sources)
            except Exception as e:
                logger.error("[Response Agent] RAG response error: %s", e)
                response = f"Не удалось сгенерировать ответ: {e}"
            return {"final_response": response}

        # --- Hybrid ---
        if query_type == "hybrid" and (use_sql or use_rag):
            logger.info("[Response Agent] Hybrid response")
            ctx = state.get("conversation_context")
            sql_chars = sum(len(str(r)) for r in (sql_result or [])[:_MAX_ROWS_IN_PROMPT])
            data_chars = sql_chars + len(rag_response or "")
            history = self._history_prefix(
                ctx, self._get_history_budget("hybrid", intent, data_chars)
            )
            try:
                response = self._generate_hybrid_response(
                    original_query,
                    sql_result or [],
                    intent,
                    rag_response or "",
                    rag_sources,
                    history=history,
                )
            except Exception as e:
                logger.error("[Response Agent] Hybrid response error: %s", e)
                response = f"Не удалось сгенерировать ответ: {e}"
            return {"final_response": response}

        # --- SQL only (default) ---
        return self._process_sql_response(state)
