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

logger = logging.getLogger(__name__)

# Maximum rows to include in the LLM prompt to avoid token overflow
_MAX_ROWS_IN_PROMPT = 20


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

    def _prepare_task_list(self, rows: list[dict[str, Any]]) -> str:
        """Format a list of tasks as a compact summary."""
        total = len(rows)
        shown = rows[:_MAX_ROWS_IN_PROMPT]
        lines = []
        for i, row in enumerate(shown, 1):
            key = row.get("issue_key", "?")
            summary = row.get("summary", "")[:80]
            status = row.get("issue_status_act", "?")
            team = row.get("feature_teams", "?")
            sp = row.get("storypoints_act", "?")
            lines.append(f"{i}. {key} | {status} | {team} | SP:{sp} | {summary}")
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
    # LLM response generators
    # ------------------------------------------------------------------

    def _generate_direct_response(self, original_query: str) -> str:
        """Generate a direct LLM response without external context."""
        prompt = (
            "Ты — ассистент для анализа данных о Jira-задачах. "
            "Пользователь задал общий вопрос, не связанный с конкретной задачей "
            "в базе данных. Ответь на вопрос на русском языке.\n"
            "\n"
            f"Вопрос пользователя: {original_query}\n"
            "\n"
            "Ответь на вопрос пользователя:"
        )
        response = self.llm_client.invoke(prompt)
        logger.info("[Response Agent] Generated direct response")
        return response

    def _generate_task_response(self, original_query: str, data: dict[str, Any]) -> str:
        """Generate response for a single task lookup."""
        structured = self._prepare_single_task(data)
        prompt = (
            "Ты — ассистент для анализа данных о Jira-задачах. "
            "Пользователь задал вопрос о задаче. Сформируй понятный "
            "ответ на основе данных.\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            f"Данные о задаче:\n{structured}\n\n"
            "Инструкции:\n"
            "1. Выбери только релевантную информацию\n"
            "2. Ответь на русском языке\n"
            "3. Если данные не указаны, не акцентируй на этом\n\n"
            "Ответ:"
        )
        return self.llm_client.invoke(prompt)

    def _generate_tasks_filter_response(
        self, original_query: str, rows: list[dict[str, Any]]
    ) -> str:
        """Generate response for a filtered task list."""
        task_list = self._prepare_task_list(rows)
        prompt = (
            "Ты — ассистент для анализа данных о Jira-задачах. "
            "Пользователь запросил список задач по фильтру. "
            "Сформируй ответ на основе данных.\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            f"Найдено задач: {len(rows)}\n\n"
            f"Данные:\n{task_list}\n\n"
            "Инструкции:\n"
            "1. Кратко опиши результат поиска\n"
            "2. Перечисли ключевые задачи\n"
            "3. Ответь на русском языке\n\n"
            "Ответ:"
        )
        return self.llm_client.invoke(prompt)

    def _generate_metric_response(self, original_query: str, rows: list[dict[str, Any]]) -> str:
        """Generate response for metric queries."""
        metrics_data = self._prepare_metrics(rows)
        prompt = (
            "Ты — ассистент для анализа данных о Jira-задачах. "
            "Пользователь запросил метрики команды/спринта. "
            "Сформируй ответ на основе данных.\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            f"Данные метрик:\n{metrics_data}\n\n"
            "Инструкции:\n"
            "1. Выдели запрошенные метрики\n"
            "2. Если есть несколько спринтов, покажи динамику\n"
            "3. Ответь на русском языке\n\n"
            "Ответ:"
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

    def _generate_hybrid_response(
        self,
        original_query: str,
        sql_result: list[dict[str, Any]],
        intent: str,
        rag_response: str,
        rag_sources: list[str],
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
            "Ты — ассистент для анализа Agile-метрик и практик.\n"
            "Используй данные из БД и контекст из базы знаний, "
            "чтобы дать развёрнутый ответ.\n\n"
            f"Вопрос пользователя: {original_query}\n\n"
            f"Данные из БД:\n{sql_section}\n\n"
            f"Контекст из базы знаний:\n{rag_response}\n\n"
            "Инструкции:\n"
            "1. Сначала приведи фактические данные из БД\n"
            "2. Затем дай рекомендации на основе базы знаний\n"
            "3. Укажи источники\n"
            "4. Ответь на русском языке\n\n"
            "Ответ:"
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

        try:
            if intent == "task":
                response = self._generate_task_response(original_query, sql_result[0])
            elif intent == "tasks_filter":
                response = self._generate_tasks_filter_response(original_query, sql_result)
            elif intent == "metric":
                response = self._generate_metric_response(original_query, sql_result)
            else:
                response = self._generate_task_response(original_query, sql_result[0])

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
            try:
                response = self._generate_direct_response(original_query)
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

        # --- Both failed ---
        if note and not use_sql and not use_rag:
            logger.warning("[Response Agent] Validation note: %s", note)
            return {
                "final_response": (
                    f"Ошибка при выполнении запроса:\n\n{note}\n\n"
                    "---\n*Попробуйте переформулировать запрос*"
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
            try:
                response = self._generate_hybrid_response(
                    original_query,
                    sql_result or [],
                    intent,
                    rag_response or "",
                    rag_sources,
                )
            except Exception as e:
                logger.error("[Response Agent] Hybrid response error: %s", e)
                response = f"Не удалось сгенерировать ответ: {e}"
            return {"final_response": response}

        # --- SQL only (default) ---
        return self._process_sql_response(state)
