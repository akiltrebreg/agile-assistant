"""Response agent for generating natural language responses using LLM.

This agent takes the SQL agent's output (list of dicts) and uses LLM
to generate a natural language response based on the user's original query.
Handles all intents: task, tasks_filter, metric, general.
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
        """Generate a direct LLM response without database context."""
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate natural language response based on intent and data.

        Args:
            state: Workflow state with intent, entities, sql_result, error, etc.

        Returns:
            State update with final_response.
        """
        logger.info("[Response Agent] Generating response...")

        route = state.get("route", "db_query")
        intent = state.get("intent", "general")
        original_query = state.get("original_query", "")

        # --- Direct response (no DB needed) ---
        if route == "direct_response":
            logger.info("[Response Agent] Generating direct response (no DB context)")
            try:
                response = self._generate_direct_response(original_query)
            except Exception as e:
                logger.error(f"[Response Agent] Direct response error: {e}")
                response = (
                    f"Не удалось сгенерировать ответ: {e}\n\n"
                    "---\n*Попробуйте переформулировать запрос*"
                )
            return {"final_response": response}

        # --- DB-based response ---
        error = state.get("error")
        sql_result = state.get("sql_result")

        # Handle errors from SQL Agent
        if error:
            logger.warning(f"[Response Agent] SQL error: {error}")
            return {
                "final_response": (
                    f"Ошибка при выполнении запроса:\n\n{error}\n\n"
                    "---\n*Попробуйте переформулировать запрос*"
                ),
            }

        # Handle empty results
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

        # Generate response based on intent
        try:
            if intent == "task":
                response = self._generate_task_response(original_query, sql_result[0])
            elif intent == "tasks_filter":
                response = self._generate_tasks_filter_response(original_query, sql_result)
            elif intent == "metric":
                response = self._generate_metric_response(original_query, sql_result)
            else:
                response = self._generate_task_response(original_query, sql_result[0])

            logger.info(f"[Response Agent] Generated response for intent={intent}")

        except Exception as e:
            logger.error(f"[Response Agent] Error generating response: {e}")
            response = (
                f"Не удалось сгенерировать ответ: {e}\n\n---\n*Попробуйте переформулировать запрос*"
            )

        return {"final_response": response}
