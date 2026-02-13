"""Response agent for generating natural language responses using LLM.

This agent takes the SQL agent's output and uses LLM to generate
a natural language response based on the user's original query.
"""

import json
import logging
from datetime import datetime
from typing import Any

from hse_prom_prog.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Agent that generates natural language responses using LLM.

    The Response agent takes data from previous agents and uses LLM
    to generate a contextual, natural language response to the user's query.

    Attributes:
        llm_client: LLM client for generating responses.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the Response agent.

        Args:
            llm_client: LLM client instance for text generation.
        """
        self.llm_client = llm_client
        logger.info("[Response Agent] Initialized with LLM client")

    def _format_value(self, value: Any) -> str:
        """Format a value for display in JSON.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation.
        """
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

    def _prepare_data_for_llm(self, data: dict[str, Any]) -> str:
        """Prepare data in a structured format for LLM.

        Args:
            data: Dictionary containing issue data from database.

        Returns:
            Formatted string with structured data.
        """
        # Format all data fields with Russian labels
        formatted_data = {
            "Ключ задачи": self._format_value(data.get("issue_key")),
            "Проект": self._format_value(data.get("issue_project")),
            "Тип задачи": self._format_value(data.get("issue_type")),
            "Описание": self._format_value(data.get("summary")),
            "Текущий статус": self._format_value(data.get("issue_status_act")),
            "Статус в конце спринта": self._format_value(data.get("issue_status_end_of_sprint")),
            "Дата создания": self._format_value(data.get("create_time")),
            "Резолюция": self._format_value(data.get("resolution")),
            "Время резолюции": self._format_value(data.get("resolution_time")),
            "ID спринта": self._format_value(data.get("jirasprint_id")),
            "Название спринта": self._format_value(data.get("sprint_name")),
            "Состояние спринта": self._format_value(data.get("sprint_state")),
            "Дата активации спринта": self._format_value(data.get("activation_date")),
            "Начало спринта": self._format_value(data.get("start_date")),
            "Конец спринта": self._format_value(data.get("end_date")),
            "Завершение спринта": self._format_value(data.get("complete_date")),
            "Команда": self._format_value(data.get("feature_teams")),
            "Команда (начало спринта)": self._format_value(
                data.get("feature_teams_start_of_sprint")
            ),
            "Команда (конец спринта)": self._format_value(data.get("feature_teams_end_of_sprint")),
            "Репортер": self._format_value(data.get("reporter")),
            "Исполнитель": self._format_value(data.get("assignee_name")),
            "Департамент": self._format_value(data.get("issue_department")),
            "Кластер": self._format_value(data.get("cluster")),
            "Подразделение": self._format_value(data.get("unit")),
            "Story Points (актуальные)": self._format_value(data.get("storypoints_act")),
            "Story Points (начало спринта)": self._format_value(
                data.get("storypoints_start_of_sprint")
            ),
            "Story Points (конец спринта)": self._format_value(
                data.get("storypoints_end_of_sprint")
            ),
            "Story Points (след. спринт)": self._format_value(data.get("storypoints_next_sprint")),
            "Время в работе (часы)": self._format_value(data.get("time_h_in_progress")),
            "Время не исправлено (часы)": self._format_value(data.get("time_h_not_fixed")),
            "Количество PR": self._format_value(data.get("merged_pr_count")),
            "Подход разработки": self._format_value(data.get("dev_approach")),
            "Эпик": self._format_value(data.get("epic_issue_key")),
            "Отчетная задача": self._format_value(data.get("is_report")),
            "Технический долг": self._format_value(data.get("is_tech_debt")),
            "Метки": self._format_value(data.get("labels")),
        }

        # Convert to JSON for better structure
        return json.dumps(formatted_data, ensure_ascii=False, indent=2)

    def _generate_response(self, original_query: str, data: dict[str, Any]) -> str:
        """Generate natural language response using LLM.

        Args:
            original_query: User's original query.
            data: Issue data from database.

        Returns:
            Generated natural language response.
        """
        # Prepare structured data
        structured_data = self._prepare_data_for_llm(data)

        # Create prompt for LLM
        prompt = (
            "Ты — ассистент для анализа данных о Jira-задачах. "
            "Пользователь задал вопрос о задаче, и тебе нужно сформировать "
            "понятный и информативный ответ на основе данных из базы данных.\n"
            "\n"
            f"Вопрос пользователя: {original_query}\n"
            "\n"
            "Данные о задаче из базы данных:\n"
            f"{structured_data}\n"
            "\n"
            "Инструкции:\n"
            "1. Внимательно проанализируй вопрос пользователя\n"
            "2. Выбери из данных только релевантную информацию, "
            "которая отвечает на вопрос\n"
            "3. Сформируй естественный ответ на русском языке\n"
            "4. Если пользователь спросил о конкретном поле "
            "(например, статус, исполнитель, story points), "
            "сфокусируйся на этом\n"
            '5. Если вопрос общий (например, "расскажи о задаче"), '
            "дай краткую сводку с ключевыми данными\n"
            "6. Используй понятный язык, избегай технического жаргона "
            "где это возможно\n"
            '7. Если какие-то данные не указаны (значение "не указано"), '
            "не акцентируй на этом внимание, если это не критично "
            "для ответа\n"
            "\n"
            "Ответь на вопрос пользователя:"
        )

        try:
            # Generate response using LLM
            response = self.llm_client.invoke(prompt)
            logger.info("[Response Agent] Successfully generated response using LLM")
            return response

        except Exception as e:
            logger.error(f"[Response Agent] Error generating response with LLM: {e}")
            # Fallback to simple formatted response
            issue_key = data.get("issue_key", "UNKNOWN")
            return f"""Не удалось сгенерировать ответ с помощью LLM: {e}

Вот основные данные по задаче {issue_key}:
- Проект: {data.get("issue_project")}
- Тип: {data.get("issue_type")}
- Статус: {data.get("issue_status_act")}
- Исполнитель: {data.get("assignee_name")}
- Спринт: {data.get("sprint_name")}"""

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate natural language response using LLM.

        Args:
            state: State dictionary containing 'sql_response' and 'original_query'.

        Returns:
            Dictionary containing the generated final response.
        """
        logger.info("[Response Agent] Generating response using LLM...")

        sql_response = state.get("sql_response")
        issue_key = state.get("issue_key", "UNKNOWN")
        original_query = state.get("original_query", "")
        error = state.get("error")

        # Handle errors
        if error:
            formatted_response = f"""## Ошибка при обработке задачи {issue_key}

❌ {error}

---
*Попробуйте другой ключ задачи или проверьте подключение к базе данных*"""

            logger.warning(f"[Response Agent] Formatted error response for {issue_key}")

            return {
                "issue_key": issue_key,
                "original_query": original_query,
                "sql_response": sql_response,
                "final_response": formatted_response,
                "error": error,
            }

        # Handle no data
        if not sql_response:
            formatted_response = f"""## Задача {issue_key} не найдена

Задача с ключом `{issue_key}` отсутствует в базе данных.

---
*Проверьте правильность ключа задачи*"""

            logger.info(f"[Response Agent] Formatted 'not found' response for {issue_key}")

            return {
                "issue_key": issue_key,
                "original_query": original_query,
                "sql_response": sql_response,
                "final_response": formatted_response,
            }

        # Generate response using LLM
        try:
            formatted_response = self._generate_response(original_query, sql_response)
            logger.info(f"[Response Agent] Successfully generated response for {issue_key}")

        except Exception as e:
            logger.error(f"[Response Agent] Error in process method: {e}")
            formatted_response = f"""## Ошибка генерации ответа

Не удалось сгенерировать ответ для задачи {issue_key}: {e}

---
*Попробуйте переформулировать запрос*"""

        return {
            "issue_key": issue_key,
            "original_query": original_query,
            "sql_response": sql_response,
            "final_response": formatted_response,
        }
