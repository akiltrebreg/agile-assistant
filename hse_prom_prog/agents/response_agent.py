"""Response agent for formatting final user responses.

This agent takes the SQL agent's output and formats it into a
user-friendly markdown table response.
"""

import logging
from datetime import datetime
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Agent that formats final responses for users.

    The Response agent takes data from previous agents and creates
    a well-formatted, user-friendly output message with tables.

    Attributes:
        None (stateless agent).
    """

    def __init__(self) -> None:
        """Initialize the Response agent."""
        logger.info("[Response Agent] Initialized")

    def _format_value(self, value: Any) -> str:
        """Format a value for display.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return "—"
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    def _create_table(self, data: dict[str, Any]) -> str:
        """Create a formatted table from issue data.

        Args:
            data: Dictionary containing issue data.

        Returns:
            Markdown-formatted table string.
        """
        # Define field groups for better organization
        field_groups = [
            (
                "Основная информация",
                [
                    ("Ключ задачи", "issue_key"),
                    ("Проект", "issue_project"),
                    ("Тип", "issue_type"),
                    ("Статус", "issue_status_end_of_sprint"),
                    ("Создана", "create_time"),
                ],
            ),
            (
                "Спринт",
                [
                    ("ID спринта", "jirasprint_id"),
                    ("Название спринта", "sprint_name"),
                    ("Состояние спринта", "sprint_state"),
                    ("Начало", "start_date"),
                    ("Конец", "end_date"),
                ],
            ),
            (
                "Команда",
                [
                    ("Команда", "feature_teams"),
                    ("Репортер", "reporter"),
                    ("Исполнитель", "assignee_name"),
                    ("Кластер", "cluster"),
                    ("Подразделение", "unit"),
                ],
            ),
            (
                "Метрики",
                [
                    ("Story Points (начало)", "storypoints_start_of_sprint"),
                    ("Story Points (конец)", "storypoints_end_of_sprint"),
                    ("Время в работе (ч)", "time_h_in_progress"),
                    ("Время не исправлено (ч)", "time_h_not_fixed"),
                    ("Резолюция", "resolution"),
                ],
            ),
        ]

        tables = []
        for group_name, fields in field_groups:
            rows = []
            for field_label, field_key in fields:
                value = data.get(field_key)
                formatted_value = self._format_value(value)
                rows.append([field_label, formatted_value])

            table = tabulate(
                rows,
                headers=["Поле", "Значение"],
                tablefmt="grid",
                colalign=("left", "left"),
            )
            tables.append(f"### {group_name}\n\n{table}")

        # Add labels if present
        labels = data.get("labels")
        if labels:
            labels_str = ", ".join(f"`{label}`" for label in labels)
            tables.append(f"### Метки\n\n{labels_str}")

        return "\n\n".join(tables)

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Format the final response for the user.

        Args:
            state: State dictionary containing 'sql_response' from SQL Agent.

        Returns:
            Dictionary containing the formatted final response.
        """
        logger.info("[Response Agent] Formatting final response...")

        sql_response = state.get("sql_response")
        issue_key = state.get("issue_key", "UNKNOWN")
        error = state.get("error")

        # Handle errors
        if error:
            formatted_response = f"""
## Ошибка при обработке задачи {issue_key}

❌ {error}

---
*Попробуйте другой ключ задачи или проверьте подключение к базе данных*
            """.strip()

            logger.warning(f"[Response Agent] Formatted error response for {issue_key}")

            return {
                "issue_key": issue_key,
                "original_query": state.get("original_query", ""),
                "sql_response": sql_response,
                "final_response": formatted_response,
                "error": error,
            }

        # Handle no data
        if not sql_response:
            # ruff: noqa: RUF001
            formatted_response = f"""
## Задача {issue_key} не найдена

Задача с ключом `{issue_key}` отсутствует в базе данных.

---
*Проверьте правильность ключа задачи*
            """.strip()

            logger.info(f"[Response Agent] Formatted 'not found' response for {issue_key}")

            return {
                "issue_key": issue_key,
                "original_query": state.get("original_query", ""),
                "sql_response": sql_response,
                "final_response": formatted_response,
            }

        # Format successful response with table
        try:
            table_content = self._create_table(sql_response)

            formatted_response = f"""
## Информация по задаче {issue_key}

{table_content}

---
*Данные получены из PostgreSQL базы данных*
            """.strip()

            logger.info(f"[Response Agent] Successfully formatted response for {issue_key}")

        except Exception as e:
            logger.error(f"[Response Agent] Error formatting table: {e}")
            formatted_response = f"""
## Результат обработки задачи {issue_key}

Данные получены, но возникла ошибка форматирования:
```
{sql_response}
```

---
*Ошибка форматирования: {e}*
            """.strip()

        return {
            "issue_key": issue_key,
            "original_query": state.get("original_query", ""),
            "sql_response": sql_response,
            "final_response": formatted_response,
        }
