"""Components for rendering task results and errors."""

import streamlit as st


def render_result(task: dict) -> str:
    """Format a COMPLETED task result for the chat.

    Returns:
        Markdown string to display as assistant message.
    """
    result = task.get("result") or {}
    response_text = result.get("final_response", "")
    issue_key = result.get("issue_key")

    parts: list[str] = []
    if issue_key:
        parts.append(f"**Задача: {issue_key}**\n")
    if response_text:
        parts.append(response_text)

    return "\n".join(parts) if parts else "Результат пуст."


def render_error(task: dict) -> str:
    """Format a FAILED task result for the chat.

    Returns:
        Markdown string to display as assistant message.
    """
    error_msg = task.get("error", "Неизвестная ошибка")
    return f"Произошла ошибка при обработке запроса:\n\n`{error_msg}`"


def render_timeout(task_id: str) -> str:
    """Format a timeout message.

    Returns:
        Markdown string with task_id for manual retry.
    """
    return (
        "Время ожидания истекло. Задача всё ещё обрабатывается.\n\n"
        f"Вы можете проверить статус позже по ID:\n`{task_id}`"
    )


def render_task_details(task: dict) -> None:
    """Show timing details in an expander."""
    with st.expander("Детали выполнения"):
        cols = st.columns(3)
        cols[0].metric("Создана", _fmt_ts(task.get("created_at")))
        cols[1].metric("Начата", _fmt_ts(task.get("started_at")))
        cols[2].metric("Завершена", _fmt_ts(task.get("completed_at")))


def _fmt_ts(ts: str | None) -> str:
    """Format an ISO timestamp to a short readable form."""
    if not ts:
        return "—"
    # "2026-02-13T16:13:56.405927Z" → "16:13:56"
    try:
        return ts.split("T")[1][:8]
    except (IndexError, AttributeError):
        return str(ts)
