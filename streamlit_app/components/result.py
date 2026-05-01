"""Components for rendering task results and errors."""


def render_result(task: dict) -> str:
    """Format a ``COMPLETED`` task result for the chat.

    Args:
        task: Task object as returned by ``GET /tasks/{id}``.

    Returns:
        Markdown string to display as the assistant message.
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
    """Format a ``FAILED`` task result for the chat.

    Args:
        task: Task object as returned by ``GET /tasks/{id}``.

    Returns:
        Markdown string to display as the assistant message.
    """
    error_msg = task.get("error", "Неизвестная ошибка")
    return f"Произошла ошибка при обработке запроса:\n\n`{error_msg}`"


def render_timeout() -> str:
    """Format a polling-timeout message for the chat.

    Returns:
        Markdown string telling the user the request is still processing.
    """
    return (
        "Время ожидания истекло. Запрос всё ещё обрабатывается — "
        "попробуйте обновить страницу через несколько секунд."
    )
