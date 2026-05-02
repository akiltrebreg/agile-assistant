"""Components for rendering task results and errors."""

import re

# Lines whose stripped content begins with one of these LaTeX commands
# are auto-wrapped in $$...$$ so Streamlit's built-in KaTeX renders them.
# Covers what the model actually emits in metric explanations: \text,
# \frac, \left/\right, \times, \approx, plus Greek letters and a handful
# of common operators.
_BARE_MATH_PREFIX_RE = re.compile(
    r"^\s*\\(?:text|frac|sqrt|sum|prod|int|left|right|times|cdot|approx|"
    r"begin|displaystyle|mathbb|mathcal|mathrm|mathbf|hat|bar|tilde|vec|"
    r"alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|phi|psi|omega)\b"
)


def _normalize_math(text: str) -> str:
    """Wrap LLM-emitted LaTeX so Streamlit's KaTeX picks it up.

    The model produces bare LaTeX (``\\text{...}``, ``\\frac{a}{b}``) or
    LaTeX-flavoured ``\\[...\\]`` / ``\\(...\\)`` delimiters that
    ``st.markdown`` does not recognise — they end up rendered as raw
    source. This helper rewrites both forms to the ``$``/``$$`` math
    delimiters that Streamlit/KaTeX understands, leaving fenced code
    blocks and lines that already contain ``$`` untouched.
    """
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside a fenced code block — preserve verbatim.
            out.append(part)
            continue

        # \[ ... \]  →  $$ ... $$  (display math)
        # \( ... \)  →  $ ... $    (inline math)
        rewritten = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", part, flags=re.DOTALL)
        rewritten = re.sub(r"\\\((.*?)\\\)", r"$\1$", rewritten, flags=re.DOTALL)

        wrapped: list[str] = []
        for line in rewritten.split("\n"):
            if _BARE_MATH_PREFIX_RE.match(line) and "$" not in line:
                stripped = line.strip()
                indent = line[: len(line) - len(line.lstrip())]
                wrapped.append(f"{indent}$${stripped}$$")
            else:
                wrapped.append(line)
        out.append("\n".join(wrapped))

    return "".join(out)


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
        parts.append(_normalize_math(response_text))

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
