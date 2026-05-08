"""CLI entry point for the Agile AI Assistant workflow.

Wires up logging, the LLM client, and the LangGraph workflow, then runs a
single user query end-to-end and prints the final response. Used for ad-hoc
checks against a live deployment; the production path is the FastAPI service
plus Celery workers.
"""

import logging
import sys

from hse_prom_prog.config import settings
from hse_prom_prog.graph.workflow import AgileWorkflow
from hse_prom_prog.llm.client import get_llm_client
from hse_prom_prog.observability.logging import setup_logging as _setup_logging


def setup_logging() -> None:
    """Configure root logger via the centralised observability module.

    Reads ``settings.log_level`` and delegates to
    :func:`hse_prom_prog.observability.logging.setup_logging`, which
    attaches stdout + a rotating file handler under ``/app/logs/cli.log``.
    """
    _setup_logging("cli", level=settings.log_level)


def print_separator() -> None:
    """Print a 60-character ``=`` separator surrounded by blank lines."""
    print("\n" + "=" * 60 + "\n")


def main(query: str | None = None) -> None:
    """Run the workflow on a single query and print the final response.

    Args:
        query: User question to process. When ``None`` (the default), the
            CLI argv is joined into a single string; if argv is empty,
            usage is printed and the process exits with code 1.

    Raises:
        SystemExit: With code 0 on Ctrl+C, code 1 on missing argv or any
            unhandled workflow exception.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Get query from argument if not provided
    if query is None:
        if len(sys.argv) < 2:
            print("Usage: python -m hse_prom_prog.main '<your query>'")
            print("Example: python -m hse_prom_prog.main 'Выведи данные по задаче ABC-123'")
            sys.exit(1)
        query = " ".join(sys.argv[1:])

    try:
        logger.info("Starting Agile AI Assistant...")
        print("\n🤖 Agile AI Assistant")
        print(f"📝 Query: {query}")
        print_separator()

        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = get_llm_client()

        # Initialize workflow
        logger.info("Initializing workflow...")
        workflow = AgileWorkflow(llm_client)

        # Run workflow
        logger.info("Running workflow...")
        result = workflow.run(query)

        # Display results
        print_separator()
        print("=== ФИНАЛЬНЫЙ ОТВЕТ ===")
        print()
        print(result.get("final_response", "No response generated"))
        print_separator()

        logger.info("Workflow completed successfully")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\n\nПрервано пользователем.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
