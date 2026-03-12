"""Main entry point for the HSE Prom Prog CLI application.

This module provides the command-line interface for running the
multi-agent Jira query processing workflow.
"""

import logging
import sys

from hse_prom_prog.config import settings
from hse_prom_prog.graph.workflow import AgileWorkflow
from hse_prom_prog.llm.client import get_llm_client


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def print_separator() -> None:
    """Print a visual separator line."""
    print("\n" + "=" * 60 + "\n")


def main(query: str | None = None) -> None:
    """Main function to run the Agile AI Assistant workflow.

    Args:
        query: Optional user query. If not provided, will use CLI argument.
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
