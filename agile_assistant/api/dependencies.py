"""FastAPI dependencies for dependency injection.

This module provides dependency functions for FastAPI endpoints,
following the FastAPI dependency injection pattern.
"""

from collections.abc import Generator

from fastapi import Depends

from agile_assistant.database.connection import DatabaseConnection, get_database
from agile_assistant.database.task_repository import TaskRepository
from agile_assistant.memory.manager import MemoryManager


def get_db() -> Generator[DatabaseConnection, None, None]:
    """Dependency to get database connection.

    Provides a database connection for the duration of a request,
    ensuring proper cleanup when request completes.

    Yields:
        DatabaseConnection instance.

    Example:
        @app.get("/tasks/{task_id}")
        def get_task(task_id: UUID, db: DatabaseConnection = Depends(get_db)):
            ...
    """
    db = get_database()
    try:
        yield db
    finally:
        db.close()


def get_task_repository(
    db: DatabaseConnection,
) -> TaskRepository:
    """Dependency to get task repository.

    Creates a TaskRepository instance using the injected database connection.

    Args:
        db: Database connection (injected by get_db dependency).

    Returns:
        TaskRepository instance.

    Example:
        @app.get("/tasks/{task_id}")
        def get_task(
            task_id: UUID,
            repo: TaskRepository = Depends(get_task_repository),
        ):
            ...
    """
    return TaskRepository(db)


def get_memory_manager(
    db: DatabaseConnection = Depends(get_db),  # noqa: B008  — FastAPI Depends in default is the standard pattern
) -> MemoryManager:
    """Dependency to get a ``MemoryManager`` bound to the request DB.

    The manager is cheap to construct (no connection pool of its own —
    it reuses ``db.get_session()``) so a per-request instance is fine.
    """
    return MemoryManager(db)
