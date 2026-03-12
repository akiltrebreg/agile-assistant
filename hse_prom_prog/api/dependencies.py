"""FastAPI dependencies for dependency injection.

This module provides dependency functions for FastAPI endpoints,
following the FastAPI dependency injection pattern.
"""

from collections.abc import Generator

from hse_prom_prog.database.connection import DatabaseConnection, get_database
from hse_prom_prog.database.task_repository import TaskRepository


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
