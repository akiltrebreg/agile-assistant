"""Database connection module for PostgreSQL.

This module provides SQLAlchemy engine and session management for
connecting to PostgreSQL database with Jira issues data.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from agile_assistant.config import settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager for PostgreSQL.

    This class handles database engine creation, session management,
    and provides context managers for safe database operations.

    Attributes:
        engine: SQLAlchemy engine instance.
        SessionLocal: Session factory for creating database sessions.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize database connection.

        Args:
            database_url: PostgreSQL connection URL. Defaults to settings value.
        """
        self.database_url = database_url or settings.database_url
        logger.info(f"Initializing database connection to {self._masked_url()}")

        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,  # Verify connections before using
                pool_size=5,
                max_overflow=10,
                echo=settings.log_level == "DEBUG",
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    def _masked_url(self) -> str:
        """Get database URL with masked password.

        Returns:
            Database URL with password replaced by asterisks.
        """
        if "@" in self.database_url:
            user_pass, rest = self.database_url.split("@", 1)
            if ":" in user_pass:
                user, _ = user_pass.rsplit(":", 1)
                return f"{user}:****@{rest}"
        return self.database_url

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions.

        Yields:
            SQLAlchemy Session object.

        Example:
            >>> db = DatabaseConnection()
            >>> with db.get_session() as session:
            ...     result = session.execute(text("SELECT 1"))
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connectivity.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Args:
            query: SQL query string.
            params: Optional query parameters.

        Returns:
            List of dictionaries containing query results.

        Raises:
            SQLAlchemyError: If query execution fails.
        """
        try:
            with self.get_session() as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))

                # Convert result to list of dicts
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row, strict=False)) for row in rows]

        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Query: {query}, Params: {params}")
            raise

    def close(self) -> None:
        """Close database engine and all connections."""
        logger.info("Closing database connections")
        self.engine.dispose()


def get_database() -> DatabaseConnection:
    """Factory function to create database connection.

    Returns:
        DatabaseConnection instance.
    """
    return DatabaseConnection()
