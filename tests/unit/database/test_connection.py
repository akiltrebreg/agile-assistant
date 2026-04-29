"""Unit tests for ``hse_prom_prog.database.connection``.

The DatabaseConnection wraps SQLAlchemy. We test:

  * ``_masked_url`` — passwords never reach logs (the URL appears in
    INFO logs at startup; an unmasked one would leak credentials).
  * ``get_session`` — commits on success, rollbacks on exception, always
    closes the session in finally.
  * ``execute_query`` — returns list[dict] (not list[tuple]); SQLAlchemy
    errors propagate with the original exception class.
  * ``test_connection`` — returns False (NOT raises) on SQLAlchemyError
    so health checks can degrade gracefully.
  * ``close`` — calls ``engine.dispose`` (Phase 1 leaks fixed by this).

We patch ``create_engine`` and ``sessionmaker`` at the module's import
site so no real PostgreSQL connection is attempted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from hse_prom_prog.database import connection as conn_module

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_engine() -> MagicMock:
    return MagicMock(name="engine")


@pytest.fixture
def fake_session() -> MagicMock:
    """A session that supports the context-manager calls the SUT uses."""
    return MagicMock(name="session")


@pytest.fixture
def patched_sqlalchemy(
    monkeypatch: pytest.MonkeyPatch,
    fake_engine: MagicMock,
    fake_session: MagicMock,
) -> dict[str, MagicMock]:
    """Replace ``create_engine`` + ``sessionmaker`` so init is hermetic."""
    create_engine_mock = MagicMock(return_value=fake_engine)
    session_factory = MagicMock(return_value=fake_session)
    sessionmaker_mock = MagicMock(return_value=session_factory)
    monkeypatch.setattr(conn_module, "create_engine", create_engine_mock)
    monkeypatch.setattr(conn_module, "sessionmaker", sessionmaker_mock)
    return {
        "create_engine": create_engine_mock,
        "sessionmaker": sessionmaker_mock,
        "session_factory": session_factory,
        "session": fake_session,
        "engine": fake_engine,
    }


# ===================================================================== #
# Construction
# ===================================================================== #


@pytest.mark.unit
class TestConstruction:
    def test_creates_engine_with_pool_pre_ping(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # pool_pre_ping=True is the only thing standing between long-idle
        # workers and stale-connection errors. Pin it.
        conn_module.DatabaseConnection("postgresql://u:p@host:5432/db")
        ctor_kwargs = patched_sqlalchemy["create_engine"].call_args.kwargs
        assert ctor_kwargs["pool_pre_ping"] is True
        assert ctor_kwargs["pool_size"] == 5
        assert ctor_kwargs["max_overflow"] == 10

    def test_uses_settings_when_no_url_passed(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        from hse_prom_prog.config import settings

        db = conn_module.DatabaseConnection()
        # The URL stored on the instance matches settings — pinning so a
        # refactor of get_database() can't accidentally wire to a stub.
        assert db.database_url == settings.database_url

    def test_engine_init_failure_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If create_engine raises (bad URL, missing driver) the ctor must
        # NOT swallow it — the API server's startup probe relies on a
        # crash-loud failure to surface bad config in CI.
        monkeypatch.setattr(
            conn_module, "create_engine", MagicMock(side_effect=ValueError("bad url"))
        )
        with pytest.raises(ValueError, match="bad url"):
            conn_module.DatabaseConnection("invalid")


# ===================================================================== #
# _masked_url — passwords must never leak
# ===================================================================== #


@pytest.mark.unit
class TestMaskedUrl:
    def test_password_replaced_with_asterisks(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        db = conn_module.DatabaseConnection("postgresql://user:secret@host:5432/db")
        masked = db._masked_url()
        assert "secret" not in masked
        assert "****" in masked
        assert "user" in masked  # username preserved
        assert "host:5432/db" in masked

    def test_url_without_credentials_returned_as_is(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Local sqlite-style URL has no @ — must round-trip unchanged.
        db = conn_module.DatabaseConnection("sqlite:///local.db")
        assert db._masked_url() == "sqlite:///local.db"

    def test_username_only_no_at_returned_as_is(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Pin: defensive — non-standard URL shapes don't crash.
        db = conn_module.DatabaseConnection("postgresql:///socket_path")
        masked = db._masked_url()
        # No `@` separator, so nothing to mask. Test asserts no crash.
        assert "socket_path" in masked


# ===================================================================== #
# get_session — commit/rollback/close lifecycle
# ===================================================================== #


@pytest.mark.unit
class TestGetSession:
    def test_commits_on_success(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        with db.get_session() as s:
            s.execute(MagicMock())
        # Pin the lifecycle: commit-then-close on the success path.
        session.commit.assert_called_once()
        session.rollback.assert_not_called()
        session.close.assert_called_once()

    def test_rolls_back_on_exception_and_reraises(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        with pytest.raises(RuntimeError, match="boom"), db.get_session():
            raise RuntimeError("boom")
        # Pin: rollback called, NO commit, close still happens.
        session.rollback.assert_called_once()
        session.commit.assert_not_called()
        session.close.assert_called_once()

    def test_session_closed_even_when_commit_fails(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # If commit itself raises (e.g. constraint violation surfaced at
        # commit time), close() must still run — otherwise a leaked
        # session pins a connection in the pool.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        session.commit.side_effect = SQLAlchemyError("constraint violated")
        with pytest.raises(SQLAlchemyError), db.get_session():
            pass
        session.close.assert_called_once()


# ===================================================================== #
# execute_query — returns dicts, propagates SQLAlchemyError
# ===================================================================== #


@pytest.mark.unit
class TestExecuteQuery:
    def _wire_session_result(
        self,
        session: MagicMock,
        *,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> MagicMock:
        result = MagicMock()
        result.keys = MagicMock(return_value=columns)
        result.fetchall = MagicMock(return_value=rows)
        session.execute = MagicMock(return_value=result)
        return result

    def test_returns_list_of_dicts_keyed_by_columns(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Pin: rows come back as list[dict[col, value]], not raw tuples.
        # SQL Agent's downstream prompt construction reads rows by column
        # name; switching to tuples would silently feed positional access.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        self._wire_session_result(
            session,
            columns=["issue_key", "team_name"],
            rows=[("AL-1", "cthulhu"), ("AL-2", "vihlena")],
        )
        out = db.execute_query("SELECT issue_key, team_name FROM tasks")
        assert out == [
            {"issue_key": "AL-1", "team_name": "cthulhu"},
            {"issue_key": "AL-2", "team_name": "vihlena"},
        ]

    def test_empty_result_is_empty_list(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        # Pin: zero rows → []. NOT None. Validator distinguishes
        # ``[]`` (empty result) from ``None`` (error) via truthiness;
        # returning None would silently mark success as failure.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        self._wire_session_result(session, columns=["x"], rows=[])
        assert db.execute_query("SELECT x WHERE FALSE") == []

    def test_with_params_passed_through(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        self._wire_session_result(session, columns=["x"], rows=[(1,)])
        db.execute_query("SELECT x WHERE k = :k", params={"k": "v"})
        # session.execute received TWO args: the text() expr + params.
        # Pin only that the params dict reached the SUT.
        args, _ = session.execute.call_args
        assert args[1] == {"k": "v"}

    def test_sqlalchemy_error_propagates(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        session.execute = MagicMock(side_effect=SQLAlchemyError("relation missing"))
        with pytest.raises(SQLAlchemyError, match="relation missing"):
            db.execute_query("SELECT * FROM nonexistent")


# ===================================================================== #
# test_connection — health-check semantics
# ===================================================================== #


@pytest.mark.unit
class TestTestConnection:
    def test_returns_true_on_success(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        # Default session.execute returns a MagicMock — that's enough.
        assert db.test_connection() is True

    def test_returns_false_on_sqlalchemy_error(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Critical: a /health endpoint relies on this NOT raising.
        # A regression that removed the try/except would crash the API
        # process the first time the DB went read-only.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        session.execute = MagicMock(side_effect=SQLAlchemyError("connection lost"))
        assert db.test_connection() is False

    def test_non_sqlalchemy_exception_does_propagate(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Pin: only SQLAlchemyError is caught — a programming bug
        # (e.g. AttributeError) should still surface so it's not hidden
        # by the broad health-check try/except.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        session = patched_sqlalchemy["session"]
        session.execute = MagicMock(side_effect=AttributeError("oops"))
        with pytest.raises(AttributeError):
            db.test_connection()


# ===================================================================== #
# close — engine disposal
# ===================================================================== #


@pytest.mark.unit
class TestClose:
    def test_calls_engine_dispose(self, patched_sqlalchemy: dict[str, MagicMock]) -> None:
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        engine = patched_sqlalchemy["engine"]
        db.close()
        # Pin: dispose() drops every connection in the pool. Memory
        # tasks rely on this — without it long-running Celery workers
        # would slowly bleed file descriptors over hours.
        engine.dispose.assert_called_once()


# ===================================================================== #
# get_database — factory
# ===================================================================== #


@pytest.mark.unit
class TestGetDatabaseFactory:
    def test_returns_database_connection_instance(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        db = conn_module.get_database()
        assert isinstance(db, conn_module.DatabaseConnection)

    def test_factory_does_not_cache_singleton(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # Pin: each call returns a fresh connection. Workflow + Celery
        # tasks instantiate their own; sharing a stale one across the
        # process boundary was the cause of an earlier production bug.
        a = conn_module.get_database()
        b = conn_module.get_database()
        assert a is not b


# ===================================================================== #
# Realistic chained context manager — integration-style sanity check
# ===================================================================== #


@pytest.mark.unit
class TestSessionContextManagerIntegration:
    def test_real_contextmanager_decorator_yields_session(
        self, patched_sqlalchemy: dict[str, MagicMock]
    ) -> None:
        # The @contextmanager decorator builds a generator-based CM.
        # Pin: the yielded object IS the session our factory returned —
        # callers do ``with db.get_session() as s`` and expect ``s`` to
        # be the actual SQLAlchemy Session, not the engine or factory.
        db = conn_module.DatabaseConnection("postgresql://u:p@h/d")
        with db.get_session() as s:
            assert s is patched_sqlalchemy["session"]
