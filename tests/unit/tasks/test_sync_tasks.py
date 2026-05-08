"""Unit tests for ``sync_tasks`` — Beat-scheduled S3 → DB / Qdrant pipelines.

Two tasks share the same shape:
  * acquire Redis SETNX lock (skip if held by another worker)
  * run the heavy work (load_csv.main / run_ingestion)
  * record outcome metrics
  * release the lock in a finally block

Tests cover both the cooperative-locking contract AND failure handling
(SystemExit from ``load_csv.main`` must be re-raised as RuntimeError so
Celery's retry machinery sees it; bare exceptions must record failure
and re-raise so retry kicks in).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agile_assistant.tasks import sync_tasks as st

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


@pytest.fixture
def lock_acquired(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the Redis client + SETNX so the lock is granted.

    Returns the lock client mock so tests can assert on .delete (release).
    """
    client = MagicMock()
    client.set.return_value = True  # SETNX succeeded
    client.delete.return_value = 1
    monkeypatch.setattr(st, "_redis_client", lambda: client)
    return client


@pytest.fixture
def lock_held(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the Redis client so SETNX fails — another worker holds the lock."""
    client = MagicMock()
    client.set.return_value = False
    monkeypatch.setattr(st, "_redis_client", lambda: client)
    return client


def _run_eager(task: Any) -> Any:
    """Execute the task body once, bypassing Celery's autoretry wrapper.

    Both sync tasks declare ``autoretry_for=(Exception,) max_retries=1``.
    Going through ``apply()`` re-fires the body on every exception (eager
    mode runs autoretry inline), which makes ``pytest.raises`` see the
    *retry chain* rather than the underlying error. ``_orig_run`` is the
    pre-wrap user function and is already bound to the task as ``self`` —
    we call it with no args so the bound ``self`` is the only positional.
    """
    return task._orig_run()


# ===================================================================== #
# Locking primitives
# ===================================================================== #


@pytest.mark.unit
class TestLockingPrimitives:
    def test_try_acquire_uses_setnx_with_ttl(self) -> None:
        client = MagicMock()
        client.set.return_value = True
        ok = st._try_acquire(client, "jira_csv", 1200)
        assert ok is True
        # Pin the lock-key namespace so collisions across env are visible.
        client.set.assert_called_once()
        kwargs = client.set.call_args.kwargs
        assert kwargs["name"] == "agile:sync:lock:jira_csv"
        assert kwargs["nx"] is True
        assert kwargs["ex"] == 1200

    def test_try_acquire_returns_false_when_held(self) -> None:
        client = MagicMock()
        client.set.return_value = None  # NX failed → set() returns None
        assert st._try_acquire(client, "jira_csv", 1200) is False

    def test_release_calls_delete(self) -> None:
        client = MagicMock()
        st._release(client, "jira_csv")
        client.delete.assert_called_once_with("agile:sync:lock:jira_csv")

    def test_release_swallows_redis_errors(self) -> None:
        # Lock TTL will reap it; release errors must not propagate.
        import redis

        client = MagicMock()
        client.delete.side_effect = redis.RedisError("connection lost")
        # Must not raise.
        st._release(client, "jira_csv")


# ===================================================================== #
# sync_jira_data
# ===================================================================== #


@pytest.mark.unit
class TestSyncJiraData:
    def test_skipped_when_lock_held(self, lock_held: MagicMock) -> None:
        # Another worker is already running — return skip status, no work done.
        out = _run_eager(st.sync_jira_data)
        assert out == {"source": "jira_csv", "status": "skipped"}
        # No release attempt — we never owned the lock.
        lock_held.delete.assert_not_called()

    def test_happy_path_runs_load_csv_and_counts_rows(
        self,
        lock_acquired: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch load_csv.main + the COUNT(*) probe path. The task issues
        # a real psycopg2 connection to count rows after load_csv, so we
        # mock that boundary too.
        monkeypatch.setattr(st.load_csv, "main", lambda: None)

        # Build a chained context-manager mock for psycopg2.connect()
        cur = MagicMock()
        cur.fetchone.side_effect = [(100,), (50,)]  # 100 main rows + 50 metric rows
        cur.__enter__ = lambda self: cur
        cur.__exit__ = lambda *_: False
        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__ = lambda self: conn
        conn.__exit__ = lambda *_: False
        monkeypatch.setattr(st.psycopg2, "connect", lambda *_a, **_kw: conn)

        out = _run_eager(st.sync_jira_data)
        assert out == {"source": "jira_csv", "status": "success", "rows": 150}
        # Lock was released in the finally block.
        lock_acquired.delete.assert_called_once_with("agile:sync:lock:jira_csv")

    def test_load_csv_systemexit_treated_as_failure(
        self,
        lock_acquired: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # load_csv.main calls sys.exit on misconfig — the task must convert
        # that to a RuntimeError so Celery's retry machinery + the failure
        # metric both fire (a raw SystemExit would silently mark "success").
        def _exit() -> None:
            raise SystemExit(2)

        monkeypatch.setattr(st.load_csv, "main", _exit)

        with pytest.raises(RuntimeError, match="load_csv aborted"):
            _run_eager(st.sync_jira_data)
        # Lock still released.
        lock_acquired.delete.assert_called_once()

    def test_db_failure_propagates_after_lock_release(
        self,
        lock_acquired: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # load_csv succeeds but the COUNT probe fails — must bubble up so
        # Celery records a failure (and the retry kicks in once).
        monkeypatch.setattr(st.load_csv, "main", lambda: None)
        monkeypatch.setattr(
            st.psycopg2,
            "connect",
            lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("db down")),
        )
        with pytest.raises(RuntimeError, match="db down"):
            _run_eager(st.sync_jira_data)
        lock_acquired.delete.assert_called_once()


# ===================================================================== #
# sync_knowledge_base
# ===================================================================== #


@pytest.mark.unit
class TestSyncKnowledgeBase:
    def test_skipped_when_lock_held(self, lock_held: MagicMock) -> None:
        out = _run_eager(st.sync_knowledge_base)
        assert out == {"source": "knowledge_base", "status": "skipped"}
        lock_held.delete.assert_not_called()

    def test_happy_path_runs_ingestion_and_returns_chunks(
        self,
        lock_acquired: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(st, "run_ingestion", lambda: 1234)
        out = _run_eager(st.sync_knowledge_base)
        assert out == {"source": "knowledge_base", "status": "success", "rows": 1234}
        lock_acquired.delete.assert_called_once_with("agile:sync:lock:knowledge_base")

    def test_ingestion_failure_propagates_after_lock_release(
        self,
        lock_acquired: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Qdrant unreachable — ingestion raises, we record failure metric
        # and re-raise (Celery autoretry then re-fires once).
        def _boom() -> int:
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(st, "run_ingestion", _boom)
        with pytest.raises(RuntimeError, match="qdrant down"):
            _run_eager(st.sync_knowledge_base)
        # Lock released even on failure (no orphaned lock for the next 2h).
        lock_acquired.delete.assert_called_once()


# ===================================================================== #
# Constants
# ===================================================================== #


@pytest.mark.unit
class TestLockTTLConstants:
    def test_jira_lock_ttl_pinned(self) -> None:
        # 1200s (20 min) — must stay >> realistic worst-case CSV runtime
        # AND < 6h Beat interval so a crashed worker's lock TTLs out
        # before the next slot fires.
        assert st._JIRA_LOCK_TTL_SECONDS == 1200

    def test_kb_lock_ttl_pinned(self) -> None:
        # 2h — KB ingestion (S3 → embeddings → Qdrant upsert) can take
        # 30-60 min; 2h gives ~2x headroom for slow embedding batches.
        assert st._KB_LOCK_TTL_SECONDS == 2 * 60 * 60

    def test_lock_key_prefix_pinned(self) -> None:
        # The key namespace is observed in Redis monitoring — changing
        # it without coordination breaks "redis-cli KEYS agile:sync:*".
        assert st._LOCK_KEY_PREFIX == "agile:sync:lock:"
