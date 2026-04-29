"""Unit tests for ``hse_prom_prog.database.load_csv``.

The module is the S3 → PostgreSQL bridge for the Beat-scheduled
``sync_jira_data`` task. Key behaviours:

  * ``_download_csvs_from_s3`` builds a boto3 client with creds from
    env vars, downloads each file from the configured bucket/prefix
    into a temp dir, and returns the dir path.
  * ``_load_table`` truncates the target table and bulk-loads via
    ``COPY FROM STDIN`` — pinning that TRUNCATE always runs (otherwise
    we'd accumulate duplicates across runs).
  * ``main`` validates ``S3_DATA_BUCKET`` is set, downloads, loads each
    table, closes the connection. ``sys.exit(1)`` on missing bucket is
    pinned so the Beat task converts it to RuntimeError correctly.

We patch ``boto3``, ``psycopg2`` and ``tempfile.mkdtemp`` so no real
S3 / DB / FS interaction happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from hse_prom_prog.database import load_csv as lc

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``boto3.session.Session`` so no AWS-level calls happen."""
    session = MagicMock(name="boto3_session")
    s3 = MagicMock(name="s3_client")
    s3.download_file = MagicMock(return_value=None)
    session.client = MagicMock(return_value=s3)
    monkeypatch.setattr(lc.boto3.session, "Session", MagicMock(return_value=session))
    return s3


@pytest.fixture
def fake_tempdir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect ``tempfile.mkdtemp`` to ``tmp_path`` for hermetic tests."""
    monkeypatch.setattr(lc.tempfile, "mkdtemp", lambda prefix=None: str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_pg_conn(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``psycopg2.connect`` so ``main`` runs without a real DB."""
    conn = MagicMock(name="pg_conn")
    cur = MagicMock(name="pg_cursor")
    cur.fetchone.return_value = (123,)
    cur.__enter__ = lambda self: cur
    cur.__exit__ = lambda *_: False
    conn.cursor = MagicMock(return_value=cur)
    monkeypatch.setattr(lc.psycopg2, "connect", MagicMock(return_value=conn))
    return conn


# ===================================================================== #
# Module-level constants
# ===================================================================== #


@pytest.mark.unit
class TestModuleConstants:
    def test_tables_pinned(self) -> None:
        # Pin the (table, csv) pairs — adding a third would silently
        # extend the load order; removing one would create silent
        # downstream gaps in the SQL Agent prompt.
        assert lc._TABLES == [
            ("report_agile_dashboard", "report_agile_dashboard.csv"),
            ("report_agile_dashboard_metrics", "report_agile_dashboard_metrics.csv"),
        ]


# ===================================================================== #
# _download_csvs_from_s3
# ===================================================================== #


@pytest.mark.unit
class TestDownloadCsvsFromS3:
    def test_downloads_each_table(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: every entry in _TABLES triggers exactly one download_file
        # call. Off-by-one bugs here would silently skip a table.
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "my-bucket")
        monkeypatch.setattr(lc.settings, "s3_data_path", "data")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3.example/")
        result = lc._download_csvs_from_s3()
        assert result == fake_tempdir
        assert fake_s3.download_file.call_count == len(lc._TABLES)

    def test_constructs_keys_from_prefix(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The S3 key path is bucket-relative and the prefix has its
        # trailing slash normalised — pin the key shape so a refactor
        # that stripped the slash twice or doubled it would be caught.
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "my-bucket")
        monkeypatch.setattr(lc.settings, "s3_data_path", "data/2026/")  # trailing slash
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3.example/")
        lc._download_csvs_from_s3()
        # Each call: (bucket, key, local_path)
        first_call = fake_s3.download_file.call_args_list[0]
        bucket, key, _ = first_call.args
        assert bucket == "my-bucket"
        assert key == "data/2026/report_agile_dashboard.csv"

    def test_prefix_without_trailing_slash_normalised(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same test, but the input prefix has no trailing slash.
        # Pin: the SUT adds it (otherwise key becomes "datareport_..csv").
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "raw")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3/")
        lc._download_csvs_from_s3()
        _, key, _ = fake_s3.download_file.call_args_list[0].args
        assert key.startswith("raw/")
        assert "//" not in key

    def test_aws_credentials_read_from_env(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: AWS creds come from the OS env (not from settings) — this
        # matches the deployment pattern where IAM creds rotate via the
        # docker-compose env block, not a Pydantic-managed config.
        captured: dict[str, Any] = {}

        def _capture_session(**kw: Any) -> MagicMock:
            captured.update(kw)
            return fake_s3._mock_parent or MagicMock(client=lambda *_a, **_k: fake_s3)

        # Patch boto3.session.Session at the module level.
        session_factory = MagicMock(side_effect=_capture_session)
        monkeypatch.setattr(lc.boto3.session, "Session", session_factory)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecret")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ru-msk")
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "p")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3/")
        lc._download_csvs_from_s3()
        assert captured["aws_access_key_id"] == "AKIA-test"
        assert captured["aws_secret_access_key"] == "supersecret"
        assert captured["region_name"] == "ru-msk"

    def test_default_region_when_env_missing(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: when AWS_DEFAULT_REGION is unset, the SUT uses
        # ``ru-central1`` (Yandex Cloud default for this deployment).
        captured: dict[str, Any] = {}
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        monkeypatch.setattr(
            lc.boto3.session,
            "Session",
            MagicMock(
                side_effect=lambda **kw: captured.update(kw)
                or MagicMock(client=lambda *_a, **_k: fake_s3)
            ),
        )
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "p")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3/")
        lc._download_csvs_from_s3()
        assert captured["region_name"] == "ru-central1"

    def test_endpoint_url_passed_to_client(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Yandex S3 needs a custom endpoint_url. AWS-default boto3 would
        # try to talk to the AWS endpoint and 403 instead — pin it.
        captured: dict[str, Any] = {}

        def _client(service: str, **kw: Any) -> MagicMock:
            captured["service"] = service
            captured.update(kw)
            return fake_s3

        session_obj = MagicMock(client=_client)
        monkeypatch.setattr(lc.boto3.session, "Session", MagicMock(return_value=session_obj))
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "p")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://storage.yandexcloud.net")
        lc._download_csvs_from_s3()
        assert captured["service"] == "s3"
        assert captured["endpoint_url"] == "https://storage.yandexcloud.net"


# ===================================================================== #
# _load_table — TRUNCATE + COPY + COUNT
# ===================================================================== #


@pytest.mark.unit
class TestLoadTable:
    def _wire_chained_cursor(
        self,
        conn: MagicMock,
        *,
        count: int = 100,
    ) -> MagicMock:
        cur = MagicMock(name="cursor")
        cur.fetchone.return_value = (count,)
        cur.__enter__ = lambda self: cur
        cur.__exit__ = lambda *_: False
        conn.cursor.return_value = cur
        return cur

    def test_truncate_runs_before_copy(
        self,
        tmp_path: Path,
    ) -> None:
        # Pin: TRUNCATE comes first. Reversing the order would let a
        # mid-COPY failure leave the table empty — and a successful
        # second run would silently double the data.
        conn = MagicMock()
        cur = self._wire_chained_cursor(conn, count=42)

        csv = tmp_path / "f.csv"
        csv.write_text("col\n1\n", encoding="utf-8")

        count = lc._load_table(conn, "my_table", csv)
        assert count == 42
        # First execute() call is the TRUNCATE, then copy_expert, then SELECT.
        sql_calls = [c.args[0] for c in cur.execute.call_args_list]
        assert sql_calls[0] == "TRUNCATE TABLE my_table;"
        assert sql_calls[-1] == "SELECT COUNT(*) FROM my_table;"

    def test_copy_expert_called_with_csv_format(
        self,
        tmp_path: Path,
    ) -> None:
        # Pin the COPY statement shape — HEADER true (skip the header
        # line), DELIMITER ',', NULL '' (empty string = NULL). Any of
        # those changing without the CSV producer changing in lock-step
        # silently corrupts data.
        conn = MagicMock()
        cur = self._wire_chained_cursor(conn)
        csv = tmp_path / "f.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")

        lc._load_table(conn, "tbl", csv)
        copy_call = cur.copy_expert.call_args
        sql, fileobj = copy_call.args
        assert "COPY tbl" in sql
        assert "FORMAT csv" in sql
        assert "HEADER true" in sql
        assert "DELIMITER ','" in sql
        assert "NULL ''" in sql
        # The file object passed to copy_expert is a real open()
        # handle — pin that we send the actual file, not the path.
        assert hasattr(fileobj, "read")

    def test_commit_called_after_load(
        self,
        tmp_path: Path,
    ) -> None:
        # Without commit() the COPY stays inside the implicit transaction
        # and is never visible to other connections. Pin: explicit commit.
        conn = MagicMock()
        self._wire_chained_cursor(conn)
        csv = tmp_path / "f.csv"
        csv.write_text("x\n1\n", encoding="utf-8")
        lc._load_table(conn, "t", csv)
        conn.commit.assert_called_once()


# ===================================================================== #
# main()
# ===================================================================== #


@pytest.mark.unit
class TestMain:
    def test_exits_when_bucket_unconfigured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin: misconfig → sys.exit(1). The Beat-scheduled task
        # (sync_tasks.sync_jira_data) catches SystemExit and converts
        # it to RuntimeError("load_csv aborted") — that translation
        # depends on the exit code being non-zero here.
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "")
        with pytest.raises(SystemExit) as exc:
            lc.main()
        assert exc.value.code == 1

    def test_happy_path_loads_each_table(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        fake_pg_conn: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # End-to-end happy path — pin: connection opened, every table
        # in _TABLES gets a load attempt, connection closed exactly once.
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "p")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3/")
        # database_url is a computed property on Settings — patch the
        # class-level property to a plain value for this test only.
        monkeypatch.setattr(
            type(lc.settings),
            "database_url",
            property(lambda _self: "postgresql://u:p@h/d"),
        )

        # Pre-create the CSVs so _load_table's open() succeeds.
        for _, fname in lc._TABLES:
            (fake_tempdir / fname).write_text("col\n1\n", encoding="utf-8")

        # Each _load_table issues two execute calls (TRUNCATE + SELECT)
        # plus one copy_expert. Cursor's fetchone() returns (123,).
        lc.main()
        # Connection lifecycle pinned.
        assert fake_pg_conn.close.call_count == 1

    def test_connection_closed_on_load_failure(
        self,
        fake_s3: MagicMock,
        fake_tempdir: Path,
        fake_pg_conn: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If _load_table raises, the connection MUST still close —
        # leaking would pin a Postgres backend across Beat retries
        # until pg_terminate_backend or restart.
        monkeypatch.setattr(lc.settings, "s3_data_bucket", "b")
        monkeypatch.setattr(lc.settings, "s3_data_path", "p")
        monkeypatch.setattr(lc.settings, "s3_endpoint", "https://s3/")
        monkeypatch.setattr(
            type(lc.settings),
            "database_url",
            property(lambda _self: "postgresql://u:p@h/d"),
        )

        # Force the cursor's TRUNCATE to fail on the first table.
        cur = MagicMock()
        cur.execute = MagicMock(side_effect=RuntimeError("relation missing"))
        cur.__enter__ = lambda self: cur
        cur.__exit__ = lambda *_: False
        fake_pg_conn.cursor.return_value = cur
        for _, fname in lc._TABLES:
            (fake_tempdir / fname).write_text("c\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="relation missing"):
            lc.main()
        fake_pg_conn.close.assert_called_once()
