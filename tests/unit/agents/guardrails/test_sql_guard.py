"""Unit tests for L2 SQL guardrail (``check_sql``).

The guard is a fail-fast 4-layer pipeline:
  1. ``limits``  — empty / too-long
  2. ``regex``   — DDL / DML / dangerous-function / comment / stacked-statement
  3. ``ast``     — sqlglot parse + mutation-node walk + table whitelist
  4. ``limits``  — JOIN count cap

Each test asserts both the pass/block outcome AND the *layer* that produced it,
so a regression that lets an attack slip past the cheap regex into the AST
layer (slow, more parser-bug-prone) is still visible.
"""

from __future__ import annotations

import pytest

from agile_assistant.agents.guardrails import sql_guard
from agile_assistant.agents.guardrails.sql_guard import check_sql

_VALID_SELECT = "SELECT * FROM report_agile_dashboard WHERE issue_key = 'AL-1'"


# ===================================================================== #
# Layer 1 — limits (empty / too long)
# ===================================================================== #


@pytest.mark.unit
class TestLimits:
    """Cheapest layer; runs first."""

    def test_empty_string_blocked(self) -> None:
        result = check_sql("")
        assert result.allowed is False
        assert result.layer == "limits"
        assert result.reason == "empty_query"

    def test_whitespace_only_blocked(self) -> None:
        # `.strip()` happens inside check_sql — pure-whitespace must fail
        # the same as truly empty input.
        result = check_sql("   \n\t  ")
        assert result.allowed is False
        assert result.layer == "limits"
        assert result.reason == "empty_query"

    def test_query_at_max_length_passes(self) -> None:
        # A SELECT padded out to exactly 2000 chars via a literal string
        # so neither comments nor JOINs interfere with the limits check.
        suffix = "' AS s FROM report_agile_dashboard"
        prefix = "SELECT '"
        padding = "a" * (2000 - len(prefix) - len(suffix))
        sql = prefix + padding + suffix
        assert len(sql) == 2000
        result = check_sql(sql)
        assert result.allowed is True

    def test_query_over_max_length_blocked(self) -> None:
        sql = "SELECT '" + "a" * 2100 + "' FROM report_agile_dashboard"
        assert len(sql) > 2000
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "limits"
        assert result.reason == "query_too_long"


# ===================================================================== #
# Layer 2 — regex blacklist
# ===================================================================== #


@pytest.mark.unit
class TestRegexBlacklist:
    """Catches DDL / DML / dangerous functions / comments / procedural blocks."""

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE report_agile_dashboard",
            "drop table users",
            "CREATE TABLE foo (id int)",
            "ALTER TABLE foo ADD COLUMN bar int",
            "TRUNCATE TABLE report_agile_dashboard",
            "RENAME TABLE foo TO bar",
            "DROP INDEX idx_foo",
            "CREATE VIEW v AS SELECT 1",
        ],
    )
    def test_ddl_keywords_blocked(self, sql: str) -> None:
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "regex"

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO users VALUES (1)",
            "insert  into  users  (a) VALUES (1)",
            "UPDATE users SET name = 'x'",
            "DELETE FROM users WHERE 1=1",
            "MERGE INTO target USING src ON ...",
        ],
    )
    def test_dml_blocked(self, sql: str) -> None:
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "regex"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT pg_sleep(10)",
            "SELECT dblink('host=evil', 'SELECT 1')",
            "SELECT lo_import('/etc/passwd')",
        ],
    )
    def test_dangerous_postgres_functions_blocked(self, sql: str) -> None:
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "regex"

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1 -- payload here",
            "SELECT /* sneaky */ 1",
            "SELECT 1 /* block */ FROM t",
        ],
    )
    def test_sql_comments_blocked(self, sql: str) -> None:
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "regex"

    def test_copy_statement_blocked(self) -> None:
        # COPY can read/write the filesystem — must not survive even though
        # PostgreSQL accepts it as a top-level statement.
        result = check_sql("COPY users FROM '/tmp/x.csv'")
        assert result.allowed is False
        assert result.layer == "regex"

    def test_do_block_blocked(self) -> None:
        # PL/pgSQL anonymous code blocks would let an attacker run arbitrary
        # procedural code under the connection's role.
        result = check_sql("DO $$ BEGIN PERFORM 1; END $$")
        assert result.allowed is False
        assert result.layer == "regex"

    @pytest.mark.parametrize(
        "sql",
        [
            "SET role = 'admin'",
            "LOAD 'pg_hba'",
            "GRANT SELECT ON t TO u",
            "REVOKE SELECT ON t FROM u",
        ],
    )
    def test_dcl_and_session_state_blocked(self, sql: str) -> None:
        result = check_sql(sql)
        assert result.allowed is False
        assert result.layer == "regex"

    def test_drop_inside_string_literal_does_not_false_positive(self) -> None:
        # The DDL pattern requires `DROP <object-keyword>` adjacency, so
        # an issue summary that just mentions "DROP-123" does not trip it.
        # A regex that fires on the bare word `DROP` would block half of
        # all bug-report queries.
        sql = (
            "SELECT issue_key FROM report_agile_dashboard WHERE summary = 'fix DROP-123 regression'"
        )
        result = check_sql(sql)
        assert result.allowed is True


# ===================================================================== #
# Layer 3 — AST (sqlglot)
# ===================================================================== #


@pytest.mark.unit
class TestAST:
    """sqlglot-based parse + walk + whitelist."""

    def test_stacked_statements_blocked_at_ast(self) -> None:
        # Two separate statements separated by `;`. There is intentionally
        # no regex for this — AST catches it via len(parsed) > 1.
        result = check_sql("SELECT 1; SELECT 2")
        assert result.allowed is False
        assert result.layer == "ast"
        assert "multiple_statements" in result.reason

    def test_unparseable_sql_blocked_at_ast(self) -> None:
        # Garbage that isn't blocked by regex but cannot be parsed.
        result = check_sql("SELECT WHERE FROM")
        assert result.allowed is False
        assert result.layer == "ast"
        # parse_error or forbidden_statement_type — both are acceptable
        # here; sqlglot may interpret this as a non-SELECT shape.
        assert "parse_error" in result.reason or "forbidden_statement_type" in result.reason

    def test_select_from_disallowed_table_blocked(self) -> None:
        # "users" is not in the allow-list. AST resolves the table reference
        # and blocks before any rows are touched.
        result = check_sql("SELECT * FROM users")
        assert result.allowed is False
        assert result.layer == "ast"
        assert "forbidden_tables" in result.reason

    def test_schema_qualified_disallowed_table_blocked(self) -> None:
        # Qualifying a forbidden table with a schema does not bypass
        # the whitelist — sqlglot exposes only the table name.
        result = check_sql("SELECT * FROM information_schema.tables")
        assert result.allowed is False
        assert result.layer == "ast"


# ===================================================================== #
# Whitelist tables — happy path
# ===================================================================== #


@pytest.mark.unit
class TestWhitelistTables:
    """Allowed tables flow through to layer 4 (JOIN count)."""

    def test_select_from_allowed_table_passes(self) -> None:
        result = check_sql("SELECT * FROM report_agile_dashboard")
        assert result.allowed is True
        assert result.layer == "ok"
        assert result.reason == "ok"

    def test_select_from_metrics_table_passes(self) -> None:
        result = check_sql("SELECT velocity FROM report_agile_dashboard_metrics")
        assert result.allowed is True

    def test_join_between_two_allowed_tables_passes(self) -> None:
        sql = (
            "SELECT d.issue_key, m.velocity "
            "FROM report_agile_dashboard d "
            "JOIN report_agile_dashboard_metrics m ON d.id = m.id"
        )
        assert check_sql(sql).allowed is True


# ===================================================================== #
# Layer 4 — JOIN complexity
# ===================================================================== #


@pytest.mark.unit
class TestJoinComplexity:
    """Caps the number of JOINs to avoid runaway plans."""

    @staticmethod
    def _make_sql_with_n_joins(n: int) -> str:
        joins = " ".join(
            f"JOIN report_agile_dashboard_metrics m{i} ON d.id = m{i}.id" for i in range(n)
        )
        return f"SELECT d.issue_key FROM report_agile_dashboard d {joins}"

    def test_five_joins_allowed(self) -> None:
        # 5 == _MAX_JOIN_COUNT — boundary is inclusive on the safe side.
        result = check_sql(self._make_sql_with_n_joins(5))
        assert result.allowed is True

    def test_six_joins_blocked(self) -> None:
        result = check_sql(self._make_sql_with_n_joins(6))
        assert result.allowed is False
        assert result.layer == "limits"
        assert "too_many_joins" in result.reason

    def test_join_keyword_count_is_case_insensitive(self) -> None:
        # `\bJOIN\b` is matched with re.IGNORECASE; lowercase keywords
        # must count exactly the same.
        sql = self._make_sql_with_n_joins(6).lower()
        assert check_sql(sql).allowed is False


# ===================================================================== #
# Graceful degradation — sqlglot missing
# ===================================================================== #


@pytest.mark.unit
class TestGracefulDegradation:
    """When ``sqlglot`` is unavailable the AST layer must be skipped, not crash."""

    def test_ast_layer_skipped_when_sqlglot_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Flip the module-level flag and ensure regex still works (so DDL
        # is still blocked) while a benign SELECT against an *unknown*
        # table now passes — the AST whitelist is intentionally bypassed.
        monkeypatch.setattr(sql_guard, "_SQLGLOT_AVAILABLE", False)

        # Regex layer still blocks dangerous statements:
        assert check_sql("DROP TABLE foo").allowed is False

        # AST whitelist is no longer enforced — the cost we accept for
        # graceful degradation. This pins the behaviour so a future
        # change to make degradation safer is a deliberate decision,
        # not an accident.
        result = check_sql("SELECT * FROM unknown_table")
        assert result.allowed is True


# ===================================================================== #
# Result shape
# ===================================================================== #


@pytest.mark.unit
class TestResultShape:
    """Pin down the SQLGuardResult contract — used by run_query."""

    def test_passing_result_fields(self) -> None:
        result = check_sql(_VALID_SELECT)
        assert result.allowed is True
        assert result.reason == "ok"
        assert result.layer == "ok"

    @pytest.mark.parametrize(
        ("sql", "expected_layer"),
        [
            ("", "limits"),
            ("DROP TABLE foo", "regex"),
            ("SELECT * FROM users", "ast"),
        ],
    )
    def test_blocking_layer_label_correct(self, sql: str, expected_layer: str) -> None:
        # Used by Prometheus metric labels and Langfuse traces — must be
        # one of the documented values.
        assert check_sql(sql).layer == expected_layer
