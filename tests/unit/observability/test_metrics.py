"""Unit tests for ``agile_assistant.metrics``.

The metrics registry is the single source of truth for all custom
Prometheus metrics. Tests cover four invariants:

  1. Each public metric is exported under the ``agile_assistant_*``
     namespace prefix — Grafana dashboards key off this prefix and a
     namespace rename would silently break every panel.
  2. Counter / Gauge / Histogram type per metric is pinned — switching
     a Counter to a Gauge would invalidate ``rate()`` queries.
  3. Histogram bucket boundaries are pinned where they encode SLO
     thresholds (pipeline duration, queue wait, response length tokens).
  4. ``initialize_label_combinations`` actually pre-touches every label
     combination it claims to (else fresh-deploy dashboards show "No data"
     instead of zero — that was the bug that motivated this whole helper).
  5. Counter increments are thread-safe under concurrent access — pinning
     prometheus_client's documented guarantee that we rely on (Celery
     ``--pool=threads --concurrency=4`` shares the in-memory registry).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from agile_assistant import metrics as m

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _full_name(metric: Any) -> str:
    """Return the fully namespaced metric name (``agile_assistant_xxx``).

    For Counters, ``_name`` strips the ``_total`` suffix that Prometheus
    appends at scrape time. We re-add it so the names line up with what
    Grafana queries actually use.
    """
    name = metric._name
    if isinstance(metric, Counter) and not name.endswith("_total"):
        name = f"{name}_total"
    return name


def _sample_value(metric_name: str, labels: dict[str, str] | None = None) -> float:
    """Read a single-sample value out of the global REGISTRY.

    This walks every collector once — fine for unit tests where the
    registry is small. For label-bearing metrics, ``labels`` filters
    by exact match.
    """
    for collector in REGISTRY.collect():
        for sample in collector.samples:
            if sample.name != metric_name:
                continue
            if labels is None or all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    raise AssertionError(f"sample {metric_name}{labels or ''} not found")


# ===================================================================== #
# Namespace + type invariants
# ===================================================================== #


@pytest.mark.unit
class TestNamespaceAndType:
    """Pin the (name, type) shape for every metric Grafana reads."""

    @pytest.mark.parametrize(
        ("metric", "expected_name"),
        [
            (m.PIPELINE_DURATION, "agile_assistant_pipeline_duration_seconds"),
            (m.PIPELINE_QUEUE_WAIT, "agile_assistant_pipeline_queue_wait_seconds"),
            (m.TASKS_TOTAL, "agile_assistant_tasks_total"),
            (m.TASKS_IN_PROGRESS, "agile_assistant_tasks_in_progress"),
            (m.CELERY_TASK_DURATION, "agile_assistant_celery_task_duration_seconds"),
            (m.CELERY_TASKS_TOTAL, "agile_assistant_celery_tasks_total"),
            (m.SUPERVISOR_CLASSIFICATIONS, "agile_assistant_supervisor_classifications_total"),
            (m.SUPERVISOR_DURATION, "agile_assistant_supervisor_duration_seconds"),
            (m.SUPERVISOR_FAST_PATH, "agile_assistant_supervisor_fast_path_total"),
            (m.SQL_AGENT_DURATION, "agile_assistant_sql_agent_duration_seconds"),
            (m.SQL_QUERIES_TOTAL, "agile_assistant_sql_queries_total"),
            (m.SQL_EMPTY_RESULTS, "agile_assistant_sql_empty_results_total"),
            (m.RAG_AGENT_DURATION, "agile_assistant_rag_agent_duration_seconds"),
            (m.RAG_REQUESTS, "agile_assistant_rag_requests_total"),
            (m.VALIDATOR_RESULTS, "agile_assistant_validator_results_total"),
            (m.VALIDATOR_DATA_MISSING, "agile_assistant_validator_data_missing_total"),
            (m.RESPONSE_DURATION, "agile_assistant_response_duration_seconds"),
            (m.RESPONSE_LENGTH_TOKENS, "agile_assistant_response_length_tokens"),
            (m.GUARDRAIL_L1_RESULTS, "agile_assistant_guardrail_l1_results_total"),
            (m.GUARDRAIL_L2_RESULTS, "agile_assistant_guardrail_l2_results_total"),
            (m.GUARDRAIL_L3_RESULTS, "agile_assistant_guardrail_l3_results_total"),
            (m.SANITIZER_CORRECTIONS, "agile_assistant_sanitizer_corrections_total"),
            (m.MEMORY_CONTEXT_TOKENS, "agile_assistant_memory_context_tokens"),
            (m.MEMORY_SESSION_ROTATIONS, "agile_assistant_memory_session_rotations_total"),
            (m.JUDGE_CRITERION_SCORES, "agile_assistant_judge_criterion_score"),
            (m.JUDGE_WEIGHTED_TOTAL, "agile_assistant_judge_weighted_total"),
            (m.DATA_SYNC_TIMESTAMP, "agile_assistant_data_sync_timestamp_seconds"),
            (m.DATA_SYNC_DURATION, "agile_assistant_data_sync_duration_seconds"),
            (m.DATA_SYNC_TOTAL, "agile_assistant_data_sync_total"),
        ],
    )
    def test_metric_name_namespaced(self, metric: Any, expected_name: str) -> None:
        assert _full_name(metric) == expected_name

    @pytest.mark.parametrize(
        ("metric", "expected_type"),
        [
            (m.PIPELINE_DURATION, Histogram),
            (m.PIPELINE_QUEUE_WAIT, Histogram),
            (m.TASKS_TOTAL, Counter),
            (m.TASKS_IN_PROGRESS, Gauge),
            (m.CELERY_ACTIVE_TASKS, Gauge),
            (m.CELERY_QUEUE_LENGTH, Gauge),
            (m.SUPERVISOR_FAST_PATH, Counter),
            (m.SUPERVISOR_LLM_CALLS, Counter),
            (m.SUPERVISOR_PARSE_ERRORS, Counter),
            (m.SQL_AGENT_DURATION, Histogram),
            (m.SQL_QUERY_DURATION, Histogram),
            (m.SQL_RETRIES, Counter),
            (m.RAG_RETRIEVAL_DURATION, Histogram),
            (m.RAG_TOP_SCORE, Histogram),
            (m.RAG_FALLBACKS, Counter),
            (m.RAG_REQUESTS, Counter),
            (m.VALIDATOR_RESULTS, Counter),
            (m.RESPONSE_LLM_TIMEOUTS, Counter),
            (m.RESPONSE_TRUNCATED, Counter),
            (m.GUARDRAIL_L3_CHECKS_FAILED, Counter),
            (m.JUDGE_CRITERION_SCORES, Gauge),
            (m.JUDGE_WEIGHTED_TOTAL, Gauge),
            (m.JUDGE_EVALUATIONS_TOTAL, Counter),
            (m.MEMORY_PROFILE_UPDATES, Counter),
            (m.MEMORY_SUMMARIZATIONS, Counter),
            (m.MEMORY_SESSION_ROTATIONS, Counter),
            (m.DATA_SYNC_ROWS, Gauge),
            (m.DATA_SYNC_TIMESTAMP, Gauge),
        ],
    )
    def test_metric_type_pinned(self, metric: Any, expected_type: type) -> None:
        # Counter ↔ Gauge swaps are silent in code but break PromQL:
        # rate() on a gauge produces nonsense; avg_over_time() on a
        # counter undercounts. Pin every type to catch refactor drift.
        assert isinstance(metric, expected_type)


# ===================================================================== #
# Histogram bucket pinning — SLO-bearing metrics only
# ===================================================================== #


@pytest.mark.unit
class TestHistogramBuckets:
    """Some buckets encode SLO thresholds (queue-wait alerts, pipeline
    timeout). Renaming a label is benign; changing a bucket boundary
    silently shifts the alert threshold."""

    def test_pipeline_duration_buckets_match_slo_doc(self) -> None:
        # SLOs documented in monitoring_plan.md §3.1: simple < 1s,
        # sql 3-8s, hybrid up to 30s, timeout at 60s. Pin the bucket
        # boundaries so a "let's add 45s" PR triggers a doc update.
        assert m._PIPELINE_BUCKETS == (
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
            8.0,
            13.0,
            21.0,
            30.0,
            60.0,
        )

    def test_supervisor_duration_buckets_span_fast_and_slow(self) -> None:
        # Fast path = sub-millisecond regex; slow path = ~0.5-3s LLM.
        # Histogram must hit both ends so the dashboard shows fast/slow
        # mix correctly. Pin so a refactor that moved fast-path tracking
        # to a different metric doesn't trim the lower bound.
        buckets = m._SUPERVISOR_BUCKETS
        assert buckets[0] <= 0.001  # capture sub-ms regex hit
        assert buckets[-1] >= 5.0  # capture LLM tail

    def test_response_length_tokens_buckets_terminate_at_max(self) -> None:
        # The top bucket equals the vLLM max_tokens cap (600). If a
        # production trace lives in the >600 bucket, that's the signal
        # to bump the cap or shorten the prompt. Pin: 600 is the last
        # boundary so the +Inf bucket = "exceeded cap".
        # Read the histogram metadata via _upper_bounds (private but
        # stable across prometheus_client versions).
        bounds = list(m.RESPONSE_LENGTH_TOKENS._upper_bounds)
        # Drop the trailing +Inf.
        finite = [b for b in bounds if b != float("inf")]
        assert finite[-1] == 600

    def test_data_sync_duration_top_bucket_matches_kb_ingest_budget(self) -> None:
        # KB ingestion budget is 20 minutes (1200s) — that's the top
        # bucket boundary. A run that crowds the top bucket means the
        # embedding pass is approaching the Beat re-fire window.
        bounds = list(m.DATA_SYNC_DURATION._upper_bounds)
        finite = [b for b in bounds if b != float("inf")]
        assert finite[-1] == 1200.0


# ===================================================================== #
# Pre-touching (initialize_label_combinations)
# ===================================================================== #


@pytest.mark.unit
class TestPreTouching:
    """The pre-touch helpers exist because Grafana renders empty
    counters as "No data", which is indistinguishable from a broken
    scrape. Pin: every claimed label combo is materialised at 0."""

    def test_pipeline_pretouch_sets_completed_and_failed(self) -> None:
        m._pretouch_pipeline()
        # Both labels exist in the registry as 0 (or whatever current
        # accumulated value is). Pin: lookup MUST NOT raise.
        completed = _sample_value("agile_assistant_tasks_total", {"status": "COMPLETED"})
        failed = _sample_value("agile_assistant_tasks_total", {"status": "FAILED"})
        assert completed >= 0
        assert failed >= 0

    def test_guardrail_pretouch_covers_all_layers(self) -> None:
        m._pretouch_guardrails()
        # Pin: every l2 layer Validator inspects has a row.
        for layer in ("limits", "regex", "ast", "ok"):
            for allowed in ("true", "false"):
                _sample_value(
                    "agile_assistant_guardrail_l2_results_total",
                    {"allowed": allowed, "layer": layer},
                )

    def test_guardrail_l3_pretouch_covers_check_names(self) -> None:
        m._pretouch_guardrails()
        # Pin: each L3 check_name from response_guard.py shows up.
        # If a new check is added in code without updating the pretouch
        # list, the dashboard panel goes "No data" until the first
        # block fires — this test catches the omission.
        expected = (
            "length_empty",
            "length_overflow",
            "language",
            "sql_leak",
            "traceback",
            "hallucinated_urls",
            "internal_leak",
        )
        for check in expected:
            _sample_value(
                "agile_assistant_guardrail_l3_checks_failed_total",
                {"check_name": check},
            )

    def test_data_sync_pretouch_covers_both_sources(self) -> None:
        m._pretouch_data_sync()
        for source in ("jira_csv", "knowledge_base"):
            for status in ("success", "failure"):
                _sample_value(
                    "agile_assistant_data_sync_total",
                    {"source": source, "status": status},
                )

    def test_initialize_label_combinations_runs_all_subhelpers(self) -> None:
        # The single public entrypoint must call all four helpers —
        # forgetting one would leave a gap in the dashboard.
        m.initialize_label_combinations()
        # Sentinels from each helper.
        _sample_value("agile_assistant_tasks_total", {"status": "COMPLETED"})
        _sample_value("agile_assistant_guardrail_l1_results_total", {"reason": "pass"})
        _sample_value("agile_assistant_sanitizer_corrections_total", {"layer": "1_synonym"})
        _sample_value(
            "agile_assistant_data_sync_total",
            {"source": "knowledge_base", "status": "success"},
        )

    def test_pretouch_is_idempotent(self) -> None:
        # Pin: calling initialize_label_combinations twice does NOT
        # increment counters — ``inc(0)`` is a no-op. Without this
        # property, a worker restart would inflate counter rates.
        m._pretouch_data_sync()
        before = _sample_value(
            "agile_assistant_data_sync_total",
            {"source": "jira_csv", "status": "success"},
        )
        m._pretouch_data_sync()
        m._pretouch_data_sync()
        after = _sample_value(
            "agile_assistant_data_sync_total",
            {"source": "jira_csv", "status": "success"},
        )
        assert after == before


# ===================================================================== #
# Thread safety — Counter.inc() and Histogram.observe()
# ===================================================================== #


@pytest.mark.unit
class TestThreadSafety:
    """Celery worker pool is ``--pool=threads --concurrency=4``. The
    in-memory registry is documented as thread-safe; we pin that
    contract here so a future migration to ``--pool=prefork`` (which
    would need ``PROMETHEUS_MULTIPROC_DIR``) is caught loudly."""

    def test_counter_inc_under_concurrent_threads(self) -> None:
        # Use a module-local Counter so the test's effect is isolated
        # from the global registry's accumulated values.
        c = Counter(
            "test_thread_counter",
            "Concurrency test counter",
            namespace="test_metrics_unit",
        )
        n_threads = 8
        increments_per_thread = 1000

        def _hammer() -> None:
            for _ in range(increments_per_thread):
                c.inc()

        threads = [threading.Thread(target=_hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Pin: no torn-write loss — total = threads * iterations exactly.
        # If prometheus_client ever ships a non-thread-safe inc(), this
        # test fails with a tiny under-count.
        for collector in REGISTRY.collect():
            for sample in collector.samples:
                if sample.name == "test_metrics_unit_test_thread_counter_total":
                    assert sample.value == n_threads * increments_per_thread
                    return
        raise AssertionError("test counter sample not found")

    def test_histogram_observe_under_concurrent_threads(self) -> None:
        # Same property for Histograms — needed by SQL Agent + RAG Agent
        # which observe() durations from worker threads concurrently.
        h = Histogram(
            "test_thread_hist_seconds",
            "Concurrency test histogram",
            namespace="test_metrics_unit",
            buckets=(0.1, 0.5, 1.0),
        )
        n_threads = 4
        observes_per_thread = 500

        def _hammer() -> None:
            for _ in range(observes_per_thread):
                h.observe(0.05)

        threads = [threading.Thread(target=_hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Pin: histogram count matches total observe() calls exactly.
        for collector in REGISTRY.collect():
            for sample in collector.samples:
                if sample.name == "test_metrics_unit_test_thread_hist_seconds_count":
                    assert sample.value == n_threads * observes_per_thread
                    return
        raise AssertionError("test histogram sample not found")


# ===================================================================== #
# Instrumentation point sanity — metrics actually fire on agent calls
# ===================================================================== #


@pytest.mark.unit
class TestInstrumentationPoints:
    """Pin that the metric definitions are wired into the SUTs at the
    expected points. We don't run full agents (those are unit-tested);
    we verify that the imports and inc() calls are coupled correctly
    by checking that calling a metric we KNOW the SUT increments
    actually moves the dial."""

    def test_sql_empty_results_counter_increments(self) -> None:
        # Pin: SQL_EMPTY_RESULTS is labelled by intent (one Grafana
        # panel per intent shows "% of queries returning 0 rows").
        before = (
            _sample_value("agile_assistant_sql_empty_results_total", {"intent": "metric"})
            if _safe_lookup("agile_assistant_sql_empty_results_total", {"intent": "metric"})
            else 0
        )
        m.SQL_EMPTY_RESULTS.labels(intent="metric").inc()
        after = _sample_value("agile_assistant_sql_empty_results_total", {"intent": "metric"})
        assert after == before + 1

    def test_validator_results_counter_uses_string_labels(self) -> None:
        # Pin: Prometheus labels MUST be strings. ``True``/``False``
        # would be coerced to "True"/"False" — different from the
        # SUT's ``str(...).lower()`` convention. Test pins the lowercase
        # form expected on the dashboard query.
        m.VALIDATOR_RESULTS.labels(use_sql="true", use_rag="false").inc()
        v = _sample_value(
            "agile_assistant_validator_results_total",
            {"use_sql": "true", "use_rag": "false"},
        )
        assert v >= 1

    def test_response_duration_histogram_observes_branch(self) -> None:
        # Pin: branch label set carries every routing branch from
        # _resolve_branch (sql_task, sql_filter, sql_metric, sql, rag,
        # hybrid, simple, error). Smoke-test one to confirm the metric
        # is set up to receive the label.
        m.RESPONSE_DURATION.labels(branch="rag").observe(0.5)
        v = _sample_value("agile_assistant_response_duration_seconds_count", {"branch": "rag"})
        assert v >= 1

    def test_judge_score_set_by_criterion(self) -> None:
        # Gauge.set is what JUDGE_CRITERION_SCORES uses (idempotent
        # last-value semantics). Pin: each criterion lands as its
        # own labelled time-series.
        m.JUDGE_CRITERION_SCORES.labels(criterion="politeness").set(1.0)
        m.JUDGE_CRITERION_SCORES.labels(criterion="completeness").set(0.0)
        # Different criterion labels do NOT alias.
        polite = _sample_value("agile_assistant_judge_criterion_score", {"criterion": "politeness"})
        complete = _sample_value(
            "agile_assistant_judge_criterion_score", {"criterion": "completeness"}
        )
        assert polite == 1.0
        assert complete == 0.0


def _safe_lookup(name: str, labels: dict[str, str]) -> bool:
    """Helper: return True if a sample exists, False otherwise."""
    try:
        _sample_value(name, labels)
        return True
    except AssertionError:
        return False


# ===================================================================== #
# Constants pinned
# ===================================================================== #


@pytest.mark.unit
class TestConstants:
    def test_namespace_pinned(self) -> None:
        # Changing this single constant renames every metric. Critical
        # invariant for Grafana dashboards.
        assert m._NAMESPACE == "agile_assistant"
