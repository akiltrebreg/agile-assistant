"""Prometheus metrics registry for the Agile AI Assistant (Phase 1).

This module is the single source of truth for all custom Prometheus
metrics used by the application. Defining the Counter / Histogram /
Gauge objects in one place avoids duplicate-registration errors in
``prometheus_client.REGISTRY`` and gives importers a stable handle.

Where each metric is populated:
  * ``PIPELINE_*`` and ``TASKS_*`` — Celery worker (workflow_task.py).
  * ``CELERY_*`` — Celery worker (celery_app.py signals).
  * ``CELERY_QUEUE_LENGTH`` — populated lazily on each /metrics scrape
    by ``QueueLengthCollector`` (api/app.py), which queries Redis LLEN.

Process model assumptions (Phase 1):
  * Celery worker runs with ``--pool=threads --concurrency=4`` — a
    single Python process, so the default in-memory registry is safe.
    If we ever switch to ``--pool=prefork``, configure
    ``PROMETHEUS_MULTIPROC_DIR`` and migrate to the multiprocess
    collector (see prometheus_client.multiprocess docs).
  * FastAPI runs under gunicorn with 2 UvicornWorker processes. Each
    worker has its own registry, so HTTP counters from
    prometheus-fastapi-instrumentator are split across workers and
    Prometheus may scrape either. Trends remain meaningful; absolute
    rates are undercounted by a factor of N. Acceptable for Phase 1;
    multiprocess mode would fix it cleanly.
"""

from prometheus_client import Counter, Gauge, Histogram

# Shared namespace and a short Fibonacci-like bucket scale chosen to
# match the SLOs documented in monitoring_plan.md (§3.1):
#   simple < 1s, sql 3-8s, hybrid up to 30s, timeout at 60s.
_NAMESPACE = "agile_assistant"
_PIPELINE_BUCKETS = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0, 60.0)

# ── E2E Pipeline ──────────────────────────────────────────────
PIPELINE_DURATION = Histogram(
    "pipeline_duration_seconds",
    "End-to-end pipeline duration from request receipt to response generation",
    labelnames=("query_type",),
    namespace=_NAMESPACE,
    buckets=_PIPELINE_BUCKETS,
)

# Queue wait covers the gap between API enqueue (tasks.created_at) and
# the worker actually starting the task. Mostly Redis-broker overhead
# (10-100ms) until backlog grows; sub-second buckets dominate.
PIPELINE_QUEUE_WAIT = Histogram(
    "pipeline_queue_wait_seconds",
    "Time spent waiting in Celery queue before processing starts",
    namespace=_NAMESPACE,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

TASKS_TOTAL = Counter(
    "tasks_total",
    "Total number of completed tasks",
    labelnames=("status",),
    namespace=_NAMESPACE,
)

TASKS_IN_PROGRESS = Gauge(
    "tasks_in_progress",
    "Number of tasks currently being processed",
    namespace=_NAMESPACE,
)

# ── Celery ────────────────────────────────────────────────────
# Buckets cover background memory tasks (summarisation < 10s) up to
# the workflow hard limit (celery_task_time_limit = 600s).
CELERY_TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Celery task execution duration",
    labelnames=("task_name",),
    namespace=_NAMESPACE,
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total Celery tasks by name and final status",
    labelnames=("task_name", "status"),
    namespace=_NAMESPACE,
)

CELERY_ACTIVE_TASKS = Gauge(
    "celery_active_tasks",
    "Currently executing Celery tasks",
    namespace=_NAMESPACE,
)

CELERY_QUEUE_LENGTH = Gauge(
    "celery_queue_length",
    "Number of tasks waiting in Redis queue",
    namespace=_NAMESPACE,
)

# ── Supervisor Agent ──────────────────────────────────────────
# Bucket scale spans both paths: fast = regex hit (sub-ms) and slow =
# vLLM call (~0.5-3s). Same Histogram so quantiles are comparable
# without a label-aware aggregator.
_SUPERVISOR_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)

SUPERVISOR_CLASSIFICATIONS = Counter(
    "supervisor_classifications_total",
    "Supervisor classifications by intent and query_type",
    labelnames=("intent", "query_type"),
    namespace=_NAMESPACE,
)
SUPERVISOR_DURATION = Histogram(
    "supervisor_duration_seconds",
    "Supervisor classification latency",
    labelnames=("path",),
    namespace=_NAMESPACE,
    buckets=_SUPERVISOR_BUCKETS,
)
SUPERVISOR_FAST_PATH = Counter(
    "supervisor_fast_path_total",
    "Requests classified by regex without LLM call",
    namespace=_NAMESPACE,
)
SUPERVISOR_LLM_CALLS = Counter(
    "supervisor_llm_calls_total",
    "LLM calls for classification (slow path)",
    namespace=_NAMESPACE,
)
SUPERVISOR_PARSE_ERRORS = Counter(
    "supervisor_parse_errors_total",
    "JSON parse failures from LLM classification response",
    namespace=_NAMESPACE,
)

# ── SQL Agent ─────────────────────────────────────────────────
SQL_AGENT_DURATION = Histogram(
    "sql_agent_duration_seconds",
    "Total SQL Agent duration (LangGraph StateGraph execution)",
    labelnames=("intent",),
    namespace=_NAMESPACE,
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0),
)
# Indexed PG queries should land < 50ms; quarter-second is the warning
# zone before we start blaming a missing index.
SQL_QUERY_DURATION = Histogram(
    "sql_query_duration_seconds",
    "PostgreSQL query execution time in run_query tool",
    namespace=_NAMESPACE,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
SQL_QUERIES_TOTAL = Counter(
    "sql_queries_total",
    "SQL queries executed by status",
    labelnames=("status",),
    namespace=_NAMESPACE,
)
SQL_RETRIES = Counter(
    "sql_retries_total",
    "SQL Agent LLM retry attempts",
    namespace=_NAMESPACE,
)
SQL_EMPTY_RESULTS = Counter(
    "sql_empty_results_total",
    "SQL queries returning zero rows",
    labelnames=("intent",),
    namespace=_NAMESPACE,
)
SQL_RESULT_ROWS = Histogram(
    "sql_result_rows",
    "Number of rows returned by SQL query",
    namespace=_NAMESPACE,
    buckets=(0, 1, 5, 10, 20, 50, 100, 200),
)

# ── RAG Agent ─────────────────────────────────────────────────
RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency by search type",
    labelnames=("search_type",),
    namespace=_NAMESPACE,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)
# bge-reranker-v2-m3 on 20 chunks typically lands at 100-500 ms.
RAG_RERANKER_DURATION = Histogram(
    "rag_reranker_duration_seconds",
    "Cross-encoder reranking latency",
    namespace=_NAMESPACE,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)
RAG_AGENT_DURATION = Histogram(
    "rag_agent_duration_seconds",
    "Total RAG Agent duration (retrieval + reranking + LLM generation)",
    namespace=_NAMESPACE,
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0),
)
RAG_CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Chunks returned by retriever before reranking",
    labelnames=("search_type",),
    namespace=_NAMESPACE,
    buckets=(0, 1, 2, 4, 8, 10, 15, 20, 30),
)
RAG_CHUNKS_AFTER_RERANKER = Histogram(
    "rag_chunks_after_reranker",
    "Chunks passing reranker threshold",
    namespace=_NAMESPACE,
    buckets=(0, 1, 2, 3, 4, 5, 10),
)
RAG_TOP_SCORE = Histogram(
    "rag_top_score",
    "Cosine similarity of the best retrieved chunk",
    labelnames=("search_type",),
    namespace=_NAMESPACE,
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_FALLBACKS = Counter(
    "rag_fallbacks_total",
    "RAG search mode fallbacks",
    labelnames=("from_mode", "to_mode"),
    namespace=_NAMESPACE,
)
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total RAG requests by search type",
    labelnames=("search_type",),
    namespace=_NAMESPACE,
)

# ── Validator Agent ───────────────────────────────────────────
VALIDATOR_RESULTS = Counter(
    "validator_results_total",
    "Validation results by data availability",
    labelnames=("use_sql", "use_rag"),
    namespace=_NAMESPACE,
)
VALIDATOR_DATA_MISSING = Counter(
    "validator_data_missing_total",
    "Validation: data source returned no data",
    labelnames=("source",),
    namespace=_NAMESPACE,
)

# ── Response Agent ────────────────────────────────────────────
RESPONSE_DURATION = Histogram(
    "response_duration_seconds",
    "Response Agent generation latency by processing branch",
    labelnames=("branch",),
    namespace=_NAMESPACE,
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0),
)
RESPONSE_LLM_TIMEOUTS = Counter(
    "response_llm_timeouts_total",
    "LLM timeouts during response generation",
    namespace=_NAMESPACE,
)
# vllm_max_tokens=600 is the hard ceiling; if the histogram crowds the
# top bucket the prompt is too tight or the cap is too low.
RESPONSE_LENGTH_TOKENS = Histogram(
    "response_length_tokens",
    "Generated response length in estimated tokens",
    labelnames=("branch",),
    namespace=_NAMESPACE,
    buckets=(10, 25, 50, 100, 150, 200, 300, 400, 500, 600),
)
RESPONSE_TRUNCATED = Counter(
    "response_truncated_total",
    "Responses truncated due to task list overflow (>20 tasks)",
    namespace=_NAMESPACE,
)

# ── Guardrails (L1 / L2 / L3) ─────────────────────────────────
GUARDRAIL_L1_RESULTS = Counter(
    "guardrail_l1_results_total",
    "L1 TopicGuard results",
    labelnames=("reason",),
    namespace=_NAMESPACE,
)
GUARDRAIL_L2_RESULTS = Counter(
    "guardrail_l2_results_total",
    "L2 SQLGuard results",
    labelnames=("allowed", "layer"),
    namespace=_NAMESPACE,
)
GUARDRAIL_L3_RESULTS = Counter(
    "guardrail_l3_results_total",
    "L3 ResponseGuard results",
    labelnames=("passed", "blocked"),
    namespace=_NAMESPACE,
)
GUARDRAIL_L3_CHECKS_FAILED = Counter(
    "guardrail_l3_checks_failed_total",
    "L3 ResponseGuard individual check failures",
    labelnames=("check_name",),
    namespace=_NAMESPACE,
)

# ── Entity Sanitizer ──────────────────────────────────────────
SANITIZER_CORRECTIONS = Counter(
    "sanitizer_corrections_total",
    "Entity corrections by sanitizer layer",
    labelnames=("layer",),
    namespace=_NAMESPACE,
)
SANITIZER_ANAPHORA_CARRIES = Counter(
    "sanitizer_anaphora_carries_total",
    "Anaphoric carry-forward events by entity type",
    labelnames=("entity_type",),
    namespace=_NAMESPACE,
)
SANITIZER_FALLBACK_EXTRACTIONS = Counter(
    "sanitizer_fallback_extractions_total",
    "Fallback enum extractions from raw query (layer 7)",
    labelnames=("field",),
    namespace=_NAMESPACE,
)

# ── Memory Layer ──────────────────────────────────────────────
# Context-window budget defaults to 1200 tokens (HISTORY_TOKEN_BUDGET).
MEMORY_CONTEXT_TOKENS = Histogram(
    "memory_context_tokens",
    "Tokens in conversation_history block injected into prompts",
    namespace=_NAMESPACE,
    buckets=(0, 100, 200, 400, 600, 800, 1000, 1200, 1500),
)
MEMORY_CONTEXT_TURNS = Histogram(
    "memory_context_turns",
    "Dialog turns fitting in the sliding window",
    namespace=_NAMESPACE,
    buckets=(0, 1, 2, 3, 4, 5, 8, 10, 15, 20),
)
MEMORY_SESSION_ROTATIONS = Counter(
    "memory_session_rotations_total",
    "Session rotation events",
    labelnames=("reason",),
    namespace=_NAMESPACE,
)
MEMORY_PROFILE_UPDATES = Counter(
    "memory_profile_updates_total",
    "Async user profile update events",
    namespace=_NAMESPACE,
)
MEMORY_SUMMARIZATIONS = Counter(
    "memory_summarizations_total",
    "Session summarization events",
    namespace=_NAMESPACE,
)

# ── LLM-as-a-Judge ────────────────────────────────────────────
# Phase 4: each completed workflow ships query+response to GPT-5.2
# (vsellm) for asynchronous quality scoring on six binary criteria
# (practicality, language_quality, text_cleanliness, agile_correctness,
# completeness, politeness). Lives in a dedicated celery queue so a
# slow judge LLM call can never block the user-facing pipeline.
#
# Gauge — not Histogram — because scores are absolute 0/1, not draws
# from a distribution. PromQL avg_over_time() gives the rolling "share
# of 1s" we want for the dashboard, equivalent to a mean of binary
# observations. The .set() race between concurrent judge runs is
# acceptable: scrape every 15s + 1h rolling window swallows it.
JUDGE_CRITERION_SCORES = Gauge(
    "judge_criterion_score",
    "LLM-as-a-Judge score per criterion (0 or 1), last evaluation",
    labelnames=("criterion",),
    namespace=_NAMESPACE,
)
JUDGE_WEIGHTED_TOTAL = Gauge(
    "judge_weighted_total",
    "Weighted total score from LLM-as-a-Judge (0.0-1.0)",
    namespace=_NAMESPACE,
)
JUDGE_EVALUATIONS_TOTAL = Counter(
    "judge_evaluations_total",
    "Total judge evaluations by outcome",
    labelnames=("status",),
    namespace=_NAMESPACE,
)
