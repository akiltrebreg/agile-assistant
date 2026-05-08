"""Execute the AgileWorkflow as a Celery task.

Wraps the existing LangGraph workflow in a Celery task, managing status
updates in PostgreSQL throughout execution. Implements retry logic with
exponential backoff for transient failures and integrates short-term /
long-term memory plus Langfuse tracing and the LLM-as-a-Judge fan-out.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from celery import Task as CeleryTask
from celery.exceptions import SoftTimeLimitExceeded

from agile_assistant.config import settings
from agile_assistant.database.connection import DatabaseConnection, get_database
from agile_assistant.database.task_repository import TaskRepository
from agile_assistant.graph.workflow import AgileWorkflow
from agile_assistant.llm.client import get_llm_client
from agile_assistant.memory.manager import MemoryManager
from agile_assistant.metrics import (
    MEMORY_SESSION_ROTATIONS,
    PIPELINE_DURATION,
    PIPELINE_QUEUE_WAIT,
    TASKS_IN_PROGRESS,
    TASKS_TOTAL,
)
from agile_assistant.models.memory import Conversation, ConversationContext
from agile_assistant.models.task import TaskStatus
from agile_assistant.rag.retriever import get_retriever
from agile_assistant.tasks.celery_app import celery_app
from agile_assistant.tracing import langfuse_client

logger = logging.getLogger(__name__)

# Default token budget for the conversation history slot injected
# into prompts. Counted in *estimate-tokens* (memory.token_estimator,
# CHARS_PER_TOKEN=3), the same scale used by Supervisor's overflow
# guard.
#
# Sized against the Supervisor classifier (the tightest prompt budget
# in the pipeline): avibe vLLM runs with --max-model-len=6144, which
# maps to ~5400 estimate-tokens after the safety factor that covers
# the estimator's undershoot on Cyrillic-heavy prompts. The static
# rubric is ~3500 estimate-tokens, so 800 here leaves ~1100 of
# headroom for the profile and safety margin. Supervisor's guard
# (:data:`_PROMPT_MAX_TOKENS` in agents/supervisor.py) still drops
# history first and profile second on the rare second-turn overflow.
HISTORY_TOKEN_BUDGET = 800

# Any message-bearing conversation idle for longer than this gets closed
# and replaced with a fresh one on the next user request. Keeps sessions
# scoped to a single working context (so the sidebar doesn't accumulate
# one giant "ever-living" chat) and lets the summariser run periodically.
INACTIVITY_THRESHOLD = timedelta(minutes=30)


class WorkflowTask(CeleryTask):
    """Custom Celery task class with lifecycle hooks."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log the permanent failure when retries are exhausted."""
        logger.error(f"[Celery] Task {task_id} failed permanently: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log the retry decision when Celery re-queues the task."""
        logger.warning(f"[Celery] Task {task_id} retrying due to: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        """Log a one-line success line for the completed task."""
        logger.info(f"[Celery] Task {task_id} completed successfully")


@celery_app.task(
    bind=True,
    base=WorkflowTask,
    name="agile_assistant.tasks.execute_workflow",
    # Retry on transient errors (network, DB, vLLM timeouts)
    autoretry_for=(ConnectionError, OSError, TimeoutError),
    retry_backoff=True,  # Exponential backoff: 1s, 2s, 4s, ...
    retry_backoff_max=60,  # Max 60s between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
    max_retries=3,  # Up to 3 retries for transient failures
)
def execute_workflow(  # noqa: PLR0913, PLR0915, PLR0912, C901
    self,
    task_id: str,
    query: str,
    conversation_id: str | None = None,
    user_external_id: str | None = None,
    user_id: str | None = None,
    created_at_ts: float | None = None,
) -> None:
    """Execute ``AgileWorkflow`` for a given task.

    Status flow: ``PENDING`` -> ``PROCESSING`` -> ``COMPLETED`` /
    ``FAILED``.

    Memory integration: if ``conversation_id`` is set, the task loads the
    short-term context and (when a user is present) the long-term profile,
    passes them through the workflow, and persists the resulting turn on
    success.

    Retry semantics: ``ConnectionError`` / ``OSError`` / ``TimeoutError``
    are auto-retried with exponential backoff (capped at 60s) up to three
    times. ``SoftTimeLimitExceeded`` and any other exception are terminal
    and immediately mark the task ``FAILED``. Queue-wait is recorded only
    on the first attempt to avoid inflating p95 with retry latency.

    Args:
        self: Celery task instance (injected by ``bind=True``).
        task_id: Application task UUID (from the ``tasks`` table).
        query: User query to process.
        conversation_id: Memory-layer conversation id (pre-created by API).
        user_external_id: External (cookie / SSO) user id.
        user_id: Deprecated alias for ``user_external_id`` — accepted so
            older Celery messages in the queue don't crash after deploy.
        created_at_ts: POSIX timestamp when the API enqueued this task,
            used to record queue wait. Optional so old in-flight messages
            from before this deploy don't crash.
    """
    task_uuid = UUID(task_id)
    db: DatabaseConnection | None = None
    external_id = user_external_id or user_id

    # Queue wait: record once on first attempt only — retries spend
    # time in the worker, not the queue, and would inflate p95 wait.
    if created_at_ts is not None and self.request.retries == 0:
        wait = max(0.0, time.time() - created_at_ts)
        PIPELINE_QUEUE_WAIT.observe(wait)

    pipeline_start = time.time()
    query_type = "error"
    # ``terminal_status`` stays None for transient retries (Celery will
    # re-queue the task; counting it as FAILED here would double-count
    # once the eventual retry settles). It flips to COMPLETED/FAILED
    # only on a terminal outcome.
    terminal_status: str | None = None
    TASKS_IN_PROGRESS.inc()

    # Langfuse tracing is now driven by the LangGraph callback handler
    # (see ``AgileWorkflow.run``). The handler receives user_id,
    # session_id and metadata at invoke time, then emits one trace per
    # ``graph.invoke`` with one span per node — this is what survives
    # LangGraph's Pregel runtime, unlike the previous imperative
    # ``langfuse_client.trace(...)`` pattern. We capture the resulting
    # trace_id from the workflow result so the judge task can attach
    # scores to the same trace.
    final_response = ""
    langfuse_trace_id = ""

    try:
        db = get_database()
        repo = TaskRepository(db)
        _mark_processing_if_first(self, repo, task_uuid, task_id)

        memory = MemoryManager(db)
        conv_uuid, internal_user_uuid = _resolve_memory_ids(
            memory,
            conversation_id,
            external_id,
            task_uuid=task_uuid,
            task_repo=repo,
        )
        ctx, user_profile = _load_memory_payloads(memory, conv_uuid, internal_user_uuid)

        llm_client = get_llm_client()
        retriever = _get_retriever_safe()
        workflow = AgileWorkflow(llm_client, db_connection=db, retriever=retriever)

        logger.info(f"[WorkflowTask {task_id}] Running AgileWorkflow for: {query[:50]}...")
        result = workflow.run(
            query,
            conversation_id=str(conv_uuid) if conv_uuid else None,
            user_id=str(internal_user_uuid) if internal_user_uuid else None,
            conversation_context=ctx,
            user_profile=user_profile,
            # Tracing metadata — picked up by the Langfuse callback
            # handler inside ``workflow.run``. Kept separate from the
            # internal ``user_id`` (a memory uuid) so the trace UI
            # shows the human-meaningful external identifier.
            trace_user_id=external_id,
            trace_metadata={
                "task_id": task_id,
                "celery_retry": self.request.retries,
            },
        )
        # The handler exposes the freshly created trace_id via
        # the result dict; pop it so it doesn't leak into the
        # workflow_state we persist in PostgreSQL.
        langfuse_trace_id = result.pop("_langfuse_trace_id", "") or ""

        if conv_uuid is not None:
            _persist_turn_safe(memory, conv_uuid, query, result, task_id)
            if internal_user_uuid is not None:
                _enqueue_profile_refresh(internal_user_uuid, conv_uuid)

        entities = result.get("entities") or {}
        query_type = result.get("query_type", "sql")
        final_response = result.get("final_response", "No response generated")
        logger.info(f"[WorkflowTask {task_id}] Workflow completed (query_type={query_type})")
        repo.update_task_status(
            task_uuid,
            TaskStatus.COMPLETED,
            result={
                "final_response": final_response,
                "issue_key": entities.get("issue_key"),
                "query_type": query_type,
                "conversation_id": str(conv_uuid) if conv_uuid else None,
            },
            workflow_state=result,
        )
        terminal_status = "COMPLETED"

    except SoftTimeLimitExceeded:
        # Hard timeout -- no retries, fail immediately
        error_msg = "Task exceeded soft time limit"
        logger.error(f"[WorkflowTask {task_id}] {error_msg}")
        _update_task_failed(db, task_uuid, task_id, error_msg)
        terminal_status = "FAILED"
        raise

    except (ConnectionError, OSError, TimeoutError):
        # Transient errors -- autoretry handles these.
        # On the last retry mark as FAILED in DB before re-raise.
        if self.request.retries >= self.max_retries:
            error_msg = f"Failed after {self.max_retries} retries (transient error)"
            _update_task_failed(db, task_uuid, task_id, error_msg)
            terminal_status = "FAILED"
        raise

    except Exception as e:
        # Non-retryable errors -- mark FAILED immediately
        error_msg = f"{type(e).__name__}: {e!s}"
        logger.error(f"[WorkflowTask {task_id}] Workflow failed: {error_msg}", exc_info=True)
        _update_task_failed(db, task_uuid, task_id, error_msg)
        terminal_status = "FAILED"
        raise

    finally:
        TASKS_IN_PROGRESS.dec()
        duration = time.time() - pipeline_start
        PIPELINE_DURATION.labels(query_type=query_type).observe(duration)
        if terminal_status is not None:
            TASKS_TOTAL.labels(status=terminal_status).inc()
        if db is not None:
            db.close()

        # The LangGraph callback handler closes the trace and flushes
        # asynchronously when ``graph.invoke`` returns, so there's no
        # imperative ``trace.update(...)`` or ``flush()`` to make here.
        # We still call ``flush`` defensively to make sure events are
        # shipped before the worker process moves on (the SDK uses a
        # background thread, and a long Celery idle gap could swallow
        # the queued events).
        if langfuse_client is not None:
            try:
                langfuse_client.flush()
            except Exception as exc:
                logger.warning("[WorkflowTask %s] Langfuse flush failed: %s", task_id, exc)

        # LLM-as-a-Judge (Phase 4) — fire-and-forget. Runs only on
        # successful completions with a real, evaluable response.
        # off_topic / error responses are scripted; rating them tells us
        # nothing about model quality and just burns vsellm tokens.
        if (
            settings.judge_enabled
            and terminal_status == "COMPLETED"
            and final_response
            and query_type not in ("off_topic", "error")
        ):
            judge_trace_id = langfuse_trace_id
            try:
                # Lazy import keeps the workflow_task <-> judge_task
                # import graph acyclic and lets the workflow worker
                # start even if the judge module fails to load
                # (e.g. openai SDK missing in the celery-worker image).
                from agile_assistant.tasks.judge_task import (  # noqa: PLC0415
                    evaluate_response_async,
                )

                evaluate_response_async.apply_async(
                    kwargs={
                        "trace_id": judge_trace_id,
                        "query": query,
                        "response": final_response,
                        "query_type": query_type,
                    },
                    queue="judge",
                )
            except Exception as exc:
                logger.warning("[WorkflowTask %s] Failed to enqueue judge task: %s", task_id, exc)


def _mark_processing_if_first(
    task: Any,
    repo: TaskRepository,
    task_uuid: UUID,
    task_id: str,
) -> None:
    """Flip the task to ``PROCESSING`` on attempt 0; log retries otherwise.

    Args:
        task: Bound Celery task (used to read ``request.retries``).
        repo: Task repository for the status update.
        task_uuid: Task UUID to update.
        task_id: Task id string used in log lines.
    """
    if task.request.retries == 0:
        logger.info(f"[WorkflowTask {task_id}] Starting workflow execution")
        repo.update_task_status(task_uuid, TaskStatus.PROCESSING)
    else:
        logger.info(
            "[WorkflowTask %s] Retry %s/%s", task_id, task.request.retries, task.max_retries
        )


def _resolve_memory_ids(
    memory: MemoryManager,
    conversation_id: str | None,
    external_id: str | None,
    *,
    task_uuid: UUID | None = None,
    task_repo: TaskRepository | None = None,
) -> tuple[UUID | None, UUID | None]:
    """Resolve request identifiers to internal UUIDs for the memory layer.

    Also rotates conversations idle for more than
    :data:`INACTIVITY_THRESHOLD` so long breaks start a fresh session
    with a summarised predecessor. ``task_uuid`` + ``task_repo`` are
    optional: when both are present and a rotation happens, the audit
    row in ``tasks`` is repointed at the freshly created conversation
    so debugging stays honest.

    Args:
        memory: Memory manager used for repo access.
        conversation_id: Caller-provided conversation id (string), or
            ``None`` to skip short-term memory.
        external_id: External (cookie / SSO) user id, or ``None``.
        task_uuid: Optional audit task uuid for repointing on rotation.
        task_repo: Optional task repository used for the audit update.

    Returns:
        ``(conversation_uuid, internal_user_uuid)``; either may be
        ``None`` for anonymous sessions or when no conversation exists.
    """
    internal_user_uuid: UUID | None = None
    if external_id:
        internal_user_uuid = memory.profile_repo.get_or_create(external_id).id

    conv_uuid: UUID | None = None
    if conversation_id:
        conv = memory.get_or_create_conversation(UUID(conversation_id), internal_user_uuid)
        conv = _maybe_rotate_stale_conversation(
            memory,
            conv,
            internal_user_uuid,
            task_uuid=task_uuid,
            task_repo=task_repo,
        )
        conv_uuid = conv.id
    return conv_uuid, internal_user_uuid


def _maybe_rotate_stale_conversation(
    memory: MemoryManager,
    conv: Conversation,
    user_uuid: UUID | None,
    *,
    task_uuid: UUID | None = None,
    task_repo: TaskRepository | None = None,
) -> Conversation:
    """Close an idle conversation and hand back a fresh one.

    Rotates only when:
      * the user is identified (rotation for anon users would leave
        orphan closed conversations nobody can see in the sidebar),
      * the conversation is still active and has at least one message
        (rotating an empty conversation just wastes a row), and
      * ``now - conv.updated_at > INACTIVITY_THRESHOLD``.

    The stale conversation is closed and submitted for summarisation;
    failures at either step degrade gracefully — the new conversation
    is still returned so the user's current request is never blocked.

    When ``task_uuid`` and ``task_repo`` are provided,
    ``tasks.conversation_id`` is updated to the new conversation so the
    audit trail reflects where the request was actually processed.

    Args:
        memory: Memory manager exposing the conversation repository.
        conv: Current conversation candidate for rotation.
        user_uuid: Internal user uuid; ``None`` short-circuits rotation.
        task_uuid: Optional audit task uuid for repointing.
        task_repo: Optional task repository used for the audit update.

    Returns:
        ``conv`` itself when no rotation is needed, otherwise the
        freshly created conversation.
    """
    if user_uuid is None or not conv.is_active or conv.updated_at is None:
        return conv

    now = datetime.now(conv.updated_at.tzinfo or UTC)
    if now - conv.updated_at <= INACTIVITY_THRESHOLD:
        return conv
    if memory.conversation_repo.count_messages(conv.id) == 0:
        return conv

    logger.info(
        "[WorkflowTask] Rotating stale conversation %s (idle=%s)",
        conv.id,
        now - conv.updated_at,
    )
    MEMORY_SESSION_ROTATIONS.labels(reason="inactivity").inc()
    try:
        memory.conversation_repo.close(conv.id)
    except Exception as e:
        logger.warning("[WorkflowTask] Failed to close stale conv %s: %s", conv.id, e)

    try:
        # Lazy import to keep the memory_tasks <-> workflow_task cycle broken.
        from agile_assistant.tasks.memory_tasks import summarize_session  # noqa: PLC0415

        summarize_session.apply_async(args=[str(conv.id), str(user_uuid)])
    except Exception as e:
        logger.warning("[WorkflowTask] Failed to enqueue summarisation: %s", e)

    new_conv = memory.conversation_repo.create(user_id=user_uuid)

    if task_uuid is not None and task_repo is not None:
        try:
            task_repo.update_conversation_id(task_uuid, new_conv.id)
        except Exception as e:
            # Audit-only update — never block the user's response on it.
            logger.warning(
                "[WorkflowTask] Failed to repoint task %s to conv %s: %s",
                task_uuid,
                new_conv.id,
                e,
            )

    return new_conv


def _load_memory_payloads(
    memory: MemoryManager,
    conv_uuid: UUID | None,
    user_uuid: UUID | None,
) -> tuple[ConversationContext | None, dict[str, Any] | None]:
    """Load short-term context and long-term profile dict for the workflow.

    Args:
        memory: Memory manager used for context and profile retrieval.
        conv_uuid: Conversation uuid, or ``None`` to skip context.
        user_uuid: Internal user uuid, or ``None`` to skip profile.

    Returns:
        ``(conversation_context, profile_dict)`` — either side may be
        ``None`` when the corresponding identifier is missing.
    """
    ctx: ConversationContext | None = None
    if conv_uuid is not None:
        ctx = memory.get_context(conv_uuid, token_budget=HISTORY_TOKEN_BUDGET)
    profile = memory.get_profile(user_uuid) if user_uuid is not None else None
    return ctx, profile


def _persist_turn_safe(
    memory: MemoryManager,
    conv_uuid: UUID,
    query: str,
    result: dict[str, Any],
    task_id: str,
) -> None:
    """Save the user + assistant turn, logging and swallowing failures.

    A persistence error must not hide the answer from the user, so any
    exception is logged and absorbed.

    Args:
        memory: Memory manager exposing ``save_turn``.
        conv_uuid: Conversation that owns the turn.
        query: User-side message body.
        result: Workflow result dict (final response and metadata).
        task_id: Task id string used in log lines.
    """
    metadata: dict[str, Any] = {
        "query_type": result.get("query_type"),
        "intent": result.get("intent"),
        "entities": result.get("entities") or {},
    }
    if sql := result.get("sql_query"):
        metadata["last_sql"] = sql
    try:
        memory.save_turn(
            conversation_id=conv_uuid,
            user_message=query,
            bot_message=result.get("final_response", ""),
            metadata=metadata,
        )
    except Exception as save_err:
        logger.error(
            "[WorkflowTask %s] save_turn failed for conv %s: %s",
            task_id,
            conv_uuid,
            save_err,
        )


def _enqueue_profile_refresh(user_id: UUID, conversation_id: UUID) -> None:
    """Schedule an async profile refresh without blocking the response.

    Uses a late import to avoid a circular dependency between
    ``workflow_task`` and ``memory_tasks`` at module load time.

    Args:
        user_id: Internal user uuid whose profile to refresh.
        conversation_id: Conversation feeding the refresh.
    """
    try:
        # Lazy import to break the workflow_task <-> memory_tasks cycle.
        from agile_assistant.tasks.memory_tasks import update_profile_async  # noqa: PLC0415

        update_profile_async.apply_async(args=[str(user_id), str(conversation_id)])
    except Exception as e:
        logger.warning("[WorkflowTask] Could not enqueue profile refresh: %s", e)


def _get_retriever_safe():
    """Return the Qdrant retriever, or ``None`` if it cannot be acquired."""
    try:
        return get_retriever()
    except Exception as e:
        logger.warning("[WorkflowTask] Qdrant retriever unavailable: %s", e)
        return None


def _update_task_failed(db, task_uuid: UUID, task_id: str, error_msg: str) -> None:
    """Update task status to ``FAILED`` in the database.

    Args:
        db: Database connection (may be ``None``).
        task_uuid: Task UUID.
        task_id: Task ID string (for logging).
        error_msg: Error message to store.
    """
    try:
        if db is None:
            db = get_database()
        repo = TaskRepository(db)
        repo.update_task_status(task_uuid, TaskStatus.FAILED, error=error_msg)
    except Exception as update_error:
        logger.error(f"[WorkflowTask {task_id}] Failed to update status to FAILED: {update_error}")
