"""Celery task for executing AgileWorkflow.

This module wraps the existing LangGraph workflow in a Celery task,
managing status updates in PostgreSQL throughout execution.
Includes retry logic with exponential backoff for transient failures.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from celery import Task as CeleryTask
from celery.exceptions import SoftTimeLimitExceeded

from hse_prom_prog.database.connection import DatabaseConnection, get_database
from hse_prom_prog.database.task_repository import TaskRepository
from hse_prom_prog.graph.workflow import AgileWorkflow
from hse_prom_prog.llm.client import get_llm_client
from hse_prom_prog.memory.manager import MemoryManager
from hse_prom_prog.models.memory import Conversation, ConversationContext
from hse_prom_prog.models.task import TaskStatus
from hse_prom_prog.rag.retriever import get_retriever
from hse_prom_prog.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Default token budget for the history slot injected into prompts.
# 1200 ≈ the Response Agent's hybrid-branch floor; context builder
# trims older turns when the total would exceed this.
HISTORY_TOKEN_BUDGET = 1200

# Any message-bearing conversation idle for longer than this gets closed
# and replaced with a fresh one on the next user request. Keeps sessions
# scoped to a single working context (so the sidebar doesn't accumulate
# one giant "ever-living" chat) and lets the summariser run periodically.
INACTIVITY_THRESHOLD = timedelta(minutes=30)


class WorkflowTask(CeleryTask):
    """Custom Celery task class with lifecycle hooks."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails after all retries exhausted."""
        logger.error(f"[Celery] Task {task_id} failed permanently: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is being retried."""
        logger.warning(f"[Celery] Task {task_id} retrying due to: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"[Celery] Task {task_id} completed successfully")


@celery_app.task(
    bind=True,
    base=WorkflowTask,
    name="hse_prom_prog.tasks.execute_workflow",
    # Retry on transient errors (network, DB, vLLM timeouts)
    autoretry_for=(ConnectionError, OSError, TimeoutError),
    retry_backoff=True,  # Exponential backoff: 1s, 2s, 4s, ...
    retry_backoff_max=60,  # Max 60s between retries
    retry_jitter=True,  # Add randomness to prevent thundering herd
    max_retries=3,  # Up to 3 retries for transient failures
)
def execute_workflow(  # noqa: PLR0913
    self,
    task_id: str,
    query: str,
    conversation_id: str | None = None,
    user_external_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Execute AgileWorkflow for a given task.

    Status flow: PENDING -> PROCESSING -> COMPLETED / FAILED.

    Memory integration: if ``conversation_id`` is set, the task loads the
    short-term context and (when a user is present) the long-term profile,
    passes them through the workflow, and persists the resulting turn on
    success.

    Args:
        self: Celery task instance (injected by bind=True).
        task_id: Application task UUID (from tasks table).
        query: User query to process.
        conversation_id: Memory-layer conversation id (pre-created by API).
        user_external_id: External (cookie / SSO) user id.
        user_id: Deprecated alias for ``user_external_id`` — accepted so
            older Celery messages in the queue don't crash after deploy.
    """
    task_uuid = UUID(task_id)
    db: DatabaseConnection | None = None
    external_id = user_external_id or user_id

    try:
        db = get_database()
        repo = TaskRepository(db)
        _mark_processing_if_first(self, repo, task_uuid, task_id)

        memory = MemoryManager(db)
        conv_uuid, internal_user_uuid = _resolve_memory_ids(memory, conversation_id, external_id)
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
        )

        if conv_uuid is not None:
            _persist_turn_safe(memory, conv_uuid, query, result, task_id)
            if internal_user_uuid is not None:
                _enqueue_profile_refresh(internal_user_uuid, conv_uuid)

        entities = result.get("entities") or {}
        query_type = result.get("query_type", "sql")
        logger.info(f"[WorkflowTask {task_id}] Workflow completed (query_type={query_type})")
        repo.update_task_status(
            task_uuid,
            TaskStatus.COMPLETED,
            result={
                "final_response": result.get("final_response", "No response generated"),
                "issue_key": entities.get("issue_key"),
                "query_type": query_type,
                "conversation_id": str(conv_uuid) if conv_uuid else None,
            },
            workflow_state=result,
        )

    except SoftTimeLimitExceeded:
        # Hard timeout -- no retries, fail immediately
        error_msg = "Task exceeded soft time limit"
        logger.error(f"[WorkflowTask {task_id}] {error_msg}")
        _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    except (ConnectionError, OSError, TimeoutError):
        # Transient errors -- autoretry handles these.
        # On the last retry mark as FAILED in DB before re-raise.
        if self.request.retries >= self.max_retries:
            error_msg = f"Failed after {self.max_retries} retries (transient error)"
            _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    except Exception as e:
        # Non-retryable errors -- mark FAILED immediately
        error_msg = f"{type(e).__name__}: {e!s}"
        logger.error(f"[WorkflowTask {task_id}] Workflow failed: {error_msg}", exc_info=True)
        _update_task_failed(db, task_uuid, task_id, error_msg)
        raise

    finally:
        if db is not None:
            db.close()


def _mark_processing_if_first(
    task: Any,
    repo: TaskRepository,
    task_uuid: UUID,
    task_id: str,
) -> None:
    """Flip the task to PROCESSING on attempt 0; log retries otherwise."""
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
) -> tuple[UUID | None, UUID | None]:
    """Resolve request identifiers to internal UUIDs for the memory layer.

    Returns ``(conversation_uuid, internal_user_uuid)``; either may be
    ``None`` (anonymous session / no conversation). Also rotates
    conversations idle for more than :data:`INACTIVITY_THRESHOLD` so
    long breaks start a fresh session with a summarised predecessor.
    """
    internal_user_uuid: UUID | None = None
    if external_id:
        internal_user_uuid = memory.profile_repo.get_or_create(external_id).id

    conv_uuid: UUID | None = None
    if conversation_id:
        conv = memory.get_or_create_conversation(UUID(conversation_id), internal_user_uuid)
        conv = _maybe_rotate_stale_conversation(memory, conv, internal_user_uuid)
        conv_uuid = conv.id
    return conv_uuid, internal_user_uuid


def _maybe_rotate_stale_conversation(
    memory: MemoryManager,
    conv: Conversation,
    user_uuid: UUID | None,
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
    try:
        memory.conversation_repo.close(conv.id)
    except Exception as e:
        logger.warning("[WorkflowTask] Failed to close stale conv %s: %s", conv.id, e)

    try:
        # Lazy import to keep the memory_tasks <-> workflow_task cycle broken.
        from hse_prom_prog.tasks.memory_tasks import summarize_session  # noqa: PLC0415

        summarize_session.apply_async(args=[str(conv.id), str(user_uuid)])
    except Exception as e:
        logger.warning("[WorkflowTask] Failed to enqueue summarisation: %s", e)

    return memory.conversation_repo.create(user_id=user_uuid)


def _load_memory_payloads(
    memory: MemoryManager,
    conv_uuid: UUID | None,
    user_uuid: UUID | None,
) -> tuple[ConversationContext | None, dict[str, Any] | None]:
    """Load short-term context and long-term profile dict for the workflow."""
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
    """Save the user + assistant turn. Logs and swallows failures — a
    persistence error must not hide the answer from the user."""
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
    """
    try:
        # Lazy import to break the workflow_task <-> memory_tasks cycle.
        from hse_prom_prog.tasks.memory_tasks import update_profile_async  # noqa: PLC0415

        update_profile_async.apply_async(args=[str(user_id), str(conversation_id)])
    except Exception as e:
        logger.warning("[WorkflowTask] Could not enqueue profile refresh: %s", e)


def _get_retriever_safe():
    """Try to get the Qdrant retriever; return None if unavailable."""
    try:
        return get_retriever()
    except Exception as e:
        logger.warning("[WorkflowTask] Qdrant retriever unavailable: %s", e)
        return None


def _update_task_failed(db, task_uuid: UUID, task_id: str, error_msg: str) -> None:
    """Update task status to FAILED in database.

    Args:
        db: Database connection (may be None).
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
