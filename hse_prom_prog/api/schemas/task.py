"""Pydantic schemas for task API requests and responses."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TaskCreateRequest(BaseModel):
    """Request schema for creating a new task.

    Attributes:
        query: User query to process (1-10000 characters).
        conversation_id: Optional existing conversation id. If omitted,
            a new conversation is created and returned in the response.
        user_id: Optional stable external user id (e.g. cookie UUID).
            ``None`` means anonymous — no profile is loaded or updated.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query to process",
        examples=["Расскажи о задаче AL-38787"],
    )
    conversation_id: UUID | None = Field(
        None,
        description="Existing conversation id (None = create new)",
    )
    user_id: str | None = Field(
        None,
        max_length=255,
        description="Stable external user id (cookie UUID, SSO id, ...)",
    )


class TaskCreateResponse(BaseModel):
    """Response schema after creating a task.

    Attributes:
        task_id: Unique task identifier (UUID).
        conversation_id: Conversation the task is bound to (existing or new).
        status: Initial task status (always PENDING).
        message: Human-readable status message.
    """

    task_id: UUID = Field(..., description="Unique task identifier")
    conversation_id: UUID = Field(
        ...,
        description="Conversation id — clients should reuse for follow-ups",
    )
    status: str = Field(..., description="Initial task status (PENDING)")
    message: str = Field(..., description="Human-readable status message")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "conversation_id": "660e8400-e29b-41d4-a716-446655440111",
                "status": "PENDING",
                "message": "Task created and queued for processing",
            }
        }
    )


class TaskResponse(BaseModel):
    """Response schema for task status and result.

    Attributes:
        task_id: Unique task identifier.
        query: Original user query.
        status: Current task status (PENDING/PROCESSING/COMPLETED/FAILED).
        result: Task result data (available when COMPLETED).
        error: Error message (available when FAILED).
        created_at: Timestamp when task was created.
        started_at: Timestamp when task execution started.
        completed_at: Timestamp when task finished.
    """

    task_id: UUID = Field(..., description="Unique task identifier")
    query: str = Field(..., description="Original user query")
    status: str = Field(
        ...,
        description="Task status (PENDING/PROCESSING/COMPLETED/FAILED)",
    )
    result: dict[str, Any] | None = Field(
        None,
        description="Task result (if completed)",
    )
    error: str | None = Field(
        None,
        description="Error message (if failed)",
    )
    created_at: datetime = Field(..., description="Task creation timestamp")
    started_at: datetime | None = Field(None, description="Task start timestamp")
    completed_at: datetime | None = Field(None, description="Task completion timestamp")
    conversation_id: UUID | None = Field(
        None,
        description="Conversation id this task belongs to (if any)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "Расскажи о задаче AL-38787",
                "status": "COMPLETED",
                "result": {
                    "final_response": "Задача AL-38787:\n\nПроект: DeepMind Logistics...",
                    "issue_key": "AL-38787",
                },
                "error": None,
                "created_at": "2026-02-13T12:00:00Z",
                "started_at": "2026-02-13T12:00:01Z",
                "completed_at": "2026-02-13T12:00:15Z",
            }
        }
    )
