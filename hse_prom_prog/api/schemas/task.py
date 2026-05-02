"""Pydantic schemas for task API requests and responses."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TaskCreateRequest(BaseModel):
    """Request schema for creating a new task."""

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
    """Response schema returned immediately after queuing a task."""

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
    """Response schema for task status and result polling."""

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
