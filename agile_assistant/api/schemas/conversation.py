"""Pydantic schemas for the conversations API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ConversationSummaryResponse(BaseModel):
    """Sidebar entry: minimal conversation metadata for list views."""

    id: UUID = Field(..., description="Conversation id")
    title: str | None = Field(None, description="Auto-generated title (first query)")
    updated_at: datetime = Field(..., description="Last activity timestamp")
    is_active: bool = Field(..., description="False once the session is closed")
    message_count: int = Field(..., ge=0, description="Total messages in the conversation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "660e8400-e29b-41d4-a716-446655440111",
                "title": "Какой velocity у cthulhu?",
                "updated_at": "2026-04-24T10:05:00Z",
                "is_active": True,
                "message_count": 4,
            }
        }
    )


class MessageResponse(BaseModel):
    """One chat message as returned by the conversations API."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Full message content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Supervisor classification, entities, last_sql, ...",
    )
    created_at: datetime = Field(..., description="Message creation timestamp")
    turn_index: int = Field(..., ge=0, description="Zero-based position within the conversation")


class ConversationCloseResponse(BaseModel):
    """Response for POST /conversations/{id}/close."""

    id: UUID = Field(..., description="Conversation id")
    is_active: bool = Field(False, description="Always False after close")
    summarize_task_id: str | None = Field(
        None,
        description="Celery task id of the async summariser; None if summarisation is disabled",
    )
