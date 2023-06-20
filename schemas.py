"""Schemas for the chat app."""
from pydantic import BaseModel, validator
from typing import Optional, List
from langchain.docstore.document import Document


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    type: str
    sources: Optional[List[str]] = None

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info"]:
            raise ValueError("type must be start, stream or end")
        return v
