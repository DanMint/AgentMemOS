from typing import Literal

from pydantic import BaseModel, Field


class MemoryCandidate(BaseModel):
    key: str
    content: str


class VehiclePurchase(BaseModel):
    """Details stated in the latest message about one completed purchase."""

    model: str = Field(description="Make/model/trim as written by the user")
    year: int
    color: str | None = None
    price: str | None = Field(default=None, description="Amount and currency as written")


class ExtractedMemories(BaseModel):
    memories: list[MemoryCandidate] = Field(default_factory=list)
    vehicle_purchase: VehiclePurchase | None = None


class MemoryRecord(BaseModel):
    id: int
    key: str
    content: str
    version: int
    scope: str = "shared"


class MemoryEdit(BaseModel):
    """A small, verifiable change to existing memory content."""
    kind: Literal["REPLACE", "APPEND"] = "REPLACE"
    old_text: str = ""
    new_text: str


class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    # For ADD, key must match an extracted MemoryCandidate.key.
    key: str | None = None
    # For UPDATE/DELETE, target_id must identify a retrieved memory.
    target_id: int | None = None
    # Only for ADD. Must exactly match one extracted candidate's content.
    new_content: str | None = None
    # Only for UPDATE. Python applies these edits, not the LLM.
    edits: list[MemoryEdit] = Field(default_factory=list)


class MemoryPlan(BaseModel):
    """All decisions for ONE user interaction (not per extracted fragment)."""
    decisions: list[MemoryDecision] = Field(default_factory=list)


class PreparedMemoryChange(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE"]
    key: str
    content: str | None = None
    target_id: int | None = None
    expected_version: int | None = None
    embedding: list[float] | None = None


class UpdateResult(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    key: str
    memory_id: int | None = None
    version: int | None = None


class Message(BaseModel):
    role: Literal["user", "agent"]
    content: str


class ExtractionContext(BaseModel):
    summary: str
    recent_messages: list[Message]
    user_message: Message
    agent_message: Message
