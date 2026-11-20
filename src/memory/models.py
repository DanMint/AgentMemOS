from pydantic import BaseModel, Field
from typing import Literal

"""
    Represents on extracted memory wi
    key - key of the extracted memory in the memory table
    content - the content of the memroy
"""
class MemoryCandidate(BaseModel):
    key: str
    content: str

"""
    Represents the set Ω of extracted memories
    memories - a list of memories
"""
class ExtractedMemories(BaseModel):
    memories: list[MemoryCandidate] = Field(
        default_factory=list
    )

"""
    A stored memory. This represents a memory that exists in the data base

    id - Unique database identifier
    key - Identifies the logical fact
    content - Stored information
    version - Tracks modifications
    scope - Determines which memory namespace it belongs to
"""
class MemoryRecord(BaseModel):
    id: int
    key: str
    content: str
    version: int
    scope: str = "shared"

"""
    What should happen to memory. This model describes what operation the updater wants to do.
    action - what should happen (ADD - Insert a new memory, UPDATE - Modify an existing memory, DELETE - Remove an existing memory, NOOP - Make no changes)
    target_id - identifies which existing memory the operation targets
    expected_version - indicates the version of the memory that the updater based its decision on
    new_content - the information that should be stored when adding or updating memory
"""
class MemoryDescion(BaseModel):
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"]
    target_id: int | None = None
    expected_version: int | None = None
    new_content: str | None = None

"""
    Represents a conversation message
    role - user or assistant
    content - what was said
"""
class Message(BaseModel):
    role: Literal["user", "agent"]
    content: str

"""
    Represents the input to the extractor
    summary - most recesnt semantic summarization of enviroment
    recent_messages - {mt-m,...,mt-2} (most recent user input and agent output messages within a window expluding the last two messages)
    user_message - mt-1 user input
    agent_message - mt agent output

    this will be the prompt: P = (summary, recent_messages, user_message, agent_message)
"""
class ExtractionContext(BaseModel):
    summary: str
    recent_messages: list[Message]
    user_message: Message
    agent_message: Message

