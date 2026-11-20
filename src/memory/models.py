from pydantic import BaseModel
from typing import Literal

"""
    Proposed memory. This is the information that the extractor deems as potentially worth remembering
    key - a logical identifier of the fact
    content - the actual information you want to remember
"""
class MemoryCandidate(BaseModel):
    key: str
    content: str

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