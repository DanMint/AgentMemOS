from src.memory.models import MemoryCandidate, MemoryDecision, UpdateResult
from src.storage.memory_repository import MemoryRepository

class MemoryConsistency:
    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def apply(self, candidate: MemoryCandidate, decision: MemoryDecision, scope: str, embedding: list[float] | None = None) -> UpdateResult:
        if decision.action == "NOOP":
            return UpdateResult(action="NOOP", key=candidate.key)

        if decision.action == "ADD":
            if embedding is None:
                raise ValueError("ADD requires an embedding")
            
            record = self.repository.add_memory(
                key=candidate.key, content=candidate.content, scope=scope, embedding=embedding,
            )

        else:
            if decision.target_id is None or decision.expected_version is None:
                raise ValueError("UPDATE/DELETE require target ID and expected version")
            
            if decision.action == "UPDATE":
                if embedding is None or not decision.new_content:
                    raise ValueError("UPDATE requires content and embedding")
                
                record = self.repository.update_memory(
                    memory_id=decision.target_id,
                    expected_version=decision.expected_version,
                    scope=scope, content=decision.new_content,
                    embedding=embedding,
                ) 

            elif decision.action == "DELETE":
                record = self.repository.delete_memory(
                    memory_id=decision.target_id,
                    expected_version=decision.expected_version,
                    scope=scope,
                )
            else:
                raise ValueError(f"Unsupported action: {decision.action}")

        return UpdateResult(
            action=decision.action,
            key=record.key,
            memory_id=record.id,
            version=record.version,
        )
