from src.memory.models import ExtractedMemories, UpdateResult
from src.storage.memory_repository import MemoryConflictError

"""
    ExtractionManager analyzes conversations and identifies facts
"""
class ExtractionManager:
    def __init__(self, repository, extractor):
        self.repository = repository
        self.extractor = extractor

    async def process_interaction(self, session_id: str, user_message_id: int, agent_message_id: int):
        context = self.repository.build_extraction_context(
            session_id=session_id,
            user_message_id=user_message_id,
            agent_message_id=agent_message_id,
        )

        return await self.extractor.extract(context)

"""
    MemoryUpdateManager compares facts against existing memories and decides whether to ADD, UPDATE 
"""
class MemoryUpdateManager:
    def __init__(self, retriever, updater, consistency, max_attempts: int = 3):
        self.retriever = retriever
        self.updater = updater
        self.consistency = consistency
        self.max_attempts = max_attempts

    async def process(self, extracted: ExtractedMemories, scope: str = "shared") -> list[UpdateResult]:
        results = []
        for candidate in extracted.memories:
            for attempt in range(self.max_attempts):
                # A fresh read is required on every retry.
                existing = await self.retriever.find_similar(candidate=candidate, scope=scope, top_k=10)

                decision = await self.updater.decide(candidate, existing)

                embedding = None
                if decision.action in ("ADD", "UPDATE"):
                    text = candidate.content if decision.action == "ADD" else decision.new_content
                    # embeddings are stored in the db
                    embedding = await self.retriever.embed_document(text)

                try:
                    result = self.consistency.apply(
                        candidate=candidate,
                        decision=decision,
                        scope=scope,
                        embedding=embedding,
                    )
                    results.append(result)
                    break
                except MemoryConflictError:
                    if attempt + 1 == self.max_attempts:
                        raise
                    # Another agent changed memory; re-retrieve and reclassify.
        return results
