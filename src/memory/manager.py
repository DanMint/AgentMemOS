from src.storage.repository import ConversationRepository
from src.memory.extractor import Extractor

class ExtractionManager:
    def __init__(self, repository, extractor):
        self.repository = repository
        self.extractor = extractor

    async def process_interaction(self, session_id: str, user_message_id: int, agent_message_id: int):
        # 1. Build context from SQLite
        context = self.repository.build_extraction_context(
            session_id=session_id,
            user_message_id=user_message_id,
            agent_message_id=agent_message_id
        )

        # 2. Run LLM extraction
        result = await self.extractor.extract(context)

        # 3. Return salient memory candidates
        return result