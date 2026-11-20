import math

from langchain_ollama import OllamaEmbeddings

from src.memory.models import MemoryCandidate, MemoryRecord
from src.storage.memory_repository import MemoryRepository


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    
    denominator = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b))
    
    return sum(x*y for x, y in zip(a, b)) / denominator if denominator else -1.0


class MemoryRetriever:
    def __init__(self, repository: MemoryRepository, embedding_model: str = "nomic-embed-text"):
        self.repository = repository
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    async def embed_document(self, text: str) -> list[float]:
        return (await self.embeddings.aembed_documents([text]))[0]

    async def find_similar(self, candidate: MemoryCandidate, scope: str = "shared", top_k: int = 10) -> list[MemoryRecord]:
        if top_k <= 0:
            return []
        
        stored = self.repository.list_active(scope)

        if not stored:
            return []

        query_vector = await self.embeddings.aembed_query(candidate.content)
        scored = []

        for record, vector in stored:
            score = cosine_similarity(query_vector, vector) if vector else -1.0
            # Force an exact key match to be considered, even with poor embeddings.
            exact_key = int(record.key == candidate.key)
            scored.append((exact_key, score, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        
        return [record for _, _, record in scored[:top_k]]
