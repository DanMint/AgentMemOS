import asyncio
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.main import DB_PATH, main as initialize_database
from src.memory.consistency import MemoryConsistency
from src.memory.extractor import Extractor
from src.memory.manager import ExtractionManager, MemoryUpdateManager
from src.memory.retriever import MemoryRetriever
from src.memory.updater import MemoryUpdater
from src.storage.memory_repository import MemoryRepository
from src.storage.repository import ConversationRepository

EXTRACTION_PROMPT = """
    You are AgentMemOS's long-term memory extractor, not a conversation summarizer.
    Extract only durable information newly stated or explicitly revised by the user
    in the LATEST USER MESSAGE.

    The conversation summary, previous messages, and latest agent response are
    CONTEXT ONLY: use them to resolve "it", "this project", "that deadline", etc.
    Never extract earlier facts again just because they appear in context.
    Never extract unsupported information from the agent's response.

    Remember: explicit user facts, preferences, ongoing projects, goals, decisions,
    constraints, deadlines, and corrections. Each memory should be one standalone,
    accurate fact with a stable descriptive key.

    Never remember greetings, jokes, general questions, assistant explanations,
    small talk, fictional statements, or guessed properties of the user or project.
    Don't invent details; if nothing qualifies, return an empty memories list.
    Do NOT decide ADD/UPDATE/DELETE/NOOP or infer whether a fact already exists in
    persistent memory. The separate updater will do that.

    Examples:
    Latest user: "Hello, how are you?" -> {"memories": []}
    Latest user: "Tell me a joke" -> {"memories": []}
    Latest user: "I'm building AgentMemOS using Python" ->
    {"memories": [{"key": "project.agentmemos.technology", "content": "AgentMemOS uses Python."}]}
    Latest user: "The deadline moved to December 10" (context: AgentMemOS) ->
    {"memories": [{"key": "project.agentmemos.deadline", "content": "AgentMemOS is due December 10."}]}
"""


class Agent:
    def __init__(self, model: str, temperature: float, agent_prompt: str):
        self.llm = ChatOllama(
            model=model, 
            temperature=temperature
        )

        self.agent = create_agent(
            model=self.llm, 
            tools=[], 
            system_prompt=agent_prompt
        )

    async def communicate(self, prompt: str) -> str:
        response = await self.agent.ainvoke(
            {
                "messages": 
                    [
                        {"role": "user", "content": prompt}
                    ]
            }
        )

        answer = response["messages"][-1].content

        if not isinstance(answer, str):
            raise TypeError("Expected text response from agent")
        return answer


async def main():
    initialize_database()

    session_id = os.getenv("AGENTMEMOS_SESSION_ID", "1")
    scope = os.getenv("AGENTMEMOS_SCOPE", "shared")
    chat_model = os.getenv("AGENTMEMOS_CHAT_MODEL", "llama3.1:8b")
    extraction_model = os.getenv("AGENTMEMOS_EXTRACTION_MODEL", "qwen3.5:9b")
    update_model = os.getenv("AGENTMEMOS_UPDATE_MODEL", extraction_model)
    embedding_model = os.getenv("AGENTMEMOS_EMBEDDING_MODEL", "nomic-embed-text")

    conversation_repo = ConversationRepository(db_path=DB_PATH)
    memory_repo = MemoryRepository(db_path=DB_PATH)

    agent = Agent(chat_model, 0, "Answer what the user asks.")
    extractor = Extractor(extraction_model, 0, EXTRACTION_PROMPT)
    extraction_manager = ExtractionManager(conversation_repo, extractor)

    retriever = MemoryRetriever(memory_repo, embedding_model)
    updater = MemoryUpdater(update_model, temperature=0)
    consistency = MemoryConsistency(memory_repo)
    update_manager = MemoryUpdateManager(retriever, updater, consistency)

    while True:
        try:
            user_prompt = input("Ask your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_prompt.lower() in ("exit", "quit"):
            break

        if not user_prompt:
            continue

        agent_response = await agent.communicate(user_prompt)
        print(f"Agent: {agent_response}")

        user_id = conversation_repo.append_message(session_id, "user", user_prompt)
        agent_id = conversation_repo.append_message(session_id, "agent", agent_response)

        extracted = await extraction_manager.process_interaction(
            session_id, user_id, agent_id,
        )
        print("\nExtracted Memories:")
        print(extracted.model_dump_json(indent=2))

        operations = await update_manager.process(extracted, scope=scope)

        print("\nMemory Operations:")

        for operation in operations:
            print(operation.model_dump_json())

        if not operations:
            print("(no changes)")

        print("\nCurrent Long-Term Memories:")

        for memory, _ in memory_repo.list_active(scope):
            print(f"[{memory.id} v{memory.version}] {memory.key}: {memory.content}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
