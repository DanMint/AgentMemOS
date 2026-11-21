import asyncio
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.main import DB_PATH, main as initialize_database
from src.memory.consistency import MemoryConsistency, MemoryValidationError
from src.memory.extractor import Extractor
from src.memory.manager import ExtractionManager, MemoryUpdateManager
from src.memory.models import Message
from src.memory.retriever import MemoryRetriever
from src.memory.updater import MemoryUpdater
from src.storage.memory_repository import MemoryRepository, MemoryConflictError
from src.storage.repository import ConversationRepository


EXTRACTION_PROMPT = """
You are AgentMemOS's long-term memory extractor, not a conversation summarizer.
Extract only durable information newly stated or explicitly revised by the user
in the LATEST USER MESSAGE.

The conversation summary, previous messages, and latest agent response are
CONTEXT ONLY: use them to resolve "it", "this project", "that deadline", etc.
Never extract earlier facts again just because they appear in context.
Never extract unsupported information from the agent's response.

Remember: explicit user facts, preferences (including casual but lasting likes
such as "I love turtles" or "I like turtles"), ongoing projects, goals,
decisions, constraints, deadlines, and corrections. Each memory should be one
standalone, accurate fact with a stable descriptive key.
Different preferences are independent: liking turtles does NOT conflict with
dogs being the user's favorite animal. A second-favorite animal is a distinct
fact from a first-favorite animal. Use second only if the latest user message
explicitly says second. An unranked favorite updates the first favorite.
Use "user.vehicle" for a newly stated vehicle identity. Use "user.vehicle.year",
"user.vehicle.color", and "user.vehicle.horsepower" for individual corrections
or details. For "My car is black in color and has 169 horse power", extract
both the color and horsepower with those keys. For "Actually my car is a 2025
not a 2026", extract the year as "The user's vehicle is a 2025."
Extract a newly stated vehicle description as one identity, including any trim
and color stated together. Preserve the user's wording for makes and models.
"My new car is a 2027 Camry Hybrid SE" does not supply a make. Do not infer
one from your knowledge, earlier vehicles, or the agent response.

Never remember greetings, jokes, general questions, assistant explanations,
small talk, fictional statements, or guessed properties of the user or project.
Don't invent details; if nothing qualifies, return an empty memories list.
A plan to buy or sell something is NOT proof that the transaction already
happened. "I'm buying a 2027 Camry" means PLANS TO BUY, not OWNS.
If the user says "I have no car", extract a current-car retraction rather
than a vague status such as "no longer available".
Do NOT decide ADD/UPDATE/DELETE/NOOP or infer whether a fact already exists in
persistent memory. The separate updater will do that.

Examples:
Latest user: "Hello, how are you?" -> {"memories": []}
Latest user: "Tell me a joke" -> {"memories": []}
Latest user: "I love turtles" ->
{"memories": [{"key": "user.likes.turtles", "content": "The user likes turtles."}]}
Latest user: "Dogs are my favorite animal" ->
{"memories": [{"key": "user.favorite_animal.first", "content": "Dogs are the user's favorite animal."}]}
Latest user: "My second favorite animal is a horse" ->
{"memories": [{"key": "user.favorite_animal.second", "content": "The user's second favorite animal is a horse."}]}
Latest user: "I'm building AgentMemOS using Python" ->
{"memories": [{"key": "project.agentmemos.technology", "content": "AgentMemOS uses Python."}]}
Latest user: "The deadline moved to December 10" (context: AgentMemOS) ->
{"memories": [{"key": "project.agentmemos.deadline", "content": "AgentMemOS is due December 10."}]}
"""


class Agent:
    def __init__(self, model: str, temperature: float, agent_prompt: str):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.agent = create_agent(
            model=self.llm, tools=[], system_prompt=agent_prompt,
        )

    async def communicate(self, prompt: str,
                          recent_messages: list[Message] | None = None) -> str:
        response = await self.agent.ainvoke({
            "messages": [
                {"role": "assistant" if message.role == "agent" else "user",
                 "content": message.content}
                for message in (recent_messages or [])
            ] + [{"role": "user", "content": prompt}]
        })
        answer = response["messages"][-1].content
        if not isinstance(answer, str):
            raise TypeError("Expected text response from agent")
        return answer


async def main():
    initialize_database()

    session_id = os.getenv("AGENTMEMOS_SESSION_ID", "1")
    scope = os.getenv("AGENTMEMOS_SCOPE", "shared")
    chat_model = os.getenv("AGENTMEMOS_CHAT_MODEL", "llama3.1:8b")
    extraction_model = os.getenv("AGENTMEMOS_EXTRACTION_MODEL", "llama3.1:8b")
    update_model = os.getenv("AGENTMEMOS_UPDATE_MODEL", extraction_model)
    embedding_model = os.getenv("AGENTMEMOS_EMBEDDING_MODEL", "nomic-embed-text")

    conversation_repo = ConversationRepository(db_path=DB_PATH)
    memory_repo = MemoryRepository(db_path=DB_PATH)

    agent = Agent(
        chat_model, 0,
        "Answer the latest user message. Treat the user's latest statements about "
        "their own preferences and belongings as context. Accept their corrections "
        "and avoid guessing or disputing personal details based on model knowledge.",
    )
    extractor = Extractor(extraction_model, 0, EXTRACTION_PROMPT)
    extraction_manager = ExtractionManager(conversation_repo, extractor)

    retriever = MemoryRetriever(memory_repo, embedding_model)
    updater = MemoryUpdater(update_model, temperature=0)
    consistency = MemoryConsistency(memory_repo)
    update_manager = MemoryUpdateManager(
        retriever, updater, consistency,
        debug=os.getenv("AGENTMEMOS_DEBUG", "0") == "1",
    )

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

        user_id = conversation_repo.append_message(session_id, "user", user_prompt)
        recent = conversation_repo.get_recent_messages(session_id, before_id=user_id)
        agent_response = await agent.communicate(user_prompt, recent)
        print(f"Agent: {agent_response}")

        agent_id = conversation_repo.append_message(session_id, "agent", agent_response)

        try:
            extracted = await extraction_manager.process_interaction(
                session_id, user_id, agent_id,
            )
            print("\nExtracted Memories:")
            print(extracted.model_dump_json(indent=2))

            operations = await update_manager.process(
                extracted, scope=scope, source_message=user_prompt,
            )
            print("\nMemory Operations:")
            for operation in operations:
                print(operation.model_dump_json())
            if not operations:
                print("(no changes)")
        except (MemoryValidationError, MemoryConflictError, ValueError, RuntimeError) as exc:
            # The interaction was logged, but an unsafe memory change is NOT written.
            print(f"\n[Memory operation rejected] {exc}")
            print("Your conversation is saved. No unsafe memory update was committed.")

        print("\nCurrent Long-Term Memories:")
        for memory, _ in memory_repo.list_active(scope):
            print(f"[{memory.id} v{memory.version}] {memory.key}: {memory.content}")
        print()


if __name__ == "__main__":
    asyncio.run(main())