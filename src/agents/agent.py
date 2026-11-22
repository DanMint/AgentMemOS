import asyncio
import os

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.agents.turn_workflow import build_turn_workflow
from src.main import DB_PATH, main as initialize_database
from src.memory.event_store import EventStore, PendingTurn
from src.memory.events import TurnInterpreter
from src.memory.models import Message
from src.memory.retriever import MemoryRetriever
from src.memory.summarizer import Summarizer, update_conversation_summary
from src.storage.memory_repository import MemoryRepository
from src.storage.repository import ConversationRepository


class Agent:
    def __init__(self, model: str, temperature: float, agent_prompt: str):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.agent = create_agent(
            model=self.llm, tools=[], system_prompt=agent_prompt,
        )

    async def communicate(self, prompt: str,
                          recent_messages: list[Message] | None = None,
                          summary: str = "", memories: list[str] | None = None) -> str:
        context = []
        if summary:
            context.append(f"Earlier conversation summary:\n{summary}")
        if memories:
            context.append("Saved long-term memories:\n" + "\n".join(memories))
        summary_context = [
            {"role": "user", "content": "\n\n".join(context)}
        ] if context else []
        response = await self.agent.ainvoke({
            "messages": summary_context + [
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
    summary_model = os.getenv("AGENTMEMOS_SUMMARY_MODEL", extraction_model)
    embedding_model = os.getenv("AGENTMEMOS_EMBEDDING_MODEL", "nomic-embed-text")

    conversation_repo = ConversationRepository(db_path=DB_PATH)
    memory_repo = MemoryRepository(db_path=DB_PATH)

    agent = Agent(
        chat_model, 0,
        "Answer the latest user message. Treat the user's latest statements about "
        "their own preferences and belongings as context. Accept their corrections "
        "and avoid guessing or disputing personal details based on model knowledge. "
        "Treat sale proceeds as money received. Keep deposits, sale proceeds, "
        "and spending distinct. Only calculate an available balance when the user "
        "has supplied the starting balance and the transactions affecting it. "
        "Otherwise state that the current balance is unknown if asked.",
    )
    summarizer = Summarizer(summary_model)
    retriever = MemoryRetriever(memory_repo, embedding_model)
    workflow = build_turn_workflow(
        TurnInterpreter(extraction_model), EventStore(memory_repo, retriever, scope), agent,
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
        pending_json = conversation_repo.get_pending_memory_turn(session_id, scope)
        pending = PendingTurn.model_validate_json(pending_json) if pending_json else None
        result = await workflow.ainvoke({
            "message": user_prompt,
            "turn_id": f"{session_id}:{user_id}",
            "recent": conversation_repo.get_recent_messages(session_id, before_id=user_id),
            "summary": conversation_repo.get_summary(session_id),
            "pending": pending,
        })
        next_pending = result["pending"]
        if next_pending != pending:
            conversation_repo.save_pending_memory_turn(
                session_id, scope, next_pending.model_dump_json() if next_pending else None,
            )
        agent_response = result["response"]
        print(f"Agent: {agent_response}")
        agent_id = conversation_repo.append_message(session_id, "agent", agent_response)

        if interpretation := result.get("interpretation"):
            print("\nInterpreted Turn:")
            print(interpretation.model_dump_json(indent=2, exclude_none=True))
        if error := result.get("error"):
            print(f"\n[Memory workflow] {error}")
        print("\nMemory Operations:")
        for operation in result.get("operations", []):
            print(operation.model_dump_json())
        if not result.get("operations"):
            print("(no changes)")

        try:
            await update_conversation_summary(
                conversation_repo, summarizer, session_id, agent_id,
            )
        except Exception as exc:
            print(f"\n[Conversation summary deferred] {exc}")

        print("\nCurrent Long-Term Memories:")
        for memory, _ in memory_repo.list_active(scope):
            print(f"[{memory.id} v{memory.version}] {memory.key}: {memory.content}")
        print()


if __name__ == "__main__":
    asyncio.run(main())