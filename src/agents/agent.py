
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import asyncio

from src.main import main as initialize_database, DB_PATH
from src.storage.repository import ConversationRepository
from src.memory.extractor import Extractor
from src.memory.manager import ExtractionManager


class Agent:
    def __init__(self, model: str, temperature: float, agent_prompt: str) -> None:
        self.model = model
        self.temperature = temperature
        self.agent_prompt = agent_prompt

        self.llm = ChatOllama(
            model=self.model,
            temperature=self.temperature
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=self.agent_prompt
        )

    async def communicate(self, prompt: str) -> str:
        response = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        # Extract the final agent response
        agent_response = response["messages"][-1].content

        if not isinstance(agent_response, str):
            raise TypeError("Expected a string response from agent")

        return agent_response

async def main():
    # Initialize database and tables
    initialize_database()

    session_id = "1"

    # Initialize repository
    repository = ConversationRepository(
        db_path=DB_PATH
    )

    # Initialize agent
    a1 = Agent(
        model="llama3.1:8b",
        temperature=0,
        agent_prompt="Answer what the user asks"
    )

    # Initialize extractor
    extractor = Extractor(model="qwen3.5:9b", temperature=0)

    # Initialize extraction manager
    manager = ExtractionManager(repository=repository, extractor=extractor)

    while True:

        try:
            user_prompt = input("Ask your question: ").strip()

        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_prompt.lower() == "exit":
            break

        if not user_prompt:
            continue

        # 1. Agent generates response
        agent_response = await a1.communicate(prompt=user_prompt)

        print(f"Agent: {agent_response}")

        # 2. Save user message (m_t-1)
        user_message_id = repository.append_message(session_id=session_id, role="user", content=user_prompt)

        # 3. Save agent response (m_t)
        agent_message_id = repository.append_message(session_id=session_id, role="agent", content=agent_response)

        # 4. Execute memory extraction
        extracted_memories = await manager.process_interaction(
            session_id=session_id,
            user_message_id=user_message_id,
            agent_message_id=agent_message_id
        )

        # 5. Display extracted memories
        print("\nExtracted Memories:")
        print(extracted_memories.model_dump_json(indent=2))
        print()

if __name__ == "__main__":
    asyncio.run(main())
