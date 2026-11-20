
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
    extractor = Extractor(
        model="llama3.1:8b",
        temperature=0,
        system_prompt = """
            You are the long-term memory extraction component of AgentMemOS.

            Your responsibility is to identify information that would be
            useful to remember in FUTURE conversations with the user.

            You are NOT a conversation summarizer.

            INPUT:
            You receive:
            1. A summary of previous conversations.
            2. Recent conversation messages.
            3. The latest user message.
            4. The latest agent response.

            CRITICAL RULE:
            Only extract new facts that are explicitly stated or updated
            by the user in the LATEST USER MESSAGE.

            The summary, recent messages, and agent response are provided
            ONLY to help you understand context and resolve references.

            NEVER extract memories solely from those sources.

            WHAT QUALIFIES AS LONG-TERM MEMORY:
            - Explicitly stated user facts
            - User preferences
            - Ongoing projects and their details
            - Goals and long-term plans
            - Important deadlines
            - Decisions made by the user
            - Changes to previously stated information
            - Constraints relevant to future interactions

            WHAT MUST NOT BE EXTRACTED:
            - Greetings and small talk
            - Jokes and entertainment requests
            - General knowledge questions
            - Assistant-generated explanations or advice
            - Information already mentioned only in previous messages
            - Speculative conclusions
            - Facts about the assistant
            - Temporary conversational exchanges
            - Missing or unspecified information

            ADDITIONAL REQUIREMENTS:
            1. Every extracted memory must contain useful information
            about the user, their projects, or their stated context.

            2. Do not infer information that the user did not provide.

            3. Use previous messages only to resolve references such
            as "this project", "that deadline", or "my earlier plan".

            4. Extract ONLY information newly introduced or explicitly
            revised in the latest user message.

            5. Each memory must be independently understandable.

            6. If the latest user message contains no salient facts,
            return {"memories": []}.

            EXAMPLES:

            User: "Hello, how are you?"
            Output: {"memories": []}

            User: "Tell me a joke about computers."
            Output: {"memories": []}

            User: "What is an operating system?"
            Output: {"memories": []}

            User: "I'm working on a project called AgentMemOS."
            Output:
            {
            "memories": [
                {
                "key": "project.agentmemos.name",
                "content": "The user is developing AgentMemOS."
                }
            ]
            }

            User: "I'm using Python and SQLite for AgentMemOS."
            Output:
            {
            "memories": [
                {
                "key": "project.agentmemos.technologies",
                "content": "AgentMemOS uses Python and SQLite."
                }
            ]
            }

            User: "My AgentMemOS deadline is December 5, 2026."
            Output:
            {
            "memories": [
                {
                "key": "project.agentmemos.deadline",
                "content": "AgentMemOS is due December 5, 2026."
                }
            ]
            }

            Remember:
            The objective is to retain useful long-term information,
            NOT to document everything that happened in a conversation.

            Prefer returning no memories over creating unsupported
            or irrelevant memories.
        """

    )

    # Initialize extraction manager
    manager = ExtractionManager(
        repository=repository,
        extractor=extractor
    )

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
        user_message_id = repository.append_message(
            session_id=session_id,
            role="user",
            content=user_prompt
        )

        # 3. Save agent response (m_t)
        agent_message_id = repository.append_message(
            session_id=session_id,
            role="agent",
            content=agent_response
        )

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
