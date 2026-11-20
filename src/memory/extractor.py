from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

from .models import ExtractionContext, ExtractedMemories

class Extractor():
    def __init__(self, model: str, temperature: int):
        self.model = model
        self.temperature = temperature

        llm = ChatOllama(
            model = self.model,
            temperature = self.temperature
        )

        self.llm = llm.with_structured_output(
            ExtractedMemories,
            method="json_schema"
        )

    async def extract(self, context: ExtractionContext) -> ExtractedMemories:
        history = "\n".join(f"{msg.role}: {msg.content}"
                            for msg in context.recent_messages)

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

        user_prompt = f"""
            Conversation Summary:
            {context.summary}

            Recent Messages:
            {history}

            Latest User Message:
            {context.user_message.content}

            Latest Assistant Response:
            {context.agent_message.content}
        """

        result = await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return result