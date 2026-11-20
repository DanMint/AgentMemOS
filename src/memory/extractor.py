from langchain_ollama import ChatOllama, OllamaEmbeddings
import asyncio

from .models import ExtractionContext, ExtractedMemories

class Extractor():
    def __init__(self, model: str, temperature: int, system_prompt: str):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

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

        user_prompt = f"""
            Conversation Summary:
            {context.summary}

            Recent Messages:
            {history}

            Latest User Message:
            {context.user_message.content}

            Latest Assistant Response:
            {context.assistant_message.content}
        """

        result = await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return result