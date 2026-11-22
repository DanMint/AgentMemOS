from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

from .models import ExtractionContext, ExtractedMemories, Message

class Extractor:
    def __init__(self, model: str, temperature: float, system_prompt: str):
        self.system_prompt = system_prompt

        llm = ChatOllama(
            model=model, 
            temperature=temperature
        )
        
        self.llm = llm.with_structured_output(
            ExtractedMemories, 
            method="json_schema"
        )

    async def extract(self, context: ExtractionContext) -> ExtractedMemories:
        history = "\n".join(
                f"{msg.role}: {msg.content}" 
                for msg in context.recent_messages
            )
        
        user_prompt = f"""
            Conversation Summary:
            {context.summary}

            Recent Messages:
            {history}

            Latest User Message:
            {context.user_message.content}

            Latest Agent Response:
            {context.agent_message.content}
        """

        return await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ])
