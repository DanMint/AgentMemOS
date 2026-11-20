from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .models import Message

class Summarizer:
    def __init__(self, model: str, temperature: int, system_prompt: str):
       self.system_prompt = system_prompt

       self.llm = ChatOllama(
           model=model,
           temperature=temperature
       )     

    async def summarize(self, previous_summary: str, new_messages: list[Message]) -> str:
        history = "\n".join(f"{msg.role}: {msg.content}" 
                            for msg in new_messages)

        user_prompt = f"""
            Previous Conversation Summary:
            {previous_summary}

            New Messages:
            {history}
        """

        response = await self.llm.ainvoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response

           
