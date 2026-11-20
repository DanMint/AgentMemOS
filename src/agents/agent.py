from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
import asyncio
import sqlite3
from datetime import datetime, timezone

class Agent():
    def __init__(self, model: str, temperature: int, agent_prompt: str) -> None:
        self.model = model
        self.temperature = temperature
        self.agent_prompt = agent_prompt

        self.llm = ChatOllama(
            model = self.model,
            temperature = self.temperature
        )

        self.agent = create_agent(
            model = self.llm,
            tools = [],
            system_prompt = self.agent_prompt
        )

    async def communicate(self, prompt:str) -> str:
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

        return response

def append_to_db(message_id, session_id, role, content, timestamp) -> None:
    conn = sqlite3.connect("db/messages.db")

    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        (message_id, "1", "user", content, datetime.now(timezone.utc).isoformat())
    )

    conn.commit()

async def main():
    a1 = Agent(model="llama3.1:8b", temperature=0, agent_prompt="Answer what the user asks")

    meessage_id = 1
    while True:
        user_prompt = input("Ask your questino: ")

        # record user input (mt-1)
        append_to_db(meessage_id, "1", "user", user_prompt, datetime.now(timezone.utc).isoformat())
        meessage_id += 1

        response = await a1.communicate(prompt=user_prompt)
        # record agents response (mt)
        append_to_db(meessage_id, "1", "agent", response.get('messages')[1].content, datetime.now(timezone.utc).isoformat())
        meessage_id += 1
        
        print(response.get('messages')[1].content)
        
        if user_prompt == "exit":
            break

if __name__ == "__main__":
    asyncio.run(main())