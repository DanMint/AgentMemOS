from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
import asyncio

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

async def main():
    a1 = Agent(model="llama3.1:8b", temperature=0, agent_prompt="Answer what the user asks")

    while True:
        user_prompt = input("Ask your questino: ")
        response = await a1.communicate(prompt=user_prompt)
        print(response.get('messages')[1].content)

        if user_prompt == "exit":
            break

if __name__ == "__main__":
    asyncio.run(main())