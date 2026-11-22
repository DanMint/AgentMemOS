import asyncio

from src.memory.models import ExtractionContext, Message
from src.memory.extractor import Extractor

import time
import json

"""
    llm benchmarking to see the best performing llm for extarction. After the test its: granite4.1:8b
"""

EXTRACTION_PROMPT = """
    You are AgentMemOS's long-term memory extractor, not a conversation summarizer.
    Extract only durable information newly stated or explicitly revised by the user
    in the LATEST USER MESSAGE.

    The conversation summary, previous messages, and latest agent response are
    CONTEXT ONLY: use them to resolve "it", "this project", "that deadline", etc.
    Never extract earlier facts again just because they appear in context.
    Never extract unsupported information from the agent's response.

    Remember: explicit user facts, preferences, ongoing projects, goals, decisions,
    constraints, deadlines, and corrections. Each memory should be one standalone,
    accurate fact with a stable descriptive key.

    Never remember greetings, jokes, general questions, assistant explanations,
    small talk, fictional statements, or guessed properties of the user or project.
    Don't invent details; if nothing qualifies, return an empty memories list.
    Do NOT decide ADD/UPDATE/DELETE/NOOP or infer whether a fact already exists in
    persistent memory. The separate updater will do that.

    Examples:
    Latest user: "Hello, how are you?" -> {"memories": []}
    Latest user: "Tell me a joke" -> {"memories": []}
    Latest user: "I'm building AgentMemOS using Python" ->
    {"memories": [{"key": "project.agentmemos.technology", "content": "AgentMemOS uses Python."}]}
    Latest user: "The deadline moved to December 10" (context: AgentMemOS) ->
    {"memories": [{"key": "project.agentmemos.deadline", "content": "AgentMemOS is due December 10."}]}
"""

async def main():
    models = [
        # Qwen family - model scaling
        "qwen3.5:2b",
        "qwen3.5:4b",
        "qwen3.5:9b",

        # Microsoft
        "phi4-mini:3.8b",

        # IBM
        "granite4.1:3b",
        "granite4.1:8b",

        # Mistral
        "ministral-3:3b",
        "ministral-3:8b",

        # Allen AI
        "olmo-3:7b-instruct",

        # Google
        "gemma4:12b",

        # Meta
        "llama3.1:8b",

        # Previous generation
        "qwen3:8b"
    ]

    # Each scenario uses the same ExtractionContext schema.
    scenarios = {
        "1. Greeting - No Memory": {
            "context": ExtractionContext(
                summary="",
                recent_messages=[],
                user_message=Message(
                    role="user",
                    content="Hello, how are you?"
                ),
                agent_message=Message(
                    role="agent",
                    content="I'm doing well! How can I help?"
                )
            ),
            "expected": "No memories"
        },

        "2. Positive and Negative Preferences": {
            "context": ExtractionContext(
                summary="",
                recent_messages=[],
                user_message=Message(
                    role="user",
                    content="I love dogs, but I hate hippos."
                ),
                agent_message=Message(
                    role="agent",
                    content="Dogs are wonderful animals!"
                )
            ),
            "expected": "User likes dogs; user dislikes hippos"
        },

        "3. Multiple Durable Facts": {
            "context": ExtractionContext(
                summary="",
                recent_messages=[],
                user_message=Message(
                    role="user",
                    content=(
                        "I'm building AgentMemOS using Python and SQLite. "
                        "My goal is to publish a research paper about "
                        "long-term memory in AI agents."
                    )
                ),
                agent_message=Message(
                    role="agent",
                    content="That sounds like an interesting project."
                )
            ),
            "expected": (
                "AgentMemOS uses Python and SQLite; "
                "user aims to publish a paper about agent memory"
            )
        },

        "4. Correction of Existing Preference": {
            "context": ExtractionContext(
                summary="The user previously said they liked cats.",
                recent_messages=[
                    Message(
                        role="user",
                        content="I really like cats."
                    ),
                    Message(
                        role="agent",
                        content="Cats are great companions."
                    )
                ],
                user_message=Message(
                    role="user",
                    content="Actually, I don't like cats anymore."
                ),
                agent_message=Message(
                    role="agent",
                    content="Understood, your preference has changed."
                )
            ),
            "expected": (
                "User no longer likes cats; "
                "do not re-extract the old positive preference"
            )
        },

        "5. Context-Dependent Fact": {
            "context": ExtractionContext(
                summary=(
                    "The user is developing AgentMemOS. "
                    "The project deadline was December 1."
                ),
                recent_messages=[
                    Message(
                        role="user",
                        content="AgentMemOS is due on December 1."
                    ),
                    Message(
                        role="agent",
                        content="I understand the deadline."
                    )
                ],
                user_message=Message(
                    role="user",
                    content="The deadline for that project moved to December 10."
                ),
                agent_message=Message(
                    role="agent",
                    content="Understood, the deadline changed."
                )
            ),
            "expected": "AgentMemOS deadline is December 10"
        },

        "6. Assistant Hallucination": {
            "context": ExtractionContext(
                summary="",
                recent_messages=[],
                user_message=Message(
                    role="user",
                    content="Can you explain how embeddings work?"
                ),
                agent_message=Message(
                    role="agent",
                    content=(
                        "Since your favorite animal is cats, "
                        "I'll explain embeddings using cats."
                    )
                )
            ),
            "expected": "No memories"
        },

        "7. Old Facts Should Not Be Re-extracted": {
            "context": ExtractionContext(
                summary=(
                    "The user is building AgentMemOS in Python "
                    "and likes dogs."
                ),
                recent_messages=[
                    Message(
                        role="user",
                        content="I like dogs."
                    ),
                    Message(
                        role="agent",
                        content="Dogs are great."
                    )
                ],
                user_message=Message(
                    role="user",
                    content="Can you explain SQLite indexes?"
                ),
                agent_message=Message(
                    role="agent",
                    content="SQLite indexes accelerate database queries."
                )
            ),
            "expected": "No memories"
        }
    }

    results = {}

    # Execute one model at a time to avoid keeping
    # multiple models loaded in GPU memory.
    for model_name in models:
        print(f"\n{'=' * 65}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'=' * 65}")

        extractor = Extractor(
            model=model_name,
            temperature=0,
            system_prompt=EXTRACTION_PROMPT
        )

        results[model_name] = {}

        for scenario_name, scenario in scenarios.items():
            print(f"\nRunning: {scenario_name}")

            start = time.perf_counter()

            try:
                extracted = await extractor.extract(
                    scenario["context"]
                )

                duration = time.perf_counter() - start

                results[model_name][scenario_name] = {
                    "memories": extracted.model_dump()["memories"],
                    "time": round(duration, 2)
                }

                print("Expected:", scenario["expected"])
                print(
                    "Extracted:",
                    json.dumps(
                        extracted.model_dump(),
                        indent=2
                    )
                )
                print(f"Time: {duration:.2f}s")

            except Exception as exc:
                print(f"Error: {exc}")
                results[model_name][scenario_name] = {
                    "error": str(exc)
                }

    # Final comparison
    print(f"\n{'=' * 65}")
    print("MODEL COMPARISON")
    print(f"{'=' * 65}")

    for scenario_name, scenario in scenarios.items():
        print(f"\n{scenario_name}")
        print(f"Expected: {scenario['expected']}")

        for model_name in models:
            result = results[model_name][scenario_name]

            if "error" in result:
                print(f"  {model_name}: ERROR")
                continue

            print(
                f"  {model_name}: "
                f"{len(result['memories'])} memories, "
                f"{result['time']:.2f}s"
            )


if __name__ == "__main__":
    asyncio.run(main())
