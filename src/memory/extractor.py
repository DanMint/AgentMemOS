"""Long-term memory extraction for AgentMemOS."""

import re

from langchain_ollama import ChatOllama

from .models import ExtractionContext, ExtractedMemories, MemoryCandidate


def _simple_preferences(message: str) -> ExtractedMemories | None:
    """
    Extract unambiguous first-person likes and dislikes.

    Return None if the message requires contextual interpretation,
    so that the normal LLM extractor handles it instead.
    """

    cleaned = message.strip().rstrip(".! ")

    match = re.fullmatch(
        r"i\s+(?:(?:really|absolutely)\s+)?"
        r"(?P<verb>don['’]?t\s+like|do\s+not\s+like|hate|dislike|love|like)"
        r"\s+(?P<objects>[a-zA-Z][a-zA-Z ,\-]*)",
        cleaned,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    negative = match.group("verb").casefold() in {
        "hate",
        "dislike",
        "don't like",
        "don’t like",
        "dont like",
        "do not like",
    }

    objects = re.split(
        r"\s*,\s*|\s+and\s+",
        match.group("objects"),
        flags=re.IGNORECASE,
    )

    if not 1 <= len(objects) <= 5:
        return None

    memories = []

    for obj in objects:
        subject = obj.strip().casefold()

        # Do not treat multi-clause statements as simple preferences.
        # The LLM must interpret those using conversation context.
        if (
            not subject
            or len(subject.split()) > 3
            or re.search(
                r"\b(?:i|we|you|they|he|she|but|because|except|"
                r"not|never|like|love|hate|dislike)\b",
                subject,
            )
        ):
            return None

        # Build a stable key from the subject.
        words = subject.split()
        noun = words[-1]

        if noun.endswith("ies") and len(noun) > 4:
            noun = noun[:-3] + "y"
        elif noun.endswith("s") and not noun.endswith(
            ("ss", "us", "is")
        ):
            noun = noun[:-1]

        words[-1] = noun
        key_subject = ".".join(words)

        if negative:
            key = f"user.dislikes.{key_subject}"
            content = f"The user dislikes {subject}."
        else:
            key = f"user.likes.{key_subject}"
            content = f"The user likes {subject}."

        memories.append(
            MemoryCandidate(
                key=key,
                content=content,
            )
        )

    return ExtractedMemories(memories=memories)


class Extractor:
    def __init__(
        self,
        model: str,
        temperature: float,
        system_prompt: str,
    ):
        self.system_prompt = system_prompt

        llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

        self.llm = llm.with_structured_output(
            ExtractedMemories,
            method="json_schema",
        )

    async def extract(
        self,
        context: ExtractionContext,
    ) -> ExtractedMemories:

        # Prefer direct extraction for simple user preferences.
        # This avoids re-extracting unrelated earlier conversation facts.
        direct = _simple_preferences(
            context.user_message.content
        )

        if direct is not None:
            return direct

        # All other messages use the existing LLM extraction pipeline.
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

"""

        return await self.llm.ainvoke([
            ("system", self.system_prompt),
            ("human", user_prompt),
        ])