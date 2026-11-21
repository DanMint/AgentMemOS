import json

from langchain_ollama import ChatOllama

from src.memory.models import MemoryCandidate, MemoryPlan, MemoryRecord


DEFAULT_UPDATE_PROMPT = """
You maintain long-term memories for an AI agent. You receive:
- the user's LATEST raw message (the source of truth for new changes),
- ALL facts extracted from that single message,
- existing related memories and their IDs.

Make ONE coherent plan for the WHOLE interaction. Output a MemoryPlan with
"decisions", which is a list of ADD/UPDATE/DELETE/NOOP operations.

CRITICAL:
- Only the latest user message can authorize a NEW memory value.
- Existing memories supply facts that must be preserved unless corrected.
- All extracted facts about the SAME stored entity must be processed together.
- At most ONE UPDATE or DELETE for a given existing memory ID per interaction.
- NEVER invent or silently change years, names, makes, models, numbers,
  project names, or other existing attributes.
- Never rewrite the entire existing memory for an UPDATE.
If the user says they no longer own an entity, DELETE its current ownership
memory rather than replacing "owns" with a vague "no longer available".
If the user is merely PLANNING to buy something, do not treat it as owned.
For multiple related extracted facts, prefer one coherent plan and no
redundant changes. A repeated fact must be NOOP ONLY if an existing memory already states the
same fact. Similar topics are NOT enough: "dogs are my favorite animal" and
"I like turtles" are different facts; "my second favorite animal is a horse"
does not replace "dogs are my favorite". For those facts, use ADD.
For a new color or horsepower on an existing vehicle, APPEND to that vehicle.
Use REPLACE for an attribute already present. When the user states a different
current vehicle, UPDATE its identity. Reserve DELETE for an explicit removal
without a replacement.
For NOOP, you must be able to point to the specific existing memory that
already contains the candidate's meaning. A NOOP cannot omit new facts.

Operations:
ADD: new independent memory. Copy key and new_content from exactly one
     extracted candidate. Python will ultimately store the extractor's content.
     Do not invent or reword a candidate. No target_id.
UPDATE: a related existing memory has genuinely changed or gained detail.
        Set target_id and one or more SMALL edits. Do NOT set new_content.
        For a changed value use kind="REPLACE", old_text EXACTLY as it appears
        in the existing memory, and new_text grounded in the latest
        user message. Prefer verbatim words, but harmless paraphrasing (e.g.
        "169 horsepower" for "169 horse power") is acceptable.
        Change the smallest possible substring.
        For new detail use kind="APPEND", old_text="", and new_text as a
        concise factual phrase supported by the user's latest message.
        Preserve every other part of the stored memory.
DELETE: user explicitly retracts an entire existing memory without a replacement.
        Set target_id, no edits.
NOOP: the fact is already reflected in persistent memory or is irrelevant.

For UPDATE and DELETE, only choose target IDs that appear in existing_memories.
For ADD, use a real extracted candidate key and content. Avoid duplicate adds.
If an attribute is ALREADY represented within an existing memory, UPDATE that
memory to change the attribute. APPEND new details about that same vehicle.
Use ADD when there is no related vehicle record.

EXAMPLE 1:
latest_user_message: "actually its a red toyota corolla"
extracted_candidates: [{"key":"car.color","content":"red"},
                       {"key":"car.model","content":"Toyota Corolla"}]
existing_memories: [{"id":1,"key":"user.vehicle",
  "content":"black Toyota Corolla 2026","version":1,"scope":"shared"}]
output: {"decisions":[{"action":"UPDATE","target_id":1,
  "edits":[{"kind":"REPLACE","old_text":"black","new_text":"red"}]}]}
This keeps the Toyota Corolla and the year 2026 unchanged.

EXAMPLE 2:
latest_user_message: "i actually have a red toyota corolla not a black one"
existing_memories: [{"id":1,"key":"user.vehicle",
  "content":"red Toyota Corolla 2026","version":2,"scope":"shared"}]
output: {"decisions":[{"action":"NOOP","target_id":1}]}
DO NOT increment a version merely because the user repeated information.

EXAMPLE 3:
latest_user_message: "My car is red"
existing_memories: [{"id":1,"key":"user.vehicle",
  "content":"black Toyota Corolla 2026","version":1,"scope":"shared"}]
output: {"decisions":[{"action":"UPDATE","target_id":1,
 "edits":[{"kind":"REPLACE","old_text":"black","new_text":"red"}]}]}

Always return a structured MemoryPlan. No prose.
"""


class MemoryUpdater:
    def __init__(self, model: str, temperature: float = 0,
                 system_prompt: str = DEFAULT_UPDATE_PROMPT):
        self.system_prompt = system_prompt
        llm = ChatOllama(model=model, temperature=temperature)
        self.llm = llm.with_structured_output(MemoryPlan, method="json_schema")

    async def plan(
        self,
        candidates: list[MemoryCandidate],
        existing_memories: list[MemoryRecord],
        source_message: str,
        feedback: str | None = None,
    ) -> MemoryPlan:
        payload = {
            "latest_user_message": source_message,
            "extracted_candidates": [c.model_dump() for c in candidates],
            "existing_memories": [record.model_dump() for record in existing_memories],
        }
        if feedback:
            payload["previous_attempt_validation_error"] = feedback
            payload["instruction"] = "Correct the previous plan; do not repeat the error."

        result = await self.llm.ainvoke([
            ("system", self.system_prompt),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ])
        if result is None:
            raise RuntimeError("Updater returned no structured plan")
        return result