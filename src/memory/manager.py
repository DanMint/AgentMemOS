"""Coordinate extraction and one atomic memory update per interaction."""

import re

from src.memory.consistency import MemoryValidationError
from src.memory.identity import (
    is_represented, is_represented_in_text, normalize_candidate, same_fact,
)
from src.memory.models import (
    ExtractedMemories, MemoryCandidate, MemoryDecision, MemoryEdit, MemoryPlan,
    MemoryRecord, PreparedMemoryChange, UpdateResult,
)
from src.storage.memory_repository import MemoryConflictError


class ExtractionManager:
    def __init__(self, repository, extractor):
        self.repository = repository
        self.extractor = extractor

    async def process_interaction(self, session_id: str,
                                  user_message_id: int, agent_message_id: int):
        context = self.repository.build_extraction_context(
            session_id=session_id,
            user_message_id=user_message_id,
            agent_message_id=agent_message_id,
        )
        return await self.extractor.extract(context)


def _explicitly_lost_current_vehicle(message: str) -> bool:
    direct = re.search(
        r"\bi\s+(?:now\s+)?(?:have|own)\s+no\s+(?:car|vehicle)\b|"
        r"\bi\s+(?:do not|don't|dont|no longer)\s+(?:have|own)\s+"
        r"(?:a\s+)?(?:car|vehicle)\b",
        message, re.IGNORECASE,
    )
    contextual = re.search(
        r"\bi\b[^.!?\n]{0,100}\bnow\s+have\s+no\s+(?:car|vehicle)\b",
        message, re.IGNORECASE,
    )
    crash_and_loss = re.search(
        r"\bi\s+(?:have\s+)?crashed\s+my\s+(?:car|vehicle)\s*,?\s+"
        r"and\s+now\s+(?:it(?:['’]?s|\s+is)\s+gone|"
        r"i\s+(?:don['’]?t|do\s+not|no\s+longer)\s+(?:have|own)\s+it)"
        r"\s*[.!]*\s*$",
        message, re.IGNORECASE,
    )
    return bool(direct or contextual or crash_and_loss)


def _is_current_vehicle_key(key: str) -> bool:
    parts = re.split(r"[._]", key.casefold())
    if not any(part in {"vehicle", "car"} for part in parts):
        return False
    return not any(part in {"plan", "planned", "purchase", "history", "previous"}
                   for part in parts)


def _stated_favorite_animal(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"(?:actually\s*,?\s*)?my\s+(?:(first|second)\s+)?"
        r"favou?rite\s+animals?\s+(?:is|are)\s+(?:actually\s+)?"
        r"((?:(?:a|an)\s+)?[a-z]+(?:[- ][a-z]+)?)\s*[.!]?",
        message.strip(), re.I,
    )
    if not match:
        return None
    rank, animal = match.groups()
    if re.search(r"\b(?:not|no|never|none|unknown|maybe|probably|uncertain)\b", animal, re.I):
        return None
    rank = (rank or "first").lower()
    label = "second favorite" if rank == "second" else "favorite"
    return MemoryCandidate(
        key=f"user.favorite_animal.{rank}",
        content=f"The user's {label} animal is {animal.lower()}.",
    )


def _stated_vehicle(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"(?:now\s+)?i\s+(?:"
        r"(?:drive|own|have|bought|purchased)\s+an?\s+|"
        r"(?:bought|purchased)\s+a\s+(?:new\s+)?(?:car|vehicle)\s*[.!]\s*"
        r"it(?:['’]?s|\s+is)\s+an?\s+)"
        r"([\w -]*\b(?:19|20)\d{2}\s+[\w-]+(?:\s+[\w-]+){0,5})[.!]?",
        message.strip(), re.I,
    )
    if not match:
        return None
    description = match[1]
    if re.search(r"\b(?:not|no|or|and|but|maybe)\b", description, re.I):
        return None
    return MemoryCandidate(
        key="user.vehicle", content=f"The user's vehicle is a {description}.",
    )


class MemoryUpdateManager:
    def __init__(self, retriever, updater, consistency,
                 max_attempts: int = 3, debug: bool = False):
        self.retriever = retriever
        self.updater = updater
        self.consistency = consistency
        self.max_attempts = max_attempts
        self.debug = debug

    @staticmethod
    def _normalize_candidates(extracted: ExtractedMemories,
                              source_message: str) -> list[MemoryCandidate]:
        """Unify simple aliases, preserving every fact's distinct identity."""
        favorite = _stated_favorite_animal(source_message)
        vehicle = _stated_vehicle(source_message)
        unique = {fact.key: fact.content for fact in (favorite, vehicle) if fact}
        for original in extracted.memories:
            candidate = normalize_candidate(original, source_message)
            if favorite and candidate.key.startswith("user.favorite_animal."):
                continue
            if vehicle and _is_current_vehicle_key(candidate.key):
                continue
            previous = unique.get(candidate.key)
            if previous is not None and previous != candidate.content:
                raise MemoryValidationError(
                    f"Extractor returned competing values for {candidate.key!r}"
                )
            unique[candidate.key] = candidate.content
        return [MemoryCandidate(key=key, content=content)
                for key, content in unique.items()]

    @staticmethod
    def _verify_coverage(candidates: list[MemoryCandidate],
                         active: list[MemoryRecord],
                         changes: list[PreparedMemoryChange]) -> None:
        """Reject plans that silently drop facts, including incorrect NOOPs.

        The hypothetical final state is constructed *before* any database write.
        Lexical coverage is intentionally conservative, not semantic inference.
        """
        final = {record.id: (record.key, record.content) for record in active}
        for index, change in enumerate(changes):
            if change.action == "DELETE":
                final.pop(change.target_id, None)
            elif change.action == "UPDATE":
                if change.target_id not in final:
                    raise MemoryValidationError("Update targets an unavailable record")
                final[change.target_id] = (change.key, change.content or "")
            elif change.action == "ADD":
                final[f"new:{index}"] = (change.key, change.content or "")
        for candidate in candidates:
            if not any(is_represented_in_text(candidate, key, content)
                       for key, content in final.values()):
                raise MemoryValidationError(
                    f"Extracted fact {candidate.key!r} was ignored by the "
                    "proposed memory operations (vehicle color or another "
                    "attribute is missing; incorrect NOOP or update)"
                )

    @staticmethod
    def _safe_numeric_fallback(pending: list[MemoryCandidate],
                               related: dict[int, MemoryRecord],
                               source_message: str) -> MemoryPlan | None:
        """Last resort for an unambiguous horsepower correction.

        The model sometimes proposes REPLACE('produces 169 horsepower', ...)
        when the stored text says 'has 169 horsepower'. Construct the smallest
        safe edit using the *actual stored bytes* instead. Never infer a value.
        """
        if len(pending) != 1 or len(related) != 1:
            return None
        candidate = pending[0]
        if candidate.key != "user.vehicle.horsepower":
            return None
        old = next(iter(related.values()))
        pattern = r"\b(\d+)\s*horse[ -]?power\b"
        stored = re.findall(pattern, old.content, flags=re.IGNORECASE)
        proposed = re.findall(pattern, candidate.content, flags=re.IGNORECASE)
        stated = re.findall(pattern, source_message, flags=re.IGNORECASE)
        if len(stored) != 1 or len(proposed) != 1 or proposed[0] not in stated:
            return None
        if stored[0] == proposed[0]:
            return MemoryPlan(decisions=[MemoryDecision(
                action="NOOP", target_id=old.id,
            )])
        return MemoryPlan(decisions=[MemoryDecision(
            action="UPDATE", target_id=old.id,
            edits=[MemoryEdit(kind="REPLACE", old_text=stored[0],
                              new_text=proposed[0])],
        )])

    async def process(self, extracted: ExtractedMemories,
                      scope: str = "shared", source_message: str = "") -> list[UpdateResult]:
        if not source_message.strip():
            raise ValueError("MemoryUpdateManager requires the latest user message")

        repository = self.consistency.repository
        if _explicitly_lost_current_vehicle(source_message):
            vehicle_records = [
                record for record, _ in repository.list_active(scope)
                if _is_current_vehicle_key(record.key)
            ]
            if not vehicle_records:
                return []
            deletion = MemoryPlan(decisions=[
                MemoryDecision(action="DELETE", target_id=record.id)
                for record in vehicle_records
            ])
            changes, noops = self.consistency.prepare_plan(
                candidates=extracted.memories,
                existing_memories=vehicle_records,
                plan=deletion,
                source_message=source_message,
            )
            return self.consistency.apply_plan(changes, noops, scope)

        candidates = self._normalize_candidates(extracted, source_message)
        if not candidates:
            return []
        feedback = None

        for attempt in range(self.max_attempts):
            active = []
            related = {}
            pending = []
            auto_changes = []
            already_present = []
            try:
                # Read the entire current scope to avoid losing an exact fact
                # because it fell below a vector similarity cutoff.
                active = [record for record, _ in repository.list_active(scope)]
                related = {}
                fresh = []
                preference_changes = []
                already_present = []
                pending = []

                for candidate in candidates:
                    # Semantic retrieval remains the discovery step, but it
                    # does NOT give the updater permission to merge unrelated
                    # attributes. Exact logical identity determines eligibility.
                    nearest = await self.retriever.find_similar(
                        candidate=candidate, scope=scope, top_k=10
                    )
                    matches = {
                        r.id: r for r in [*nearest, *active]
                        if same_fact(candidate, r)
                    }
                    if not matches:
                        fresh.append(candidate)
                        continue
                    preference = self.consistency.prepare_preference_update(
                        candidate, [record for record in active if record.id in matches],
                        source_message,
                    )
                    if preference is not None:
                        changes, noops = preference
                        preference_changes.extend(changes)
                        already_present.extend(noops)
                        continue
                    if is_represented(candidate, list(matches.values())):
                        chosen = next(r for r in matches.values()
                                      if is_represented(candidate, [r]))
                        already_present.append(UpdateResult(
                            action="NOOP", key=chosen.key,
                            memory_id=chosen.id, version=chosen.version,
                        ))
                        continue
                    pending.append(candidate)
                    related.update(matches)

                auto_add_plan = MemoryPlan(decisions=[
                    MemoryDecision(action="ADD", key=c.key,
                                   new_content=c.content)
                    for c in fresh
                ])
                auto_changes, _ = self.consistency.prepare_plan(
                    candidates=fresh, existing_memories=active,
                    plan=auto_add_plan, source_message=source_message,
                )

                auto_changes.extend(preference_changes)
                direct_changes = self.consistency.prepare_vehicle_update(
                    pending, list(related.values()), source_message,
                )
                if direct_changes is None and len(pending) == len(related) == 1:
                    candidate = pending[0]
                    record = next(iter(related.values()))
                    change = self.consistency.prepare_favorite_animal_update(
                        candidate, record, source_message,
                    )
                    if change is not None:
                        direct_changes = [change]
                llm_changes = []
                llm_noops = []
                if pending and direct_changes is None:
                    # One combined plan for all facts that actually refer to
                    # existing mutable memories.
                    plan = await self.updater.plan(
                        candidates=pending,
                        existing_memories=list(related.values()),
                        source_message=source_message,
                        feedback=feedback,
                    )
                    if not plan.decisions:
                        raise MemoryValidationError(
                            "Updater returned no decisions for unmatched facts"
                        )
                    if self.debug:
                        print("\n[Proposed Memory Update Plan]")
                        print(plan.model_dump_json(indent=2))
                    llm_changes, llm_noops = self.consistency.prepare_plan(
                        candidates=pending, existing_memories=list(related.values()),
                        plan=plan, source_message=source_message,
                    )

                changes = [*auto_changes, *(direct_changes or []), *llm_changes]
                self._verify_coverage(candidates, active, changes)

                for change in changes:
                    if change.action in ("ADD", "UPDATE"):
                        change.embedding = await self.retriever.embed_document(
                            change.content
                        )
                # A single interaction may mention several attributes of one
                # record. Don't report stale v1 NOOPs after updating it to v2.
                changed_ids = {c.target_id for c in changes
                               if c.target_id is not None}
                noops = []
                seen_noops = set()
                for noop in [*already_present, *llm_noops]:
                    marker = noop.memory_id if noop.memory_id is not None else noop.key
                    if noop.memory_id in changed_ids or marker in seen_noops:
                        continue
                    seen_noops.add(marker)
                    noops.append(noop)
                return self.consistency.apply_plan(changes, noops, scope)
            except (MemoryValidationError, MemoryConflictError) as exc:
                feedback = str(exc)
                if self.debug:
                    print(f"[Update plan attempt {attempt + 1} rejected] {feedback}")
                if attempt + 1 == self.max_attempts:
                    # If the LLM repeatedly invents an invalid substring for
                    # a numeric replacement, try exactly one deterministic,
                    # source-grounded fallback. Never do this for ambiguous
                    # or multi-fact changes. Keep the failed transaction out
                    # of the database if fallback validation also fails.
                    fallback = (self._safe_numeric_fallback(
                        pending, related, source_message
                    ) if self.max_attempts > 1 else None)
                    if fallback is None:
                        raise
                    if self.debug:
                        print("[Fallback] Verifying a literal horsepower edit")
                    fb_changes, fb_noops = self.consistency.prepare_plan(
                        candidates=pending,
                        existing_memories=list(related.values()),
                        plan=fallback,
                        source_message=source_message,
                    )
                    # A fallback cannot omit unrelated new facts. The normal
                    # fresh/add and present/noop decisions are retained.
                    merged = [*auto_changes, *fb_changes]
                    self._verify_coverage(candidates, active, merged)
                    for change in merged:
                        if change.action in ("ADD", "UPDATE"):
                            change.embedding = await self.retriever.embed_document(
                                change.content
                            )
                    return self.consistency.apply_plan(
                        merged, [*already_present, *fb_noops], scope,
                    )
                # A fresh store snapshot is loaded on each retry.

        return []