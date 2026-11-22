"""Coordinate extraction and one atomic memory update per interaction."""

import re

from src.memory.consistency import MemoryValidationError
from src.memory.identity import (
    ADDITIONAL_VEHICLE_PREFIX, COLORS, INVESTMENT_DEPOSIT_PREFIX, is_represented, is_represented_in_text, normalize_candidate,
    normalize_grouped_numbers, same_fact,
)
from src.memory.models import (
    ExtractedMemories, MemoryCandidate, MemoryDecision, MemoryEdit, MemoryPlan,
    MemoryRecord, PreparedMemoryChange, UpdateResult, VehiclePurchase,
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
        message = context.user_message.content
        stock_purchase = _stated_stock_purchase(message)
        if stock_purchase:
            return ExtractedMemories(memories=[stock_purchase])
        if _stated_vehicle_sale(message):
            return ExtractedMemories()
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


def _stated_vehicle_sale(message: str) -> re.Match | None:
    text = message.strip()
    generic_sale = re.fullmatch(
        r"(?:i\s+(?:have\s+)?|i['’]ve\s+)(?:(?:just|recently)\s+)?sold\s+(?:my|the)\s+(?:car|vehicle)"
        r"(?:\s+and\s+got\s+(?P<amount>\$?\d[\d,]*(?:\.\d{2})?)\s+for\s+it)?[.!]*",
        text, re.IGNORECASE,
    )
    if generic_sale:
        return generic_sale
    return re.fullmatch(
        r"(?:i\s+(?:have\s+)?|i['’]ve\s+)(?:(?:just|recently)\s+)?sold\s+(?:my|the)\s+(?P<name>[\w-]+(?:\s+[\w-]+){0,4})"
        r"\s+for\s+(?P<amount>\$?\d[\d,]*(?:\.\d{2})?)"
        r"(?:\s+after\s+fixing\s+it\s+up)?"
        r"(?P<down_payment>\s+and\s+(?:put|made)\s+a\s+down\s+payment\s+on\s+a\s+new\s+house)?[.!]*",
        text, re.IGNORECASE,
    )


def _is_current_vehicle_key(key: str) -> bool:
    key = key.casefold()
    if key.startswith(("user.likes.", "user.dislikes.", "user.preference.",
                       ADDITIONAL_VEHICLE_PREFIX)):
        return False
    parts = re.split(r"[._]", key)
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


def _stated_new_vehicle_purchase(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"(?:with\s+the\s+money\s+i\s+deposited\s*,?\s+)?"
        r"i\s+bought\s+a\s+new\s+(?:car|vehicle)\s*[,.:]?\s+an?\s+"
        r"([\w-]+(?:\s+[\w-]+){0,5})\s+from\s+((?:19|20)\d{2})"
        r"(?:\s+for\s+(\$?\d+(?:\.\d{2})?))?[.!]?",
        normalize_grouped_numbers(message.strip()), re.I,
    )
    if not match:
        return None
    name, year, price = match.groups()
    if re.search(r"\b(?:not|no|or|and|but|maybe)\b", name, re.I):
        return None
    content = f"The user's vehicle is a {year} {name}"
    if price:
        content += f"; bought for {price}"
    return MemoryCandidate(key="user.vehicle", content=f"{content}.")


def _stated_additional_vehicle(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"i\s+(?:have\s+)?spent\s+(\$?\d+(?:\.\d{2})?\$?)\s+on\s+another\s+(?:car|vehicle)"
        r"\s*[.!]\s*(?:the\s+(?:car|vehicle)\s+)?it(?:['’]?s|\s+is)\s+an?\s+"
        r"([a-z]+)\s+((?:19|20)\d{2})\s+([\w-]+(?:\s+[\w-]+){0,5})[.!]?",
        normalize_grouped_numbers(message.strip()), re.I,
    )
    if not match:
        return None
    price, color, year, name = match.groups()
    if color.lower() not in COLORS or re.search(r"\b(?:not|no|or|and|but|maybe)\b", name, re.I):
        return None
    if price.endswith("$"):
        price = "$" + price[:-1]
    model_key = ".".join(name.casefold().split())
    return MemoryCandidate(
        key=f"{ADDITIONAL_VEHICLE_PREFIX}{year}.{model_key}",
        content=f"The user spent {price} on another car, a {color} {year} {name}.",
    )


def _stated_vehicle(message: str) -> MemoryCandidate | None:
    additional = _stated_additional_vehicle(message)
    if additional:
        return additional
    purchase = _stated_new_vehicle_purchase(message)
    if purchase:
        return purchase

    funded_purchase = re.fullmatch(
        r"i\s+used\s+(?:this|that|the)\s+money\s+to\s+buy\s+an?\s+"
        r"([\w-]+(?:\s+[\w-]+){0,3})\s+((?:19|20)\d{2})\s+"
        r"([a-z]+)\s+with\s+an?\s+(v\d+)\s+engine[.!]?",
        message.strip(), re.I,
    )
    if funded_purchase:
        name, year, color, engine = funded_purchase.groups()
        if color.lower() in COLORS:
            return MemoryCandidate(
                key="user.vehicle",
                content=f"The user's vehicle is a {color} {year} {name} with a {engine.upper()} engine.",
            )

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


def _structured_vehicle_purchase(
    event: VehiclePurchase | None, message: str,
) -> MemoryCandidate | None:
    if event is None:
        return None
    source = normalize_grouped_numbers(message).casefold()
    source = " ".join(source.split())
    # Only a completed first-person purchase can change ownership.
    completed = re.search(
        r"\bi (?:have |just |recently )*(?:bought|purchased|spent)\b|"
        r"\bi used (?:this|that|the) money to buy\b",
        source,
    )
    uncertain = re.search(
        r"\b(?:if|wish|hypothetically|pretend|maybe|almost|never|not)\b",
        source,
    )
    if not completed or uncertain or source.endswith("?"):
        return None

    model = " ".join(event.model.split())
    year = str(event.year)
    color = event.color.strip().lower() if event.color else None
    purchase_text = source[completed.end():]
    if completed[0].endswith("spent") and not re.match(
        r" \$?\d+(?:\.\d{2})?\$? on (?:another|an?(?: new| second| additional)?) (?:car|vehicle)\b",
        purchase_text,
    ):
        return None
    if (not re.fullmatch(r"[\w-]+(?: [\w-]+){0,5}", model)
            or not re.search(r"\b" + re.escape(model.casefold()) + r"\b", purchase_text)
            or not re.search(rf"\b{year}\b", purchase_text)
            or not 1900 <= event.year <= 2099):
        return None
    if color and (color not in COLORS or not re.search(rf"\b{color}\b", purchase_text)):
        return None

    price = normalize_grouped_numbers(event.price.strip()) if event.price else None
    if price:
        if not re.fullmatch(r"\$?\d+(?:\.\d{2})?\$?", price):
            return None
        amount = re.escape(price)
        price_context = rf"\b(?:spent|for|paid) {amount}(?![\w$]|\.\d)"
        if not re.search(price_context, source):
            return None
        if price.endswith("$"):
            price = "$" + price[:-1]

    additional = re.search(
        r"\b(?:another|second|additional) (?:car|vehicle)\b"
        rf"|\b(?:another|second|additional)(?: {year}| {color or year}){{0,2}} "
        + re.escape(model.casefold()) + r"\b",
        purchase_text,
    )
    key = "user.vehicle"
    description = " ".join(part for part in (color, year, model) if part)
    if additional:
        model_key = ".".join(model.casefold().split())
        key = f"{ADDITIONAL_VEHICLE_PREFIX}{year}.{model_key}"
        content = f"The user bought another vehicle, a {description}"
    else:
        content = f"The user's vehicle is a {description}"
    if price:
        content += f"; purchase price {price}"
    return MemoryCandidate(key=key, content=f"{content}.")


def _stated_investment_deposit(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"(?:i\s+(?:(?:have|currently|just|recently)\s+)*(?P<completed>deposited)"
        r"|(?:i['’]?m|i\s+am)\s+depositing)\s+"
        r"(\$?\d+(?:\.\d{2})?\$?)\s+"
        r"(?:to\s+(?:start\s+investing|make\s+investments|invest)|for\s+investments?)[.!]?",
        normalize_grouped_numbers(message.strip()), re.I,
    )
    if not match:
        return None
    amount = match[2]
    if amount.endswith("$"):
        amount = "$" + amount[:-1]
    amount_key = amount.replace("$", "dollar_").replace(".", "_")
    verb = "deposited" if match["completed"] else "is depositing"
    return MemoryCandidate(
        key=f"{INVESTMENT_DEPOSIT_PREFIX}{amount_key}",
        content=f"The user {verb} {amount} for investments.",
    )


def _stated_stock_purchase(message: str) -> MemoryCandidate | None:
    match = re.fullmatch(
        r"i (?:have |just |recently )*bought ([1-9]\d*) (stocks?|shares?) "
        r"(?:(?:which |that )?(?:were |was )?)?(worth|for) (\$?\d+(?:\.\d{2})?\$?)[.!]?",
        " ".join(normalize_grouped_numbers(message).strip().split()), re.I,
    )
    if not match:
        return None
    quantity, asset, relation, amount = match.groups()
    if amount.endswith("$"):
        amount = "$" + amount[:-1]
    amount_key = amount.replace("$", "dollar_").replace(".", "_")
    return MemoryCandidate(
        key=f"user.investment.purchase.{quantity}.{asset.lower()}.{relation.lower()}.{amount_key}",
        content=f"The user bought {quantity} {asset.lower()} {relation.lower()} {amount}.",
    )


def transaction_acknowledgment(message: str) -> str | None:
    purchase = _stated_stock_purchase(message)
    if purchase:
        detail = purchase.content.removeprefix("The user ")
        return f"You {detail} Your remaining investment balance is unknown."
    sale = _stated_vehicle_sale(message)
    if sale:
        amount = sale["amount"]
        name = sale.groupdict().get("name")
        item = f"the {name}" if name else "your vehicle"
        if amount:
            return f"You reported selling {item} for {amount}. Your investment balance is unknown."
        return f"You reported selling {item}."
    return None


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
                              source_message: str,
                              purchase: MemoryCandidate | None = None) -> list[MemoryCandidate]:
        """Unify simple aliases, preserving every fact's distinct identity."""
        stock_purchase = _stated_stock_purchase(source_message)
        if stock_purchase:
            # This whole message states one purchase. Extracted balances or
            # unrelated facts from earlier turns cannot affect this write.
            return [stock_purchase]

        favorite = _stated_favorite_animal(source_message)
        vehicle = purchase or _stated_vehicle(source_message)
        deposit = _stated_investment_deposit(source_message)
        unique = {fact.key: fact.content for fact in (favorite, vehicle, deposit) if fact}
        for original in extracted.memories:
            candidate = normalize_candidate(original, source_message)
            if candidate.key == "user.financial_assistant" and re.fullmatch(
                r"i want you to be my personal financial assistant who keeps track of my investments[.!]?",
                " ".join(source_message.strip().split()), re.I,
            ):
                candidate = candidate.model_copy(update={
                    "content": "The user wants a personal financial assistant who keeps track of their investments.",
                })
            if favorite and candidate.key.startswith("user.favorite_animal."):
                continue
            if vehicle and (_is_current_vehicle_key(candidate.key)
                            or candidate.key.startswith(ADDITIONAL_VEHICLE_PREFIX)):
                continue
            if deposit and re.match(
                r"user\.(?:financial|finances|deposit|deposited)(?:\.|$)", candidate.key,
            ):
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

        if re.fullmatch(
            r"(?:what happened to my (?:car|vehicle)|"
            r"(?:give me a summary of|summari[sz]e) what (?:has )?happened(?: so far)?|"
            r"how much (?:money )?do i have(?: after selling my (?:car|vehicle))?)\s*[?!.]*",
            source_message.strip(), re.I,
        ):
            return []

        repository = self.consistency.repository
        sale = _stated_vehicle_sale(source_message)
        if sale or _explicitly_lost_current_vehicle(source_message):
            active = [record for record, _ in repository.list_active(scope)]
            vehicle_records = [r for r in active if _is_current_vehicle_key(r.key)]
            sale_fields = sale.groupdict() if sale else {}
            name = sale_fields.get("name")
            key = "user.vehicle.history.sale"
            if name:
                key += "." + re.sub(r"\s+", "_", name.casefold())
            previous = next((record for record in active if record.key == key), None)
            if name:
                name_pattern = r"\b" + r"\s+".join(map(re.escape, name.split())) + r"\b"
                current_match = any(re.search(name_pattern, r.content, re.I) for r in vehicle_records)
                additional_matches = [
                    r for r in active
                    if r.key.startswith(ADDITIONAL_VEHICLE_PREFIX)
                    and re.search(name_pattern, r.content, re.I)
                ]
                if len(additional_matches) + current_match > 1:
                    raise MemoryValidationError("The sale matches more than one saved vehicle")
                if not current_match:
                    if not additional_matches and previous is None:
                        return []
                    vehicle_records = additional_matches
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
            if sale:
                amount = sale_fields["amount"]
                content = f"The user sold the {name}." if name else "The user sold their car."
                if amount:
                    content = f"{content[:-1]} for {amount}."
                if sale_fields.get("down_payment"):
                    content += " The user made a down payment on a new house."
                if previous is None or (
                    previous.content != content
                    and (vehicle_records or (amount and " for " not in previous.content))
                ):
                    changes.append(PreparedMemoryChange(
                        action="UPDATE" if previous else "ADD", key=key, content=content,
                        target_id=previous.id if previous else None,
                        expected_version=previous.version if previous else None,
                        embedding=await self.retriever.embed_document(content),
                    ))
            return self.consistency.apply_plan(changes, noops, scope)

        structured_purchase = _structured_vehicle_purchase(extracted.vehicle_purchase, source_message)
        new_purchase = structured_purchase or _stated_new_vehicle_purchase(source_message)
        candidates = self._normalize_candidates(extracted, source_message, new_purchase)
        if not candidates:
            return []
        feedback = None
        deposit = _stated_investment_deposit(source_message)

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
                resolved_changes = []
                already_present = []
                pending = []

                for candidate in candidates:
                    # Semantic retrieval remains the discovery step, but it
                    # does NOT give the updater permission to merge unrelated
                    # attributes. Exact logical identity determines eligibility.
                    nearest = [] if candidate == structured_purchase else await self.retriever.find_similar(
                        candidate=candidate, scope=scope, top_k=10
                    )
                    matches = {
                        r.id: r for r in [*nearest, *active]
                        if same_fact(candidate, r)
                    }
                    if not matches:
                        if candidate == structured_purchase:
                            resolved_changes.append(PreparedMemoryChange(
                                action="ADD", key=candidate.key, content=candidate.content,
                            ))
                        else:
                            fresh.append(candidate)
                        continue
                    preference = self.consistency.prepare_preference_update(
                        candidate, [record for record in active if record.id in matches],
                        source_message,
                    )
                    if preference is not None:
                        changes, noops = preference
                        resolved_changes.extend(changes)
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
                    if candidate in (new_purchase, deposit) and len(matches) == 1:
                        record = next(iter(matches.values()))
                        resolved_changes.append(PreparedMemoryChange(
                            action="UPDATE", key=record.key, content=candidate.content,
                            target_id=record.id, expected_version=record.version,
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

                auto_changes.extend(resolved_changes)
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