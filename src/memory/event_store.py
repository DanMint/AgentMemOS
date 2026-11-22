"""Ground interpreted events and apply their state transitions to SQLite."""

import hashlib
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from src.memory.consistency import MemoryValidationError, _grounded_content
from src.memory.events import (
    Deposit, Event, Fact, StockPurchase, TurnInterpretation, VehicleDetail,
    VehiclePurchased, VehicleRemoved, VehicleSelection,
)
from src.memory.identity import (
    ADDITIONAL_VEHICLE_PREFIX, COLORS, INVESTMENT_DEPOSIT_PREFIX, is_represented,
    normalize_candidate, normalize_grouped_numbers, same_fact, tokens,
)
from src.memory.models import MemoryCandidate, MemoryRecord, PreparedMemoryChange, UpdateResult
from src.storage.memory_repository import MemoryConflictError


class PendingTurn(BaseModel):
    source: str
    turn_id: str
    interpretation: TurnInterpretation
    event_index: int
    options: list[int]
    selections: dict[int, int] = Field(default_factory=dict)


class InvalidEvidence(MemoryValidationError):
    """The interpretation can be retried with specific feedback."""


class ClarifyVehicle(Exception):
    def __init__(self, options: list[MemoryRecord]):
        self.options = options


def _quote(value: str, evidence: str) -> None:
    quoted = " ".join(value.casefold().split())
    source = " ".join(evidence.casefold().split())
    if not quoted or not re.search(rf"(?<!\w){re.escape(quoted)}(?!\w)", source):
        raise MemoryValidationError(f"The user's message does not support {value!r}.")


def _validate(event: Event, source: str) -> None:
    _quote(event.evidence, source)
    if event.evidence.rstrip().endswith("?"):
        raise MemoryValidationError("A question does not establish a memory event.")
    for field in ("description", "vehicle", "replaces", "destination",
                  "value", "quantity", "security"):
        value = getattr(event, field, None)
        if value is not None:
            _quote(value, event.evidence)
    if isinstance(event, Fact):
        _grounded_content(event.content, event.evidence)
    if isinstance(event, (Deposit, StockPurchase, VehicleRemoved, VehiclePurchased)) and event.amount:
        amount = normalize_grouped_numbers(event.amount)
        evidence = normalize_grouped_numbers(event.evidence)
        if not re.search(
            rf"(?<![\w$€£]){re.escape(amount)}(?![\w$€£]|\.\d)",
            evidence, re.I,
        ):
            raise MemoryValidationError("The amount or currency is incomplete.")
        _amount(event.amount)


def validate_vehicle_selection(event: VehicleSelection, source: str) -> None:
    _validate(event, source)


def _amount(amount: str) -> str:
    value = " ".join(normalize_grouped_numbers(amount).split())
    # Preserve spoken units such as “14 grand” and any stated currency.
    numeric = r"[$€£]?\d+(?:\.\d+)?(?:[$€£k])?(?: (?:grand|thousand|million|dollars?|euros?|pounds?|usd|eur|gbp))?"
    word = (
        r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
        r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
        r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|grand"
    )
    spoken = rf"(?:{word})(?:[ -](?:{word}|and))*"
    if not re.fullmatch(rf"{numeric}|{spoken}", value, re.I):
        raise MemoryValidationError("The amount needs a clear value or unit.")
    if value[-1] in "$€£" and value[0].isdigit():
        value = value[-1] + value[:-1]
    return value


def _key_part(value: str) -> str:
    return hashlib.sha256(value.casefold().encode()).hexdigest()[:20]


def _vehicle_match(reference: str, record: MemoryRecord) -> bool:
    # Compare literal words, retaining digits in names such as BMW 3 Series.
    needle = tokens(reference)
    words = tokens(record.content.split(";", 1)[0])
    return bool(needle) and any(
        words[i:i + len(needle)] == needle for i in range(len(words) - len(needle) + 1)
    )


class EventBatch:
    def __init__(self, records: Iterable[MemoryRecord]):
        self.original = {record.id: record for record in records}
        self.current = self.original.copy()
        self.next_id = -1

    def by_key(self, key: str) -> MemoryRecord | None:
        return next((record for record in self.current.values() if record.key == key), None)

    def put(self, key: str, content: str, record: MemoryRecord | None = None) -> None:
        if record is None:
            record = self.by_key(key)
        if record is None:
            self.current[self.next_id] = MemoryRecord(id=self.next_id, key=key, content=content, version=1)
            self.next_id -= 1
        else:
            self.current[record.id] = record.model_copy(update={"key": key, "content": content})

    def vehicles(self) -> list[MemoryRecord]:
        return [record for record in self.current.values()
                if record.key == "user.vehicle" or record.key.startswith(ADDITIONAL_VEHICLE_PREFIX)]

    def vehicle(self, reference: str | None, selected: int | None) -> MemoryRecord:
        options = [record for record in self.vehicles()
                   if not reference or _vehicle_match(reference, record)]
        if selected is not None:
            options = [record for record in options if record.id == selected]
        if len(options) > 1:
            raise ClarifyVehicle(options)
        if not options:
            raise MemoryValidationError("I couldn't match that statement to a saved vehicle.")
        return options[0]

    def retire(self, record: MemoryRecord) -> None:
        self.current.pop(record.id)
        if record.key == "user.vehicle":
            # Old databases may store the primary car's attributes separately.
            for attr in ("color", "year", "horsepower", "transmission", "engine", "trim", "make", "model"):
                fragment = self.by_key(f"user.vehicle.{attr}")
                if fragment:
                    self.current.pop(fragment.id)

    def changes(self) -> list[PreparedMemoryChange]:
        result = []
        for old in self.original.values():
            new = self.current.get(old.id)
            if new is None:
                result.append(PreparedMemoryChange(
                    action="DELETE", key=old.key, target_id=old.id, expected_version=old.version,
                ))
            elif (new.key, new.content) != (old.key, old.content):
                result.append(PreparedMemoryChange(
                    action="UPDATE", key=new.key, content=new.content,
                    target_id=old.id, expected_version=old.version,
                ))
        result.extend(PreparedMemoryChange(action="ADD", key=r.key, content=r.content)
                      for r in self.current.values() if r.id < 0)
        # Retire conflicting keys before renaming or adding a record.
        return sorted(result, key=lambda change: change.action != "DELETE")


def _apply_vehicle(batch: EventBatch, event: VehiclePurchased | VehicleRemoved | VehicleDetail,
                   event_key: str, selected: int | None) -> None:
    if isinstance(event, VehiclePurchased):
        content = f"The user's vehicle is {event.description.strip().rstrip('.')}."
        if event.amount:
            content = f"{content[:-1]}; bought for {_amount(event.amount)}."
        if any(record.content.casefold() == content.casefold() for record in batch.vehicles()):
            return
        if event.replaces:
            old = batch.vehicle(event.replaces, selected)
            batch.put(f"user.vehicle.history.replaced.{event_key}", f"Previously saved vehicle: {old.content}")
            batch.retire(old)
            key = old.key
        elif not batch.vehicles():
            key = "user.vehicle"
        else:
            key = f"{ADDITIONAL_VEHICLE_PREFIX}{event_key}"
        batch.put(key, content)
        return

    if isinstance(event, VehicleRemoved):
        history_key = f"user.vehicle.history.{event.kind}.{event_key}"
        if batch.by_key(history_key):
            return
        record = batch.vehicle(event.vehicle, selected)
        description = record.content.split(";", 1)[0].removeprefix("The user's vehicle is ").rstrip(".")
        action = "sold" if event.kind == "vehicle_sold" else "no longer possesses"
        content = f"The user {action} {description}"
        if event.kind == "vehicle_sold" and event.amount:
            content += f" for {_amount(event.amount)}"
        batch.retire(record)
        batch.put(history_key, f"{content}.")
        return

    record = batch.vehicle(event.vehicle, selected)
    value = event.value.strip().rstrip(".")
    description, *details = record.content.rstrip(".").split("; ")
    pattern = {
        "year": r"\b(?:19|20)\d{2}\b",
        "horsepower": r"\b\d+\s*horse[ -]?power\b",
        "color": r"\b(?:" + "|".join(sorted(COLORS)) + r")\b",
    }.get(event.attribute)
    if pattern:
        if event.attribute == "horsepower" and re.fullmatch(r"\d+", value):
            value += " horsepower"
        parts = [description, *details]
        for index, part in enumerate(parts):
            if part.startswith("bought for "):
                continue
            if re.search(pattern, part, re.I):
                parts[index] = re.sub(pattern, lambda _: value, part, count=1, flags=re.I)
                batch.put(record.key, "; ".join(parts) + ".", record)
                break
        else:
            batch.put(record.key, f"{record.content.rstrip('.')}; {event.attribute} is {value}.", record)
    else:
        prefix = event.attribute + " is "
        details = [part for part in details if not part.startswith(prefix)]
        batch.put(record.key, "; ".join([description, *details, prefix + value]) + ".", record)
    if record.key == "user.vehicle":
        legacy = batch.by_key(f"user.vehicle.{event.attribute}")
        if legacy:
            batch.current.pop(legacy.id)


def _apply_deposit(batch: EventBatch, event: Deposit, event_key: str) -> None:
    amount = _amount(event.amount) if event.amount else None
    destination = f" for {event.destination}" if event.destination else ""
    if amount is None:
        pending = [record for record in batch.current.values()
                   if record.key.startswith(INVESTMENT_DEPOSIT_PREFIX)
                   and record.content.startswith("The user is depositing ")
                   and (not destination or record.content.casefold().endswith(f"{destination}.".casefold()))]
        if event.status != "completed" or len(pending) != 1:
            raise MemoryValidationError("Please specify the deposit amount and which deposit you mean.")
        record = pending[0]
        batch.put(record.key, record.content.replace("is depositing", "deposited", 1), record)
        return
    action = "is depositing" if event.status == "pending" else "deposited"
    content = f"The user {action} {amount}{destination}."
    if any(record.content in {content, content.replace("is depositing", "deposited", 1)}
           for record in batch.current.values() if record.key.startswith(INVESTMENT_DEPOSIT_PREFIX)):
        return
    pending = [
        record for record in batch.current.values()
        if record.key.startswith(INVESTMENT_DEPOSIT_PREFIX)
        and record.content.startswith(f"The user is depositing {amount}")
        and record.content[len(f"The user is depositing {amount}"):].startswith((" ", "."))
        and (not destination or record.content.endswith(f"{destination}."))
    ]
    if event.status == "completed" and len(pending) == 1:
        previous = pending[0]
        batch.put(previous.key, previous.content.replace("is depositing", "deposited", 1), previous)
    elif event.status == "completed" and len(pending) > 1:
        raise MemoryValidationError("More than one pending deposit matches that amount. Which deposit cleared?")
    else:
        batch.put(INVESTMENT_DEPOSIT_PREFIX + event_key, content)


def _apply_fact(batch: EventBatch, event: Fact) -> None:
    key = event.key.casefold().strip()
    if "." not in key and re.fullmatch(r"[a-z0-9_-]+", key):
        key = f"user.{key}"
    candidate = normalize_candidate(MemoryCandidate(key=key, content=event.content), event.evidence)
    key = candidate.key
    if not re.fullmatch(r"user\.[a-z0-9_.-]+|project\.[a-z0-9_.-]+", key):
        raise MemoryValidationError("The fact has an invalid memory key.")
    if key == "user.vehicle":
        # Ownership statements may introduce a first vehicle. Further changes
        # need a vehicle event with an identity or a specific attribute.
        if batch.vehicles():
            if any(is_represented(candidate, [record]) for record in batch.vehicles()):
                return
            raise MemoryValidationError("Please say which vehicle you are adding or correcting.")
    elif key.startswith(("user.vehicle.", "user.investment.purchase.", INVESTMENT_DEPOSIT_PREFIX)):
        raise MemoryValidationError("That fact needs a specific vehicle or transaction event.")
    related = [record for record in batch.current.values() if same_fact(candidate, record)]
    keeper = next((record for record in related if record.key == key), related[0] if related else None)
    if key.startswith(("user.likes.", "user.dislikes.")):
        for record in related:
            if record != keeper:
                batch.current.pop(record.id)
    if keeper and is_represented(candidate, [keeper]):
        return
    batch.put(key, candidate.content, keeper)


class EventStore:
    def __init__(self, repository, retriever, scope: str):
        self.repository = repository
        self.retriever = retriever
        self.scope = scope

    async def apply(self, interpretation: TurnInterpretation, source: str, turn_id: str,
                    selections: dict[int, int]) -> tuple[list[UpdateResult], list[str], PendingTurn | None]:
        try:
            for event in interpretation.events:
                _validate(event, source)
        except MemoryValidationError as exc:
            raise InvalidEvidence(str(exc)) from exc
        for attempt in range(3):
            batch = EventBatch(record for record, _ in self.repository.list_active(self.scope))
            for index, event in enumerate(interpretation.events):
                event_key = _key_part(f"{turn_id}:{index}")
                try:
                    if isinstance(event, (VehiclePurchased, VehicleRemoved, VehicleDetail)):
                        _apply_vehicle(batch, event, event_key, selections.get(index))
                    elif isinstance(event, Deposit):
                        _apply_deposit(batch, event, event_key)
                    elif isinstance(event, StockPurchase):
                        security = f" of {event.security}" if event.security else ""
                        amount = ""
                        if event.amount:
                            label = "costing" if event.amount_kind == "cost" else "worth"
                            amount = f" {label} {_amount(event.amount)}"
                        batch.put(f"user.investment.purchase.{event_key}",
                                  f"The user bought {event.quantity}{security}{amount}.")
                    elif isinstance(event, Fact):
                        _apply_fact(batch, event)
                except ClarifyVehicle as exc:
                    pending = PendingTurn(
                        source=source, turn_id=turn_id, interpretation=interpretation,
                        event_index=index, options=[record.id for record in exc.options],
                        selections=selections,
                    )
                    labels = [record.content for record in exc.options]
                    return [], labels, pending
            changes = batch.changes()
            if not changes:
                return [], [], None
            for change in changes:
                if change.content is not None:
                    change.embedding = await self.retriever.embed_document(change.content)
            try:
                operations = self.repository.apply_batch(
                    changes, self.scope, expected_active={
                        record.id: record.version for record in batch.original.values()
                    },
                )
                return operations, [change.content for change in changes if change.content], None
            except MemoryConflictError:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def select_vehicle(self, pending: PendingTurn, event: VehicleSelection) -> int:
        matches = [
            record.id for record, _ in self.repository.list_active(self.scope)
            if record.id in pending.options and _vehicle_match(event.vehicle, record)
        ]
        if len(matches) != 1:
            raise MemoryValidationError("Please name one of the saved vehicles.")
        return matches[0]
