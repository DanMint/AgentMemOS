import re

from src.memory.identity import (
    COLORS, PREFERENCE_SPELLINGS, is_represented, normalize_grouped_numbers,
    normalized_key,
)
from src.memory.models import (
    MemoryCandidate, MemoryDecision, MemoryPlan, MemoryRecord,
    PreparedMemoryChange, UpdateResult,
)
from src.storage.memory_repository import MemoryRepository


class MemoryValidationError(ValueError):
    """A proposed memory change is not sufficiently grounded in the user message."""


def _words(text: str) -> list[str]:
    """
    Normalize common lexical variants before grounding checks.

    In particular, hate/dislike/don't like must be treated
    equivalently when validating negative preferences.
    """

    text = re.sub(
        r"\b(?:don['’]?t|doesn['’]?t|do(?:es)? not)\s+like\b",
        "dislike",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bhorse[\s-]*power\b",
        "horsepower",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\btransimission\b",
        "transmission",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:automobile|vehicle)\b",
        "car",
        text,
        flags=re.IGNORECASE,
    )

    text = normalize_grouped_numbers(text)
    words = [
        word.casefold()
        for word in re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE,
        )
    ]

    aliases = {
        "plans": "plan",
        "planning": "plan",
        "buys": "buy",
        "wants": "want",
        "investing": "invest",
        "investment": "invest",
        "investments": "invest",

        "likes": "like",
        "loves": "like",
        "love": "like",
        "liked": "like",

        "hates": "dislike",
        "hate": "dislike",
        "hated": "dislike",
        "dislikes": "dislike",
        "disliked": "dislike",

        "dogs": "dog",
        "turtles": "turtle",
        "horses": "horse",
        "animals": "animal",
        "pets": "pet",

        "favourite": "favorite",
    }

    corrected = [PREFERENCE_SPELLINGS.get(word, word) for word in words]
    return [aliases.get(word, word) for word in corrected]

# These terms express grammar, not the value of the fact. A model can say
# "has a CVT transmission" even when the user said "transmission ... is a CVT".
_GRAMMAR = {
    "a", "an", "the", "i", "me", "my", "mine", "you", "your", "user", "users",
    "his", "her", "their", "it", "its", "this", "that", "those", "these",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "with", "to", "in", "on", "of", "for", "from", "by", "and", "or",
    "now", "currently", "already", "also", "which", "who", "named", "called",
    "owns", "own", "possesses", "drives", "at", "as", "uses", "used", "car", "s",
}


def _grounded_content(content: str, source_message: str,
                      existing_content: str = "") -> None:
    """Conservative lexical grounding for *new* material in LLM output.

    Accept harmless paraphrasing and word-order changes, but do not allow the
    model to invent a year, numeric value, model name, or descriptive detail.
    Existing memory can supply context for pronouns such as 'it'. This is a
    safeguard, not a complete semantic entailment check.
    """
    source_words = set(_words(source_message))
    context_words = set(_words(existing_content))
    allowed_words = source_words | context_words
    significant = set(_words(content)) - _GRAMMAR
    missing = significant - allowed_words
    # In a car/horsepower fact, 'engine' often supplies grammatical context
    # rather than a separate, novel engine specification.
    if "horsepower" in source_words and "horsepower" in significant:
        missing.discard("engine")
    if missing:
        raise MemoryValidationError(
            f"Unsupported information in proposed memory: {', '.join(sorted(missing))}. "
            "Use wording supported by the latest user message and existing memory."
        )

    # Preserve the distinction between purchasing and already owning a car.
    # The extractor may hallucinate an ownership predicate for future plans.
    prospective = bool(re.search(
        r"\b(?:buying|plan(?:ning)? to buy|going to buy|intend(?:ing)? to buy|will buy)\b",
        source_message, flags=re.IGNORECASE))
    completed_ownership = bool(re.search(
        r"\b(?:i (?:already )?(?:own|have|bought|purchased)|i've (?:bought|purchased))\b",
        source_message, flags=re.IGNORECASE))
    claims_ownership = bool(re.search(r"\b(?:user )?(?:owns|owned)\b", content,
                                      flags=re.IGNORECASE))
    if prospective and claims_ownership and not completed_ownership:
        raise MemoryValidationError(
            "A plan to buy something is not evidence of current ownership"
        )


def _is_contiguous_quote(quote: str, source: str) -> bool:
    """Ensure changed/added words occur consecutively in the current user message."""
    expected = _words(quote)
    actual = _words(source)
    if not expected:
        return False
    size = len(expected)
    return any(actual[i:i + size] == expected
               for i in range(len(actual) - size + 1))


def _guard_protected_tokens(old: str, new: str, source: str) -> None:
    """Conservatively prevent removing an unchanged year or named entity.

    We allow editing a protected span when its *old value* is explicitly
    mentioned in the latest message (e.g. 'not a 2026 model') or a replacement
    of the same category is explicitly provided ('2027').
    """
    old_numbers = set(re.findall(r"\b\d+\b", old))
    new_numbers = set(re.findall(r"\b\d+\b", new))
    source_numbers = set(re.findall(r"\b\d+\b", source))
    if old_numbers - new_numbers and not (old_numbers & source_numbers or new_numbers & source_numbers):
        raise MemoryValidationError("Edit would discard a year/number not mentioned in the user message")

    # An atomic replacement of a proper name must remain a proper name.
    # In particular "Toyota" -> "red" must NOT pass just because red is
    # mentioned in the latest message. Broader rewrites must preserve old names
    # unless those old identities are themselves explicitly discussed.
    colors = {"black", "white", "red", "blue", "green", "yellow", "orange",
              "purple", "pink", "brown", "gray", "grey", "silver", "gold",
              "beige", "maroon", "navy", "teal"}
    old_words = _words(old)
    new_words = _words(new)
    if len(old_words) == 1 and old_words[0] in colors:
        if not any(word in colors for word in new_words):
            raise MemoryValidationError("A color cannot be replaced with a non-color value")

    old_names = {word.casefold() for word in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", old)}
    new_names = {word.casefold() for word in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", new)}
    old_names -= colors
    new_names -= colors
    source_words = set(_words(source))
    removed = old_names - new_names
    if len(old_words) == 1 and removed and not new_names:
        # CVT -> six speed manual is a genuine attribute change, not a
        # vehicle make/model hallucination. Both terms must refer to a
        # transmission, and the user must support the new specification.
        transmission_kinds = {"cvt", "dct", "automatic", "manual"}
        if not (old_words[0] in transmission_kinds
                and any(word in transmission_kinds for word in new_words)):
            raise MemoryValidationError(
                "A named entity cannot be replaced by an unrelated attribute"
            )
    if len(old_words) > 1 and removed and not removed.issubset(source_words):
        raise MemoryValidationError(
            "Edit would remove a named entity that the user did not explicitly correct"
        )


def apply_grounded_edits(existing_content: str, edits: list, source_message: str) -> str:
    """Python, not the LLM, constructs updated memory content."""
    if not edits:
        raise MemoryValidationError("UPDATE requires at least one explicit edit")
    content = existing_content
    for edit in edits:
        old = edit.old_text
        new = edit.new_text
        if not new.strip():
            raise MemoryValidationError("An edit cannot introduce empty content")
        # The model may paraphrase or reorder words ("has a CVT transmission"
        # versus "the transmission ... is a CVT"). Check factual tokens,
        # not exact contiguous wording.
        _grounded_content(new, source_message, existing_content)

        if edit.kind == "APPEND":
            if old.strip():
                raise MemoryValidationError("APPEND must have empty old_text")
            if _is_contiguous_quote(new, content):
                continue  # Repeated fact: no change.
            colors = {"black", "white", "red", "blue", "green", "yellow",
                      "orange", "purple", "pink", "brown", "gray", "grey",
                      "silver", "gold", "beige", "maroon", "navy", "teal"}
            old_colors = set(_words(content)) & colors
            new_colors = set(_words(new)) & colors
            if old_colors and new_colors and not new_colors.issubset(old_colors):
                raise MemoryValidationError(
                    "Cannot APPEND a conflicting color to an existing fact; REPLACE the old color"
                )
            old_hp = re.search(r"\b(\d+)\s*horsepower\b", " ".join(_words(content)))
            new_hp = re.search(r"\b(\d+)\s*horsepower\b", " ".join(_words(new)))
            if old_hp and new_hp and old_hp.group(1) != new_hp.group(1):
                raise MemoryValidationError(
                    "Cannot APPEND a contradictory horsepower value; REPLACE the old value"
                )
            if (re.search(r"\bCVT\b", content, flags=re.IGNORECASE)
                    and "manual" in _words(new)):
                raise MemoryValidationError(
                    "Cannot APPEND a conflicting transmission; REPLACE the old transmission"
                )
            content = content.rstrip(". ") + "; " + new.strip().rstrip(". ") + "."
        elif edit.kind == "REPLACE":
            if not old.strip():
                raise MemoryValidationError("REPLACE requires old_text")
            matches = list(re.finditer(re.escape(old), content, flags=re.IGNORECASE))
            if len(matches) != 1:
                raise MemoryValidationError(
                    f"Cannot safely replace {old!r}: expected one exact occurrence, found {len(matches)}"
                )
            _guard_protected_tokens(old, new, source_message)
            match = matches[0]
            content = content[:match.start()] + new + content[match.end():]
        else:
            raise MemoryValidationError(f"Unsupported edit kind: {edit.kind}")

    return content


class MemoryConsistency:
    def __init__(self, repository: MemoryRepository):
        self.repository = repository

    def prepare_plan(
        self,
        candidates: list[MemoryCandidate],
        existing_memories: list[MemoryRecord],
        plan: MemoryPlan,
        source_message: str,
    ) -> tuple[list[PreparedMemoryChange], list[UpdateResult]]:
        by_id = {record.id: record for record in existing_memories}
        by_key = {}
        for candidate in candidates:
            by_key.setdefault(candidate.key, set()).add(candidate.content)

        changes = []
        noops = []
        targeted = set()
        added_keys = set()

        for decision in plan.decisions:
            action = decision.action
            if action == "NOOP":
                record = by_id.get(decision.target_id)
                noops.append(UpdateResult(
                    action="NOOP", key=record.key if record else (decision.key or "unchanged"),
                    memory_id=record.id if record else None,
                    version=record.version if record else None,
                ))
                continue

            if action == "ADD":
                # Only the extractor supplies new memory content; the updater
                # is allowed to choose ADD, not to invent a new key or value.
                # If its key drifts, match the exact candidate content and
                # restore the candidate's canonical key.
                matches = []
                if decision.key in by_key:
                    if decision.new_content is None:
                        if len(by_key[decision.key]) == 1:
                            matches = [c for c in candidates if c.key == decision.key]
                    else:
                        matches = [c for c in candidates if c.key == decision.key
                                   and c.content == decision.new_content]
                    if not matches:
                        raise MemoryValidationError(
                            "ADD content must exactly match an extracted candidate"
                        )
                elif decision.new_content is not None:
                    matches = [c for c in candidates
                               if c.content == decision.new_content]
                elif len(candidates) == 1:
                    matches = [candidates[0]]

                if len(matches) != 1:
                    raise MemoryValidationError(
                        "Cannot unambiguously resolve ADD to an extracted candidate"
                    )
                candidate = matches[0]
                # Two different key spellings can still refer to the same
                # attribute ("horsepower" vs "engine.power", for example).
                # Do not let an ADD create conflicting copies of a property.
                parts = set(re.split(r"[._]", candidate.key.casefold()))
                if parts & {"vehicle", "car"}:
                    is_hp = ("horsepower" in parts or
                             {"engine", "power"}.issubset(parts))
                    is_transmission = bool(parts & {"transmission", "gearbox"})
                    for record in existing_memories:
                        record_parts = set(re.split(r"[._]", record.key.casefold()))
                        if not (record_parts & {"vehicle", "car"}):
                            continue
                        record_text = record.content.casefold()
                        hp_already_stored = (
                            "horsepower" in record_parts or
                            {"engine", "power"}.issubset(record_parts) or
                            bool(re.search(r"\b\d+\s*horsepower\b", record_text))
                        )
                        transmission_already_stored = (
                            bool(record_parts & {"transmission", "gearbox"}) or
                            "transmission" in record_text or
                            bool(re.search(r"\bCVT\b", record.content))
                        )
                        if (is_hp and hp_already_stored) or (is_transmission and transmission_already_stored):
                            raise MemoryValidationError(
                                "Vehicle attribute is already stored; UPDATE or NOOP instead of ADD"
                            )
                # Fragmented car-color/model candidates must not become separate
                # memories when the same user's vehicle is already represented
                # by a composite record. Replan as one UPDATE instead.
                normalized_key = candidate.key.casefold().split(".")
                car_fragment = (
                    any(part in {"car", "vehicle"} for part in normalized_key)
                    and normalized_key[-1] in {"color", "make", "model", "year"}
                )
                if car_fragment and any(
                    any(part in {"vehicle", "car"} for part in record.key.casefold().split("."))
                    for record in existing_memories
                ):
                    raise MemoryValidationError(
                        "Related vehicle memory already exists; update it rather than ADDing a fragment"
                    )
                if candidate.key in added_keys:
                    raise MemoryValidationError("Duplicate ADD key in a single interaction")
                _grounded_content(
                    candidate.content, source_message,
                    " ".join(record.content for record in existing_memories),
                )
                added_keys.add(candidate.key)
                changes.append(PreparedMemoryChange(
                    action="ADD", key=candidate.key, content=candidate.content,
                ))
                continue

            if action not in ("UPDATE", "DELETE"):
                raise MemoryValidationError(f"Unknown operation {action}")
            if decision.target_id not in by_id:
                raise MemoryValidationError("Updater selected an unknown memory ID")
            if decision.target_id in targeted:
                raise MemoryValidationError("Same memory targeted more than once in one interaction")
            targeted.add(decision.target_id)
            record = by_id[decision.target_id]

            if action == "UPDATE":
                content = apply_grounded_edits(record.content, decision.edits, source_message)
                if content == record.content:
                    noops.append(UpdateResult(action="NOOP", key=record.key,
                                              memory_id=record.id, version=record.version))
                else:
                    changes.append(PreparedMemoryChange(
                        action="UPDATE", key=record.key, content=content,
                        target_id=record.id, expected_version=record.version,
                    ))
            else:
                if decision.edits:
                    raise MemoryValidationError("DELETE cannot contain edits")
                # A deletion needs explicit user-authorized removal. A change
                # in color is UPDATE, not permission to delete the vehicle.
                if not re.search(
                    r"\b(?:no longer|not anymore|don't have|do not have|"
                    r"don't own|do not own|no car|no vehicle|sold|gave away|"
                    r"remove|delete|forget|lost|crashed)\b",
                    source_message, flags=re.IGNORECASE,
                ):
                    raise MemoryValidationError(
                        "DELETE requires an explicit user retraction or removal"
                    )
                changes.append(PreparedMemoryChange(
                    action="DELETE", key=record.key,
                    target_id=record.id, expected_version=record.version,
                ))

        return changes, noops

    def prepare_preference_update(
        self,
        candidate: MemoryCandidate,
        records: list[MemoryRecord],
        source_message: str,
    ) -> tuple[list[PreparedMemoryChange], list[UpdateResult]] | None:
        if not records or not candidate.key.startswith(("user.likes.", "user.dislikes.")):
            return None
        _, polarity, subject = candidate.key.split(".", 2)
        verb = "like" if polarity == "likes" else "dislike"
        if _words(candidate.content) != ["the", "user", verb, subject]:
            return None
        source = " ".join(_words(source_message))
        if not re.fullmatch(
            rf"(?:actually )?(?:forget everything )?i "
            rf"(?:(?:really|{verb}) )*{verb} {re.escape(subject)}",
            source,
        ):
            return None
        _grounded_content(candidate.content, source_message)

        # Preserve the canonical record when one exists. Retire any other
        # versions of this subject in the same transaction as the correction.
        keeper = min(records, key=lambda record: (record.key != candidate.key, record.id))
        changes = [
            PreparedMemoryChange(
                action="DELETE", key=record.key, target_id=record.id,
                expected_version=record.version,
            )
            for record in records if record.id != keeper.id
        ]
        if keeper.key == candidate.key and is_represented(candidate, [keeper]):
            noops = [] if changes else [UpdateResult(
                action="NOOP", key=keeper.key, memory_id=keeper.id, version=keeper.version,
            )]
            return changes, noops
        changes.append(PreparedMemoryChange(
            action="UPDATE", key=candidate.key, content=candidate.content,
            target_id=keeper.id, expected_version=keeper.version,
        ))
        return changes, []

    def prepare_favorite_animal_update(
        self,
        candidate: MemoryCandidate,
        record: MemoryRecord,
        source_message: str,
    ) -> PreparedMemoryChange | None:
        if candidate.key not in {"user.favorite_animal.first", "user.favorite_animal.second"}:
            return None
        if normalized_key(record.key, record.content) != candidate.key:
            return None

        rank = "second " if candidate.key.endswith(".second") else "(?:first )?"
        source = " ".join(_words(source_message))
        if not re.search(rf"\bmy {rank}favorite animal\b", source):
            return None
        _grounded_content(candidate.content, source_message)
        return PreparedMemoryChange(
            action="UPDATE", key=record.key, content=candidate.content,
            target_id=record.id, expected_version=record.version,
        )

    def prepare_vehicle_update(
        self,
        candidates: list[MemoryCandidate],
        related: list[MemoryRecord],
        source_message: str,
    ) -> list[PreparedMemoryChange] | None:
        """Apply unambiguous current-vehicle changes directly from user facts."""
        if len(related) != 1:
            return None
        record = related[0]
        if normalized_key(record.key, record.content) != "user.vehicle":
            return None

        content = record.content
        patterns = {
            "user.vehicle.year": r"\b(?:19|20)\d{2}\b",
            "user.vehicle.color": r"\b(?:" + "|".join(sorted(COLORS)) + r")\b",
            "user.vehicle.horsepower": r"\b\d+\s*horse[ -]?power\b",
        }
        for candidate in candidates:
            if candidate.key == "user.vehicle":
                # A fully stated vehicle identity can replace the old vehicle
                # in place. Attributes of the previous model no longer apply.
                colors = "|".join(sorted(COLORS))
                ownership = (
                    rf"The user(?: owns|'s (?:vehicle|car) is) an? "
                    rf"(?:(?:{colors}) )?((?:19|20)\d{{2}}) ([\w -]+)"
                )
                previous = re.fullmatch(ownership, content.split(";", 1)[0].rstrip("."), re.I)
                replacement = re.fullmatch(ownership, candidate.content.rstrip("."), re.I)
                if not previous or not replacement:
                    return None
                year, name = replacement.groups()
                if len(name.split()) < 2 or not re.search(
                    r"\b(?:I\s+(?:own|have)|my\s+(?:car|vehicle)\s+is)\s+an?\s+"
                    + re.escape(f"{year} {name}") + r"\b", source_message, re.I,
                ):
                    return None
                _grounded_content(candidate.content, source_message)
                if previous[2].casefold() == name.casefold():
                    content = content[:previous.start(1)] + year + content[previous.end(1):]
                else:
                    content = candidate.content
                continue

            pattern = patterns.get(candidate.key)
            if pattern is None:
                return None
            proposed = re.findall(pattern, candidate.content, re.I)
            stored = list(re.finditer(pattern, content, re.I))
            if len(proposed) != 1 or len(stored) > 1:
                return None
            value = proposed[0]
            if candidate.key == "user.vehicle.horsepower":
                amount = _words(value)[0]
                source = " ".join(_words(source_message))
                if not re.search(rf"\b(?:{amount} horsepower|horsepower is {amount})\b", source):
                    return None
            elif not _is_contiguous_quote(value, source_message):
                return None
            if stored:
                old = stored[0]
                if _words(old[0]) != _words(value):
                    content = content[:old.start()] + value + content[old.end():]
            else:
                label = candidate.key.rsplit(".", 1)[-1]
                phrase = f"has {value}" if label == "horsepower" else f"{label} is {value}"
                content = f"{content.rstrip('. ')}; {phrase}."

        if content == record.content:
            return None
        return [PreparedMemoryChange(
            action="UPDATE", key=record.key, content=content,
            target_id=record.id, expected_version=record.version,
        )]

    def apply_plan(
        self,
        changes: list[PreparedMemoryChange],
        noops: list[UpdateResult],
        scope: str,
    ) -> list[UpdateResult]:
        if not changes:
            return noops
        # All writes + history rows share a single SQL transaction.
        return self.repository.apply_batch(changes, scope=scope) + noops