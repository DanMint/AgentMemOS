"""Small, deterministic fact-identity checks for the memory update pipeline.

Embeddings tell us that two sentences discuss similar topics; identity tells us
whether they describe the *same mutable fact*. This deliberately recognizes
common preference/vehicle patterns and falls back to exact keys elsewhere.
It is not a general-purpose natural-language entailment engine.
"""

import re

from src.memory.models import MemoryCandidate, MemoryRecord


PREFERENCE_SPELLINGS = {"turtel": "turtle", "turtels": "turtles"}

ADDITIONAL_VEHICLE_PREFIX = "user.vehicle.additional."
INVESTMENT_DEPOSIT_PREFIX = "user.financial.investment_deposit."

COLORS = frozenset({
    "black", "white", "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey", "silver", "gold",
    "beige", "maroon", "navy", "teal",
})

_STOP = frozenset({
    "a", "an", "the", "i", "my", "me", "you", "your", "user", "users",
    "theirs", "his", "her", "their", "it", "its", "this", "that",
    "has", "have", "had", "is", "are", "was", "were", "owns", "own",
    "with", "of", "to", "for", "at", "from", "by", "in", "on", "and",
    "now", "currently", "also", "named", "called", "car", "vehicle",
    "engine", "produces", "produce", "using", "uses", "used", "s", "does",
})


def normalize_grouped_numbers(value: str) -> str:
    return re.sub(
        r"(?<![\w,])\d{1,3}(?:,\d{3})+(?![\w,])",
        lambda match: match[0].replace(",", ""),
        value,
    )


def tokens(value: str) -> list[str]:
    value = normalize_grouped_numbers(value)
    value = re.sub(
        r"\b(?:do(?:es)? not|don['’]?t|doesn['’]?t)\s+like\b",
        "dislike", value, flags=re.I,
    )
    value = re.sub(r"\bhorse[ -]?power\b", "horsepower", value, flags=re.I)
    value = re.sub(r"\b(?:automobile|vehicle)\b", "car", value, flags=re.I)
    value = re.sub(r"\bsix[ -]speed\b", "six speed", value, flags=re.I)
    parts = re.findall(r"[\w]+", value.casefold())
    aliases = {
        "loves": "like", "love": "like", "likes": "like", "liked": "like",
        "dogs": "dog", "turtles": "turtle", "horses": "horse", "animals": "animal",
        "favourite": "favorite", "favourites": "favorite",
        "hates": "dislike", "hate": "dislike", "dislikes": "dislike",
    }
    corrected = [PREFERENCE_SPELLINGS.get(part, part) for part in parts]
    return [aliases.get(part, part) for part in corrected]


def meaningful(value: str) -> set[str]:
    return set(tokens(value)) - _STOP


def normalized_key(key: str, content: str = "", source_message: str = "") -> str:
    """Give a stable key to a few well-defined attribute aliases.

    Don't make up a preference subject when neither source nor extracted
    content identifies it. Other application domains keep extractor keys.
    """
    k = key.casefold().replace("_", ".").replace("-", ".")
    parts = set(filter(None, k.split(".")))
    c = content.casefold()
    source = source_message.casefold()

    if key.startswith(ADDITIONAL_VEHICLE_PREFIX):
        return key

    if "vehicle" in parts or "car" in parts:
        words = tokens(content)
        if (len(words) == 4 and words[:2] == ["the", "user"]
                and words[2] in {"like", "dislike"} and words[3] in {"car", "cars"}):
            return f"user.{words[2]}s.cars"
        if "horsepower" in parts or ("engine" in parts and "power" in parts):
            return "user.vehicle.horsepower"
        if "transmission" in parts or "gearbox" in parts:
            return "user.vehicle.transmission"
        if parts & {"color", "colour"}:
            return "user.vehicle.color"
        if "make" in parts:
            return "user.vehicle.make"
        if "model" in parts:
            return "user.vehicle.model"
        if "year" in parts:
            return "user.vehicle.year"
        if parts & {"plan", "purchase", "history", "previous"}:
            return key
        return "user.vehicle"

    if ("favorite" in parts or "favourite" in parts) and (
        "animal" in parts or "animal" in c or "animal" in source
    ):
        ordinal = "second" if "second" in parts or "second favorite" in c or "second favorite" in source else "first"
        return f"user.favorite_animal.{ordinal}"

    preference = re.fullmatch(r"user\.(?:likes|dislikes|preference)\.([\w-]+)", key, re.I)
    if preference:
        words = tokens(content)
        if len(words) == 4 and words[:2] == ["the", "user"] and words[2] in {"like", "dislike"}:
            return f"user.{words[2]}s.{words[3]}"

    if ("preference" in parts or "likes" in parts or "like" in parts
            or "love" in parts):
        # e.g. 'The user likes turtles' -> user.likes.turtle
        match = re.search(r"\b(?:likes?|loves?|enjoys?)\s+([\w-]+)", c)
        if not match:
            match = re.search(r"\bi\s+(?:like|love|enjoy)\s+([\w-]+)", source)
        if match:
            subject = tokens(match.group(1))[0]
            if subject not in _STOP:
                return f"user.likes.{subject}"

    return key


def normalize_candidate(candidate: MemoryCandidate, source_message: str) -> MemoryCandidate:
    key = normalized_key(candidate.key, candidate.content, source_message)
    content = candidate.content
    if key.startswith(("user.likes.", "user.dislikes.")):
        content = re.sub(
            r"\b(?:turtel|turtels)\b",
            lambda match: PREFERENCE_SPELLINGS[match[0].lower()],
            content, flags=re.I,
        )
    return candidate.model_copy(update={"key": key, "content": content})


def _attribute(key: str) -> str:
    if key == "user.vehicle" or key == "user.vehicle.identity":
        return "identity"
    if key.startswith("user.vehicle."):
        return key.removeprefix("user.vehicle.")
    return ""


def same_fact(candidate: MemoryCandidate, record: MemoryRecord) -> bool:
    """True only if record can hold the candidate's *same property*.

    A generic vehicle description may contain embedded sub-attributes; it is a
    destination for vehicle details, even before those details are stored.
    """
    ckey = normalized_key(candidate.key, candidate.content)
    rkey = normalized_key(record.key, record.content)
    if ckey == rkey:
        return True
    preference_prefixes = ("user.likes.", "user.dislikes.")
    if ckey.startswith(preference_prefixes) and rkey.startswith(preference_prefixes):
        return ckey.split(".", 2)[2] == rkey.split(".", 2)[2]
    if ckey.startswith("user.vehicle") and rkey.startswith("user.vehicle"):
        ca, ra = _attribute(ckey), _attribute(rkey)
        if ca == "identity" and ra == "identity":
            return True
        if ca in {"make", "model", "year", "color", "horsepower"} and ra == "identity":
            return True
        if ca == "transmission" and ra == "identity":
            return bool(re.search(r"\b(?:cvt|manual|automatic|dct|transmission)\b", record.content, re.I))
        if ca == "identity" and ra in {"horsepower", "transmission"}:
            return False
    return False


def is_represented(candidate: MemoryCandidate, records: list[MemoryRecord]) -> bool:
    """Conservative lexical check that a candidate is ALREADY present.

    Identity match prevents 'likes turtles' being swallowed by 'likes dogs'.
    Meaningful token inclusion prevents 'red' being swallowed by 'black'.
    """
    needed = meaningful(candidate.content)
    if not needed:
        return False
    for record in records:
        if not same_fact(candidate, record):
            continue
        stored = meaningful(record.content)
        if needed.issubset(stored):
            return True
        if candidate.key.startswith(INVESTMENT_DEPOSIT_PREFIX) and "deposited" in stored:
            # Repeating an earlier in-progress statement must not undo confirmation.
            if "depositing" in needed and (needed - {"depositing"}).issubset(stored):
                return True
    return False


def is_represented_in_text(candidate: MemoryCandidate, key: str, content: str) -> bool:
    synthetic = MemoryRecord(id=-1, key=key, content=content, version=1)
    return is_represented(candidate, [synthetic])