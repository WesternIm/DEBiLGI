"""Exact-lemma matching policy and explicit guards for lexical false friends."""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache

from debilgi.language.tokenizer import normalize_word

# Members of one group are intentionally close in spelling but not interchangeable.
# Extend this list when diagnostics expose another recurring collision.
FALSE_FRIEND_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"варенье", "варение"}),
    frozenset({"стать", "статья"}),
    frozenset({"вол", "вёл", "вел", "вести"}),
    frozenset({"компания", "кампания"}),
    frozenset({"адресат", "адресант"}),
    frozenset({"эффектный", "эффективный"}),
    frozenset({"невежа", "невежда"}),
    frozenset({"одеть", "надеть"}),
)


def false_friend_group(value: str) -> frozenset[str] | None:
    normalized = normalize_word(value)
    return next((group for group in FALSE_FRIEND_GROUPS if normalized in group), None)


def fuzzy_match_allowed(left: str, right: str) -> bool:
    """Disallow fuzzy equivalence between different members of a protected group."""

    normalized_left = normalize_word(left)
    normalized_right = normalize_word(right)
    if not normalized_left or not normalized_right or normalized_left == normalized_right:
        return bool(normalized_left and normalized_left == normalized_right)
    group = false_friend_group(normalized_left)
    return group is None or normalized_right not in group


def same_lexical_family(
    left: str,
    right: str,
    *,
    minimum_prefix: int = 4,
    minimum_similarity: float = 0.58,
) -> bool:
    """Compare related forms only after exact identity and false-friend protection."""

    normalized_left = normalize_word(left)
    normalized_right = normalize_word(right)
    if normalized_left == normalized_right:
        return bool(normalized_left)
    if not fuzzy_match_allowed(normalized_left, normalized_right):
        return False
    shared = 0
    for left_character, right_character in zip(
        normalized_left, normalized_right, strict=False
    ):
        if left_character != right_character:
            break
        shared += 1
    return shared >= minimum_prefix and SequenceMatcher(
        None, normalized_left, normalized_right
    ).ratio() >= minimum_similarity


@lru_cache(maxsize=8192)
def expressive_family(left: str, right: str) -> bool:
    """Restricted Russian derivation for affect, never identity of factual concepts.

    Productive evaluative suffixes and velar alternations explain such forms
    without adding a separate list of moods or slang insults.
    """
    def stems(word):
        word = normalize_word(word)
        values = {word}
        for ending in ("енький", "онький", "оватый", "еватый", "ость", "ный", "ной",
                       "кий", "но", "ара", "яра", "а", "я", "о"):
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                values.add(word[:-len(ending)])
        return {value[:-1] + {"к": "ч", "г": "ж", "х": "ш"}.get(value[-1], value[-1])
                for value in values if value}

    return same_lexical_family(left, right) or bool(stems(left) & stems(right))


@lru_cache(maxsize=8192)
def predicate_family(left: str, right: str) -> bool:
    """A prefixed verb is an analogy candidate, not a synonymous fact predicate."""
    from debilgi.language.capabilities import _ACTION_PREFIXES
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    if left == right:
        return True
    if not all(any(r.is_known and r.part_of_speech == "verb"
                   for r in DEFAULT_MORPHOLOGY.parse_word(word)) for word in (left, right)):
        return False
    return any(left == prefix + right or right == prefix + left for prefix in _ACTION_PREFIXES)


# Verbal prefixes (a closed morphological class, not words): prefixation turns an imperfective
# verb into its perfective partner («делать»→«с-делать»).
_VERBAL_PREFIXES = (
    "пере", "пред", "при", "про", "под", "над", "раз", "рас", "воз", "вос", "пре", "обо", "ото",
    "изо", "недо", "вз", "вс", "ис", "низ", "вы", "по", "на", "за", "об", "от", "из", "до", "со",
    "с", "у", "в", "о",
)


def _verb_aspects(word: str) -> set[str]:
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    return {
        "perfective" if "perf" in reading.grammemes else "imperfective"
        for reading in DEFAULT_MORPHOLOGY.parse_word(word)
        if reading.is_known and reading.part_of_speech == "verb"
        and ({"perf", "impf"} & set(reading.grammemes))
    }


def _verb_stem(lemma: str) -> str:
    """The lemma without reflexive particle and infinitive ending."""
    for ending in ("ться", "тись", "чься", "ть", "ти", "чь", "ся", "сь"):
        if lemma.endswith(ending) and len(lemma) - len(ending) >= 2:
            return lemma[: -len(ending)]
    return lemma


def _without_prefix(word: str) -> set[str]:
    return {word} | {
        word[len(prefix):]
        for prefix in _VERBAL_PREFIXES
        if word.startswith(prefix) and len(word) - len(prefix) >= 4
    }


@lru_cache(maxsize=8192)
def verb_family(left: str, right: str) -> bool:
    """One verb in any aspect, with or without a verbal prefix («летать», «лететь», «полететь»).

    Identity, an aspect pair, or two known verbs of the same reflexivity whose stems (with at
    most one verbal prefix peeled from each) share a start of ≥3 letters that is at least half
    the shorter stem, unless the false-friend protection separates them.
    """
    if left == right or aspect_pair(left, right):
        return True
    if left.endswith(("ся", "сь")) != right.endswith(("ся", "сь")):
        return False
    if not (_verb_aspects(left) and _verb_aspects(right)):
        return False
    if not fuzzy_match_allowed(normalize_word(left), normalize_word(right)):
        return False
    for first in _without_prefix(_verb_stem(left)):
        for second in _without_prefix(_verb_stem(right)):
            shared = 0
            for x, y in zip(first, second, strict=False):
                if x != y:
                    break
                shared += 1
            if shared >= 3 and shared * 2 >= min(len(first), len(second)):
                return True
    return False


@lru_cache(maxsize=8192)
def aspect_pair(left: str, right: str) -> bool:
    """Two lemmas of one verb in opposite aspect («просыпаться»/«проснуться»), by morphology.

    The analyzer must read one lemma as imperfective and the other as perfective (homonym
    readings are enough). They pair when the perfective is the imperfective with a verbal
    prefix, or when, with at most one verbal prefix peeled from each, they share a stem start
    of ≥3 letters that is at least half the shorter lemma (suffixal pairs -ыва-/-ива-/-а-/-я-
    and alternations) and the lexical false-friend protection allows it. Reflexivity must
    agree. Identity is not an aspect pair.
    """
    if left == right or (left.endswith(("ся", "сь")) != right.endswith(("ся", "сь"))):
        return False
    left_aspects, right_aspects = _verb_aspects(left), _verb_aspects(right)
    if not (
        "imperfective" in left_aspects
        and "perfective" in right_aspects
        or "perfective" in left_aspects
        and "imperfective" in right_aspects
    ):
        return False
    if any(
        left == prefix + right or right == prefix + left for prefix in _VERBAL_PREFIXES
    ):
        return True
    if not fuzzy_match_allowed(normalize_word(left), normalize_word(right)):
        return False
    for first in _without_prefix(_verb_stem(left)):
        for second in _without_prefix(_verb_stem(right)):
            shared = 0
            for x, y in zip(first, second, strict=False):
                if x != y:
                    break
                shared += 1
            if shared >= 3 and shared * 2 >= min(len(first), len(second)):
                return True
    return False
