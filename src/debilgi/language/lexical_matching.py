"""Exact-lemma matching policy and explicit guards for lexical false friends."""

from __future__ import annotations

from difflib import SequenceMatcher

from debilgi.language.character_markov import normalize_word

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
