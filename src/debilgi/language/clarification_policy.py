"""Recognize contact and keep lexical research out of questions to a speaker."""

from __future__ import annotations

import re

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize

CONTACT_WORDS = frozenset({"йо", "алло", "але", "алё", "эй", "ау", "привет", "здравствуй"})
_MEANING_QUESTION = re.compile(
    r"\bподразумеваешь\s+под\s+[«\"']?(.+?)[»\"']?\s*\?+$", re.IGNORECASE,
)
_DEFINITION_REQUEST = re.compile(
    r"(?:\bчто\s+(?:такое|значит|означает)\b|\bподразумеваешь\s+под\b|"
    r"\bобъясни\w*\b.*\b(?:слово|значение)\b)", re.IGNORECASE,
)


def requests_word_definition(text: str) -> bool:
    return bool(_DEFINITION_REQUEST.search(text)) and (
        text.rstrip().endswith("?") or "объясни" in text.casefold()
    )


def is_contact_request(prompt: str) -> bool:
    words = [t for t in tokenize(prompt) if t[:1].isalnum()]
    return bool(words) and len(words) <= 3 and any(w in CONTACT_WORDS for w in words) and all(
        w in CONTACT_WORDS or w in {"ну", "ты", "там", "да"} for w in words
    )


def lexical_question_target(text: str) -> str | None:
    match = _MEANING_QUESTION.search(text.strip())
    if match is None:
        return None
    target = match.group(1).strip(' «»"\'')
    # These questions request one lexical item, not an arbitrary quoted claim.
    if len(tokenize(target)) != 1 or not target[:1].isalpha():
        return None
    return DEFAULT_MORPHOLOGY.lemma(target) or target.casefold()


def clarification_rejection(text: str, allowed: tuple[str, ...]) -> str | None:
    if not allowed and requests_word_definition(text):
        return "autonomous_dictionary_lookup_required"
    target = lexical_question_target(text)
    if target is not None and target not in allowed:
        return "lexical_clarification_without_unresolved_gap"
    return None
