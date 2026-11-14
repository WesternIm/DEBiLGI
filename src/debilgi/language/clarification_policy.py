"""Recognize contact and keep lexical research out of questions to a speaker."""

from __future__ import annotations

import re

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import normalize_chat_token, tokenize

CONTACT_WORDS = frozenset(
    {"йо", "ку", "алло", "але", "алё", "эй", "ау", "привет", "здравствуй"}
)
GREETING_WORDS = frozenset({"йо", "ку", "привет", "здравствуй"})
_MEANING_QUESTION = re.compile(
    r"\bподразумеваешь\s+под\s+[«\"']?(.+?)[»\"']?\s*\?+$",
    re.IGNORECASE,
)
_DEFINITION_REQUEST = re.compile(
    r"(?:\bчто\s+(?:такое|значит|означает)\b|\bподразумеваешь\s+под\b|"
    r"\bобъясни\w*\b.*\b(?:слово|значение)\b)",
    re.IGNORECASE,
)


def requests_word_definition(text: str) -> bool:
    return bool(_DEFINITION_REQUEST.search(text)) and (
        text.rstrip().endswith("?") or "объясни" in text.casefold()
    )


def is_contact_request(prompt: str) -> bool:
    # Contact calls often repeat a letter only twice ("ауу"). General spelling
    # normalization intentionally retains doubles in ordinary lexical words.
    def contact_form(word: str) -> str:
        return re.sub(r"(.)\1+", r"\1", normalize_chat_token(word).canonical)

    contacts = {contact_form(word) for word in CONTACT_WORDS}
    words = [contact_form(t) for t in tokenize(prompt) if t[:1].isalnum()]
    return (
        bool(words)
        and len(words) <= 3
        and any(w in contacts for w in words)
        and all(w in contacts or w in {"ну", "ты", "там", "да"} for w in words)
    )


def is_attention_call(prompt: str) -> bool:
    """Return whether a contact turn summons the agent instead of greeting it."""

    if not is_contact_request(prompt):
        return False

    def contact_form(word: str) -> str:
        return re.sub(r"(.)\1+", r"\1", normalize_chat_token(word).canonical)

    greetings = {contact_form(word) for word in GREETING_WORDS}
    words = [contact_form(token) for token in tokenize(prompt) if token[:1].isalnum()]
    return not any(word in greetings for word in words)


def lexical_question_target(text: str) -> str | None:
    match = _MEANING_QUESTION.search(text.strip())
    if match is None:
        return None
    target = match.group(1).strip(" «»\"'")
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
