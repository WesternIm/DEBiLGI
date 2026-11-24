"""Deterministic tokenizer for local Discord language statistics."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

_WORD_RE = re.compile(r"^[^\W_]+(?:[-'][^\W_]+)*$", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ROLE_RE = re.compile(r"<@&\d+>")
_USER_RE = re.compile(r"<@!?\d+>")
_CHANNEL_RE = re.compile(r"<#\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
_TOKEN_RE = re.compile(
    r"<(?:url|user|role|channel|emoji:[A-Za-z0-9_]+)>"
    r"|(?:[A-Za-zА-Яа-яЁё]\.){2,}[A-Za-zА-Яа-яЁё]?\.?"
    r"|\d+(?:(?:\.\d+)+|,\d+)?(?:[%‰])?"
    r"|[\w]+(?:['’\-][\w]+)*"
    r"|[^\s\w]",
    re.UNICODE,
)
_CHAT_JOIN_RE = re.compile(r"(?<!\w)всмысле(?!\w)")
# The enclitic question particle «ли» written with a hyphen («что-ли») or glued to the pronoun it
# follows («чтоли») is the same two words as «что ли» (4.46); the parser read the three spellings as
# a conjunction, a conjunction + particle and a guessed verb.
_CHAT_LI_RE = re.compile(r"(?<=\w)-ли(?!\w)|(?<!\w)(что|чё|че)ли(?!\w)")
_ELONGATED_RUN_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE | re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChatTokenNormalization:
    """Canonical form plus the expressive information removed from spelling."""

    surface: str
    canonical: str
    kind: str | None = None
    intensity: float = 0.0
    repetitions: int = 1


def normalize_discord_markup(text: str) -> str:
    text = _URL_RE.sub("<url>", text)
    text = _ROLE_RE.sub("<role>", text)
    text = _USER_RE.sub("<user>", text)
    text = _CHANNEL_RE.sub("<channel>", text)
    return _CUSTOM_EMOJI_RE.sub(lambda match: f"<emoji:{match.group(1).lower()}>", text)


def tokenize(text: str, *, max_tokens: int = 500, max_token_length: int = 100) -> tuple[str, ...]:
    if max_tokens <= 0 or max_token_length <= 0:
        raise ValueError("token limits must be positive")
    normalized = normalize_discord_markup(text).lower()
    normalized = _CHAT_JOIN_RE.sub("в смысле", normalized)
    from debilgi.config import thought_content_enabled

    if thought_content_enabled():  # (4.46) a reading, so flag 0 keeps the earlier tokens
        normalized = _CHAT_LI_RE.sub(
            lambda m: f"{m.group(1)} ли" if m.group(1) else " ли", normalized)
    tokens = _TOKEN_RE.findall(normalized)
    return tuple(token[:max_token_length] for token in tokens[:max_tokens])


def normalize_chat_token(
    token: str,
    *,
    is_known: Callable[[str], bool] | None = None,
) -> ChatTokenNormalization:
    """Normalize chat elongation/reduplication without losing its intensity."""

    if is_known is None:
        return _cached_chat_token(token)
    return _normalize_chat_token(token, is_known=is_known)


@lru_cache(maxsize=32768)
def _cached_chat_token(token: str) -> ChatTokenNormalization:
    return _normalize_chat_token(token)


def _normalize_chat_token(token, *, is_known=None):
    surface = token.casefold().replace("’", "'").strip("-' ")
    if not surface:
        return ChatTokenNormalization(token, surface)

    parts = tuple(part for part in surface.split("-") if part)
    if len(parts) >= 2 and len(set(parts)) == 1:
        canonical = _canonical_elongation(parts[0], is_known)
        return ChatTokenNormalization(
            token,
            canonical,
            "reduplication",
            min(1.0, 0.42 + 0.14 * (len(parts) - 1)),
            len(parts),
        )

    runs = tuple(_ELONGATED_RUN_RE.finditer(surface))
    if not runs:
        return ChatTokenNormalization(token, surface)
    canonical = _canonical_elongation(surface, is_known)
    longest = max(len(match.group(0)) for match in runs)
    removed = max(1, len(surface) - len(canonical))
    return ChatTokenNormalization(
        token,
        canonical,
        "elongation",
        min(1.0, 0.38 + longest * 0.06 + removed * 0.035),
        longest,
    )


def canonical_chat_tokens(text: str) -> tuple[str, ...]:
    """Tokenize a chat turn into comparison forms for acts and social signals."""

    return tuple(
        normalize_chat_token(token).canonical if token[:1].isalnum() else token
        for token in tokenize(text)
    )


def _canonical_elongation(
    word: str,
    is_known: Callable[[str], bool] | None,
) -> str:
    variants = elongation_variants(word)
    if is_known is not None:
        known = tuple(candidate for candidate in variants if is_known(candidate))
        if known:
            # Doubled consonants can be lexical; a doubled vowel is not
            # evidence for a governed pronoun in an isolated reaction.
            consonant = any(
                m.group(1) not in "аеёиоуыэюяaeiou" for m in _ELONGATED_RUN_RE.finditer(word)
            )
            return (max if consonant else min)(known, key=len)
    return min(variants, key=len)


def elongation_variants(word: str) -> tuple[str, ...]:
    """Keep both reductions available to contextual morphology."""
    runs = tuple(_ELONGATED_RUN_RE.finditer(word))
    if not runs:
        return (word,)
    variants = [""]
    cursor = 0
    for match in runs:
        prefix = word[cursor : match.start()]
        character = match.group(1)
        variants = [
            f"{variant}{prefix}{character * width}" for variant in variants for width in (1, 2)
        ]
        cursor = match.end()
    variants = [f"{variant}{word[cursor:]}" for variant in variants]
    return tuple(variants)


@lru_cache(maxsize=16384)
def normalize_word(word: str) -> str:
    return word.casefold().replace("’", "'").strip("-' ")


def is_word_form(word: str) -> bool:
    return bool(word and _WORD_RE.fullmatch(word))


def words_from_text(text: str) -> tuple[str, ...]:
    return tuple(word for token in tokenize(text) if is_word_form(word := normalize_word(token)))
