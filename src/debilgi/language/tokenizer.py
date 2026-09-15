"""Deterministic tokenizer for local Discord language statistics."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

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
    normalized = re.sub(r"(?<!\w)всмысле(?!\w)", "в смысле", normalized)
    tokens = _TOKEN_RE.findall(normalized)
    return tuple(token[:max_token_length] for token in tokens[:max_tokens])


def normalize_chat_token(
    token: str,
    *,
    is_known: Callable[[str], bool] | None = None,
) -> ChatTokenNormalization:
    """Normalize chat elongation/reduplication without losing its intensity."""

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
    runs = tuple(_ELONGATED_RUN_RE.finditer(word))
    if not runs:
        return word
    variants = [""]
    cursor = 0
    for match in runs:
        prefix = word[cursor : match.start()]
        character = match.group(1)
        variants = [
            f"{variant}{prefix}{character * width}"
            for variant in variants
            for width in (1, 2)
        ]
        cursor = match.end()
    variants = [f"{variant}{word[cursor:]}" for variant in variants]
    if is_known is not None:
        known = tuple(candidate for candidate in variants if is_known(candidate))
        if known:
            # Preserve a legitimate Russian double consonant when both the
            # one- and two-letter reductions happen to exist in a dictionary.
            return max(known, key=len)
    return min(variants, key=len)
