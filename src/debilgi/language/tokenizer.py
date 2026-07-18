"""Deterministic tokenizer for local Discord language statistics."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ROLE_RE = re.compile(r"<@&\d+>")
_USER_RE = re.compile(r"<@!?\d+>")
_CHANNEL_RE = re.compile(r"<#\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
_TOKEN_RE = re.compile(
    r"<(?:url|user|role|channel|emoji:[A-Za-z0-9_]+)>"
    r"|[\w]+(?:['’\-][\w]+)*"
    r"|[^\s\w]",
    re.UNICODE,
)


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
    tokens = _TOKEN_RE.findall(normalized)
    return tuple(token[:max_token_length] for token in tokens[:max_tokens])
