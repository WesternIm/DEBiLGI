"""Aggregate reply shapes without attaching opinions to authors."""

from __future__ import annotations

from collections.abc import Sequence


def build_reply_pattern(
    prompt_tokens: Sequence[str], response_tokens: Sequence[str], *, width: int = 3
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if width <= 0:
        raise ValueError("width must be positive")
    if not prompt_tokens or not response_tokens:
        return None
    return tuple(prompt_tokens[-width:]), tuple(response_tokens[:width])
