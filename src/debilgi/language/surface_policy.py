"""Shared policy for text that describes internals instead of answering a user."""

from __future__ import annotations

import re

INTERNAL_SURFACE_MARKERS = (
    "получился несвязным",
    "удачный исход",
    "повышаю доверие",
    "уровень доверия",
    "внутреннее напряжение",
    "вектор реакции",
    "коэффициент уверенности",
    "тогда ответ бориса к словам",
    "команды узнай названного объекта",
    "связь пока неоднозначна",
    "уточни, что именно с этим произошло",
    "имя не попадать разговора в поиск",
    "по реплике вижу тему",
    "в памяти пока не хватает фактов",
    "значения и памяти. но не хватает фактов",
    "продолжаю прежнюю тему",
    "это продолжение темы",
    "internal state",
    "confidence score",
    "reward signal",
    "внутренний разбор",
    "кандидат ответа",
    "итоговый score",
    "очередь на изучение",
    "скрытый тест",
)

_INTERNAL_NARRATION_PATTERNS = (
    re.compile(
        r"^(?:этот\s+)?пользователь\b.{0,48}\b(?:спрашивает|переспрашивает|просит|"
        r"интересуется|реагирует|сообщает|утверждает|выражает|обращается)\b"
    ),
    re.compile(r"^(?:ответ|реплика)\s+(?:агента|пользователя|собеседника)\b"),
    # Raw evidence counters are internal diagnostics, not answers.  The
    # response layer must name the unresolved slot or refer to the question.
    re.compile(
        r"\b(?:(?:нет\s+|не\s+хватает\s+|недостаточно\s+)"
        r"(?:достаточн\w*\s+)?(?:данн|факт|сведен)\w*|"
        r"достаточн\w*\s+(?:данн|факт|сведен)\w*"
        r"\s+(?:нет|не\s+хватает)|"
        r"(?:данн|факт|сведен)\w*\s+недостаточно)\b"
    ),
    re.compile(r"\b(?:нет|не\s+хватает)\s+достаточн\w*\s+основан\w*\b"),
)


def internal_surface_reason(text: str) -> str | None:
    normalized = " ".join(text.casefold().replace("ё", "е").split())
    marker = next(
        (marker for marker in INTERNAL_SURFACE_MARKERS if marker in normalized),
        None,
    )
    if marker is not None:
        return marker
    pattern = next(
        (pattern.pattern for pattern in _INTERNAL_NARRATION_PATTERNS if pattern.search(normalized)),
        None,
    )
    return pattern
