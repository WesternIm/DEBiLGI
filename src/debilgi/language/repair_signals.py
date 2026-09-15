"""Shared, conservative signals that the user is correcting a prior answer."""

from __future__ import annotations

import re

from debilgi.language.tokenizer import tokenize

_CLAIM_VERBS = frozenset(
    {
        "говорил",
        "говорила",
        "говорили",
        "написал",
        "написала",
        "написали",
        "сказал",
        "сказала",
        "сказали",
    }
)
_CONTRADICTION_MARKERS = (
    "противореч",
    "до этого ты",
    "до этого вы",
    "раньше ты",
    "раньше вы",
    "сам же сказал",
    "сама же сказала",
    "сами же сказали",
    "ты же сказал",
    "ты же говорила",
    "ты же говорил",
    "вы же сказали",
)
_PAST_ACTION = re.compile(r"^[а-я][а-я-]{2,}(?:л|ла|ло|ли)$", re.IGNORECASE)


def signals_prior_answer_correction(text: str) -> bool:
    """Return true only for language that points out a prior self-claim.

    The short elliptical form ``ты сказал делал`` is important here: it is a
    correction of the preceding denial, not neutral small talk.  Requiring a
    direct-address claim verb plus a reported past action keeps ordinary
    phrases such as ``ты сказал правду`` out of this class.
    """

    normalized = " ".join(text.casefold().replace("ё", "е").split())
    if any(marker in normalized for marker in _CONTRADICTION_MARKERS):
        return True
    words = tuple(token for token in tokenize(normalized) if token[:1].isalnum())
    # Discourse particles do not change who is being held accountable.
    while words and words[0] in {"но", "ведь", "а", "ну"}:
        words = words[1:]
    if (len(words) >= 4 and words[0] in {"ты", "вы"}
            and words[1] in _CLAIM_VERBS and words[2] == "что"):
        return True
    if len(words) < 3 or words[0] not in {"ты", "вы"}:
        return False
    try:
        claim_index = next(index for index, word in enumerate(words) if word in _CLAIM_VERBS)
    except StopIteration:
        return False
    reported = words[claim_index + 1 :]
    if reported[:1] == ("что",):
        reported = reported[1:]
    return bool(reported) and any(_PAST_ACTION.fullmatch(word) for word in reported)
