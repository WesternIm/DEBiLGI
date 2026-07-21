"""Lexical evidence and surface affinity for reaction primitives, never response phrases."""

from __future__ import annotations

from debilgi.language.tokenizer import normalize_chat_token
from debilgi.social.models import PrimitiveVector

AFFIRMATION = frozenset(
    {
        "да",
        "ага",
        "верно",
        "согласен",
        "согласна",
        "поддерживаю",
        "принимаю",
        "подходит",
        "разумно",
        "точно",
        "правда",
        "ок",
        "okay",
        "yes",
    }
)
REFUSAL = frozenset(
    {
        "нет",
        "не",
        "никогда",
        "отказываюсь",
        "отклоняю",
        "отвергаю",
        "останавливаю",
        "недопустимо",
        "хватит",
        "нельзя",
        "прекрати",
        "no",
    }
)
APPROVAL = frozenset(
    {
        "хорошо",
        "отлично",
        "круто",
        "красава",
        "красавчик",
        "молодец",
        "норм",
        "нравится",
        "правильно",
        "логично",
        "логичный",
        "разумно",
        "разумный",
        "убедительно",
        "убедительный",
        "верно",
        "полезно",
        "польза",
        "рабочий",
        "удачно",
        "успех",
        "умница",
        "охуенно",
        "good",
        "great",
    }
)
DISAPPROVAL = frozenset(
    {
        "плохо",
        "ошибка",
        "неправильно",
        "сломано",
        "провал",
        "ужасно",
        "риск",
        "ненадёжно",
        "противоречит",
        "вредно",
        "bad",
        "wrong",
    }
)
CURIOSITY = frozenset(
    {
        "кто",
        "что",
        "где",
        "куда",
        "когда",
        "зачем",
        "почему",
        "как",
        "интересно",
        "любопытно",
        "узнать",
        "понять",
    }
)
AVERSION = frozenset(
    {
        "фу",
        "мерзко",
        "противно",
        "отвратительно",
        "ненавижу",
        "тошнит",
        "говно",
        "хуйня",
        "disgusting",
    }
)


def evidence_counts(tokens: tuple[str, ...]) -> dict[str, int]:
    token_set = {
        normalize_chat_token(token).canonical if token[:1].isalnum() else token
        for token in tokens
    }
    return {
        "affirmation": len(token_set & AFFIRMATION),
        "refusal": len(token_set & REFUSAL),
        "approval": len(token_set & APPROVAL),
        "disapproval": len(token_set & DISAPPROVAL),
        "curiosity": len(token_set & CURIOSITY),
        "aversion": len(token_set & AVERSION),
    }


def reaction_affinity(
    tokens: tuple[str, ...], vector: PrimitiveVector | None, speech_act: str
) -> float:
    if vector is None or not tokens:
        return 0.0
    counts = evidence_counts(tokens)
    has_question = "?" in tokens
    has_exclamation = "!" in tokens
    agreement_fit = min(1.0, counts["affirmation"] + counts["approval"] * 0.4)
    resistance_fit = min(1.0, counts["refusal"] + counts["disapproval"] * 0.35)
    approval_fit = min(1.0, counts["approval"] + counts["affirmation"] * 0.3)
    aversion_fit = min(1.0, counts["aversion"] + counts["disapproval"] * 0.25)
    curiosity_fit = min(1.0, counts["curiosity"] * 0.5 + float(has_question))
    arousal_fit = float(has_exclamation)
    score = (
        vector.agreement * agreement_fit
        + vector.resistance * resistance_fit
        + vector.approval * approval_fit
        + vector.aversion * aversion_fit
        + vector.curiosity * curiosity_fit
        + vector.arousal * arousal_fit * 0.5
    ) / 3.0
    if speech_act == "clarifying_question" and has_question:
        score += vector.curiosity * 0.25
    return min(1.0, score)
