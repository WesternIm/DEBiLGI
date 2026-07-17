"""Soft surface grammar learned from token-role sequences in local memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

_START = "start"
_END = "end"
_PUNCTUATION = frozenset(",.!?:;—-()[]{}«»…")
_PRONOUNS = frozenset(
    [
        "я",
        "ты",
        "он",
        "она",
        "оно",
        "мы",
        "вы",
        "они",
        "меня",
        "мне",
        "тебя",
        "тебе",
        "его",
        "ему",
        "её",
        "ей",
        "нас",
        "нам",
        "вас",
        "вам",
        "их",
        "им",
        "кто",
        "что",
        "который",
        "которая",
        "которые",
        "этот",
        "эта",
        "это",
        "эти",
        "тот",
        "та",
        "те",
        "себя",
        "свой",
        "моя",
        "твой",
    ]
)
_PREPOSITIONS = frozenset(
    [
        "в",
        "во",
        "на",
        "над",
        "под",
        "за",
        "из",
        "от",
        "до",
        "к",
        "ко",
        "у",
        "о",
        "об",
        "обо",
        "про",
        "для",
        "без",
        "при",
        "через",
        "между",
        "среди",
        "после",
        "перед",
        "вокруг",
        "возле",
        "вместо",
        "внутри",
        "вне",
        "с",
        "со",
        "по",
    ]
)
_CONJUNCTIONS = frozenset(
    [
        "и",
        "а",
        "но",
        "или",
        "либо",
        "потому",
        "поэтому",
        "если",
        "хотя",
        "чтобы",
        "когда",
        "пока",
        "будто",
        "словно",
        "зато",
        "однако",
        "ведь",
        "причем",
        "причём",
    ]
)
_PARTICLES = frozenset(
    [
        "не",
        "ни",
        "бы",
        "же",
        "ли",
        "вот",
        "даже",
        "уже",
        "ещё",
        "еще",
        "только",
        "просто",
        "именно",
        "типа",
        "типо",
    ]
)
_QUESTION_WORDS = frozenset(
    [
        "кто",
        "что",
        "где",
        "куда",
        "откуда",
        "когда",
        "зачем",
        "почему",
        "как",
        "какой",
        "какая",
        "какие",
    ]
)
_VERB_SUFFIXES = (
    "ться",
    "ить",
    "ать",
    "ять",
    "еть",
    "уть",
    "ешь",
    "ишь",
    "ете",
    "ите",
    "ется",
    "ится",
    "ются",
    "ятся",
    "ет",
    "ит",
    "ут",
    "ют",
    "ат",
    "ят",
    "ал",
    "ала",
    "али",
    "ило",
    "ing",
    "ed",
)
_ADJECTIVE_SUFFIXES = (
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ому",
    "ему",
    "ым",
    "им",
    "ом",
    "ем",
    "ую",
    "юю",
    "ой",
    "ей",
    "ых",
    "их",
)
_ROLES = (
    _END,
    "punctuation",
    "pronoun",
    "preposition",
    "conjunction",
    "particle",
    "question",
    "verb",
    "adjective",
    "number",
    "content",
)
_BAD_ENDINGS = frozenset({"preposition", "conjunction", "question"})


@dataclass(slots=True)
class SoftGrammar:
    """Probabilistic role transitions; no authored phrases or mandatory templates."""

    transitions: dict[str, Counter[str]]
    observations: int

    @classmethod
    def learn(cls, messages: tuple[tuple[str, ...], ...]) -> SoftGrammar:
        transitions: dict[str, Counter[str]] = defaultdict(Counter)
        observations = 0
        for tokens in messages:
            roles = tuple(role(token) for token in tokens)
            if not roles:
                continue
            previous = _START
            for current in (*roles, _END):
                transitions[previous][current] += 1
                previous = current
                observations += 1
        return cls(dict(transitions), observations)

    def probability(self, previous: str, current: str) -> float:
        options = self.transitions.get(previous, Counter())
        alpha = 0.35
        return (float(options.get(current, 0)) + alpha) / (
            float(sum(options.values())) + alpha * len(_ROLES)
        )

    def reweight(self, options: Counter[str], history: tuple[str, ...]) -> Counter[str]:
        previous = role(history[-1]) if history else _START
        baseline = 1.0 / len(_ROLES)
        weighted: Counter[str] = Counter()
        for token, count in options.items():
            ratio = self.probability(previous, role(token)) / baseline
            multiplier = max(0.35, min(2.40, ratio))
            weighted[token] = max(1, round(float(count) * multiplier * 100))
        return weighted

    def score(self, tokens: tuple[str, ...]) -> float:
        if not tokens:
            return 0.0
        roles = tuple(role(token) for token in tokens)
        previous = _START
        probabilities: list[float] = []
        for current in (*roles, _END):
            probabilities.append(self.probability(previous, current))
            previous = current
        baseline = 1.0 / len(_ROLES)
        transition_fit = sum(min(1.0, value / baseline) for value in probabilities) / len(
            probabilities
        )
        content_roles = {"content", "pronoun", "adjective", "number"}
        has_subject_matter = any(current in content_roles for current in roles)
        has_predicate = "verb" in roles or len(tokens) <= 5
        complete_ending = roles[-1] not in _BAD_ENDINGS
        completion = (
            float(has_subject_matter) * 0.40
            + float(has_predicate) * 0.30
            + float(complete_ending) * 0.30
        )
        function_chain = max(
            (
                sum(item in {"preposition", "conjunction", "particle"} for item in roles[i : i + 3])
                for i in range(max(1, len(roles) - 2))
            ),
            default=0,
        )
        penalty = 0.18 if function_chain >= 3 else 0.0
        return max(0.0, min(1.0, transition_fit * 0.62 + completion * 0.38 - penalty))


def role(token: str) -> str:
    normalized = token.casefold().replace("ё", "е")
    if token == "\0end":
        return _END
    if normalized in _PUNCTUATION:
        return "punctuation"
    if normalized.isdigit():
        return "number"
    if normalized in _QUESTION_WORDS:
        return "question"
    if normalized in _PRONOUNS:
        return "pronoun"
    if normalized in _PREPOSITIONS:
        return "preposition"
    if normalized in _CONJUNCTIONS:
        return "conjunction"
    if normalized in _PARTICLES:
        return "particle"
    if len(normalized) >= 4 and normalized.endswith(_VERB_SUFFIXES):
        return "verb"
    if len(normalized) >= 4 and normalized.endswith(_ADJECTIVE_SUFFIXES):
        return "adjective"
    return "content"
