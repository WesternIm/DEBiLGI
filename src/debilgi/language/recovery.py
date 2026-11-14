"""Minimal recovery surfaces for failures that cannot be reasoned through.

Content answers do not belong here.  Identity, self-state, capability,
relation, causal and lexical questions are handled by their semantic
reasoners.  This module retains only social protocol, explicit repair and
terminal lookup-failure language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from debilgi.language.abbreviations import (
    is_abbreviation_candidate,
    normalize_abbreviation,
)
from debilgi.language.content import content_lemmas
from debilgi.language.lexicon import UnknownKind, WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize
from debilgi.social.signals import APPROVAL, DISAPPROVAL

_WELLBEING_RE = re.compile(
    r"^(?:ну\s+)?(?:а\s+)?(?:как\s+(?:у\s+тебя\s+)?дела|как\s+ты|"
    r"как\s+жизнь|что\s+нового)[?!.…]*$",
    re.IGNORECASE,
)
_THANKS = frozenset({"спасибо", "благодарю", "пасиб", "спс"})
_CLARIFICATION_RE = re.compile(
    r"^(?:а\s+)?(?:чего|чё|че|что|в\s*смысле)(?:\s+(?:бля|блин|нахуй))?[?!.…]*[)\]]*$",
    re.IGNORECASE,
)
_ATTENTION_CALL_RE = re.compile(
    r"^(?:а+у+|ал+о+|ал+е+|эй+)[?!.…]*$",
    re.IGNORECASE,
)
_DIFFICULT_TO_EXPLAIN_RE = re.compile(
    r"^(?:это\s+)?(?:очень\s+)?сложно\s+(?:объяснить|описать|сказать)[?!.…]*$",
    re.IGNORECASE,
)
_DO_NOT_LAUGH_RE = re.compile(r"^не\s+смейся[?!.…]*$", re.IGNORECASE)
_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^ты\s+(?:(?:ведь|что)\s+)?(?P<negation>не\s+)?"
    r"(?P<trait>[а-яёa-z-]{2,40})(?:\s+теперь)?"
    r"(?:\s+что[\s-]*ли)?[?!.…]*$",
    re.IGNORECASE,
)
_REVERSED_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^(?P<trait>[а-яёa-z][а-яёa-z0-9_-]{1,47})\s+(?:это\s+)?ты[?!.…]*$",
    re.IGNORECASE,
)
_IMPLIED_SECOND_PERSON_PREDICATE_RE = re.compile(
    r"^(?P<trait>[а-яёa-z-]{3,40})(?:\s+что[\s-]*ли)?[?!.…]*$",
    re.IGNORECASE,
)
_SEMANTIC_SELF_STATE_TRAITS = frozenset(
    {"осознанный", "сознательный"}
)
_IDENTITY_QUESTION_RE = re.compile(
    r"^(?:а\s+)?(?:имя\s+какое\s+у\s+тебя|какое\s+у\s+тебя\s+имя|"
    r"как\s+тебя\s+зовут|у\s+тебя\s+есть\s+имя|кто\s+ты|"
    r"кем\s+ты\s+себя\s+считаешь|какой\s+у\s+тебя\s+ник|"
    r"твой\s+ник\s+какой|кто\s+я|как\s+меня\s+зовут|"
    r"какой\s+у\s+меня\s+ник|кто\s+тебе\s+пишет)[?!.…]*$",
    re.IGNORECASE,
)
_LEXICAL_LOOKUP_RE = re.compile(
    r"\b(?:значение\s+слова|что\s+(?:такое|значит|означает)"
    r"(?:\s+слово)?)\s+[«\"']?((?:[A-Za-zА-Яа-яЁё]\.){2,}|"
    r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_-]{1,47})",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    prompt: str
    analyses: tuple[WordAnalysis, ...] = ()
    facts: tuple[str, ...] = ()
    similar_words: tuple[tuple[str, tuple[str, ...]], ...] = ()
    context_messages: tuple[str, ...] = ()


def compose_dialogue_reply(evidence: RecoveryEvidence) -> str | None:
    """Return only an irreducible conversational protocol response."""

    normalized = _normalized(evidence.prompt)
    if _ATTENTION_CALL_RE.fullmatch(normalized):
        return "Я здесь. Говори."
    predicate = _SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    reversed_predicate = _REVERSED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    implied_predicate = _IMPLIED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized)
    if (
        predicate is not None
        or reversed_predicate is not None
        or (
            implied_predicate is not None
            and _is_implied_second_person_predicate(
                implied_predicate.group("trait")
            )
        )
    ):
        match = predicate or reversed_predicate or implied_predicate
        assert match is not None
        surface = match.group("trait")
        analysis = _analysis_for(surface, evidence.analyses)
        lemma = analysis.lemma if analysis and analysis.lemma else surface
        if _is_semantic_self_state_trait(lemma):
            return None
        negated = bool(
            predicate is not None
            and predicate.groupdict().get("negation")
        )
        label = f"не {lemma}" if negated else lemma
        return (
            f"Ты оцениваешь меня как «{label}». "
            "На каком ответе основана эта оценка?"
        )
    return None


def lexical_lookup_target(prompt: str) -> str | None:
    """Extract the lexical subject of an explicit definition request."""

    match = _LEXICAL_LOOKUP_RE.search(prompt)
    if match is None:
        return None
    target = match.group(1)
    return (
        normalize_abbreviation(target)
        if is_abbreviation_candidate(target)
        else target.casefold().replace("ё", "е")
    )


def is_social_routine(prompt: str) -> bool:
    """Identify only acts whose response is genuinely conventional."""

    normalized = _normalized(prompt)
    words = _words(normalized)
    return bool(
        _WELLBEING_RE.fullmatch(normalized)
        or _ATTENTION_CALL_RE.fullmatch(normalized)
        or _CLARIFICATION_RE.fullmatch(normalized)
        or _DIFFICULT_TO_EXPLAIN_RE.fullmatch(normalized)
        or _DO_NOT_LAUGH_RE.fullmatch(normalized)
        or (
            (predicate := _SECOND_PERSON_PREDICATE_RE.fullmatch(normalized))
            is not None
            and not _is_semantic_self_state_trait(predicate.group("trait"))
        )
        or (
            (reversed_predicate := _REVERSED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized))
            is not None
            and not _is_semantic_self_state_trait(
                reversed_predicate.group("trait")
            )
        )
        or (
            (implied := _IMPLIED_SECOND_PERSON_PREDICATE_RE.fullmatch(normalized))
            is not None
            and _is_implied_second_person_predicate(implied.group("trait"))
        )
        or words & _THANKS
        or (len(words) <= 6 and words & (APPROVAL | DISAPPROVAL))
    )


def is_clarification_request(prompt: str) -> bool:
    """Return whether a short turn challenges the preceding answer."""

    return _CLARIFICATION_RE.fullmatch(_normalized(prompt)) is not None


def compose_recovery(evidence: RecoveryEvidence) -> str:
    """Use evidence first and keep the remaining repair language minimal."""

    normalized = _normalized(evidence.prompt)
    words = _words(normalized)
    direct = compose_dialogue_reply(evidence)
    if direct is not None:
        return direct
    if _CLARIFICATION_RE.fullmatch(normalized):
        return "Предыдущий ответ не сработал. Сформулируй вопрос ещё раз."
    if _DIFFICULT_TO_EXPLAIN_RE.fullmatch(normalized):
        return "Скажи сначала, что произошло и что в этом главное."
    if _DO_NOT_LAUGH_RE.fullmatch(normalized):
        return "Хорошо, без шуток."
    if _WELLBEING_RE.fullmatch(normalized):
        return "Я в работе. А ты как?"
    if words & _THANKS:
        return "Пожалуйста."
    if len(words) <= 6 and words & APPROVAL:
        return "Понял."
    if len(words) <= 6 and words & DISAPPROVAL:
        return "Понял. Пересобираю ответ."

    fact = _best_fact(evidence.prompt, evidence.facts)
    if fact:
        return fact.rstrip()

    ambiguous = next(
        (
            item
            for item in evidence.analyses
            if item.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT
            and len(item.dictionary_meanings) > 1
        ),
        None,
    )
    if ambiguous is not None:
        variants = "; ".join(ambiguous.dictionary_meanings[:3])
        return (
            f"У «{ambiguous.surface_form}» несколько найденных значений: "
            f"{variants}. Уточни нужное."
        )

    unknown = next(
        (item for item in evidence.analyses if item.unknown_kind is not None),
        None,
    )
    if unknown is not None:
        similar = dict(evidence.similar_words).get(unknown.normalized_form, ())
        if similar:
            options = ", ".join(f"«{word}»" for word in similar[:3])
            return (
                f"После поиска для «{unknown.surface_form}» остались "
                f"ближайшие варианты: {options}. Уточни написание."
            )
        return unknown_word_fallback(unknown.surface_form, 0)

    selected = next(
        (item for item in evidence.analyses if item.selected_meaning),
        None,
    )
    if selected is not None:
        return f"«{selected.surface_form}» здесь означает: {selected.selected_meaning}."

    topics = content_lemmas(evidence.prompt)
    if topics:
        return f"Уточни, что именно ты спрашиваешь о «{topics[-1]}»."
    return "Сформулируй вопрос ещё раз одним полным предложением."


def unknown_word_fallback(word: str, variant: int) -> str:
    """Report a terminal lookup failure; ``variant`` remains API-compatible."""

    del variant
    safe_word = word.strip().strip("«»\"'")[:64] or "это слово"
    return f"Поиск не дал значения для «{safe_word}». Уточни написание."


def is_identity_question(prompt: str) -> bool:
    """Recognize identity prompts without prescribing their answer."""

    return _IDENTITY_QUESTION_RE.fullmatch(_normalized(prompt)) is not None


def _best_fact(prompt: str, facts: tuple[str, ...]) -> str | None:
    prompt_terms = set(content_lemmas(prompt))
    ranked = sorted(
        (fact.strip() for fact in facts if fact.strip()),
        key=lambda fact: (
            len(prompt_terms & set(content_lemmas(fact))),
            -len(fact),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else None


def _analysis_for(
    surface: str,
    analyses: tuple[WordAnalysis, ...],
) -> WordAnalysis | None:
    normalized = surface.casefold().replace("ё", "е")
    return next(
        (
            item
            for item in analyses
            if normalized
            in {
                item.normalized_form,
                (item.lemma or "").replace("ё", "е"),
            }
        ),
        None,
    )


def _is_implied_second_person_predicate(surface: str) -> bool:
    reading = DEFAULT_MORPHOLOGY.best(surface)
    return bool(
        reading is not None
        and reading.part_of_speech == "verb"
        and reading.person == "second"
    )


def _is_semantic_self_state_trait(surface: str) -> bool:
    reading = DEFAULT_MORPHOLOGY.best(surface)
    lemma = reading.lemma if reading is not None else surface.casefold()
    return lemma in _SEMANTIC_SELF_STATE_TRAITS


def _normalized(prompt: str) -> str:
    return " ".join(prompt.strip().casefold().replace("ё", "е").split())


def _words(normalized: str) -> set[str]:
    return {
        token
        for token in tokenize(normalized)
        if token[:1].isalnum()
    }
