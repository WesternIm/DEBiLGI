"""Expose runtime observations as facts, never as response wording."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.structured_utterance import StructuredUtterance

_LAYOUT_TERMS = frozenset({"раскладка"})
_LANGUAGE_TERMS = frozenset({"язык", "русский", "английский"})
_ADDRESSEE_FORMS = frozenset({"ты", "вы", "тебя", "вас", "твой", "ваш"})


@dataclass(frozen=True, slots=True)
class RuntimeStateFrame:
    state: str
    target_lemmas: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    fact_id: str
    value: str
    confidence: float


@dataclass(frozen=True, slots=True)
class RuntimeStateResolution:
    frame: RuntimeStateFrame
    observations: tuple[RuntimeObservation, ...]
    confidence: float
    repairs_previous_answer: bool = False


def analyze_runtime_state(
    structure: StructuredUtterance,
) -> RuntimeStateFrame | None:
    """Recognize questions about interface state owned by this runtime."""

    if not any(clause.clause_kind == "question" for clause in structure.clauses):
        return None
    lemmas = tuple(
        token.lemma
        for token in structure.tokens
        if token.part_of_speech is not None
    )
    lemma_set = set(lemmas)
    addressed = bool(lemma_set & _ADDRESSEE_FORMS)
    asks_layout = bool(lemma_set & _LAYOUT_TERMS) and bool(
        lemma_set & _LANGUAGE_TERMS
    )
    if not addressed or not asks_layout:
        return None
    return RuntimeStateFrame(
        state="input_language_layout",
        target_lemmas=tuple(
            dict.fromkeys(
                lemma
                for lemma in lemmas
                if lemma in _LAYOUT_TERMS | _LANGUAGE_TERMS
            )
        ),
        confidence=min(0.96, structure.confidence * 0.82 + 0.14),
    )


def reason_about_runtime_state(
    frame: RuntimeStateFrame | None,
    *,
    repairs_previous_answer: bool = False,
) -> RuntimeStateResolution | None:
    """Publish observable state for an independent response planner."""

    if frame is None or frame.state != "input_language_layout":
        return None
    observations = (
        *(
            (
                RuntimeObservation(
                    "previous_answer.topic_match",
                    (
                        "predicate=ответить;subject=agent;topic=тема;"
                        "topic_preposition=по;topic_case=dative;"
                        "tense=past;gender=feminine;truth=false. "
                        "Предыдущий ответ агента был не по теме."
                    ),
                    0.99,
                ),
            )
            if repairs_previous_answer
            else ()
        ),
        RuntimeObservation(
            "runtime.keyboard_layout.observable",
            (
                "predicate=наблюдать;subject=agent;"
                "direct_object=раскладка;"
                "direct_object_modifiers=системный;"
                "tense=present;truth=false. "
                "Агент не способен наблюдать системную раскладку клавиатуры."
            ),
            1.0,
        ),
        RuntimeObservation(
            "message.language",
            (
                "predicate=приходить;subject=сообщение;"
                "manner=по-русски;tense=present;truth=true. "
                "Текущее сообщение пользователя написано по-русски."
            ),
            0.98,
        ),
    )
    return RuntimeStateResolution(
        frame=frame,
        observations=observations,
        confidence=min(0.95, frame.confidence * 0.84 + 0.10),
        repairs_previous_answer=repairs_previous_answer,
    )
