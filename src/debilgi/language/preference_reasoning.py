"""Grounded self-preferences with reusable reasons rather than random stances."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.character_markov import normalize_word
from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.personality.perspective import PersonalityPerspective

_APPRAISAL_DIMENSIONS: tuple[tuple[str, float, frozenset[str]], ...] = (
    (
        "curiosity",
        0.34,
        frozenset(
            {
                "астрономический",
                "история",
                "знание",
                "изменение",
                "искусство",
                "космос",
                "наука",
                "необычный",
                "панорама",
                "природа",
                "пространство",
                "развитие",
                "сложный",
                "творчество",
            }
        ),
    ),
    (
        "expressiveness",
        0.30,
        frozenset(
            {
                "выражение",
                "идеограмма",
                "пиктограмма",
                "символ",
                "смайлик",
                "сообщение",
                "эмоция",
                "язык",
            }
        ),
    ),
    (
        "usefulness",
        0.28,
        frozenset(
            {
                "защита",
                "инструмент",
                "общение",
                "помощь",
                "польза",
                "практический",
                "связь",
                "создание",
                "удобный",
            }
        ),
    ),
    (
        "harm",
        -0.42,
        frozenset(
            {
                "агрессия",
                "боль",
                "драка",
                "вред",
                "побои",
                "опасность",
                "разрушение",
                "риск",
                "смерть",
                "столкновение",
                "травма",
                "удар",
                "угроза",
            }
        ),
    ),
    (
        "coercion",
        -0.30,
        frozenset({"давление", "запрет", "контроль", "насилие", "принуждение"}),
    ),
)


@dataclass(frozen=True, slots=True)
class PreferenceAppraisal:
    topic: str
    stance: str
    score: float
    confidence: float
    reason_lemma: str | None
    dimension: str | None
    evidence: tuple[str, ...]


def appraise_preference(
    topic: str,
    evidence_texts: tuple[str, ...] = (),
    *,
    perspective: PersonalityPerspective | None = None,
) -> PreferenceAppraisal:
    """Appraise known properties using a small personality value profile.

    The profile contains semantic criteria, never answer wording.  With no
    matching evidence the result remains undecided instead of inventing a
    yes/no position from a hash.
    """

    topic_reading = DEFAULT_MORPHOLOGY.best(topic)
    topic_lemma = (
        topic_reading.lemma
        if topic_reading is not None and topic_reading.is_known
        else normalize_word(topic) or topic.casefold()
    )
    evidence_lemmas = tuple(
        dict.fromkeys(
            lemma
            for text in evidence_texts
            for lemma in content_lemmas(text)
            if lemma != topic_lemma
        )
    )
    matches: list[tuple[float, str, str]] = []
    for dimension, weight, cues in _APPRAISAL_DIMENSIONS:
        for lemma in evidence_lemmas:
            if lemma in cues:
                adjusted = weight
                if perspective is not None:
                    drive_name = {
                        "curiosity": "curiosity",
                        "expressiveness": "expressiveness",
                        "usefulness": "coherence",
                        "harm": "caution",
                        "coercion": "independence",
                    }.get(dimension, dimension)
                    drive = perspective.drive(drive_name)
                    # Positive criteria become more important when their
                    # corresponding drive is strong. Negative criteria use
                    # caution/independence as sensitivity, not a global rule.
                    adjusted = weight * (0.42 + drive * 0.86)
                matches.append((adjusted, dimension, lemma))
    perspective_appraisal = (
        perspective.appraise((*evidence_lemmas, topic_lemma))
        if perspective is not None
        else None
    )
    if (
        perspective_appraisal is not None
        and perspective_appraisal.dimensions
        and abs(perspective_appraisal.score) >= 0.12
    ):
        matches.append(
            (
                perspective_appraisal.score * 0.42,
                "personal_values",
                perspective_appraisal.terms[0],
            )
        )
    if not matches:
        return PreferenceAppraisal(
            topic_lemma,
            "undecided",
            0.0,
            0.42,
            None,
            None,
            ("no_appraisal_evidence",),
        )
    score = max(-1.0, min(1.0, sum(item[0] for item in matches)))
    strongest = max(matches, key=lambda item: abs(item[0]))
    stance = "like" if score >= 0.18 else "dislike" if score <= -0.18 else "undecided"
    confidence = min(0.92, 0.58 + len(matches) * 0.07)
    appraisal_evidence = tuple(
        dict.fromkeys(
            (
                *(f"appraisal:{dimension}:{lemma}" for _, dimension, lemma in matches),
                *(
                    perspective_appraisal.evidence
                    if perspective_appraisal is not None
                    else ()
                ),
            )
        )
    )
    return PreferenceAppraisal(
        topic_lemma,
        stance,
        score,
        confidence,
        strongest[2],
        strongest[1],
        appraisal_evidence,
    )


def preference_position_thought(appraisal: PreferenceAppraisal) -> ClauseThought:
    if appraisal.stance == "undecided":
        return ClauseThought(
            predicate=_first_person_verb("определить", "определил", tense="past"),
            dependencies=(
                SemanticDependency("subject", _speaker()),
                SemanticDependency(
                    "direct_object",
                    ConceptArgument(
                        ConceptLexeme("отношение", "отношение", "noun"),
                        grammatical_case="accusative",
                    ),
                ),
                SemanticDependency(
                    "topic",
                    ConceptArgument(
                        _topic_lexeme(appraisal.topic),
                        preposition="к",
                        grammatical_case="dative",
                    ),
                ),
            ),
            negated=True,
            confidence=appraisal.confidence,
            evidence=appraisal.evidence,
        )
    positive = appraisal.stance == "like"
    return ClauseThought(
        predicate=_first_person_verb("любить", "люблю"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    _topic_lexeme(appraisal.topic),
                    grammatical_case="accusative",
                ),
            ),
        ),
        negated=not positive,
        response_polarity=positive,
        focus_role="direct_object",
        confidence=appraisal.confidence,
        evidence=appraisal.evidence,
    )


def preference_reason_thought(
    appraisal: PreferenceAppraisal,
    *,
    correcting_negative_premise: bool = False,
) -> ClauseThought | None:
    if appraisal.reason_lemma is None or appraisal.stance == "undecided":
        return None
    positive = appraisal.stance == "like"
    return ClauseThought(
        predicate=ConceptLexeme(
            "привлекать" if positive else "отталкивать",
            "привлекает" if positive else "отталкивает",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(_topic_lexeme(appraisal.reason_lemma)),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "меня",
                        "pronoun",
                        (("case", "accusative"), ("number", "singular")),
                    ),
                    grammatical_case="accusative",
                ),
            ),
        ),
        response_polarity=False if positive and correcting_negative_premise else None,
        confidence=appraisal.confidence,
        evidence=appraisal.evidence,
    )


def _speaker() -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(
            "я",
            "я",
            "pronoun",
            (("number", "singular"), ("person", "first")),
        )
    )


def _topic_lexeme(topic: str) -> ConceptLexeme:
    reading = DEFAULT_MORPHOLOGY.best(topic)
    trusted_reading = reading if reading is not None and reading.is_known else None
    features = (
        tuple(
            (name, value)
            for name, value in (
                ("number", trusted_reading.number),
                ("gender", trusted_reading.gender),
                ("animacy", trusted_reading.animacy),
            )
            if value is not None
        )
        if trusted_reading is not None
        else (("indeclinable", "true"),)
    )
    return ConceptLexeme(
        trusted_reading.lemma if trusted_reading is not None else normalize_word(topic) or topic,
        topic,
        trusted_reading.part_of_speech if trusted_reading is not None else "noun",
        features,
    )


def _first_person_verb(
    lemma: str,
    surface: str,
    *,
    tense: str = "present",
) -> ConceptLexeme:
    features = [("tense", tense), ("number", "singular")]
    if tense == "past":
        features.append(("gender", "masculine"))
    else:
        features.append(("person", "first"))
    return ConceptLexeme(lemma, surface, "verb", tuple(features))
