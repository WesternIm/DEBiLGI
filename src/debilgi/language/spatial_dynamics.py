"""Local deductions about motion through a bounded field of view.

This module produces a semantic thought, not a stored answer sentence.  The
Russian surface is assembled later by the productive NLG layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    ProductiveSurface,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    TokenInterpretation,
    analyze_structured_utterance,
)

_MOTION_PREDICATES = frozenset(
    {
        "бежать",
        "двигаться",
        "ехать",
        "идти",
        "катиться",
        "лететь",
        "мчаться",
        "плыть",
        "ползти",
    }
)
_APPEARANCE_PREDICATES = frozenset(
    {
        "въехать",
        "войти",
        "вплыть",
        "вползти",
        "влететь",
        "появиться",
    }
)
_BOUNDED_VIEW_LEMMAS = frozenset({"вид", "кадр", "монитор", "окно", "поле", "сцена", "экран"})
_DIRECTION_TO_SOURCE_EDGE = {
    "влево": "справа",
    "налево": "справа",
    "вправо": "слева",
    "направо": "слева",
    "вверх": "снизу",
    "вниз": "сверху",
}
_PERSONAL_PRONOUNS = frozenset({"он", "она", "оно", "они"})


@dataclass(frozen=True, slots=True)
class SpatialDynamicsResolution:
    thought: ClauseThought
    confidence: float
    focus_terms: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpatialDynamicsExplanation:
    effect: ClauseThought
    cause: ClauseThought
    confidence: float
    focus_terms: tuple[str, ...]
    evidence: tuple[str, ...]


def resolve_boundary_entry_question(
    structure: StructuredUtterance,
) -> SpatialDynamicsResolution | None:
    """Infer the entry edge opposite to an object's direction of motion."""

    question = next(
        (
            clause
            for clause in structure.clauses
            if clause.clause_kind == "question" and _asks_for_boundary_source(structure, clause)
        ),
        None,
    )
    if question is None:
        return None

    premise = next(
        (
            clause
            for clause in reversed(structure.clauses)
            if clause.clause_index < question.clause_index
            and clause.predicate_index is not None
            and structure.tokens[clause.predicate_index].lemma in _MOTION_PREDICATES
            and clause.subject_indices
        ),
        None,
    )
    if premise is None:
        return None
    direction_token = next(
        (
            token
            for token in structure.tokens[premise.token_start : premise.token_end]
            if token.lemma in _DIRECTION_TO_SOURCE_EDGE
        ),
        None,
    )
    if direction_token is None:
        return None

    premise_subject = structure.tokens[premise.subject_indices[0]]
    question_subject = (
        structure.tokens[question.subject_indices[0]] if question.subject_indices else None
    )
    if not _same_moving_entity(premise_subject, question_subject):
        return None

    edge = _DIRECTION_TO_SOURCE_EDGE[direction_token.lemma]
    predicate = structure.tokens[question.predicate_index]  # guarded by query check
    evidence = (
        f"motion_direction:{direction_token.lemma}",
        f"bounded_view_source_edge:{edge}",
        "entry_edge_opposes_motion_direction",
        f"referent:{premise_subject.lemma}",
    )
    thought = ClauseThought(
        predicate=_lexeme(predicate),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    _lexeme(premise_subject),
                    grammatical_case="nominative",
                ),
            ),
            SemanticDependency(
                "manner",
                ConceptArgument(ConceptLexeme(edge, edge, "adverb")),
            ),
        ),
        confidence=0.94,
        evidence=evidence,
    )
    return SpatialDynamicsResolution(
        thought=thought,
        confidence=0.94,
        focus_terms=(premise_subject.lemma, direction_token.lemma, edge),
        evidence=evidence,
    )


def resolve_boundary_entry_explanation(
    structure: StructuredUtterance,
    context_messages: tuple[str, ...],
) -> SpatialDynamicsExplanation | None:
    """Explain a prior entry-edge deduction when the user asks ``почему``."""

    if not _asks_for_previous_boundary_reason(structure):
        return None
    antecedent_resolution = next(
        (
            resolution
            for message in reversed(context_messages[-8:])
            if " ".join(message.casefold().split()) != " ".join(structure.text.casefold().split())
            if (
                resolution := resolve_boundary_entry_question(analyze_structured_utterance(message))
            )
            is not None
        ),
        None,
    )
    if antecedent_resolution is None:
        return None
    subject_lemma, direction, edge = antecedent_resolution.focus_terms
    subject = next(
        (
            dependency.argument.head
            for dependency in antecedent_resolution.thought.dependencies
            if dependency.role == "subject"
        ),
        None,
    )
    if subject is None:
        return None
    pronoun = _subject_pronoun(subject)
    evidence = (
        *antecedent_resolution.evidence,
        "follow_up_reason_request",
        "effect_explained_by_motion_direction",
    )
    cause = ClauseThought(
        predicate=ConceptLexeme(
            "двигаться",
            "движется",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(pronoun)),
            SemanticDependency(
                "manner",
                ConceptArgument(ConceptLexeme(direction, direction, "adverb")),
            ),
        ),
        confidence=0.94,
        evidence=evidence,
    )
    return SpatialDynamicsExplanation(
        antecedent_resolution.thought,
        cause,
        0.94,
        (subject_lemma, direction, edge),
        evidence,
    )


def realize_boundary_entry_explanation(
    explanation: SpatialDynamicsExplanation,
    patterns: tuple[DependencyPattern, ...] = (),
) -> tuple[ProductiveSurface, ...]:
    """Compose two productively realized propositions with a causal relation."""

    effects = realize_thought(explanation.effect, patterns, limit=2)
    causes = realize_thought(explanation.cause, patterns, limit=2)
    surfaces: list[ProductiveSurface] = []
    seen: set[str] = set()
    for effect in effects:
        for cause in causes:
            effect_body = effect.text.rstrip(" .!?…")
            cause_body = cause.text.rstrip(" .!?…")
            if not effect_body or not cause_body:
                continue
            cause_body = cause_body[:1].casefold() + cause_body[1:]
            text = f"{effect_body}, потому что {cause_body}."
            normalized = " ".join(text.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)
            surfaces.append(
                ProductiveSurface(
                    text,
                    (
                        *(f"effect:{role}" for role in effect.role_order),
                        "relation:cause",
                        *(f"cause:{role}" for role in cause.role_order),
                    ),
                    min(
                        explanation.confidence,
                        effect.confidence,
                        cause.confidence,
                    ),
                    "causal_clause_composition",
                )
            )
    return tuple(surfaces[:3])


def _asks_for_boundary_source(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> bool:
    if clause.predicate_index is None:
        return False
    predicate = structure.tokens[clause.predicate_index]
    if predicate.lemma not in _APPEARANCE_PREDICATES:
        return False
    tokens = structure.tokens[clause.token_start : clause.token_end]
    lemmas = {token.lemma for token in tokens}
    asks_source = "откуда" in lemmas or {"какой", "сторона"} <= lemmas
    return asks_source and bool(lemmas & _BOUNDED_VIEW_LEMMAS)


def _asks_for_previous_boundary_reason(structure: StructuredUtterance) -> bool:
    question = next(
        (clause for clause in structure.clauses if clause.clause_kind == "question"),
        None,
    )
    if question is None:
        return False
    lemmas = {token.lemma for token in structure.tokens[question.token_start : question.token_end]}
    return bool(lemmas & {"почему", "отчего"}) and bool(
        lemmas & {"оттуда", "там", "слева", "справа", "сверху", "снизу"}
    )


def _same_moving_entity(
    premise_subject: TokenInterpretation,
    question_subject: TokenInterpretation | None,
) -> bool:
    if question_subject is None:
        return True
    if premise_subject.lemma == question_subject.lemma:
        return True
    if question_subject.lemma not in _PERSONAL_PRONOUNS:
        return False
    premise_features = dict(premise_subject.grammatical_features)
    question_features = dict(question_subject.grammatical_features)
    for feature in ("number", "gender"):
        left = premise_features.get(feature)
        right = question_features.get(feature)
        if left is not None and right is not None and left != right:
            return False
    return True


def _subject_pronoun(subject: ConceptLexeme) -> ConceptLexeme:
    features = dict(subject.features)
    number = features.get("number", "singular")
    gender = features.get("gender")
    lemma = (
        "они"
        if number == "plural"
        else "она"
        if gender == "feminine"
        else "оно"
        if gender == "neuter"
        else "он"
    )
    return ConceptLexeme(
        lemma,
        lemma,
        "pronoun",
        tuple(
            (name, value)
            for name, value in (
                ("number", number),
                ("gender", gender),
                ("person", "third"),
            )
            if value is not None
        ),
    )


def _lexeme(token: TokenInterpretation) -> ConceptLexeme:
    return ConceptLexeme(
        token.lemma,
        token.corrected_form or token.surface,
        token.part_of_speech,
        token.grammatical_features,
        token.entity_kind,
    )
