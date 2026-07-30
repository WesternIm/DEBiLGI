"""Answer questions about the agent's own observed action history."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.capabilities import action_requires_personal_world_history
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import StructuredUtterance


@dataclass(frozen=True, slots=True)
class PersonalActionFrame:
    action_lemma: str
    action_surface: str
    tense: str
    object: ConceptArgument | None
    complements: tuple[SemanticDependency, ...]
    negated: bool
    confidence: float

    @property
    def key_points(self) -> tuple[str, ...]:
        return (
            self.action_lemma,
            *((self.object.head.lemma,) if self.object is not None else ()),
            *(item.argument.head.lemma for item in self.complements),
        )


@dataclass(frozen=True, slots=True)
class PersonalActionResolution:
    frame: PersonalActionFrame
    actual: bool | None
    thought: ClauseThought | None
    confidence: float
    evidence: tuple[str, ...]


def analyze_personal_action(
    structure: StructuredUtterance,
) -> PersonalActionFrame | None:
    """Find a polar question about an action performed by the addressee."""

    clause = next(
        (
            item
            for item in structure.clauses
            if item.clause_kind == "question" and item.predicate_index is not None
        ),
        None,
    )
    if clause is None or clause.question_target_index is not None:
        return None
    addressed = clause.implicit_subject == "addressee" or any(
        structure.tokens[index].lemma in {"ты", "вы"}
        for index in clause.subject_indices
    )
    if not addressed:
        return None
    predicate = structure.tokens[clause.predicate_index]
    tense = dict(predicate.grammatical_features).get("tense")
    if predicate.part_of_speech != "verb" or tense not in {"past", "present"}:
        return None
    object_argument = (
        _argument(
            structure,
            clause.direct_object_indices[0],
            grammatical_case=_object_case(
                structure,
                clause.direct_object_indices[0],
            ),
        )
        if clause.direct_object_indices
        else None
    )
    complements = tuple(
        SemanticDependency(
            "topic",
            _argument(
                structure,
                complement.object_indices[0],
                postmodifier_indices=complement.object_indices[1:],
                preposition=structure.tokens[complement.preposition_index].lemma,
                grammatical_case=dict(
                    structure.tokens[complement.object_indices[0]].grammatical_features
                ).get("case"),
            ),
        )
        for complement in clause.complements
        if complement.object_indices
    )
    return PersonalActionFrame(
        action_lemma=predicate.lemma,
        action_surface=predicate.corrected_form or predicate.surface,
        tense=tense,
        object=object_argument,
        complements=complements,
        negated=clause.negated,
        confidence=min(0.94, clause.confidence * 0.78 + predicate.confidence * 0.16),
    )


def reason_about_personal_action(
    frame: PersonalActionFrame | None,
) -> PersonalActionResolution | None:
    """Resolve only actions covered by observable local embodiment boundaries."""

    if frame is None or not action_requires_personal_world_history(frame.action_lemma):
        return None
    evidence = (
        "local_action_history",
        "non_physical_runtime",
        f"unobserved_external_action:{frame.action_lemma}",
    )
    thought = ClauseThought(
        predicate=ConceptLexeme(
            frame.action_lemma,
            frame.action_surface,
            "verb",
            (
                ("tense", frame.tense),
                ("number", "singular"),
                *(
                    (("gender", "masculine"),)
                    if frame.tense == "past"
                    else (("person", "first"),)
                ),
            ),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            *(
                (SemanticDependency("direct_object", frame.object),)
                if frame.object is not None
                else ()
            ),
            *frame.complements,
        ),
        negated=True,
        response_polarity=False,
        confidence=min(0.94, frame.confidence * 0.82 + 0.12),
        evidence=evidence,
    )
    return PersonalActionResolution(
        frame=frame,
        actual=False,
        thought=thought,
        confidence=thought.confidence,
        evidence=evidence,
    )


def _argument(
    structure: StructuredUtterance,
    index: int,
    *,
    modifier_indices: tuple[int, ...] = (),
    postmodifier_indices: tuple[int, ...] = (),
    preposition: str | None = None,
    grammatical_case: str | None = None,
) -> ConceptArgument:
    token = structure.tokens[index]
    return ConceptArgument(
        ConceptLexeme(
            token.lemma,
            token.corrected_form or token.surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        ),
        modifiers=tuple(
            ConceptLexeme(
                structure.tokens[item].lemma,
                structure.tokens[item].corrected_form or structure.tokens[item].surface,
                structure.tokens[item].part_of_speech,
                structure.tokens[item].grammatical_features,
                structure.tokens[item].entity_kind,
            )
            for item in modifier_indices
        ),
        preposition=preposition,
        grammatical_case=grammatical_case,
        postmodifiers=tuple(
            ConceptLexeme(
                structure.tokens[item].lemma,
                structure.tokens[item].corrected_form or structure.tokens[item].surface,
                structure.tokens[item].part_of_speech,
                structure.tokens[item].grammatical_features,
                structure.tokens[item].entity_kind,
            )
            for item in postmodifier_indices
        ),
    )


def _object_case(
    structure: StructuredUtterance,
    index: int,
) -> str:
    observed = dict(structure.tokens[index].grammatical_features).get("case")
    return (
        observed
        if observed in {"dative", "genitive", "instrumental", "prepositional"}
        else "accusative"
    )
