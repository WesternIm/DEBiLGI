"""Grounded capability reasoning for explicit modal questions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.relation_reasoning import classify_object_terms
from debilgi.language.structured_utterance import StructuredUtterance

_CAPABILITY_PREDICATES = frozenset({"мочь", "уметь"})
_COMPUTATIONAL_ACTIONS = frozenset(
    {
        "анализировать",
        "выбирать",
        "думать",
        "оценивать",
        "оценить",
        "определять",
        "определить",
        "изучать",
        "искать",
        "объяснять",
        "обрабатывать",
        "обсуждать",
        "обсудить",
        "описывать",
        "описать",
        "отвечать",
        "писать",
        "написать",
        "помнить",
        "понимать",
        "понять",
        "продолжать",
        "продолжить",
        "проверять",
        "прочитать",
        "рассуждать",
        "рассказывать",
        "рассказать",
        "решать",
        "сказать",
        "сравнивать",
        "считать",
        "сообщать",
        "сообщить",
        "устанавливать",
        "установить",
        "строить",
        "учиться",
        "читать",
    }
)
_UNASSISTED_HUMAN_LIMITS = frozenset({"летать"})
_KNOWN_HUMAN_SUBJECTS = frozenset(
    {
        "человек",
        "люди",
        "астроном",
        "космонавт",
        "учёный",
        "врач",
        "учитель",
        "пилот",
    }
)
_NON_ACTION_STATES = frozenset(
    {
        "быть",
        "верить",
        "знать",
        "иметь",
        "казаться",
        "любить",
        "нравиться",
        "помнить",
        "понимать",
        "считать",
        "хотеть",
    }
)
_ABSTRACT_INTERNAL_OBJECTS = frozenset(
    {
        "взгляд",
        "идея",
        "мнение",
        "отношение",
        "оценка",
        "позиция",
        "предпочтение",
        "решение",
        "смысл",
    }
)
_EXTERNAL_ACTIONS = frozenset(
    {
        "бежать",
        "бить",
        "брать",
        "взять",
        "выкинуть",
        "держать",
        "есть",
        "закрыть",
        "идти",
        "копать",
        "летать",
        "нести",
        "носить",
        "открыть",
        "пить",
        "плыть",
        "поднять",
        "положить",
        "разбить",
        "съесть",
        "трогать",
        "увидеть",
        "упасть",
        "хватать",
    }
)
_OBJECT_SENSITIVE_EXTERNAL_ACTIONS = frozenset(
    {
        "купить",
        "покупать",
        "потерять",
        "ронять",
        "снять",
        "снимать",
        "терять",
        "уронить",
    }
)


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    subject_lemma: str
    subject_surface: str
    subject_kind: str
    action_lemma: str
    supported: bool
    reason: str
    confidence: float
    subject_number: str | None = None
    mode: str = "capability"
    action_object: ConceptArgument | None = None

    @property
    def patterns(self) -> tuple[DependencyPattern, ...]:
        if self.action_object is None or self.mode == "experience":
            return ()
        roles = (
            ("subject", "negation", "predicate", "action_complement", "direct_object")
            if not self.supported
            else ("subject", "predicate", "action_complement", "direct_object")
        )
        return (
            DependencyPattern(
                roles,
                confidence=0.97,
                evidence_count=8,
                source="russian_modal_action_structure",
                predicate_class="modal",
                negation_mode="required" if not self.supported else "forbidden",
            ),
        )

    @property
    def thought(self) -> ClauseThought:
        subject_lemma = "я" if self.subject_kind == "bot" else self.subject_lemma
        subject_surface = "я" if self.subject_kind == "bot" else self.subject_surface
        subject = ConceptArgument(
            ConceptLexeme(
                subject_lemma,
                subject_surface,
                "pronoun" if subject_lemma == "я" else "noun",
                (("number", "singular"), ("person", "first"))
                if subject_lemma == "я"
                else (
                    ("number", self.subject_number or "singular"),
                    ("person", "third"),
                ),
            )
        )
        if self.mode == "experience":
            dependencies = [SemanticDependency("subject", subject)]
            if self.action_object is not None:
                dependencies.append(
                    SemanticDependency("direct_object", self.action_object)
                )
            return ClauseThought(
                predicate=ConceptLexeme(
                    self.action_lemma,
                    self.action_lemma,
                    "verb",
                    (
                        ("tense", "past"),
                        ("number", "singular"),
                        ("gender", "masculine"),
                    ),
                ),
                dependencies=tuple(dependencies),
                negated=not self.supported,
                response_polarity=self.supported,
                confidence=self.confidence,
                evidence=(self.reason,),
            )
        predicate_lemma = (
            "мочь"
            if self.mode == "directive"
            else "уметь"
            if self.subject_kind == "bot"
            else "мочь"
        )
        predicate_surface = (
            "могу"
            if self.mode == "directive"
            else "умею"
            if self.subject_kind == "bot"
            else "может"
        )
        action = ConceptArgument(
            ConceptLexeme(self.action_lemma, self.action_lemma, "verb")
        )
        dependencies = [
            SemanticDependency("subject", subject),
            SemanticDependency("action_complement", action),
        ]
        if self.action_object is not None:
            dependencies.append(
                SemanticDependency("direct_object", self.action_object)
            )
        return ClauseThought(
            predicate=ConceptLexeme(
                predicate_lemma,
                predicate_surface,
                "verb",
                (
                    ("tense", "present"),
                    ("number", self.subject_number or "singular"),
                ),
            ),
            dependencies=tuple(dependencies),
            negated=not self.supported,
            response_polarity=(self.supported if self.mode != "directive" else None),
            confidence=self.confidence,
            evidence=(self.reason,),
            predicate_class="modal",
        )


def assess_capability_question(
    structure: StructuredUtterance,
    *,
    analyses: Sequence[WordAnalysis] = (),
) -> CapabilityAssessment | None:
    """Ground self-directed capability, experience and external-action requests."""

    for frame in reversed(structure.clauses):
        if frame.predicate_index is None:
            continue
        predicate = structure.tokens[frame.predicate_index]
        subject = (
            structure.tokens[frame.subject_indices[-1]]
            if frame.subject_indices
            else None
        )
        addressed_bot = frame.implicit_subject == "addressee" or bool(
            subject is not None and subject.lemma in {"ты", "вы"}
        )
        action_object = _action_object(
            structure,
            frame,
            subject.index if subject is not None else None,
        )
        requires_external_agency = action_requires_external_agency(
            predicate.lemma
        ) or _object_sensitive_external_action(
            predicate.lemma,
            action_object,
            analyses,
        )
        if predicate.lemma in _CAPABILITY_PREDICATES:
            action = next(
                (
                    token
                    for token in structure.tokens[frame.predicate_index + 1 : frame.token_end]
                    if token.part_of_speech == "verb" and "INFN" in token.grammemes
                ),
                None,
            )
            if action is None or subject is None:
                continue
            action_object = _action_object(structure, frame, subject.index)
        elif addressed_bot and frame.clause_kind == "command" and action_requires_external_agency(
            predicate.lemma
        ):
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "directive_requires_unavailable_external_agency",
                0.94,
                "singular",
                "directive",
                _action_object(structure, frame, None),
            )
        elif (
            addressed_bot
            and frame.clause_kind == "question"
            and dict(predicate.grammatical_features).get("tense") == "past"
            and requires_external_agency
        ):
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "past_action_requires_body_or_external_agency",
                0.95,
                "singular",
                "experience",
                action_object,
            )
        elif (
            addressed_bot
            and frame.clause_kind == "question"
            and dict(predicate.grammatical_features).get("tense")
            in {"present", "future"}
            and requires_external_agency
        ):
            # Russian requests are often phrased as a bare future-tense
            # question: ``откроешь бутылку?``.  This is an action decision,
            # not a request to retrieve some old fact containing ``открыл``.
            return CapabilityAssessment(
                "ты",
                "ты",
                "bot",
                predicate.lemma,
                False,
                "prospective_action_requires_unavailable_external_agency",
                0.95,
                "singular",
                "directive",
                action_object,
            )
        else:
            continue
        if subject.lemma in {"ты", "вы"}:
            abstract_internal_state = bool(
                action.lemma in _NON_ACTION_STATES
                and action_object is not None
                and action_object.head.lemma in _ABSTRACT_INTERNAL_OBJECTS
            )
            if action.lemma in _COMPUTATIONAL_ACTIONS:
                supported = True
                reason = "internal_computational_action"
            elif abstract_internal_state:
                supported = True
                reason = "internal_cognitive_or_evaluative_state"
            elif action_requires_external_agency(action.lemma):
                supported = False
                reason = "action_requires_unavailable_external_agency"
            else:
                # An unknown action is uncertainty, not evidence that a body
                # is required.  Let other candidates or study resolve it.
                continue
            return CapabilityAssessment(
                subject.lemma,
                subject.surface,
                "bot",
                action.lemma,
                supported,
                reason,
                0.96 if supported else 0.92,
                dict(subject.grammatical_features).get("number"),
                "capability",
                action_object,
            )
        if subject.lemma in _KNOWN_HUMAN_SUBJECTS and action.lemma in _UNASSISTED_HUMAN_LIMITS:
            return CapabilityAssessment(
                subject.lemma,
                subject.surface,
                "person",
                action.lemma,
                False,
                "human_requires_technology",
                0.94,
                dict(subject.grammatical_features).get("number"),
                "capability",
                action_object,
            )
    return None


def realize_capability_assessment(assessment: CapabilityAssessment | None) -> str | None:
    if assessment is None:
        return None
    surfaces = realize_thought(assessment.thought, assessment.patterns)
    return surfaces[0].text if surfaces else None


def action_requires_external_agency(action_lemma: str) -> bool:
    """Separate internal language work from bodily or world-changing actions."""

    return action_lemma.casefold() in _EXTERNAL_ACTIONS if action_lemma else False


def _object_sensitive_external_action(
    action_lemma: str,
    action_object: ConceptArgument | None,
    analyses: Sequence[WordAnalysis],
) -> bool:
    """Resolve embodied/polysemous actions from the object's learned type."""

    if (
        action_lemma.casefold() not in _OBJECT_SENSITIVE_EXTERNAL_ACTIONS
        or action_object is None
    ):
        return False
    object_lemma = action_object.head.lemma.casefold()
    terms: set[str] = set()
    for analysis in analyses:
        if (analysis.lemma or analysis.normalized_form).casefold() != object_lemma:
            continue
        explanations = (
            *analysis.dictionary_meanings,
            analysis.selected_meaning,
            analysis.local_meaning,
            analysis.personal_interpretation,
        )
        terms.update(
            term
            for explanation in explanations
            if explanation
            for term in content_lemmas(explanation)
        )
    object_class, _, _ = classify_object_terms(terms)
    return object_class == "physical"


def _action_object(structure, frame, subject_index: int | None) -> ConceptArgument | None:
    indices = frame.direct_object_indices
    if not indices:
        indices = tuple(
            token.index
            for token in structure.tokens[frame.token_start : frame.token_end]
            if token.index != subject_index
            and token.part_of_speech == "noun"
            and token.index != frame.predicate_index
        )
    if not indices:
        return None
    token = structure.tokens[indices[0]]
    return ConceptArgument(
        ConceptLexeme(
            token.lemma,
            token.corrected_form or token.surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        ),
        grammatical_case="accusative",
    )
