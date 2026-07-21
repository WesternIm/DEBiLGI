"""Reason about addressed cognitive states and emit semantic thoughts only."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.content import content_lemmas
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.sentence_understanding import UtteranceUnderstanding
from debilgi.language.structured_utterance import ClauseFrame, StructuredUtterance

_STATE_PREDICATES = {
    "understanding": frozenset({"понять", "понимать", "разобраться", "осознать", "уловить"}),
    "knowledge": frozenset({"знать"}),
    "memory": frozenset({"помнить", "вспомнить"}),
    "self_assessment": frozenset({"считать"}),
    # Open first-person-position requests such as ``что ты думаешь о себе``
    # are states of reflection, not fact questions about the verb ``думать``.
    "self_reflection": frozenset({"думать", "полагать", "оценивать", "оценить"}),
    "self_location": frozenset({"находиться", "располагаться", "быть"}),
}
_NON_ATTESTED_SELF_TRAITS = frozenset({"осознанный", "сознательный"})


@dataclass(frozen=True, slots=True)
class SelfStateFrame:
    state: str
    predicate_lemma: str
    predicate_surface: str
    predicate_features: tuple[tuple[str, str], ...]
    experiencer_kind: str
    target: ConceptArgument | None
    clause_kind: str
    negated: bool
    complete: bool
    confidence: float

    @property
    def target_lemmas(self) -> tuple[str, ...]:
        if self.target is None:
            return ()
        return tuple(
            dict.fromkeys(
                (
                    self.target.head.lemma,
                    *(item.lemma for item in self.target.modifiers),
                )
            )
        )


@dataclass(frozen=True, slots=True)
class SelfStateResolution:
    thought: ClauseThought | None
    state: str
    actual: bool | None
    confidence: float
    evidence: tuple[str, ...]


def analyze_self_state(structure: StructuredUtterance) -> SelfStateFrame | None:
    frame = max(structure.clauses, key=lambda item: item.confidence, default=None)
    if frame is None or frame.predicate_index is None:
        return None
    predicate = structure.tokens[frame.predicate_index]
    state = next(
        (name for name, lemmas in _STATE_PREDICATES.items() if predicate.lemma in lemmas),
        None,
    )
    if state is None:
        return None
    question_targets = {
        structure.tokens[item.question_target_index].lemma
        for item in structure.clauses
        if item.question_target_index is not None
    }
    if state == "self_location" and not question_targets & {"где", "куда"}:
        return None
    subject = structure.tokens[frame.subject_indices[0]] if frame.subject_indices else None
    experiencer_kind = (
        "self"
        if (subject is not None and subject.lemma in {"ты", "вы"})
        or (subject is None and frame.implicit_subject == "addressee")
        else "speaker"
        if subject is not None and subject.lemma in {"я", "мы"}
        else "other"
    )
    target = (
        _self_assessment_target(structure, frame)
        if state == "self_assessment"
        else _target_argument(structure, frame)
    )
    reflexive_self = any(
        token.lemma == "себя"
        for token in structure.tokens[frame.token_start : frame.token_end]
    )
    return SelfStateFrame(
        state=state,
        predicate_lemma=predicate.lemma,
        predicate_surface=predicate.surface,
        predicate_features=predicate.grammatical_features,
        experiencer_kind=experiencer_kind,
        target=target,
        clause_kind=frame.clause_kind,
        negated=frame.negated,
        complete=experiencer_kind != "other"
        and (
            target is not None
            or state == "understanding"
            or (state == "self_reflection" and reflexive_self)
            or state == "self_location"
        ),
        confidence=min(0.98, frame.confidence * 0.82 + predicate.confidence * 0.18),
    )


def reason_about_self_state(
    frame: SelfStateFrame,
    understanding: UtteranceUnderstanding,
    *,
    context_messages: tuple[str, ...] = (),
    knowledge_facts: tuple[str, ...] = (),
) -> SelfStateResolution:
    """Resolve truth from actual local state, never from a memorized reply."""

    if not frame.complete:
        return SelfStateResolution(None, frame.state, None, frame.confidence * 0.6, ())
    if frame.clause_kind == "statement":
        if frame.experiencer_kind == "speaker" and frame.state == "understanding" and frame.negated:
            thought = _repair_question(frame)
            return SelfStateResolution(
                thought,
                frame.state,
                False,
                min(0.94, frame.confidence),
                ("speaker_reported_misunderstanding",),
            )
        return SelfStateResolution(None, frame.state, None, frame.confidence * 0.6, ())
    if frame.experiencer_kind != "self":
        return SelfStateResolution(
            None,
            frame.state,
            None,
            frame.confidence * 0.64,
            ("foreign_internal_state",),
        )

    if frame.state == "self_location":
        return SelfStateResolution(
            _self_location_thought(frame),
            frame.state,
            True,
            min(0.94, frame.confidence * 0.84 + 0.12),
            ("discord_runtime_environment", "non_physical_agent_location"),
        )

    if frame.state == "self_reflection":
        return SelfStateResolution(
            _self_reflection_thought(frame),
            frame.state,
            True,
            min(0.92, frame.confidence * 0.84 + 0.10),
            ("local_learning_architecture", "open_self_reflection"),
        )

    if frame.state == "self_assessment":
        trait = frame.target.head.lemma if frame.target is not None else ""
        actual = bool(
            understanding.sufficient
            and frame.target is not None
            and trait not in _NON_ATTESTED_SELF_TRAITS
        )
        thought = _self_assessment_thought(frame, actual)
        return SelfStateResolution(
            thought,
            frame.state,
            actual,
            min(0.94, frame.confidence * 0.84 + 0.12),
            (
                "local_self_assessment",
                "trait_not_supported_by_self_model"
                if trait in _NON_ATTESTED_SELF_TRAITS
                else "structural_parse_sufficient",
            ),
        )

    target_terms = set(frame.target_lemmas)
    contextual_terms = {
        term for message in context_messages[-12:] for term in content_lemmas(message)
    }
    knowledge_terms = {
        term for fact in knowledge_facts[:24] for term in content_lemmas(fact)
    }
    known_terms = set(understanding.known_terms) | knowledge_terms
    evidence: list[str] = []
    actual: bool | None
    if frame.state == "understanding":
        contextual_target = bool(target_terms & contextual_terms)
        target_known = not target_terms or target_terms <= known_terms
        actual = bool(understanding.sufficient and target_known and contextual_target)
        if not contextual_target:
            actual = None
            evidence.append("target_not_grounded_in_dialogue")
        else:
            evidence.extend(("complete_structural_parse", "target_in_dialogue_context"))
    elif frame.state == "knowledge":
        actual = bool(target_terms and target_terms <= known_terms)
        evidence.append(
            "target_in_local_knowledge" if actual else "target_absent_from_local_knowledge"
        )
    else:
        actual = bool(target_terms and target_terms & contextual_terms)
        evidence.append(
            "target_in_dialogue_memory" if actual else "target_absent_from_dialogue_memory"
        )

    if actual is None:
        return SelfStateResolution(
            None,
            frame.state,
            None,
            frame.confidence * 0.72,
            tuple(evidence),
        )
    thought = _answer_thought(frame, actual)
    return SelfStateResolution(
        thought,
        frame.state,
        actual,
        min(0.96, frame.confidence * 0.82 + 0.14),
        tuple(evidence),
    )


def _answer_thought(frame: SelfStateFrame, actual: bool) -> ClauseThought:
    subject = ConceptArgument(
        ConceptLexeme(
            "я",
            "я",
            "pronoun",
            (("case", "nominative"), ("number", "singular"), ("person", "first")),
        )
    )
    predicate_features = dict(frame.predicate_features)
    predicate_features["number"] = "singular"
    if predicate_features.get("tense") == "present":
        predicate_features["person"] = "first"
    dependencies = [SemanticDependency("subject", subject)]
    if frame.target is not None:
        dependencies.append(SemanticDependency("direct_object", frame.target))
    return ClauseThought(
        predicate=ConceptLexeme(
            frame.predicate_lemma,
            frame.predicate_surface,
            "verb",
            tuple(predicate_features.items()),
        ),
        dependencies=tuple(dependencies),
        negated=not actual,
        response_polarity=(actual != frame.negated),
        focus_role="direct_object" if frame.target is not None else None,
        confidence=frame.confidence,
        evidence=(f"resolved_state:{frame.state}",),
    )


def _repair_question(frame: SelfStateFrame) -> ClauseThought:
    subject = ConceptArgument(
        ConceptLexeme(
            "что",
            "что",
            "pronoun",
            (("case", "nominative"), ("number", "singular"), ("gender", "neuter")),
        )
    )
    complement = ConceptArgument(
        ConceptLexeme(
            "непонятный",
            "непонятно",
            "adjective",
            (("case", "instrumental"), ("number", "singular"), ("gender", "neuter")),
        )
    )
    return ClauseThought(
        predicate=ConceptLexeme(
            "остаться",
            "осталось",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "neuter")),
        ),
        dependencies=(
            SemanticDependency("subject", subject),
            SemanticDependency("predicate_complement", complement),
        ),
        clause_kind="question",
        focus_role="subject",
        confidence=frame.confidence,
        evidence=("request_missing_understanding_focus",),
    )


def _self_assessment_thought(frame: SelfStateFrame, actual: bool) -> ClauseThought:
    assert frame.target is not None
    return ClauseThought(
        predicate=ConceptLexeme(
            "считать",
            "считаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("case", "nominative"), ("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("себя", "себя", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency("predicate_complement", frame.target),
        ),
        negated=not actual,
        response_polarity=actual,
        focus_role="predicate_complement",
        confidence=frame.confidence,
        evidence=("resolved_state:self_assessment",),
    )


def _self_reflection_thought(frame: SelfStateFrame) -> ClauseThought:
    """Describe the agent's real process as dependencies, not a stored reply."""

    return ClauseThought(
        predicate=ConceptLexeme(
            "считать",
            "считаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("case", "nominative"), ("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("себя", "себя", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "predicate_complement",
                ConceptArgument(
                    ConceptLexeme(
                        "собеседник",
                        "собеседник",
                        "noun",
                        (("number", "singular"), ("gender", "masculine")),
                    ),
                    modifiers=(
                        ConceptLexeme(
                            "обучающийся",
                            "обучающийся",
                            "participle",
                        ),
                    ),
                    grammatical_case="instrumental",
                ),
            ),
        ),
        focus_role="predicate_complement",
        confidence=frame.confidence,
        evidence=("self_reflection_from_learning_process",),
        predicate_class="lexical",
    )


def _self_location_thought(frame: SelfStateFrame) -> ClauseThought:
    """Describe the active runtime environment as a compositional proposition."""

    return ClauseThought(
        predicate=ConceptLexeme(
            "работать",
            "работаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("case", "nominative"), ("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "location",
                ConceptArgument(
                    ConceptLexeme("discord", "Discord", "noun", entity_kind="platform"),
                    preposition="в",
                    grammatical_case="prepositional",
                ),
            ),
        ),
        focus_role="location",
        confidence=frame.confidence,
        evidence=("self_location_from_runtime_environment",),
        predicate_class="lexical",
    )


def _self_assessment_target(
    structure: StructuredUtterance,
    frame: ClauseFrame,
) -> ConceptArgument | None:
    reflexive = any(
        structure.tokens[index].lemma == "себя"
        for index in frame.direct_object_indices
    )
    if not reflexive:
        return None
    occupied = {
        *frame.subject_indices,
        *frame.direct_object_indices,
        *frame.indirect_object_indices,
    }
    target_index = next(
        (
            index
            for index in range(frame.token_end - 1, frame.token_start - 1, -1)
            if index not in occupied
            and index != frame.predicate_index
            and structure.tokens[index].part_of_speech in {"adjective", "noun", "participle"}
        ),
        None,
    )
    if target_index is None:
        return None
    token = structure.tokens[target_index]
    return ConceptArgument(
        ConceptLexeme(
            token.lemma,
            token.corrected_form or token.surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        )
    )


def _target_argument(
    structure: StructuredUtterance,
    frame: ClauseFrame,
) -> ConceptArgument | None:
    indices = (
        frame.direct_object_indices
        or frame.predicate_complement_indices
        or frame.indirect_object_indices
    )
    if not indices:
        return None
    head_index = next(
        (
            index
            for index in reversed(indices)
            if structure.tokens[index].part_of_speech in {"noun", "pronoun", "number"}
        ),
        indices[-1],
    )
    head = structure.tokens[head_index]
    modifier_indices = set(index for index in indices if index != head_index)
    previous = head_index - 1
    while previous >= frame.token_start and structure.tokens[previous].part_of_speech in {
        "adjective",
        "participle",
    }:
        modifier_indices.add(previous)
        previous -= 1
    return ConceptArgument(
        ConceptLexeme(
            head.lemma,
            head.corrected_form or head.surface,
            head.part_of_speech,
            head.grammatical_features,
            head.entity_kind,
        ),
        tuple(
            ConceptLexeme(
                structure.tokens[index].lemma,
                structure.tokens[index].corrected_form or structure.tokens[index].surface,
                structure.tokens[index].part_of_speech,
                structure.tokens[index].grammatical_features,
                structure.tokens[index].entity_kind,
            )
            for index in sorted(modifier_indices)
        ),
    )
