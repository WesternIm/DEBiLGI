"""Context-bound question answering without authored reply sentences.

The module distinguishes an actual knowledge gap from a follow-up about the
agent's previous stance.  It emits dependency thoughts only; surface wording
is still produced by the Russian realizer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.capabilities import action_requires_external_agency
from debilgi.language.content import content_lemmas
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
)

_EPISTEMIC_PREDICATES = frozenset(
    {
        "знать",
        "понимать",
        "понять",
        "помнить",
        "вспомнить",
        "разобраться",
        "определять",
        "определить",
        "устанавливать",
        "установить",
    }
)
_COGNITIVE_PROCESS_PREDICATES = frozenset(
    {"думать", "мыслить", "рассуждать", "анализировать", "решать"}
)
_PURPOSE_MARKERS = frozenset({"зачем", "для чего"})
_CAUSE_MARKERS = frozenset({"почему", "отчего"})
_QUESTION_LEMMAS = frozenset(
    {"как", "какой", "когда", "кто", "куда", "откуда", "почему", "сколько", "что", "где", "зачем"}
)


@dataclass(frozen=True, slots=True)
class ContextualQuestionResolution:
    """A selected conversational meaning, before Russian surface generation."""

    kind: str
    thought: ClauseThought
    confidence: float
    evidence: tuple[str, ...]
    antecedent: str | None = None
    target_terms: tuple[str, ...] = ()
    patterns: tuple[DependencyPattern, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerGapPlan:
    """One compositional recovery move competing with other recovery moves."""

    kind: str
    strategy: str
    thought: ClauseThought
    confidence: float
    evidence: tuple[str, ...]
    target_terms: tuple[str, ...] = ()
    patterns: tuple[DependencyPattern, ...] = ()


def reason_about_contextual_question(
    structure: StructuredUtterance,
    context_messages: tuple[str, ...] = (),
) -> ContextualQuestionResolution | None:
    """Resolve elliptical questions against the latest usable proposition."""

    if not _is_question(structure):
        return None
    lemmas = _all_lemmas(structure)
    question_markers = _question_markers(structure)
    antecedent = _latest_antecedent(structure.text, context_messages)
    antecedent_structure = (
        analyze_structured_utterance(antecedent) if antecedent is not None else None
    )
    target_terms = tuple(content_lemmas(antecedent))[-3:] if antecedent else ()

    correction = _internal_capability_correction(
        structure,
        antecedent_structure,
        question_markers,
    )
    if correction is not None:
        thought, correction_targets = correction
        evidence = (
            "cause_follow_up",
            "speaker_supplied_counterevidence",
            "internal_capability_reconsidered",
        )
        return ContextualQuestionResolution(
            "internal_capability_correction",
            replace(thought, evidence=evidence),
            0.95,
            evidence,
            antecedent,
            correction_targets,
        )

    if (
        question_markers & {"где", "куда", "откуда"}
        and antecedent_structure is not None
        and _main_predicate(structure) is not None
        and _main_predicate(structure) == _main_predicate(antecedent_structure)
        and not _antecedent_has_location(antecedent_structure)
    ):
        evidence = ("location_follow_up", "antecedent_location_unspecified")
        thought, patterns = _unspecified_location_thought(evidence)
        return ContextualQuestionResolution(
            "antecedent_location_unspecified",
            thought,
            0.92,
            evidence,
            antecedent,
            target_terms,
            patterns,
        )

    directive_restatement = (
        _restated_directive_thought(antecedent_structure)
        if antecedent_structure is not None
        and "что" in question_markers
        and _main_predicate(structure) in {"понять", "понимать"}
        else None
    )
    if directive_restatement is not None:
        thought, patterns = directive_restatement
        evidence = ("understanding_follow_up", "previous_directive_restored")
        return ContextualQuestionResolution(
            "restated_directive",
            thought,
            0.91,
            evidence,
            antecedent,
            target_terms,
            patterns,
        )

    if question_markers & _CAUSE_MARKERS and _antecedent_denies_external_action(
        antecedent_structure
    ):
        evidence = ("cause_follow_up", "external_action_requires_body")
        return ContextualQuestionResolution(
            "embodiment_cause",
            _no_body_thought(evidence),
            0.94,
            evidence,
            antecedent,
            target_terms,
        )

    if (
        question_markers & _CAUSE_MARKERS
        and antecedent_structure is not None
        and _main_predicate(structure) in {"любить", "нравиться", "предпочитать"}
        and _main_predicate(structure) == _main_predicate(antecedent_structure)
    ):
        evidence = ("preference_cause_follow_up", "preference_has_no_grounded_cause")
        return ContextualQuestionResolution(
            "preference_cause_unestablished",
            _insufficient_grounds_thought(evidence, include_answer_topic=False),
            0.93,
            evidence,
            antecedent,
            target_terms,
            (
                DependencyPattern(
                    ("location", "predicate", "subject"),
                    clause_kind="statement",
                    predicate_class="existential",
                    required_roles=("location", "predicate", "subject"),
                    negation_mode="forbidden",
                    confidence=0.99,
                    evidence_count=16,
                    source="russian_negative_existential_structure",
                ),
            ),
        )

    if _asks_for_cognitive_method(structure, lemmas, question_markers):
        evidence = ("cognitive_method_question", "local_reasoning_process")
        return ContextualQuestionResolution(
            "cognitive_method",
            _reasoning_process_thought(evidence),
            0.92,
            evidence,
            antecedent,
            target_terms,
        )

    if _asks_about_epistemic_desire(lemmas):
        evidence = ("volitional_question", "curiosity_drive")
        return ContextualQuestionResolution(
            "epistemic_desire",
            _learning_desire_thought(evidence),
            0.90,
            evidence,
            antecedent,
            target_terms,
        )

    if (
        antecedent_structure is not None
        and lemmas & {"если"}
        and _is_elliptical(structure)
    ):
        evidence = ("conditional_follow_up", "antecedent_restored")
        return ContextualQuestionResolution(
            "conditional_verification",
            _verification_thought(evidence),
            0.86,
            evidence,
            antecedent,
            target_terms,
        )

    if question_markers & _PURPOSE_MARKERS and _antecedent_has_epistemic_gap(
        antecedent_structure
    ):
        evidence = ("purpose_follow_up", "antecedent_epistemic_gap")
        return ContextualQuestionResolution(
            "epistemic_purpose",
            _non_fabrication_thought(evidence),
            0.90,
            evidence,
            antecedent,
            target_terms,
        )

    if question_markers & _CAUSE_MARKERS and (
        _asks_why_epistemic_state(structure)
        or _antecedent_has_epistemic_gap(antecedent_structure)
    ):
        evidence = ("cause_follow_up", "evidence_gap")
        return ContextualQuestionResolution(
            "epistemic_cause",
            _insufficient_grounds_thought(evidence),
            0.91,
            evidence,
            antecedent,
            target_terms,
        )
    return None


def epistemic_gap_for_question(
    structure: StructuredUtterance,
) -> ContextualQuestionResolution | None:
    """Return the strongest recovery plan for legacy single-result callers."""

    plans = epistemic_gap_candidates_for_question(structure)
    if not plans:
        return None
    selected = plans[0]
    return ContextualQuestionResolution(
        selected.kind,
        selected.thought,
        selected.confidence,
        selected.evidence,
        target_terms=selected.target_terms,
    )


def epistemic_gap_candidates_for_question(
    structure: StructuredUtterance,
) -> tuple[AnswerGapPlan, ...]:
    """Build several meaning-level recovery moves instead of one stock reply."""

    if not _is_question(structure):
        return ()
    target = _question_topic(structure)
    target_terms = (target.head.lemma,) if target is not None else ()
    markers = _question_markers(structure)
    base_evidence = ("answer_sources_exhausted", "no_grounded_claim")
    plans: list[AnswerGapPlan] = []
    if "кто" in markers and target is not None:
        evidence = (*base_evidence, "person_attribution_question")
        plans.append(
            AnswerGapPlan(
                "person_attribution_gap",
                "withhold_unsupported_label",
                _no_person_attribution_thought(target, evidence),
                0.78,
                evidence,
                target_terms,
                (
                    DependencyPattern(
                        (
                            "subject",
                            "direct_object",
                            "predicate_complement",
                            "negation",
                            "predicate",
                        ),
                        confidence=0.96,
                        evidence_count=8,
                        source="russian_negative_information_structure",
                    ),
                ),
            )
        )
    if "сколько" in markers:
        evidence = (*base_evidence, "quantity_question")
        plans.append(
            AnswerGapPlan(
                "quantity_gap",
                "withhold_unverified_number",
                _unknown_quantity_thought(evidence),
                0.76,
                evidence,
                target_terms,
                (_modal_action_pattern(),),
            )
        )
    if markers & {"где", "куда", "откуда"} and target is not None:
        evidence = (*base_evidence, "location_question")
        plans.append(
            AnswerGapPlan(
                "location_gap",
                "withhold_unobserved_location",
                _unknown_location_thought(target, evidence),
                0.77,
                evidence,
                target_terms,
                (_modal_action_pattern(),),
            )
        )
    if markers & (_CAUSE_MARKERS | _PURPOSE_MARKERS):
        evidence = (*base_evidence, "reason_question")
        plans.append(
            AnswerGapPlan(
                "reason_gap",
                "state_missing_grounds",
                _insufficient_grounds_thought(evidence, include_answer_topic=False),
                0.75,
                evidence,
                target_terms,
                (
                    DependencyPattern(
                        ("location", "predicate", "subject"),
                        clause_kind="statement",
                        predicate_class="existential",
                        required_roles=("location", "predicate", "subject"),
                        negation_mode="forbidden",
                        confidence=0.99,
                        evidence_count=16,
                        source="russian_negative_existential_structure",
                    ),
                ),
            )
        )
    for strategy, thought, confidence in (
        (
            "withhold_unverified_confirmation",
            _cannot_confirm_thought((*base_evidence, "confirmation_unavailable")),
            0.72,
        ),
        (
            "withhold_unverified_identification",
            _cannot_establish_thought((*base_evidence, "identification_unavailable")),
            0.70,
        ),
        *(
            (
                (
                    "name_missing_topic_evidence",
                    _missing_evidence_thought(
                        target,
                        (*base_evidence, "topic_evidence_missing"),
                    ),
                    0.68,
                ),
            )
            if not markers & {"кто", "сколько", "где", "куда", "откуда"}
            else ()
        ),
    ):
        plans.append(
            AnswerGapPlan(
                "topic_evidence_gap" if target is not None else "general_evidence_gap",
                strategy,
                thought,
                confidence,
                (*base_evidence, strategy),
                target_terms,
            )
        )
    return tuple(plans)


def _reasoning_process_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("анализировать", "анализирую"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(ConceptLexeme("контекст", "контекст", "noun")),
            ),
        ),
        confidence=0.92,
        evidence=evidence,
    )


def _unspecified_location_thought(
    evidence: tuple[str, ...],
) -> tuple[ClauseThought, tuple[DependencyPattern, ...]]:
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "указать",
            "указал",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
        ),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("направление", "направление", "noun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "location",
                ConceptArgument(
                    ConceptLexeme("фраза", "фразе", "noun"),
                    modifiers=(
                        ConceptLexeme("предыдущий", "предыдущей", "adjective"),
                    ),
                    preposition="в",
                    grammatical_case="prepositional",
                ),
            ),
        ),
        negated=True,
        confidence=0.92,
        evidence=evidence,
    )
    pattern = DependencyPattern(
        ("location", "subject", "negation", "predicate", "direct_object"),
        clause_kind="statement",
        predicate_class="lexical",
        required_roles=("subject", "predicate", "negation", "location"),
        optional_roles=("direct_object",),
        negation_mode="required",
        confidence=0.97,
        evidence_count=12,
        source="location_follow_up_repair",
    )
    return thought, (pattern,)


def _restated_directive_thought(
    antecedent: StructuredUtterance,
) -> tuple[ClauseThought, tuple[DependencyPattern, ...]] | None:
    frame = next(
        (item for item in reversed(antecedent.clauses) if item.clause_kind == "command"),
        None,
    )
    if frame is None or frame.predicate_index is None:
        return None
    action = antecedent.tokens[frame.predicate_index]
    object_index = next(iter(frame.direct_object_indices), None)
    if object_index is None:
        return None
    object_token = antecedent.tokens[object_index]
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "просить",
            "просишь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "ты",
                        "ты",
                        "pronoun",
                        (("number", "singular"), ("person", "second")),
                    )
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme(action.lemma, action.lemma, "verb")),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        object_token.lemma,
                        object_token.corrected_form or object_token.surface,
                        object_token.part_of_speech,
                        object_token.grammatical_features,
                        object_token.entity_kind,
                    ),
                    grammatical_case="accusative",
                ),
            ),
        ),
        confidence=0.91,
        evidence=("restated_previous_directive",),
    )
    pattern = DependencyPattern(
        ("subject", "predicate", "action_complement", "direct_object"),
        confidence=0.97,
        evidence_count=8,
        source="russian_request_content_structure",
    )
    return thought, (pattern,)


def _no_body_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme("нет", "нет", "predicate"),
        dependencies=(
            SemanticDependency("location", _speaker_location()),
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme("тело", "тела", "noun"),
                    grammatical_case="genitive",
                ),
            ),
        ),
        confidence=0.94,
        evidence=evidence,
        predicate_class="existential",
    )


def _learning_desire_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("хотеть", "хочу"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "это",
                        "это",
                        "pronoun",
                        (("case", "accusative"), ("number", "singular")),
                    ),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme("узнать", "узнать", "verb")),
            ),
        ),
        response_polarity=True,
        confidence=0.90,
        evidence=evidence,
    )


def _verification_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "требовать",
            "требует",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "third")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "это",
                        "это",
                        "pronoun",
                        (("case", "nominative"), ("number", "singular")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("проверка", "проверку", "noun"),
                    grammatical_case="genitive",
                ),
            ),
        ),
        confidence=0.86,
        evidence=evidence,
    )


def _non_fabrication_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("выдумывать", "выдумываю"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("ответ", "ответ", "noun"),
                    grammatical_case="accusative",
                ),
            ),
        ),
        negated=True,
        confidence=0.90,
        evidence=evidence,
    )


def _insufficient_grounds_thought(
    evidence: tuple[str, ...],
    *,
    include_answer_topic: bool = True,
) -> ClauseThought:
    dependencies: list[SemanticDependency] = [
        SemanticDependency("location", _speaker_location()),
        SemanticDependency(
            "subject",
            ConceptArgument(
                ConceptLexeme(
                    "основание",
                    "оснований",
                    "noun",
                    (("number", "plural"),),
                ),
                modifiers=(ConceptLexeme("достаточный", "достаточных", "adjective"),),
                grammatical_case="genitive",
            ),
        ),
    ]
    if include_answer_topic:
        dependencies.append(
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme("ответ", "ответ", "noun"),
                    preposition="для",
                    grammatical_case="genitive",
                ),
            )
        )
    return ClauseThought(
        predicate=ConceptLexeme("нет", "нет", "predicate"),
        dependencies=tuple(dependencies),
        confidence=0.91,
        evidence=evidence,
        predicate_class="existential",
    )


def _cannot_confirm_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return _negative_modal_thought("подтвердить", evidence)


def _cannot_establish_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return _negative_modal_thought("установить", evidence)


def _negative_modal_thought(
    action_lemma: str,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("мочь", "могу"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "это",
                        "это",
                        "pronoun",
                        (("case", "accusative"), ("number", "singular")),
                    ),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme(action_lemma, action_lemma, "verb")),
            ),
        ),
        negated=True,
        confidence=0.72,
        evidence=evidence,
        predicate_class="modal",
    )


def _unknown_quantity_thought(evidence: tuple[str, ...]) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("мочь", "могу"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("число", "число", "noun"),
                    modifiers=(ConceptLexeme("точный", "точное", "adjective"),),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme("назвать", "назвать", "verb")),
            ),
        ),
        negated=True,
        confidence=0.76,
        evidence=evidence,
        predicate_class="modal",
    )


def _unknown_location_thought(
    target: ConceptArgument,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("мочь", "могу"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    target.head,
                    (),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme("найти", "найти", "verb")),
            ),
        ),
        negated=True,
        confidence=0.77,
        evidence=evidence,
        predicate_class="modal",
    )


def _no_person_attribution_thought(
    target: ConceptArgument,
    evidence: tuple[str, ...],
) -> ClauseThought:
    return ClauseThought(
        predicate=_verb("называть", "называю"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("никто", "никого", "pronoun"),
                    grammatical_case="genitive",
                ),
            ),
            SemanticDependency(
                "predicate_complement",
                ConceptArgument(
                    target.head,
                    (),
                    grammatical_case="instrumental",
                ),
            ),
        ),
        negated=True,
        confidence=0.78,
        evidence=evidence,
    )


def _missing_evidence_thought(
    target: ConceptArgument | None,
    evidence: tuple[str, ...],
) -> ClauseThought:
    dependencies: list[SemanticDependency] = [
        SemanticDependency("location", _speaker_location()),
        SemanticDependency(
            "subject",
            ConceptArgument(
                ConceptLexeme(
                    "данные",
                    "данных",
                    "noun",
                    (("number", "plural"),),
                ),
                modifiers=(ConceptLexeme("достаточный", "достаточных", "adjective"),),
                grammatical_case="genitive",
            ),
        ),
    ]
    if target is not None:
        dependencies.append(
            SemanticDependency(
                "topic",
                ConceptArgument(
                    target.head,
                    (),
                    preposition="о",
                    grammatical_case="prepositional",
                ),
            )
        )
    return ClauseThought(
        predicate=ConceptLexeme("нет", "нет", "predicate"),
        dependencies=tuple(dependencies),
        confidence=0.74 if target is not None else 0.68,
        evidence=evidence,
        predicate_class="existential",
    )


def _asks_for_cognitive_method(
    structure: StructuredUtterance,
    lemmas: frozenset[str],
    question_markers: frozenset[str],
) -> bool:
    if "как" not in question_markers:
        return False
    predicate = _main_predicate(structure)
    if predicate not in _COGNITIVE_PROCESS_PREDICATES:
        return False
    return bool(lemmas & {"ты", "вы"}) or any(
        clause.implicit_subject == "addressee" for clause in structure.clauses
    )


def _asks_about_epistemic_desire(lemmas: frozenset[str]) -> bool:
    return "хотеть" in lemmas and bool(lemmas & (_EPISTEMIC_PREDICATES | {"узнать"}))


def _asks_why_epistemic_state(structure: StructuredUtterance) -> bool:
    predicate = _main_predicate(structure)
    return predicate in _EPISTEMIC_PREDICATES


def _antecedent_has_epistemic_gap(
    structure: StructuredUtterance | None,
) -> bool:
    if structure is None:
        return False
    lemmas = _all_lemmas(structure)
    has_gap_predicate = bool(lemmas & (_EPISTEMIC_PREDICATES | {"хватать", "ответ"}))
    has_negation = bool(lemmas & {"не", "нет", "недостаточно"}) or any(
        clause.negated for clause in structure.clauses
    )
    return has_gap_predicate and has_negation


def _antecedent_denies_external_action(
    structure: StructuredUtterance | None,
) -> bool:
    if structure is None:
        return False
    negated = bool(_all_lemmas(structure) & {"не", "нет"}) or any(
        clause.negated for clause in structure.clauses
    )
    if not negated:
        return False
    return any(
        token.part_of_speech == "verb"
        and action_requires_external_agency(token.lemma)
        for token in structure.tokens
    )


def _internal_capability_correction(
    current: StructuredUtterance,
    antecedent: StructuredUtterance | None,
    question_markers: frozenset[str],
) -> tuple[ClauseThought, tuple[str, ...]] | None:
    if antecedent is None or not question_markers & _CAUSE_MARKERS:
        return None
    antecedent_frame = next(
        (
            frame
            for frame in antecedent.clauses
            if frame.negated and frame.action_complement_indices
        ),
        None,
    )
    if antecedent_frame is None:
        return None
    action_index = antecedent_frame.action_complement_indices[0]
    action = antecedent.tokens[action_index]
    objects = tuple(
        antecedent.tokens[index]
        for index in antecedent_frame.direct_object_indices
        if index > action_index
    )
    abstract_targets = {"мнение", "отношение", "оценка", "позиция", "решение"}
    target = next((item for item in objects if item.lemma in abstract_targets), None)
    current_evidence = _all_lemmas(current) & {
        "анализировать",
        "выбирать",
        "думать",
        "оценивать",
        "отвечать",
        "рассуждать",
        "сравнивать",
    }
    if action.lemma not in {"иметь", "формировать", "выражать"}:
        return None
    if target is None or not current_evidence:
        return None
    thought = ClauseThought(
        predicate=_verb("мочь", "могу"),
        dependencies=(
            SemanticDependency("subject", _speaker()),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme(action.lemma, action.lemma, "verb")),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        target.lemma,
                        target.corrected_form or target.surface,
                        "noun",
                        target.grammatical_features,
                    ),
                    grammatical_case="accusative",
                ),
            ),
        ),
        response_polarity=True,
        confidence=0.95,
        predicate_class="modal",
    )
    return thought, (target.lemma, *sorted(current_evidence))


def _antecedent_has_location(structure: StructuredUtterance) -> bool:
    location_words = {
        "там",
        "тут",
        "здесь",
        "туда",
        "сюда",
        "домой",
        "внутри",
        "снаружи",
        "рядом",
        "около",
        "возле",
    }
    if _all_lemmas(structure) & location_words:
        return True
    return any(
        complement.role in {"location", "source", "destination"}
        for clause in structure.clauses
        for complement in clause.complements
    )


def _question_topic(structure: StructuredUtterance) -> ConceptArgument | None:
    candidates = [
        token
        for token in structure.tokens
        if token.part_of_speech in {"noun", "adjective"}
        and token.semantic_role not in {"function_word", "pragmatic_marker", "question_target"}
        and token.lemma not in _EPISTEMIC_PREDICATES
    ]
    if not candidates:
        return None
    head = next((item for item in reversed(candidates) if item.part_of_speech == "noun"), None)
    if head is None:
        head = candidates[-1]
    modifiers = tuple(
        ConceptLexeme(
            item.lemma,
            item.corrected_form or item.surface,
            item.part_of_speech,
            item.grammatical_features,
            item.entity_kind,
        )
        for item in candidates
        if item.index != head.index and item.part_of_speech == "adjective"
    )
    return ConceptArgument(
        ConceptLexeme(
            head.lemma,
            head.corrected_form or head.surface,
            head.part_of_speech,
            head.grammatical_features,
            head.entity_kind,
        ),
        modifiers,
    )


def _speaker() -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(
            "я",
            "я",
            "pronoun",
            (("case", "nominative"), ("number", "singular"), ("person", "first")),
        )
    )


def _speaker_location() -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(
            "я",
            "меня",
            "pronoun",
            (("number", "singular"), ("person", "first")),
        ),
        preposition="у",
        grammatical_case="genitive",
    )


def _verb(lemma: str, surface: str) -> ConceptLexeme:
    return ConceptLexeme(
        lemma,
        surface,
        "verb",
        (("tense", "present"), ("number", "singular"), ("person", "first")),
    )


def _modal_action_pattern() -> DependencyPattern:
    return DependencyPattern(
        ("subject", "negation", "predicate", "action_complement", "direct_object"),
        confidence=0.96,
        evidence_count=8,
        source="russian_modal_action_structure",
    )


def _main_predicate(structure: StructuredUtterance) -> str | None:
    clause = max(structure.clauses, key=lambda item: item.confidence, default=None)
    if clause is None or clause.predicate_index is None:
        return None
    return structure.tokens[clause.predicate_index].lemma


def _question_markers(structure: StructuredUtterance) -> frozenset[str]:
    markers = {
        token.lemma
        for token in structure.tokens
        if token.semantic_role == "question_target"
        or token.lemma in _QUESTION_LEMMAS
    }
    lemmas = {token.lemma for token in structure.tokens}
    if markers & {"какой", "что"} and lemmas & {"причина", "основание", "повод"}:
        markers.add("почему")
    if markers & {"какой", "что"} and lemmas & {"цель", "смысл"}:
        markers.add("зачем")
    return frozenset(markers)


def _all_lemmas(structure: StructuredUtterance) -> frozenset[str]:
    return frozenset(
        token.lemma for token in structure.tokens if token.semantic_role != "punctuation"
    )


def _is_question(structure: StructuredUtterance) -> bool:
    return any(clause.clause_kind == "question" for clause in structure.clauses)


def _is_elliptical(structure: StructuredUtterance) -> bool:
    return any(clause.elliptical for clause in structure.clauses)


def _latest_antecedent(text: str, messages: tuple[str, ...]) -> str | None:
    current = " ".join(text.casefold().split())
    return next(
        (
            message.strip()
            for message in reversed(messages[-10:])
            if message.strip() and " ".join(message.casefold().split()) != current
        ),
        None,
    )
