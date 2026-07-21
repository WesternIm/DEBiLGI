"""Meaning-bound dialogue repair without exposing internal diagnostics.

This layer is reached only after ordinary reasoning and optional consultation
failed to produce a reply.  It asks about the actual unresolved semantic slot,
not about parser concepts such as "missing action", "candidate" or "facts".
"""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.language.pragmatics import PragmaticAnalysis
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.structured_utterance import ClauseFrame, StructuredUtterance
from debilgi.language.valency import predicate_valency


@dataclass(frozen=True, slots=True)
class ConversationalRepair:
    thought: ClauseThought
    reason: str
    truly_unresolved: bool
    patterns: tuple[DependencyPattern, ...] = ()

    def realize(self) -> str | None:
        surfaces = realize_thought(self.thought, self.patterns)
        return surfaces[0].text if surfaces else None


def repair_for_utterance(
    structure: StructuredUtterance,
    pragmatics: PragmaticAnalysis,
) -> ConversationalRepair:
    """Continue a understood turn or ask about one concrete missing slot."""

    clause = max(structure.clauses, key=lambda item: item.confidence, default=None)
    if clause is not None:
        valency_repair = _missing_valency_repair(structure, clause)
        if valency_repair is not None:
            return valency_repair
        personal_repair = _personal_nominal_repair(structure, clause)
        if personal_repair is not None:
            return personal_repair

    cue_repair = _pragmatic_repair(pragmatics)
    if cue_repair is not None:
        return cue_repair

    topic = _topic_argument(structure, clause)
    if topic is not None:
        return ConversationalRepair(
            _topic_intention_question(topic),
            "unresolved_topic_intention",
            True,
        )
    return ConversationalRepair(
        _generic_intention_question(),
        "unresolved_reference",
        True,
    )


def unknown_meaning_repair(word: str) -> ConversationalRepair:
    """Ask about an actually unresolved word after internal study failed."""

    safe = word.strip().strip("«»\"'")[:64] or "это"
    return ConversationalRepair(
        ClauseThought(
            predicate=ConceptLexeme(
                "значить",
                "значит",
                "verb",
                (("tense", "present"), ("number", "singular"), ("person", "third")),
            ),
            dependencies=(
                SemanticDependency(
                    "subject",
                    ConceptArgument(
                        ConceptLexeme(
                            "что",
                            "что",
                            "pronoun",
                            (("case", "nominative"), ("number", "singular")),
                        )
                    ),
                ),
                SemanticDependency(
                    "direct_object",
                    ConceptArgument(
                        ConceptLexeme(
                            safe.casefold(),
                            f"«{safe}»",
                            "noun",
                            entity_kind="quoted_expression",
                        )
                    ),
                ),
            ),
            clause_kind="question",
            focus_role="subject",
            confidence=0.78,
            evidence=("meaning_unknown_after_internal_study",),
        ),
        "unknown_word_meaning",
        True,
    )


def _missing_valency_repair(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> ConversationalRepair | None:
    if clause.predicate_index is None:
        return None
    predicate = structure.tokens[clause.predicate_index]
    profile = predicate_valency(predicate.lemma)
    if profile is None or not _speaker_is_experiencer(structure, clause):
        return None
    present = {
        *(("subject",) if clause.subject_indices or clause.implicit_subject else ()),
        *(("direct_object",) if clause.direct_object_indices else ()),
        *(("indirect_object",) if clause.indirect_object_indices else ()),
        *(item.role for item in clause.complements),
    }
    missing = next(
        (
            slot
            for slot in profile.slots
            if slot.required_in_isolated_fact and slot.role not in present
        ),
        None,
    )
    if missing is None or missing.role == "subject":
        return None
    role = (
        missing.role
        if missing.role in {"direct_object", "indirect_object", "topic", "location"}
        else "direct_object"
    )
    question = _question_argument(
        role,
        grammatical_case=missing.grammatical_case,
        prepositions=missing.prepositions,
    )
    thought = ClauseThought(
        predicate=ConceptLexeme(
            predicate.lemma,
            predicate.corrected_form or predicate.surface,
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency("subject", _addressee()),
            SemanticDependency(role, question),
        ),
        clause_kind="question",
        focus_role=role,
        confidence=min(0.92, clause.confidence + 0.18),
        evidence=(f"missing_valency:{predicate.lemma}:{missing.role}",),
    )
    return ConversationalRepair(
        thought,
        f"missing_semantic_role:{missing.role}",
        True,
    )


def _pragmatic_repair(pragmatics: PragmaticAnalysis) -> ConversationalRepair | None:
    function = next(
        (
            cue.function
            for cue in sorted(pragmatics.cues, key=lambda item: item.intensity, reverse=True)
            if cue.category.startswith("interjection")
        ),
        None,
    )
    predicate = (
        "рассмешить"
        if function == "amusement"
        else "смутить"
        if function in {"hesitation", "skepticism_or_disappointment"}
        else "удивить"
        if function
        in {"surprise_or_admiration", "surprise_delight_or_regret", "strong_impression"}
        else None
    )
    if predicate is None:
        return None
    thought = ClauseThought(
        predicate=ConceptLexeme(
            predicate,
            predicate,
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "neuter")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "что",
                        "что",
                        "pronoun",
                        (("case", "nominative"), ("number", "singular")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme(
                        "ты",
                        "тебя",
                        "pronoun",
                        (("number", "singular"), ("person", "second")),
                    ),
                    grammatical_case="accusative",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="subject",
        confidence=0.84,
        evidence=(f"pragmatic_cue:{function}",),
    )
    return ConversationalRepair(
        thought,
        f"continue_pragmatic_cue:{function}",
        False,
        (
            DependencyPattern(
                ("subject", "direct_object", "predicate"),
                clause_kind="question",
                confidence=0.96,
                evidence_count=8,
                source="russian_question_information_structure",
            ),
        ),
    )


def _personal_nominal_repair(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> ConversationalRepair | None:
    if (
        clause.implicit_predicate != "быть"
        or not clause.subject_indices
        or not clause.predicate_complement_indices
        or structure.tokens[clause.subject_indices[0]].lemma not in {"я", "мы"}
    ):
        return None
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "считать",
            "считаешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency("subject", _addressee()),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme(
                        "почему",
                        "почему",
                        "adverb",
                        entity_kind="question_adverb",
                    )
                ),
            ),
            SemanticDependency(
                "manner",
                ConceptArgument(
                    ConceptLexeme("так", "так", "adverb", entity_kind="deictic_adverb")
                ),
            ),
        ),
        clause_kind="question",
        focus_role="topic",
        confidence=min(0.88, clause.confidence + 0.16),
        evidence=("request_reason_for_personal_evaluation",),
    )
    return ConversationalRepair(
        thought,
        "continue_personal_evaluation",
        False,
        (
            DependencyPattern(
                ("topic", "subject", "manner", "predicate"),
                clause_kind="question",
                confidence=0.96,
                evidence_count=8,
                source="russian_question_information_structure",
            ),
        ),
    )


def _topic_intention_question(topic: ConceptArgument) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "хотеть",
            "хочешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency("subject", _addressee()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "action_complement",
                ConceptArgument(ConceptLexeme("сказать", "сказать", "verb")),
            ),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    topic.head,
                    topic.modifiers,
                    preposition="про",
                    grammatical_case="accusative",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=0.76,
        evidence=("ask_speaker_intention_about_topic",),
    )


def _generic_intention_question() -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "подразумевать",
            "подразумеваешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency("subject", _addressee()),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=0.68,
        evidence=("unresolved_reference_after_internal_analysis",),
    )


def _question_argument(
    role: str,
    *,
    grammatical_case: str | None,
    prepositions: frozenset[str],
) -> ConceptArgument:
    preposition = sorted(prepositions)[0] if prepositions else None
    return ConceptArgument(
        ConceptLexeme("что", "что", "pronoun", (("number", "singular"),)),
        preposition=preposition,
        grammatical_case=grammatical_case,
    )


def _addressee() -> ConceptArgument:
    return ConceptArgument(
        ConceptLexeme(
            "ты",
            "ты",
            "pronoun",
            (("case", "nominative"), ("number", "singular"), ("person", "second")),
        )
    )


def _speaker_is_experiencer(
    structure: StructuredUtterance,
    clause: ClauseFrame,
) -> bool:
    if clause.implicit_subject == "speaker":
        return True
    return bool(
        clause.subject_indices
        and structure.tokens[clause.subject_indices[0]].lemma in {"я", "мы"}
    )


def _topic_argument(
    structure: StructuredUtterance,
    clause: ClauseFrame | None,
) -> ConceptArgument | None:
    if clause is None:
        return None
    excluded = {
        *clause.subject_indices,
        *clause.direct_object_indices,
        *clause.indirect_object_indices,
        *clause.predicate_complement_indices,
        *((clause.predicate_index,) if clause.predicate_index is not None else ()),
    }
    token = next(
        (
            item
            for item in structure.tokens[clause.token_start : clause.token_end]
            if item.index not in excluded
            and item.part_of_speech in {"noun", "pronoun", "adjective"}
            and item.semantic_role not in {"function_word", "pragmatic_marker"}
        ),
        None,
    )
    if token is None and clause.predicate_index is None:
        token = next(
            (
                item
                for item in structure.tokens[clause.token_start : clause.token_end]
                if item.part_of_speech in {"noun", "adjective"}
            ),
            None,
        )
    if token is None:
        return None
    return ConceptArgument(
        ConceptLexeme(
            token.lemma,
            token.corrected_form or token.surface,
            token.part_of_speech,
            token.grammatical_features,
            token.entity_kind,
        )
    )
