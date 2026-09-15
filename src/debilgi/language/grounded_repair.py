"""Compose clarification goals from the current utterance, never stored replies."""

from __future__ import annotations

from debilgi.language.clarification_policy import CONTACT_WORDS
from debilgi.language.content import content_lemmas
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import StructuredUtterance, analyze_structured_utterance

# Lexical speech-act recognition, not reply text.
_CONTACT = CONTACT_WORDS


def unmet_requirement_thought(requirement: str) -> ClauseThought:
    """State a gap in justification rather than invent a cause or a denial."""
    target = "причина" if requirement == "explain_reason" else "обоснование"
    return ClauseThought(
        predicate=ConceptLexeme("знать", "знаю", "verb",
                               (("person", "first"), ("tense", "present"))),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme(target, target, "noun"), grammatical_case="genitive")),
        ), negated=True, confidence=0.85,
        evidence=(f"unresolved_requirement:{requirement}", "no_eligible_grounded_explanation"),
    )


def prior_claim_review_thought(
    prompt: str, consultation_messages: tuple[str, ...],
) -> tuple[ClauseThought, str] | None:
    """Acknowledge an actual own utterance, without treating it as a true fact."""
    focus = set(content_lemmas(prompt)) - {"сказать", "говорить", "написать"}
    if not focus:
        return None
    for message in reversed(consultation_messages):
        role, sep, text = message.partition(": ")
        if not sep or role.casefold() != "agent" or len(text) > 240:
            continue
        structure = analyze_structured_utterance(text)
        if any(c.clause_kind == "question" for c in structure.clauses):
            continue
        if not (focus & set(content_lemmas(text))):
            continue
        quoted = f"«{text.strip().rstrip('.!?')}»"
        return ClauseThought(
            predicate=ConceptLexeme("говорить", "говорил", "verb", (("tense", "past"),)),
            dependencies=(
                SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
                SemanticDependency("direct_object", ConceptArgument(
                    ConceptLexeme(text, quoted, "noun", entity_kind="quoted_claim"))),
            ), confidence=0.9, evidence=("source:own_dialogue_turn", f"reviewed_claim:{text}"),
        ), text
    return None


def grounded_repair_thought(
    structure: StructuredUtterance, *, clarifiable_terms: tuple[str, ...] | None = None,
) -> ClauseThought | None:
    words = tuple(t for t in structure.tokens if t.surface[:1].isalnum())
    if not words:
        return None
    # A command with a missing object asks for that object, not a definition of
    # its verb. The action lexeme is supplied by the utterance itself.
    command = next((c for c in structure.clauses if c.clause_kind == "command"
                    and c.predicate_index is not None and not c.direct_object_indices), None)
    if command is not None:
        verb = structure.tokens[command.predicate_index]
        if dict(verb.grammatical_features).get("transitivity") == "transitive":
            return ClauseThought(
                predicate=ConceptLexeme("хотеть", "хочешь", "verb",
                    (("person", "second"), ("number", "singular"))),
                dependencies=(
                    SemanticDependency("subject", ConceptArgument(
                        ConceptLexeme("ты", "ты", "pronoun"))),
                    SemanticDependency("action_complement", ConceptArgument(
                        ConceptLexeme(verb.lemma, verb.lemma, "verb"))),
                    SemanticDependency("direct_object", ConceptArgument(
                        ConceptLexeme("что", "что", "pronoun"), grammatical_case="accusative")),
                ), clause_kind="question", focus_role="direct_object", predicate_class="modal",
                confidence=0.9, evidence=(f"current_message:missing_command_object:{verb.lemma}",),
            )
    if all(t.part_of_speech in {"particle", "pronoun", "adverb"} for t in words) and any(
        t.lemma in {"почему", "зачем", "как", "ну"} for t in words
    ):
        # There is no proposition to explain; request the missing object.
        return ClauseThought(
            predicate=ConceptLexeme(
                "хотеть", "хочешь", "verb", (("person", "second"), ("number", "singular"))
            ),
            dependencies=(
                SemanticDependency(
                    "subject", ConceptArgument(ConceptLexeme("ты", "ты", "pronoun"))
                ),
                SemanticDependency(
                    "action_complement",
                    ConceptArgument(ConceptLexeme("узнать", "узнать", "verb")),
                ),
                SemanticDependency(
                    "direct_object",
                    ConceptArgument(
                        ConceptLexeme("что", "что", "pronoun"), grammatical_case="accusative"
                    ),
                ),
            ),
            clause_kind="question",
            focus_role="direct_object",
            predicate_class="modal",
            confidence=0.9,
            evidence=("current_message:missing_explanation_target",),
        )
    if len(words) <= 2 and any(
        t.lemma in _CONTACT or t.surface.casefold() in _CONTACT for t in words
    ):
        return ClauseThought(
            predicate=ConceptLexeme(
                "слушать",
                "слушаю",
                "verb",
                (("person", "first"), ("number", "singular"), ("tense", "present")),
            ),
            dependencies=(
                SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
                SemanticDependency(
                    "direct_object",
                    ConceptArgument(
                        ConceptLexeme("ты", "тебя", "pronoun"), grammatical_case="accusative"
                    ),
                ),
            ),
            confidence=0.94,
            evidence=("current_message:contact_request",),
        )
    # Ask for the referent of a subject instead of asserting a guessed event.
    clause = next(
        (
            c
            for c in structure.clauses
            if c.predicate_index is not None and c.subject_indices and c.clause_kind == "statement"
        ),
        None,
    )
    if clause is not None:
        subject = structure.tokens[clause.subject_indices[0]]
        predicate = structure.tokens[clause.predicate_index]
        if subject.part_of_speech == "noun" and not (
            clause.direct_object_indices
            or clause.indirect_object_indices
            or clause.action_complement_indices
            or clause.predicate_complement_indices
        ):
            return ClauseThought(
                predicate=ConceptLexeme(
                    predicate.lemma,
                    predicate.surface,
                    predicate.part_of_speech,
                    predicate.grammatical_features,
                ),
                dependencies=(
                    SemanticDependency(
                        "subject",
                        ConceptArgument(
                            ConceptLexeme(
                                subject.lemma,
                                subject.surface,
                                subject.part_of_speech,
                                subject.grammatical_features,
                            ),
                            modifiers=(ConceptLexeme("какой", "какой", "adjective"),),
                            grammatical_case="nominative",
                        ),
                    ),
                ),
                clause_kind="question",
                focus_role="subject",
                negated=clause.negated,
                confidence=0.9,
                evidence=("current_message:unresolved_subject_reference",),
            )
    # Quote an input concept as data. No dictionary definition or invented cause.
    target = next((t for t in words if t.part_of_speech == "noun"), None)
    if target is None:
        target = next((t for t in words if t.part_of_speech == "adjective"), None)
    if target is None:
        target = next((t for t in words if t.part_of_speech == "verb"), None)
    if target is None and len(words) == 1:
        target = words[0]
    if target is None or (
        clarifiable_terms is not None and target.lemma not in clarifiable_terms
    ):
        return unresolved_answer_thought()
    return ClauseThought(
        predicate=ConceptLexeme(
            "подразумевать",
            "подразумеваешь",
            "verb",
            (("person", "second"), ("number", "singular"), ("tense", "present")),
        ),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("ты", "ты", "pronoun"))),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"), grammatical_case="accusative"
                ),
            ),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme(
                        target.lemma,
                        f"«{target.lemma}»",
                        target.part_of_speech,
                        entity_kind="quoted_lexeme",
                    ),
                    preposition="под",
                    grammatical_case="instrumental",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=0.9,
        evidence=(f"current_message:ambiguous_term:{target.lemma}",),
    )


def unresolved_answer_thought() -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme("знать", "знаю", "verb",
                               (("person", "first"), ("tense", "present"))),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("ответ", "ответ", "noun"), grammatical_case="genitive")),
        ), negated=True, confidence=0.85,
        evidence=("answer_not_derived", "no_blocking_lexical_gap"),
    )
