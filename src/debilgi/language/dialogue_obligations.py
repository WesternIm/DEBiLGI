"""Reconstruct an unanswered semantic slot from adjacent turns, then bind an answer.

Events remain the durable source. Nothing here stores a reply phrase or promotes
a user's explanation to a dictionary fact. Bound propositions are confirmations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.events.models import Event
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import (
    TokenInterpretation,
    analyze_structured_utterance,
)


@dataclass(frozen=True, slots=True)
class QuestionObligation:
    expected_role: str
    proposition: ClauseThought
    source_message: str


@dataclass(frozen=True, slots=True)
class DialogueExchange:
    obligation: QuestionObligation
    status: str
    answer_message: str
    bound_proposition: ClauseThought | None
    reason: str
    question_event_id: str | None = None
    answer_event_id: str | None = None


def exchange_from_events(events: tuple[Event, ...], current: Event) -> DialogueExchange | None:
    """Use actual recent agent turns, never archive examples or another channel's question."""
    if current.channel_id is None or current.payload.get("author_is_bot"):
        return None
    recent = sorted(
        (
            event
            for event in events
            if event.event_id != current.event_id
            and event.guild_id == current.guild_id
            and event.channel_id == current.channel_id
            and 0 <= (current.utc_timestamp - event.utc_timestamp).total_seconds() <= 300
            and isinstance(event.payload.get("content"), str)
        ),
        key=lambda event: event.utc_timestamp,
    )
    if not recent:
        return None
    question = recent[-1]
    reply_to = current.payload.get("reply_to_message_id")
    if reply_to is not None and str(question.payload.get("message_id")) != str(reply_to):
        return None
    if not question.payload.get("author_is_bot"):
        return None
    bot_id = current.payload.get("bot_user_id")
    if bot_id is not None and question.actor_id != bot_id:
        return None
    if question.target_id is not None and question.target_id != current.actor_id:
        return None
    parent = question.payload.get("reply_to_message_id")
    if parent is not None:
        owner = next((e for e in recent if str(e.payload.get("message_id")) == str(parent)), None)
        if owner is not None and owner.actor_id != current.actor_id:
            return None
    result = resolve_dialogue_exchange(
        str(current.payload.get("content", "")), (str(question.payload["content"]),)
    )
    return (
        replace(result, question_event_id=question.event_id, answer_event_id=current.event_id)
        if result
        else None
    )


def _lexeme(token: TokenInterpretation) -> ConceptLexeme:
    return ConceptLexeme(
        token.lemma, token.surface, token.part_of_speech, token.grammatical_features
    )


def question_obligation(text: str) -> QuestionObligation | None:
    structure = analyze_structured_utterance(text)
    if len(structure.clauses) != 1:
        return None
    clause = structure.clauses[0]
    if clause.clause_kind != "question" or clause.question_target_index is None:
        return None
    if clause.negated or clause.coordinations:
        return None
    tokens = structure.tokens
    wh = tokens[clause.question_target_index].lemma
    predicate = tokens[clause.predicate_index] if clause.predicate_index is not None else None
    lemma = predicate.lemma if predicate else clause.implicit_predicate
    definition = wh == "что" and lemma in {"быть", "значить", "означать", "подразумевать"}
    if definition:
        target_indices = (
            tuple(i for c in clause.complements for i in c.object_indices)
            if lemma == "подразумевать"
            else (
                *clause.subject_indices,
                *clause.direct_object_indices,
            )
        )
        targets = [tokens[i] for i in target_indices if tokens[i].part_of_speech == "noun"]
        if len(targets) != 1:
            return None
        thought = ClauseThought(
            ConceptLexeme("являться", "является", "verb", (("tense", "present"),)),
            (SemanticDependency("subject", ConceptArgument(_lexeme(targets[0]))),),
        )
        return QuestionObligation("predicate_complement", thought, text)
    if predicate is None:
        return None
    role = (
        "subject"
        if wh == "кто" and not clause.subject_indices
        else "direct_object"
        if wh == "что" and clause.subject_indices and not clause.direct_object_indices
        else "location"
        if wh == "где" and clause.subject_indices
        else None
    )
    if role is None or clause.action_complement_indices or clause.predicate_complement_indices:
        return None
    dependencies = []
    for indices, argument_role in (
        (clause.subject_indices, "subject"),
        (clause.direct_object_indices, "direct_object"),
        (clause.indirect_object_indices, "indirect_object"),
    ):
        if len(indices) > 1:
            return None
        for index in indices:
            dependencies.append(
                SemanticDependency(
                    argument_role,
                    ConceptArgument(
                        _lexeme(tokens[index]),
                        grammatical_case={
                            "subject": "nominative",
                            "direct_object": "accusative",
                            "indirect_object": "dative",
                        }[argument_role],
                    ),
                )
            )
    # Do not silently discard already supplied locations or modifiers.
    if clause.complements:
        return None
    return QuestionObligation(role, ClauseThought(_lexeme(predicate), tuple(dependencies)), text)


def resolve_dialogue_exchange(prompt: str, context: tuple[str, ...]) -> DialogueExchange | None:
    if not context:
        return None
    obligation = question_obligation(context[-1])
    if obligation is None:
        return None

    def unmatched(reason: str) -> DialogueExchange:
        return DialogueExchange(obligation, "unmatched", prompt, None, reason)

    structure = analyze_structured_utterance(prompt)
    if len(structure.clauses) != 1 or "?" in prompt:
        return unmatched("not_a_single_answer_clause")
    clause = structure.clauses[0]
    if clause.predicate_index is not None or clause.coordinations:
        return unmatched("independent_proposition_or_alternatives")
    tokens = structure.tokens
    if any(t.lemma in {"не", "ни", "нет", "или", "либо"} for t in tokens):
        return unmatched("qualified_answer")
    allowed_fillers = {"это", "этот", "такой"}
    lexical = [
        i
        for i, t in enumerate(tokens)
        if t.surface[:1].isalnum() and t.lemma not in allowed_fillers
    ]
    complement_indices = {
        i for c in clause.complements for i in (c.preposition_index, *c.object_indices)
    }
    main_indices = [i for i in lexical if i not in complement_indices]
    heads = [i for i in main_indices if tokens[i].part_of_speech == "noun"]
    if obligation.expected_role == "location":
        if main_indices or len(clause.complements) != 1:
            return unmatched("expected_location")
        complement = clause.complements[0]
        if len(complement.object_indices) != 1:
            return unmatched("ambiguous_location")
        head = tokens[complement.object_indices[0]]
        prep = tokens[complement.preposition_index].lemma
        if prep not in {"в", "на", "под", "над", "около", "возле", "за"}:
            return unmatched("not_a_location_preposition")
        argument = ConceptArgument(
            _lexeme(head),
            preposition=prep,
            grammatical_case=dict(head.grammatical_features).get("case"),
        )
    else:
        if len(heads) != 1 or any(
            tokens[i].part_of_speech not in {"noun", "adjective"} for i in main_indices
        ):
            return unmatched("expected_one_nominal_argument")
        head = tokens[heads[0]]
        argument = ConceptArgument(
            _lexeme(head),
            modifiers=tuple(_lexeme(tokens[i]) for i in main_indices if i != heads[0]),
            grammatical_case={
                "subject": "nominative",
                "direct_object": "accusative",
                "predicate_complement": "instrumental",
            }[obligation.expected_role],
        )
    additions = [SemanticDependency(obligation.expected_role, argument)]
    if obligation.expected_role != "location":
        for complement in clause.complements:
            if len(complement.object_indices) != 1:
                return unmatched("ambiguous_nominal_modifier")
            prep = tokens[complement.preposition_index].lemma
            if prep not in {"для", "из"} or obligation.expected_role != "predicate_complement":
                return unmatched("unsupported_nominal_modifier")
            head = tokens[complement.object_indices[0]]
            if (
                head.part_of_speech != "noun"
                or dict(head.grammatical_features).get("case") != "genitive"
            ):
                return unmatched("modifier_case_mismatch")
            additions.append(
                SemanticDependency(
                    "topic",
                    ConceptArgument(_lexeme(head), preposition=prep, grammatical_case="genitive"),
                )
            )
    proposition = ClauseThought(
        obligation.proposition.predicate,
        (*obligation.proposition.dependencies, *additions),
        clause_kind="question",
        confidence=0.9,
        evidence=(
            "dialogue_obligation:slot_filled",
            f"expected_role:{obligation.expected_role}",
            f"question:{obligation.source_message}",
            f"user_answer:{prompt}",
        ),
    )
    return DialogueExchange(obligation, "matched", prompt, proposition, "compatible_argument")
