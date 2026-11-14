"""Evidence -> independent meaning plans -> grammar; no act-to-reply table.

Rules below describe reusable operations on propositions (retrieve, bind a
slot, attribute a report, appraise, ask for a missing argument). They do not
contain completed response sentences. A plan retains its actual premises.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace

from debilgi.language.answer_selection import select_answer_shape
from debilgi.language.capabilities import assess_capability_question
from debilgi.language.clarification_policy import is_contact_request
from debilgi.language.communication import MessagePlan, reader_goal
from debilgi.language.content import content_lemmas
from debilgi.language.dialogue_search import query_unobserved_relation, search_continuations
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.dialogue_task_reasoning import plan_task_resolution
from debilgi.language.lexical_choice import LexicalOption, choose_lexeme
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.planning import ReplyContext
from debilgi.language.premise_reasoning import lexical_premise
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
)
from debilgi.language.relation_reasoning import (
    analyze_relation_question,
    reason_about_relation,
)
from debilgi.language.request_focus import focus_request, is_response_control
from debilgi.language.semantic_graph import (
    extract_meanings,
    goal_effect,
    infer_meanings,
    learned_result_link,
    matches_proposition,
    meaning_query,
    proposition_key,
    subject_of,
    with_subject,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    TokenInterpretation,
    analyze_structured_utterance,
)
from debilgi.language.valency import ValencySlot, predicate_valency

_CONTENT_PREDICATES = frozenset(
    {
        "знать",
        "помнить",
        "вспомнить",
        "понять",
        "понимать",
        "слышать",
        "услышать",
        "читать",
        "прочитать",
        "узнать",
        "рассказать",
        "сообщить",
        "описать",
        "объяснить",
    }
)
_REPORT_PREDICATES = frozenset({"услышать", "слышать", "прочитать", "читать"})
_INTERNAL_STATE_PREDICATES = frozenset(
    {
        "знать",
        "понимать",
        "помнить",
        "хотеть",
        "мочь",
        "уметь",
    }
)
_ACTION_BEARING_MODAL_PREDICATES = frozenset({"мочь", "уметь", "хотеть"})
_NEXT_ACTION_MARKERS = frozenset({"теперь", "дальше", "сейчас", "потом"})
_INTERROGATIVES = frozenset({"что", "кто", "какой", "где", "куда", "когда", "как", "сколько"})
_QUESTION_ROLE = {
    "где": "location",
    "куда": "location",
    "откуда": "location",
    "когда": "manner",
    "как": "manner",
    "почему": "cause",
    "зачем": "cause",
}
_PRONOUNS = {"я": "ты", "ты": "я", "мы": "вы", "вы": "мы"}
_POSSESSIVES = {"мой": "твой", "твой": "мой", "наш": "ваш", "ваш": "наш"}


@dataclass(frozen=True, slots=True)
class DialogueObjective:
    kind: str
    predicate: str | None
    question_word: str | None
    topics: tuple[str, ...]
    known_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    evidence_id: str
    kind: str
    text: str
    thought: ClauseThought


@dataclass(frozen=True, slots=True)
class DerivedPlan:
    plan: MessagePlan
    operation: str
    premises: tuple[SemanticEvidence, ...]
    answer_slot: str | None = None
    patterns: tuple[DependencyPattern, ...] = ()

    @property
    def confidence(self) -> float:
        return self.plan.claim.confidence


@dataclass(frozen=True, slots=True)
class PlanningResult:
    objective: DialogueObjective
    proposals: tuple[DerivedPlan, ...]
    evidence: tuple[SemanticEvidence, ...]
    unresolved: tuple[str, ...]


def objective_for(structure: StructuredUtterance) -> DialogueObjective:
    goal = reader_goal(structure)
    clause = next(
        (c for c in reversed(structure.clauses) if c.clause_kind == "question"),
        next(iter(structure.clauses), None),
    )
    predicate = (
        structure.tokens[clause.predicate_index].lemma
        if clause is not None and clause.predicate_index is not None
        else None
    )
    topics = tuple(
        dict.fromkeys(
            structure.tokens[i].lemma
            for c in structure.clauses
            for comp in c.complements
            if comp.role == "topic"
            for i in comp.object_indices
            if structure.tokens[i].part_of_speech in {"noun", "adjective", "number"}
        )
    )
    # The request is identified by its information slot and grounded topic,
    # not by a whitelist of verbs. New predicates can therefore participate
    # without adding another scripted branch.
    content_request = bool(
        goal.question_word
        and any(comp.role == "topic" for c in structure.clauses for comp in c.complements)
    ) or bool(clause and clause.clause_kind == "command" and goal.topics)
    kind = (
        "content"
        if content_request
        else "slot"
        if goal.question_word
        else "polar"
        if any(c.clause_kind == "question" for c in structure.clauses)
        else "command"
        if any(c.clause_kind == "command" for c in structure.clauses)
        else "respond"
    )
    return DialogueObjective(
        kind, predicate, goal.question_word, topics or goal.topics, goal.known_roles
    )


def _lexeme(token: TokenInterpretation, *, speaker: str | None = None) -> ConceptLexeme:
    lemma, surface = token.lemma, token.corrected_form or token.surface
    features = dict(token.grammatical_features)
    if speaker == "user" and lemma in _PRONOUNS:
        lemma = surface = _PRONOUNS[lemma]
        features["person"] = "first" if lemma in {"я", "мы"} else "second"
    elif speaker == "user" and lemma in _POSSESSIVES:
        lemma = surface = _POSSESSIVES[lemma]
    return ConceptLexeme(
        lemma, surface, token.part_of_speech, tuple(features.items()), token.entity_kind
    )


def statement_thought(text: str, *, speaker: str | None = None) -> ClauseThought | None:
    """Conservatively preserve one proposition, including its qualifications.

    Unsupported relations are rejected, never silently removed to obtain a
    simpler but materially different fact.
    """
    structure = analyze_structured_utterance(text)
    if len(structure.clauses) != 1:
        return None
    c = structure.clauses[0]
    if c.predicate_index is not None and c.subject_indices:
        verb_features = dict(structure.tokens[c.predicate_index].grammatical_features)
        subjects = tuple(structure.tokens[index] for index in c.subject_indices)
        # A dependency guess must not turn an omitted first/second-person
        # subject into a visible noun ("ногти потеряешь" -> "ногти потеряют").
        if verb_features.get("person") in {"first", "second"} and all(
            token.part_of_speech != "pronoun" for token in subjects
        ):
            return None
    if (
        c.clause_kind not in {"statement", "fragment"}
        or c.coordinations
        or c.modal_lemmas
        or (not c.subject_indices and not (speaker and c.implicit_subject == "speaker"))
        or (c.predicate_index is None and c.implicit_predicate != "быть")
    ):
        return _recover_topic_subject_thought(structure)
    used = {c.predicate_index} if c.predicate_index is not None else set()

    def argument(indices: tuple[int, ...], preposition: str | None = None):
        heads = [
            i
            for i in indices
            if structure.tokens[i].part_of_speech in {"noun", "pronoun", "number", "verb"}
        ]
        if not heads:
            heads = list(indices[-1:])
        if len(heads) != 1:
            return None
        index = heads[0]
        modifiers = set(indices) - {index}
        # Parser role groups may omit adjacent attributive adjectives.
        before = index - 1
        while before >= c.token_start and structure.tokens[before].part_of_speech == "adjective":
            modifiers.add(before)
            before -= 1
        if any(structure.tokens[i].part_of_speech != "adjective" for i in modifiers):
            return None
        used.update((index, *modifiers))
        token = structure.tokens[index]
        return ConceptArgument(
            _lexeme(token, speaker=speaker),
            tuple(_lexeme(structure.tokens[i], speaker=speaker) for i in sorted(modifiers)),
            preposition=preposition,
            grammatical_case=dict(token.grammatical_features).get("case"),
        )

    dependencies = []
    if not c.subject_indices and speaker == "user" and c.implicit_subject == "speaker":
        dependencies.append(
            SemanticDependency(
                "subject",
                ConceptArgument(ConceptLexeme("ты", "ты", "pronoun")),
            )
        )
    for role, indices in (
        ("subject", c.subject_indices),
        ("direct_object", c.direct_object_indices),
        ("indirect_object", c.indirect_object_indices),
        ("predicate_complement", c.predicate_complement_indices),
        ("action_complement", c.action_complement_indices),
    ):
        if indices:
            arg = argument(indices)
            if arg is None:
                return None
            dependencies.append(SemanticDependency(role, arg))
    for complement in c.complements:
        if complement.role not in {"topic", "location", "location_or_destination"}:
            return None
        prep = structure.tokens[complement.preposition_index].lemma
        arg = argument(complement.object_indices, prep)
        if arg is None:
            return None
        used.add(complement.preposition_index)
        dependencies.append(
            SemanticDependency(
                "location" if complement.role.startswith("location") else "topic",
                arg,
            )
        )
    for token in structure.tokens:
        if (
            token.index not in used
            and token.part_of_speech == "adverb"
            and token.lemma not in {"возможно", "наверное", "якобы", "вероятно"}
        ):
            dependencies.append(SemanticDependency("manner", ConceptArgument(_lexeme(token))))
            used.add(token.index)
    remaining = [
        t
        for t in structure.tokens
        if t.index not in used
        and t.surface[:1].isalnum()
        and t.lemma not in {"не", "это"}
        and not (t.semantic_role == "function_word" and t.lemma in {"а", "то", "ну", "да"})
        and not (
            t.semantic_role == "pragmatic_marker"
            and (
                t.part_of_speech == "interjection"
                or (
                    bool(t.pragmatic_functions)
                    and set(t.pragmatic_functions)
                    <= {
                        "coarse_emphasis",
                        "turn_management",
                        "parenthetical_attention_marker",
                        "summary_or_topic_shift",
                    }
                )
            )
        )
    ]
    # Unbound content and particles may change scope or certainty.
    if remaining:
        return _recover_topic_subject_thought(structure)
    predicate = (
        _lexeme(structure.tokens[c.predicate_index])
        if c.predicate_index is not None
        else ConceptLexeme("быть", "быть", "verb", (("tense", "present"),))
    )
    return ClauseThought(
        predicate,
        tuple(dependencies),
        negated=c.negated,
        confidence=min(0.92, max(0.72, c.confidence)),
        predicate_class="copular" if c.predicate_index is None else "lexical",
    )


def _recover_topic_subject_thought(structure: StructuredUtterance) -> ClauseThought | None:
    """Recover a subject split by discourse particles or quantifiers.

    The parser can label the topical noun as unassigned when a discourse
    particle (``-то``/``а``) precedes it. We recover the dependency from the
    token roles and morphology, without tying it to particular vocabulary.
    """
    if len(structure.clauses) != 1:
        return None
    clause = structure.clauses[0]
    tokens = structure.tokens
    if clause.predicate_index is None or not clause.action_complement_indices:
        return None
    predicate = _lexeme(tokens[clause.predicate_index])
    action = tokens[clause.action_complement_indices[0]]
    if action.part_of_speech != "verb":
        return None
    subject = next((tokens[i] for i in clause.subject_indices), None)
    topical_noun = next(
        (
            token
            for token in tokens
            if token.part_of_speech == "noun" and token.semantic_role == "unassigned_content"
        ),
        None,
    )
    quantifier = next(
        (
            tokens[i]
            for i in clause.subject_indices
            if tokens[i].part_of_speech in {"pronoun", "adjective"}
        ),
        None,
    )
    if topical_noun is None or subject is None or quantifier is None:
        return None
    head_token = _lexeme(topical_noun)
    head_features = dict(head_token.features)
    # The quantifier determines the noun's agreement in this construction;
    # the parser's isolated genitive reading is a surface ambiguity.
    head_features.update({"case": "nominative", "number": "plural"})
    head = replace(head_token, features=tuple(head_features.items()))
    dependencies = (
        SemanticDependency(
            "subject",
            ConceptArgument(head, modifiers=(_lexeme(quantifier),), grammatical_case="nominative"),
        ),
        SemanticDependency("action_complement", ConceptArgument(_lexeme(action))),
    )
    return ClauseThought(
        predicate,
        dependencies,
        confidence=min(0.88, clause.confidence + 0.12),
    )


def _record(kind: str, text: str, thought: ClauseThought) -> SemanticEvidence:
    key = hashlib.sha256(f"{kind}|{text}".encode()).hexdigest()[:20]
    evidence_id = f"premise:{key}"
    return SemanticEvidence(
        evidence_id, kind, text, replace(thought, evidence=(evidence_id, *thought.evidence))
    )


def _terms(thought: ClauseThought) -> set[str]:
    return {
        thought.predicate.lemma,
        *(
            lexeme.lemma
            for dep in thought.dependencies
            for lexeme in (dep.argument.head, *dep.argument.modifiers)
        ),
    }


def _proposition_key(thought: ClauseThought) -> tuple:
    features = dict(thought.predicate.features)
    return (
        thought.predicate.lemma,
        tuple(features.get(key) for key in ("tense", "aspect", "mood")),
        tuple(
            sorted(
                (
                    dep.role,
                    dep.argument.head.lemma,
                    dep.argument.preposition,
                    tuple(m.lemma for m in dep.argument.modifiers),
                    tuple(m.lemma for m in dep.argument.postmodifiers),
                )
                for dep in thought.dependencies
            )
        ),
    )


def _conflicted(record: SemanticEvidence, records: list[SemanticEvidence]) -> bool:
    return any(
        _proposition_key(other.thought) == _proposition_key(record.thought)
        and other.thought.negated != record.thought.negated
        for other in records
    )


def _supports(objective: DialogueObjective, thought: ClauseThought) -> bool:
    roles = {d.role: d.argument.head.lemma for d in thought.dependencies}
    topics = set(objective.topics) - _INTERROGATIVES
    if objective.kind == "content" and not topics:
        return False
    if topics and not topics <= _terms(thought):
        return False
    if objective.kind in {"slot", "polar"}:
        if objective.predicate and thought.predicate.lemma != objective.predicate:
            return False
        if any(
            lemma not in _INTERROGATIVES and roles.get(role) != lemma
            for role, lemma in objective.known_roles
        ):
            return False
        wanted = _QUESTION_ROLE.get(objective.question_word)
        if wanted and wanted not in roles:
            return False
    return bool(topics or objective.predicate)


def _is_incomplete_modal_request(
    structure: StructuredUtterance,
    objective: DialogueObjective,
) -> bool:
    """Tell a complete modal question from one that inherited its action.

    Russian freely omits an already mentioned infinitive: ``можешь?`` can be
    a full conversational turn while still being an incomplete proposition.
    Treating the finite modal itself as sufficient loses the action and makes
    the planner search evidence for an unbound ``can`` relation.
    """

    if (
        objective.kind != "polar"
        or objective.predicate not in _ACTION_BEARING_MODAL_PREDICATES
    ):
        return False
    for clause in reversed(structure.clauses):
        if clause.predicate_index is None:
            continue
        predicate = structure.tokens[clause.predicate_index]
        if predicate.lemma != objective.predicate:
            continue
        return not any(
            token.index != clause.predicate_index
            and token.part_of_speech == "verb"
            and "INFN" in token.grammemes
            for token in structure.tokens[clause.token_start : clause.token_end]
        )
    return False


def resolve_evidence_context(context: ReplyContext) -> ReplyContext:
    """Recover an omitted request from a labelled turn, never from bot wording."""
    active_task = (
        context.dialogue_situation.active_task if context.dialogue_situation is not None else None
    )
    if (
        active_task is not None
        and (
            active_task.source_event_id == context.source_event_id
            or active_task.objective.target_event_id == context.source_event_id
        )
        and active_task.goal == "repair_shared_model"
    ):
        if active_task.kind == "response_quality_repair":
            passed_agent_turn = False
            for message in reversed(context.consultation_messages):
                speaker, separator, previous = message.partition(": ")
                if not separator:
                    continue
                if speaker.casefold() == "agent":
                    passed_agent_turn = True
                    continue
                if (
                    speaker.casefold() == "user"
                    and passed_agent_turn
                    and not is_response_control(previous)
                    and content_lemmas(previous)
                ):
                    return replace(
                        context,
                        prompt=previous,
                        turn_prompt=context.turn_prompt or context.prompt,
                        focus_reason="repair_previous_answer_request",
                        dialogue_state=None,
                    )
        # A correction is new evidence about the shared model.  Replacing it
        # with an older unanswered question recreates the very error it reports.
        return replace(
            context,
            turn_prompt=context.turn_prompt or context.prompt,
            focus_reason="active_repair_task",
        )
    if (
        context.dialogue_state
        and context.dialogue_state.exchange
        and (context.dialogue_state.exchange.status == "matched")
    ):
        return context
    focus = focus_request(context.turn_prompt or context.prompt, context.consultation_messages)
    context = replace(
        context,
        prompt=focus.request_text,
        turn_prompt=focus.turn_text,
        response_options=focus.alternatives,
        focus_reason=focus.reason,
        dialogue_state=None if focus.request_text != context.prompt else context.dialogue_state,
    )
    structure = analyze_structured_utterance(context.prompt)
    objective = objective_for(structure)
    incomplete_modal = _is_incomplete_modal_request(structure, objective)
    if (
        focus.reason in {"current_question", "answer_control_keeps_request"}
        and not incomplete_modal
    ):
        return context
    contact = is_contact_request(context.prompt)
    if not contact and (
        objective.topics or statement_thought(context.prompt, speaker="user") is not None
    ):
        return context
    if (
        objective.kind == "respond"
        and content_lemmas(context.prompt)
        and any(
            clause.clause_kind == "statement" and clause.predicate_index is not None
            for clause in structure.clauses
        )
    ):
        return context
    # A request with its own predicate/slot is independent of earlier requests.
    if (
        objective.kind in {"slot", "polar"}
        and (objective.predicate or objective.known_roles)
        and not incomplete_modal
    ) or (objective.kind == "command" and objective.predicate not in _CONTENT_PREDICATES):
        return context
    for message in reversed(context.consultation_messages):
        speaker, sep, text = message.partition(": ")
        if not sep or speaker.casefold() != "user":
            continue
        prior_structure = analyze_structured_utterance(text)
        prior = objective_for(prior_structure)
        prior_incomplete_modal = _is_incomplete_modal_request(prior_structure, prior)
        if (
            prior.kind in {"content", "slot", "polar"}
            and (prior.topics or prior.predicate or prior.known_roles)
            and not prior_incomplete_modal
        ) or (contact and prior.kind == "respond" and statement_thought(text, speaker="user")):
            return replace(context, prompt=text, dialogue_state=None)
        # Do not chase an older request across a new substantive user turn.
        if prior.topics or statement_thought(text, speaker="user") is not None:
            break
    return context


def proposition_sentences(text: str) -> tuple[str, ...]:
    """Split explicit sentence boundaries; unsupported clauses remain rejected."""
    return tuple(
        part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ])", text) if part.strip()
    )


def plan_from_evidence(
    context: ReplyContext,
    analyses: tuple[WordAnalysis, ...] = (),
    *, explore_meanings: bool = False,
) -> PlanningResult:
    context = resolve_evidence_context(context)
    structure = analyze_structured_utterance(context.prompt)
    goal, objective = reader_goal(structure), objective_for(structure)
    state = context.dialogue_state or analyze_dialogue_state(
        context.prompt, context.dialogue_messages
    )
    bindings = dict(state.resolved_bindings)
    if bindings:
        known = tuple((role, bindings.get(role, lemma)) for role, lemma in objective.known_roles)
        objective = replace(objective, known_roles=known)
        goal = replace(goal, known_roles=known)
        if objective.kind == "content" and not objective.topics and "topic" in bindings:
            objective = replace(objective, topics=(bindings["topic"],))
            goal = replace(goal, topics=objective.topics)
    records: list[SemanticEvidence] = []
    semantic = _plan_meaning_graph(context, goal)
    if semantic:
        evidence = tuple(dict.fromkeys(p for proposal in semantic for p in proposal.premises))
        return PlanningResult(objective, semantic, evidence, ())
    modal_capability_question = any(
        clause.predicate_index is not None
        and structure.tokens[clause.predicate_index].lemma in {"мочь", "уметь"}
        for clause in structure.clauses
    )
    capability = (
        assess_capability_question(
            structure,
            analyses=analyses,
            context_messages=context.dialogue_messages,
            knowledge_facts=context.knowledge_facts,
            study_completed=True,
            perspective=context.perspective,
        )
        if modal_capability_question
        else None
    )
    if capability is not None and not capability.supported:
        record = _record(
            "capability_reasoning",
            " ".join((capability.action_lemma, capability.reason, *capability.evidence)),
            capability.thought,
        )
        plan = DerivedPlan(
            MessagePlan(goal, record.thought),
            "resolve_capability",
            (record,),
            "polarity",
            capability.patterns,
        )
        return PlanningResult(objective, (plan,), (record,), ())
    relation_frame = analyze_relation_question(structure)
    if relation_frame is not None:
        object_terms = set(relation_frame.object_lemmas)
        has_lexical_evidence = any(
            bool({word.lemma, word.normalized_form} & object_terms)
            and bool(
                word.personal_interpretation
                or word.local_meaning
                or word.selected_meaning
                or word.dictionary_meanings
            )
            for word in analyses
        )
        has_reference_evidence = any(
            bool(
                object_terms
                & (
                    set(content_lemmas(title))
                    | {title.casefold().replace("ё", "е")}
                )
            )
            and bool(definition)
            for title, definition in context.reference_definitions
        )
        relation = reason_about_relation(
            relation_frame,
            analyses,
            context.reference_definitions,
            study_completed=has_lexical_evidence or has_reference_evidence,
        )
        relation_thought = relation.productive_thought
        if relation_thought is not None:
            evidence_text = " ".join(
                (
                    relation.object_lemma or relation_frame.object_surface,
                    *relation.evidence_terms,
                )
            ).strip()
            record = _record("relation_reasoning", evidence_text, relation_thought)
            plan = DerivedPlan(
                MessagePlan(goal, record.thought),
                "resolve_relation",
                (record,),
                "quantity" if relation_frame.query_quantity else "polarity",
            )
            return PlanningResult(objective, (plan,), (record,), ())
    exchange = state.exchange
    if exchange and exchange.status == "matched" and exchange.bound_proposition is not None:
        evidence = tuple(e for e in (exchange.question_event_id, exchange.answer_event_id) if e)
        kind = (
            "statement"
            if exchange.obligation.expected_role in {"cause", "name", "polarity"}
            else "question"
        )
        thought = replace(exchange.bound_proposition, clause_kind=kind, evidence=evidence)
        record = _record("bound_user_answer", exchange.answer_message, thought)
        plan = DerivedPlan(
            MessagePlan(goal, record.thought),
            "integrate_user_answer",
            (record,),
            exchange.obligation.expected_role,
        )
        return PlanningResult(objective, (plan,), (record,), ())
    for text in dict.fromkeys(
        sentence
        for fact in (*context.knowledge_facts, *context.required_facts)
        for sentence in proposition_sentences(fact)
    ):
        thought = statement_thought(text)
        if thought is not None and not (
            any(
                dependency.role == "subject" and dependency.argument.head.lemma in {"я", "мы"}
                for dependency in thought.dependencies
            )
        ):
            records.append(_record("knowledge", text, thought))
    for claim in context.assertion_memory.assertions:
        if claim.speaker != "agent":
            records.append(_record("dialogue_report", claim.source_event_id, claim.thought))
    for source_event_id, text in context.proposition_reports:
        thought = statement_thought(text, speaker="user")
        if thought is not None:
            records.append(_record("remembered_user_report", source_event_id, thought))
    for report in context.dialogue_reports:
        thought = replace(
            report.thought, evidence=(report.question_event_id, report.answer_event_id)
        )
        records.append(_record("remembered_dialogue_answer", report.answer_text, thought))
    for word in analyses:
        target = word.lemma or word.normalized_form
        chosen = word.personal_interpretation or word.local_meaning or word.selected_meaning
        meanings = (chosen,) if chosen else word.dictionary_meanings
        # Competing senses need sense resolution; frequency is not proof.
        if len(meanings) != 1:
            continue
        options = [
            LexicalOption(target, target, meanings[0], evidence=(word.sense_id or word.source,))
        ]
        options.extend(
            LexicalOption(link.target, link.target, meanings[0], evidence=word.relation_evidence)
            for link in word.typed_relations
            if link.relation == "synonym"
        )
        choice = choose_lexeme(target, tuple(options))
        if choice is not None:
            target = choice.option.lemma
        premise = lexical_premise(meanings[0])
        if premise is None:
            continue
        thought = ClauseThought(
            ConceptLexeme("являться", "являться", "verb", (("tense", "present"),)),
            (
                SemanticDependency(
                    "subject", ConceptArgument(ConceptLexeme(target, target, "noun"))
                ),
                SemanticDependency(
                    "predicate_complement",
                    replace(premise.category, grammatical_case="instrumental"),
                ),
            ),
            confidence=0.86,
        )
        records.append(_record("lexical_meaning", f"{target}: {meanings[0]}", thought))
    proposals: list[DerivedPlan] = []
    if objective.kind in {"content", "slot", "polar"}:
        for record in records:
            if not _supports(objective, record.thought):
                continue
            if _conflicted(record, records):
                continue
            # A claim that someone knows X is not the content of X.
            if (
                objective.kind == "content"
                and record.thought.predicate.lemma in _CONTENT_PREDICATES
            ):
                continue
            shape = select_answer_shape(objective, record.thought)
            proposals.append(
                DerivedPlan(
                    MessagePlan(
                        goal, record.thought, answer_shape=shape.mode, focus_role=shape.focus_role
                    ),
                    "retrieve_proposition",
                    (record,),
                    objective.question_word,
                )
            )
    # Reconstruct what was actually said from labelled dialogue, not a report
    # about the success/failure of processing it.
    if objective.kind in {"content", "slot"} and objective.predicate in _REPORT_PREDICATES:
        for message in reversed(context.consultation_messages):
            speaker, sep, text = message.partition(": ")
            if not sep or speaker.casefold() != "user" or not text.strip():
                continue
            if objective.topics and not set(objective.topics) <= set(content_lemmas(text)):
                continue
            thought = ClauseThought(
                ConceptLexeme("написать", "написать", "verb", (("tense", "past"),)),
                (
                    SemanticDependency(
                        "subject", ConceptArgument(ConceptLexeme("ты", "ты", "pronoun"))
                    ),
                    SemanticDependency(
                        "direct_object",
                        ConceptArgument(
                            ConceptLexeme(
                                text, f"«{text.strip()}»", "noun", entity_kind="quoted_claim"
                            )
                        ),
                    ),
                ),
                confidence=0.95,
            )
            record = _record("quoted_user_turn", text, thought)
            proposals.append(
                DerivedPlan(
                    MessagePlan(goal, record.thought),
                    "attribute_dialogue_turn",
                    (record,),
                    objective.question_word,
                )
            )
            break
    if not proposals and objective.kind in {"respond", "command"}:
        current = statement_thought(context.prompt, speaker="user")
        if current is not None and bindings:
            current = replace(
                current,
                dependencies=tuple(
                    replace(
                        dep,
                        argument=replace(
                            dep.argument,
                            head=replace(
                                dep.argument.head,
                                lemma=bindings[dep.role],
                                surface=bindings[dep.role],
                                part_of_speech="noun",
                            ),
                        ),
                    )
                    if dep.role in bindings
                    else dep
                    for dep in current.dependencies
                ),
            )
        if current is not None:
            record = _record("current_user_report", context.prompt, current)
            records.append(record)
            # Continue the actual topic with another remembered proposition.
            # Personal pronouns cannot connect otherwise unrelated memories.
            anchors = {
                dep.argument.head.lemma
                for dep in current.dependencies
                if dep.argument.head.part_of_speech == "noun"
            }
            for remembered in records[:-1]:
                if _conflicted(remembered, records):
                    continue
                if remembered.kind == "lexical_meaning" or not anchors.intersection(
                    _terms(remembered.thought)
                ):
                    continue
                if remembered.thought.predicate.lemma == current.predicate.lemma:
                    continue
                proposals.append(
                    DerivedPlan(
                        MessagePlan(goal, remembered.thought),
                        "continue_from_memory",
                        (record, remembered),
                    )
                )
            for continuation in search_continuations(goal, current, evidence_id=record.evidence_id):
                proposals.append(
                    DerivedPlan(
                        continuation.plan,
                        continuation.operation,
                        (record,),
                    )
                )
            valency = predicate_valency(current.predicate.lemma)
            present = {d.role for d in current.dependencies}
            if current.negated and current.predicate.lemma in _INTERNAL_STATE_PREDICATES:
                slots = ()
            elif valency:
                slots = valency.slots
            elif dict(current.predicate.features).get("transitivity") == "transitive":
                slots = (ValencySlot("direct_object", grammatical_case="accusative"),)
            else:
                slots = ()
            for slot in slots:
                if slot.role in present or slot.role == "subject":
                    continue
                if slot.role not in {"direct_object", "indirect_object", "location"}:
                    continue
                word = "где" if slot.role == "location" else "что"
                arg = ConceptArgument(
                    ConceptLexeme(word, word, "adverb" if word == "где" else "pronoun"),
                    grammatical_case=slot.grammatical_case,
                )
                thought = replace(
                    current,
                    clause_kind="question",
                    response_polarity=None,
                    dependencies=(*current.dependencies, SemanticDependency(slot.role, arg)),
                    focus_role=slot.role,
                    evidence=(record.evidence_id,),
                )
                proposals.append(
                    DerivedPlan(MessagePlan(goal, thought), "request_missing_argument", (record,))
                )
            if (explore_meanings or not proposals) and not (
                current.negated and current.predicate.lemma in _INTERNAL_STATE_PREDICATES
            ):
                for continuation in query_unobserved_relation(
                    goal,
                    current,
                    evidence_id=record.evidence_id,
                ):
                    proposals.append(
                        DerivedPlan(
                            continuation.plan,
                            continuation.operation,
                            (record,),
                        )
                    )
    active_task = (
        context.dialogue_situation.active_task if context.dialogue_situation is not None else None
    )
    if not proposals and active_task is not None:
        resolution = plan_task_resolution(active_task, goal)
        if resolution is not None:
            evidence = SemanticEvidence(
                active_task.task_id,
                "dialogue_task_state",
                resolution.evidence_text,
                resolution.plan.claim,
            )
            records.append(evidence)
            proposals.append(
                DerivedPlan(
                    resolution.plan,
                    resolution.operation,
                    (evidence,),
                    resolution.plan.claim.focus_role,
                )
            )
    unique = {proposal.plan.key: proposal for proposal in proposals}
    unresolved = () if unique else (f"unresolved_{objective.kind}",)
    return PlanningResult(objective, tuple(unique.values()), tuple(records), unresolved)


def _plan_meaning_graph(context: ReplyContext, goal) -> tuple[DerivedPlan, ...]:
    current = extract_meanings(context.prompt, context.source_event_id or "current_turn")
    # An explicit state update in this turn replaces the older same scoped atom.
    updates = {
        proposition_key(t, signed=False) for r in current if r.kind == "fact" for t in r.conclusions
    }
    retained = tuple(
        replace(
            r,
            conclusions=tuple(
                t for t in r.conclusions if proposition_key(t, signed=False) not in updates
            ),
        )
        if r.kind == "fact"
        else r
        for r in context.meaning_records
    )
    meanings = (*retained, *current)
    if not meanings and not context.meaning_goals:
        return ()
    sources = {r.meaning_id: r for r in meanings}

    def evidence_for(
        ids: tuple[str, ...], thought: ClauseThought, hypothesis: ClauseThought | None = None
    ) -> tuple[SemanticEvidence, ...]:
        records = tuple(
            SemanticEvidence(
                r.meaning_id,
                "learned_" + r.kind,
                r.text,
                replace((r.premises or r.conclusions)[0], evidence=(r.source_event_id,)),
            )
            for key in ids
            if (r := sources.get(key)) and (r.premises or r.conclusions)
        )
        if hypothesis:
            records += (
                SemanticEvidence(
                    context.source_event_id or "current_turn",
                    "hypothesis",
                    context.prompt,
                    hypothesis,
                ),
            )
        if not records:
            records = tuple(
                SemanticEvidence(
                    r.meaning_id,
                    "learned_fact",
                    r.text,
                    replace(thought, evidence=(r.source_event_id,)),
                )
                for r in meanings
                if r.kind == "fact"
                and any(proposition_key(t) == proposition_key(thought) for t in r.conclusions)
            )
        return records

    query = meaning_query(context.prompt)
    prompt_terms = tuple(content_lemmas(context.prompt))
    next_action_marker = next(
        (term for term in prompt_terms if term in _NEXT_ACTION_MARKERS),
        None,
    )
    self_goals = tuple(
        active
        for active in context.meaning_goals
        if subject_of(active.action).head.lemma in {"я", "мы"}
    )
    if (
        self_goals
        and goal.kind == "answer_slot"
        and goal.question_word == "что"
        and goal.predicate is None
        and not goal.topics
        and next_action_marker is not None
        and set(prompt_terms) <= _NEXT_ACTION_MARKERS
    ):
        # An open ``what now/next?`` asks which action currently wins.  The
        # answer comes from a valued operational goal, then goes through the
        # ordinary morphology and dependency realizer.  No stored utterance is
        # retrieved or replayed.
        active = max(
            self_goals,
            key=lambda item: (item.weight, item.evidence_id),
        )
        predicate_features = dict(active.action.predicate.features)
        predicate_features.update(
            {"tense": "present", "number": "singular", "person": "first"}
        )
        dependencies = active.action.dependencies
        if not any(dependency.role == "manner" for dependency in dependencies):
            dependencies = (
                *dependencies,
                SemanticDependency(
                    "manner",
                    ConceptArgument(
                        ConceptLexeme(
                            next_action_marker,
                            next_action_marker,
                            "adverb",
                        )
                    ),
                ),
            )
        claim = replace(
            active.action,
            predicate=replace(
                active.action.predicate,
                surface=active.action.predicate.lemma,
                features=tuple(predicate_features.items()),
            ),
            dependencies=dependencies,
            clause_kind="statement",
            confidence=min(0.92, 0.72 + active.weight * 0.20),
            evidence=(active.evidence_id,),
        )
        premise = SemanticEvidence(
            active.evidence_id,
            "operational_goal",
            "",
            replace(active.action, evidence=(active.evidence_id,)),
        )
        return (
            DerivedPlan(
                MessagePlan(
                    replace(
                        goal,
                        kind="respond",
                        question_word=None,
                        predicate=claim.predicate.lemma,
                        known_roles=(),
                    ),
                    claim,
                ),
                "continue_active_goal",
                (premise,),
            ),
        )
    assumption = query.assumption
    if query.kind == "preference" and not assumption and query.proposition:
        wanted = query.proposition
        action = next(
            (d.argument.head for d in wanted.dependencies if d.role == "action_complement"), None
        )
        if action:
            desired_action = replace(
                wanted,
                predicate=action,
                predicate_class="lexical",
                negated=False,
                dependencies=tuple(d for d in wanted.dependencies if d.role != "action_complement"),
            )
            for active in context.meaning_goals:
                if proposition_key(desired_action, signed=False) != proposition_key(
                    active.action, signed=False
                ):
                    continue
                claim = replace(
                    wanted,
                    negated=active.action.negated,
                    confidence=0.84,
                    evidence=(active.evidence_id,),
                )
                premise = SemanticEvidence(
                    active.evidence_id,
                    "operational_goal",
                    "",
                    replace(active.action, evidence=(active.evidence_id,)),
                )
                return (
                    DerivedPlan(
                        MessagePlan(goal, claim), "express_active_goal", (premise,), "polarity"
                    ),
                )
    if assumption and query.kind == "consequence":
        link = learned_result_link(
            assumption, meanings, context.source_event_id or "current_turn", context.prompt
        )
        if link:
            meanings = (*meanings, link)
            sources[link.meaning_id] = link
    generic = ConceptArgument(
        ConceptLexeme("кто-то", "кто-то", "pronoun", entity_kind="generic_actor")
    )

    def express_variables(thought: ClauseThought) -> ClauseThought:
        return (
            with_subject(thought, generic)
            if subject_of(thought).head.entity_kind == "semantic_variable"
            else thought
        )

    if (
        "?" not in context.prompt
        and len(current) == 1
        and current[0].kind in {"definition", "condition"}
    ):
        rule = current[0]
        if len(rule.premises) == 1 and rule.conclusions:
            return (
                DerivedPlan(
                    MessagePlan(
                        replace(goal, kind="respond", question_word=None, known_roles=()),
                        replace(
                            express_variables(rule.conclusions[0]), evidence=(rule.source_event_id,)
                        ),
                        condition=express_variables(rule.premises[0]),
                        additional_claims=tuple(express_variables(t) for t in rule.conclusions[1:]),
                    ),
                    "learn_semantic_relation",
                    evidence_for((rule.meaning_id,), rule.conclusions[0]),
                ),
            )
    # The older syntactic reader can mistake a definition for a question.
    # Definition extraction is independent of that classification.
    if len(current) == 1 and current[0].kind == "definition" and "?" not in context.prompt:
        rule = current[0]
        return (
            DerivedPlan(
                MessagePlan(
                    replace(goal, kind="respond", question_word=None, known_roles=()),
                    replace(
                        express_variables(rule.conclusions[0]), evidence=(rule.source_event_id,)
                    ),
                    condition=express_variables(rule.premises[0]),
                    additional_claims=tuple(express_variables(t) for t in rule.conclusions[1:]),
                ),
                "learn_semantic_relation",
                evidence_for((rule.meaning_id,), rule.conclusions[0]),
            ),
        )
    if assumption and subject_of(assumption).head.entity_kind == "semantic_variable":
        # A scoped definition can answer a nominal request about its own state.
        candidates = [
            with_subject(assumption, subject_of(r.premises[0]))
            for r in meanings
            if r.kind == "definition"
            and len(r.premises) == 1
            and proposition_key(with_subject(assumption, subject_of(r.premises[0])))
            == proposition_key(r.premises[0])
        ]
        assumption = candidates[0] if candidates else express_variables(assumption)
    if assumption and query.kind in {"consequence", "definition", "preference"}:
        marker = "hypothesis:" + (context.source_event_id or "current_turn")
        assumption = replace(assumption, evidence=(marker,))
        proofs = tuple(
            p
            for p in infer_meanings(meanings, assumptions=(assumption,))
            if marker in p.source_event_ids
            and p.rule_ids
            and proposition_key(p.thought) != proposition_key(assumption)
        )
        if query.kind == "preference" and query.proposition:
            effects = [
                (sum(goal_effect(p.thought, g) for g in context.meaning_goals), p) for p in proofs
            ]
            score = sum(effect for effect, _ in effects)
            relevant = [
                (effect, p) for effect, p in effects if effect and (effect > 0) == (score > 0)
            ]
            if not relevant or not score:
                return ()
            _, proof = max(relevant, key=lambda item: abs(item[0]))
            premises = evidence_for(proof.rule_ids, proof.thought, assumption)
            premises += tuple(
                SemanticEvidence(
                    g.evidence_id,
                    "operational_goal",
                    "",
                    replace(g.action, evidence=(g.evidence_id,)),
                )
                for g in context.meaning_goals
                if goal_effect(proof.thought, g)
            )
            claim = replace(
                query.proposition,
                negated=score < 0,
                confidence=0.84,
                evidence=tuple(
                    dict.fromkeys((*proof.source_event_ids, *(p.evidence_id for p in premises)))
                ),
            )
            return (
                DerivedPlan(
                    MessagePlan(goal, claim, reason=proof.thought, reason_condition=assumption),
                    "appraise_learned_consequence",
                    premises,
                    "polarity",
                ),
            )
        if proofs and query.kind in {"consequence", "definition"}:
            rules = tuple(dict.fromkeys(r for p in proofs for r in p.rule_ids))
            return (
                DerivedPlan(
                    MessagePlan(
                        goal,
                        proofs[0].thought,
                        condition=assumption,
                        additional_claims=tuple(p.thought for p in proofs[1:]),
                    ),
                    "infer_conditional_consequences",
                    evidence_for(rules, proofs[0].thought, assumption),
                ),
            )
        return ()
    if query.kind in {"polar", "cause", "subject_slot"} and query.proposition:
        requested = proposition_key(query.proposition, signed=False)
        proofs = infer_meanings(meanings)
        for proof in proofs:
            if query.kind == "subject_slot":
                if not matches_proposition(query.proposition, proof.thought):
                    continue
            elif proposition_key(proof.thought, signed=False) != requested:
                continue
            reason = None
            if query.kind == "cause":
                rules = [
                    sources[r]
                    for r in proof.rule_ids
                    if sources[r].kind == "cause" and len(sources[r].premises) == 1
                ]
                if not rules:
                    continue
                reason = rules[-1].premises[0]
            premises = evidence_for(proof.rule_ids, proof.thought)
            # Rules and the actual enabling facts both remain reviewable.
            premises += tuple(
                SemanticEvidence(
                    r.meaning_id,
                    "learned_fact",
                    r.text,
                    replace(r.conclusions[0], evidence=(r.source_event_id,)),
                )
                for r in meanings
                if r.kind == "fact"
                and r.conclusions
                and r.source_event_id in proof.source_event_ids
                and all(p.evidence_id != r.meaning_id for p in premises)
            )
            claim = replace(proof.thought, evidence=proof.source_event_ids)
            return (
                DerivedPlan(
                    MessagePlan(goal, claim, reason=reason),
                    "infer_from_learned_relations",
                    premises,
                    "polarity",
                ),
            )
    return ()
