"""Evidence -> independent meaning plans -> grammar; no act-to-reply table.

Rules below describe reusable operations on propositions (retrieve, bind a
slot, attribute a report, appraise, ask for a missing argument). They do not
contain completed response sentences. A plan retains its actual premises.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from math import log2

from debilgi.language.answer_selection import select_answer_shape
from debilgi.language.canonical_claims import answer_thought, argument_ref
from debilgi.language.capabilities import assess_capability_question
from debilgi.language.clarification_policy import is_contact_request
from debilgi.language.communication import KnowledgeLimit, MessagePlan, reader_goal
from debilgi.language.content import content_lemmas
from debilgi.language.dialogue_search import query_unobserved_relation, search_continuations
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.dialogue_task_reasoning import plan_task_resolution
from debilgi.language.lexical_choice import LexicalOption, choose_lexeme
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import deictic_counterpart, person_cell
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


@dataclass(frozen=True, slots=True)
class ResponseDecision:
    mode: str
    reason: str
    act: str
    parse_confidence: float
    intent_confidence: float
    fact_confidence: float | None
    expression_confidence: float | None = None
    ambiguous_slot: str | None = None
    alternatives: tuple[str, ...] = ()
    expected_information_gain: float = 0.0
    allow_hook_question: bool = False
    turn_kind: str | None = None
    tone: tuple[str, ...] = ()
    confidence_level: str = "answer"
    target_meaning: str | None = None
    rule_id: str | None = None
    rule_weight: float = 0.0
    rule_source_ids: tuple[str, ...] = ()
    activation: float = 0.0
    input_features: tuple[str, ...] = ()
    target_features: tuple[str, ...] = ()
    mixture: tuple[tuple[str, float], ...] = ()
    known_slots: tuple[tuple[str, str, str], ...] = ()
    hypothesis: tuple[tuple[str, str], ...] = ()
    compatible_supports: tuple[dict, ...] = ()
    selection_context: dict | None = None

    def __post_init__(self):
        if self.act == "withhold":
            object.__setattr__(self, "confidence_level", "silence")


def plans_for_claim_result(result, structure) -> tuple[DerivedPlan, ...]:
    """The same graph operation serves interlocutor and direct parser requests."""
    from debilgi.language.canonical_claims import self_query

    goal = reader_goal(structure)
    records = tuple(
        SemanticEvidence(
            c.claim_id, "canonical_claim", c.source_event_id, answer_thought(c, result.query)
        )
        for c in result.claims
    )
    return tuple(
        DerivedPlan(
            MessagePlan(goal, r.thought),
            "retrieve_self_claim" if self_query(result.query) else "retrieve_proposition",
            (r,),
            goal.question_word,
        )
        for r in records
    )


def knowledge_limit_plan(context: ReplyContext) -> DerivedPlan | None:
    """Bind an epistemic assertion to the unfilled query, without asserting its body."""
    from debilgi.language.semantic_graph import parse_proposition

    result, structure = context.claim_result, context.structure
    if result is None or result.status != "Unknown":
        return None
    goal = reader_goal(structure)
    query = result.query
    # The clause is retold by the bot: the interlocutor's possessives turn to the addressee's.
    question = parse_proposition(
        structure, allow_question=True, possessive_deixis=True, irrealis=True, tree=True
    )
    from debilgi.language.canonical_claims import self_query

    if self_query(query):
        nouns = [
            t
            for t in structure.tokens
            if t.semantic_role == "predicate_complement" and t.part_of_speech == "noun"
        ]
        if query.predicate == "быть" and not goal.question_word and len(nouns) == 1:
            modifiers = tuple(
                ConceptLexeme(t.lemma, t.surface, t.part_of_speech, t.grammatical_features)
                for t in structure.tokens
                if t.semantic_role == "predicate_complement" and t.part_of_speech == "adjective"
            )
            question = ClauseThought(
                ConceptLexeme("быть", "быть", "verb", (("tense", "present"),)),
                (
                    SemanticDependency(
                        "subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))
                    ),
                    SemanticDependency(
                        "predicate_complement",
                        ConceptArgument(
                            ConceptLexeme(
                                nouns[0].lemma,
                                nouns[0].surface,
                                "noun",
                                nouns[0].grammatical_features,
                            ),
                            modifiers=modifiers,
                            grammatical_case="nominative",
                        ),
                    ),
                ),
                predicate_class="copular",
                confidence=0.86,
            )
        elif (
            query.predicate == "быть"
            and goal.question_word == "что"
            and any(t.lemma == "у" for t in structure.tokens)
        ):
            question = ClauseThought(
                ConceptLexeme("быть", "быть", "verb", (("tense", "present"),)),
                (
                    SemanticDependency(
                        "subject", ConceptArgument(ConceptLexeme("что", "что", "pronoun"))
                    ),
                    SemanticDependency(
                        "location",
                        ConceptArgument(
                            ConceptLexeme("я", "я", "pronoun"),
                            preposition="у",
                            grammatical_case="genitive",
                        ),
                    ),
                ),
                predicate_class="existential",
                confidence=0.86,
            )
    if question is None and goal.question_word in {"где", "кто"} and goal.subject:
        subject = deictic_counterpart(goal.subject[0], possessives=False) or goal.subject[0]
        question = ClauseThought(
            ConceptLexeme("быть", "быть", "verb", (("tense", "present"),)),
            (
                SemanticDependency(
                    "subject", ConceptArgument(ConceptLexeme(subject, subject, "pronoun"))
                ),
            ),
            predicate_class="copular",
            confidence=0.86,
        )
    if (
        question is None
        and goal.question_word
        and goal.addressed
        and self_query(query)
        and query.predicate
        and query.requested_role
        and query.predicate != "быть"
    ):
        # An open slot about the agent's own action whose clause the proposition reader cannot
        # take whole (a modal without its infinitive: «что ты хочешь?»): the limit is still
        # about that predicate of the agent, the slot being the asked word.
        clause = next((c for c in structure.clauses if c.predicate_index is not None), None)
        tense = clause.semantic_time.tense if clause and clause.semantic_time else None
        question = ClauseThought(
            ConceptLexeme(
                query.predicate, query.predicate, "verb", (("tense", tense or "present"),)
            ),
            (SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),),
            predicate_class="lexical",
            confidence=0.86,
        )
    condition = None
    if question is None and not goal.question_word:
        # «а если C, то P бы?»: what is unknown is the consequent under the condition,
        # asked in its own mood, not an answer to the whole question.
        from debilgi.language.semantic_graph import conditional_proposition

        conditional = conditional_proposition(structure, possessive_deixis=True)
        if (
            conditional is not None
            and conditional.consequent.predicate.lemma == (query.predicate or "")
            and any(
                d.role == "subject" and d.argument.head.lemma == "я"
                for d in conditional.consequent.dependencies
            )
        ):
            question = conditional.consequent
            if conditional.irrealis:
                question = replace(
                    question,
                    predicate=replace(
                        question.predicate,
                        features=(
                            *(f for f in question.predicate.features if f[0] != "mood"),
                            ("mood", "conditional"),
                        ),
                    ),
                )
            condition = conditional.condition
    implicit_agent = any(
        c.implicit_subject == "addressee" and c.predicate_index is not None
        for c in structure.clauses
    )
    if question is None and implicit_agent and not goal.question_word and self_query(query):
        # A past-tense clause without a subject addresses the bot (the grammar gives it no
        # person to read the subject from). Retell it with the bot as its subject, but only
        # when that clause can be said; otherwise the general limit below stays the answer.
        from debilgi.language.productive_nlg import realize_thought

        retold = parse_proposition(
            structure,
            allow_question=True,
            fallback_subject=ConceptArgument(ConceptLexeme("я", "я", "pronoun")),
            possessive_deixis=True,
            tree=True,
        )
        if retold is not None and realize_thought(retold, limit=1):
            question = retold
            implicit_agent = False
    if context.response_options or question is None and implicit_agent:
        # An observed answer domain is not a copular proposition A=B.
        # Until its predicate is supported, retain the query scope and express
        # the existing epistemic act without manufacturing an embedded clause.
        from debilgi.language.productive_nlg import unresolved_answer_thought

        assertion = replace(
            unresolved_answer_thought(confidence=structure.confidence),
            evidence=(query.exclude_event_id,),
        )
        return DerivedPlan(
            MessagePlan(
                goal, assertion, knowledge_limit=KnowledgeLimit(query, None, goal.question_word)
            ),
            "express_query_unknown",
            (),
            goal.question_word,
        )
    if question is None or not (query.predicate or query.arguments):
        return None
    role = query.requested_role or ("direct_object" if query.predicate_parent else None)
    if question.predicate_class == "existential" and goal.question_word == "что":
        role = "subject"
    wh = next((t for t in structure.tokens if t.semantic_role == "question_target"), None)
    if role is None and wh and wh.part_of_speech == "adjective":
        role = "direct_object" if any(c.direct_object_indices for c in structure.clauses) else None
    prep = (
        structure.tokens[wh.index - 1].lemma
        if wh and wh.index and structure.tokens[wh.index - 1].part_of_speech == "preposition"
        else None
    )
    if prep:
        role = "topic"
    if goal.question_word and role and not any(d.role == role for d in question.dependencies):
        question = replace(
            question,
            dependencies=(
                *question.dependencies,
                SemanticDependency(
                    role,
                    ConceptArgument(
                        ConceptLexeme(
                            goal.question_word,
                            goal.question_word,
                            "adverb" if role in {"location", "cause", "manner"} else "pronoun",
                            wh.grammatical_features if wh else (),
                        ),
                        preposition=prep,
                        grammatical_case=(
                            "nominative"
                            if query.requested_role == "predicate_complement"
                            else dict(wh.grammatical_features).get("case")
                            if wh
                            else None
                        ),
                    ),
                ),
            ),
        )
    question = replace(question, clause_kind="question", focus_role=role)
    # Reuse the epistemic predicate, replacing its old generic answer/topic
    # arguments by an explicitly scoped interrogative complement.
    from debilgi.language.productive_nlg import unresolved_answer_thought

    assertion = unresolved_answer_thought(confidence=0.95)
    assertion = replace(
        assertion,
        dependencies=tuple(d for d in assertion.dependencies if d.role == "subject"),
        evidence=(query.exclude_event_id,),
    )
    return DerivedPlan(
        MessagePlan(
            goal,
            assertion,
            knowledge_limit=KnowledgeLimit(query, question, goal.question_word),
            condition=condition,
        ),
        "express_query_unknown",
        (),
        goal.question_word,
    )


def decide_response_mode(
    context,
    planning,
    hypotheses=(),
    *,
    clarification_allowed=True,
    observed_reaction=False,
    hook_allowed=True,
    observed_forms=(),
    hypothesis_allowed=True,
):
    """Commit the response act before any wording or surface scores exist."""
    from debilgi.language.dialogue_learning import reactive_act

    structure = context.structure
    intent_confidence = max((h.confidence for h in hypotheses), default=0.0)
    form = context.turn_form
    base = dict(
        parse_confidence=structure.confidence,
        intent_confidence=intent_confidence,
        turn_kind=form.kind if form else None,
        tone=form.tone if form else (),
        selection_context={
            "enabled": False,
            "personality": asdict(context.personality) if context.personality else {},
            "affect": dict(context.affect_features),
            "relation": asdict(context.relation.vector) if context.relation else {},
        },
    )
    if form is not None and (not form.addressed or form.kind == "empty"):
        return ResponseDecision(
            "silence_or_unknown",
            "not_addressed" if not form.addressed else "empty_turn",
            "withhold",
            **base,
            fact_confidence=None,
        )
    bound = next(
        (
            p
            for p in planning.proposals
            if p.operation
            in {
                "ask_unbound_variable",
                "bind_delivered_variable",
                "explain_delivered_basis",
                "verify_reported_position",
                "supposition",
            }
        ),
        None,
    )
    if bound is not None:
        return ResponseDecision(
            "answer",
            "bound_interrogative_variable",
            "clarify_variable",
            **base,
            fact_confidence=None,
            mixture=bound.plan.mixture,
            known_slots=bound.plan.known_slots,
            activation=(
                context.prior_bot_turn.get("activation", 0.0) if context.prior_bot_turn else 0.0
            ),
        )
    task = context.dialogue_situation.active_task if context.dialogue_situation else None
    if task and task.goal == "perform_utterance":
        return ResponseDecision(
            "answer", "explicit_utterance_request", "perform", **base, fact_confidence=None
        )
    state = context.dialogue_state or analyze_dialogue_state(
        context.prompt,
        context.dialogue_messages,
        structure=structure,
    )
    gap = next(
        (
            r
            for r in state.unresolved_references
            if structure.supported
            and len(structure.clauses) == 1
            and r.kind == "explicit_pronoun"
            and len({a.lemma for a in r.alternatives}) >= 2
        ),
        None,
    )
    if gap is not None:
        options = tuple(dict.fromkeys(a.lemma for a in gap.alternatives))[:3]
        return ResponseDecision(
            "clarification" if clarification_allowed else "silence_or_unknown",
            (
                "ambiguous_reference"
                if clarification_allowed
                else "clarification_chain_budget_exhausted"
            ),
            "clarify_reference" if clarification_allowed else "withhold",
            **base,
            fact_confidence=None,
            ambiguous_slot=gap.semantic_role,
            alternatives=options,
            expected_information_gain=log2(len(options)),
        )
    if (
        context.response_options
        and knowledge_limit_plan(context) is not None
        and not any(p.plan.status == "inform" and p.premises for p in planning.proposals)
    ):
        return ResponseDecision(
            "silence_or_unknown",
            "answer_domain_has_no_supported_fact",
            "express_limit",
            **base,
            fact_confidence=None,
        )
    reaction = reactive_act(context.prompt)
    claim_result = context.claim_result
    confirmed_own_action = (
        claim_result is not None
        and claim_result.status == "Known"
        and any(c.source_kind == "agent_observed" for c in claim_result.claims)
        and not any(c.clause_kind == "question" for c in structure.clauses)
    )  # «ну ты же X»: the statement about the agent is verified by its own record
    factual_context = (
        form is None
        or confirmed_own_action
        or planning.objective.kind in {"slot", "polar", "content"}
        or reaction is None
        and any(
            c.predicate_index is not None
            and c.clause_kind == "statement"
            and c.implicit_subject not in {"speaker", "addressee"}
            and not any(
                structure.tokens[i].lemma in {"я", "ты", "мы", "вы"} for i in c.subject_indices
            )
            for c in structure.clauses
        )
    )
    delivered = tuple(
        p
        for p in planning.proposals
        if any(e.kind == "sent_turn" and e.evidence_id.startswith("receipt:") for e in p.premises)
    )
    if (factual_context or delivered) and any(
        p.plan.status == "inform"
        and p.plan.claim.clause_kind != "question"
        or p in delivered
        and p.plan.knowledge_limit is not None
        for p in planning.proposals
    ):
        return ResponseDecision(
            "answer",
            "relevant_evidence_found",
            "inform",
            **base,
            fact_confidence=max(p.confidence for p in planning.proposals),
            mixture=(("known", 1.0),),
            activation=(
                context.prior_bot_turn.get("activation", 0.0)
                if delivered and context.prior_bot_turn
                else 0.0
            ),
            known_slots=tuple(s for p in delivered for s in p.plan.known_slots),
        )

    def free_reaction(act):
        if form is not None and not observed_reaction:
            return ResponseDecision(
                "silence_or_unknown",
                "no_observed_social_form",
                "withhold",
                **base,
                fact_confidence=None,
            )
        cores = [
            f
            for f in observed_forms
            if f.component == "core"
            and f.activation > 0
            and not f.recovery_only
            and (f.clause_kind != "question" or hook_allowed)
        ]
        if observed_forms and not cores:
            return ResponseDecision(
                "silence_or_unknown",
                "zero_rule_activation",
                "withhold",
                **base,
                fact_confidence=None,
                confidence_level="silence",
            )
        if cores:
            eligible = [f for f in cores if hypothesis_allowed or f.confidence_level == "answer"]
            if not eligible:
                return ResponseDecision(
                    "silence_or_unknown",
                    "hypothesis_chain_budget_exhausted",
                    "withhold",
                    **base,
                    fact_confidence=None,
                    confidence_level="silence",
                )
            winner = max(eligible, key=lambda f: (f.activation, f.support, f.rule_id or ""))
            level = winner.confidence_level
            return ResponseDecision(
                "free_form",
                "activated_transformation",
                act,
                **base,
                fact_confidence=None,
                allow_hook_question=hook_allowed,
                confidence_level=level,
                target_meaning=winner.meaning,
                rule_id=winner.rule_id,
                rule_weight=winner.rule_weight,
                rule_source_ids=tuple(
                    dict.fromkeys(
                        e for f in cores if f.rule_id == winner.rule_id for e in f.evidence_ids
                    )
                ),
                activation=winner.activation,
                input_features=winner.input_features,
                target_features=winner.target_features,
                mixture=winner.mixture,
                known_slots=winner.known_slots,
                hypothesis=winner.hypothesis,
                compatible_supports=tuple(
                    {
                        "meaning": f.meaning,
                        "activation": f.activation,
                        "rule_id": f.rule_id,
                        "rule_weight": f.rule_weight,
                        "source_ids": f.evidence_ids,
                        "hypothesis": f.hypothesis,
                    }
                    for f in sorted(
                        cores, key=lambda f: (-f.activation, -f.support, f.rule_id or "")
                    )[:16]
                ),
            )
        return ResponseDecision(
            "free_form",
            "reactive_" + act,
            act,
            **base,
            fact_confidence=None,
            allow_hook_question=hook_allowed,
        )

    from debilgi.language.canonical_claims import self_query

    explicit_unknown = bool(
        context.claim_result is not None
        and context.claim_result.status == "Unknown"
        and self_query(context.claim_result.query)
        and knowledge_limit_plan(context) is not None
    )
    if (reaction or observed_reaction and form is None) and not explicit_unknown:
        return free_reaction(reaction or "fragment")
    if planning.objective.kind in {"slot", "polar", "content"}:
        expressible = knowledge_limit_plan(context) is not None
        bid = (
            form is not None
            and form.addressed
            and structure.tokens
            and not planning.objective.question_word
            and not any(t.part_of_speech == "noun" for t in structure.tokens)
            and any(
                dict(t.grammatical_features).get("tense") == "future"
                and dict(t.grammatical_features).get("person") == "second"
                for t in structure.tokens
            )
        )
        if (not expressible or bid) and form is not None and form.addressed:
            return free_reaction(form.kind)
        reason = (
            "query_has_no_supported_fact"
            if expressible
            else "unsupported_query_construction"
            if context.claim_result is not None
            else "query_not_executed"
        )
        return ResponseDecision(
            "silence_or_unknown",
            reason,
            "express_limit" if expressible else "withhold",
            **base,
            fact_confidence=None,
        )
    if form is not None and form.addressed:
        return free_reaction(form.kind)
    return ResponseDecision(
        "silence_or_unknown", "no_reactive_evidence", "withhold", **base, fact_confidence=None
    )


def compatible_response_decisions(decision, forms, *, hypothesis_allowed=True, limit=4):
    """Keep the original choice first, then independently supported alternatives.

    The budget bounds extra search, never removes the original reply path.
    Different wording alone does not manufacture another semantic candidate.
    """
    review = decision.selection_context if decision.selection_context is not None else {}
    review["enabled"] = limit > 1
    review["phase"] = "supported"
    yield decision
    if decision.mode != "free_form" or limit <= 1:
        return
    seen = {(decision.target_meaning, decision.hypothesis, False)}
    committed_support = max(
        (
            f.support
            for f in forms
            if f.component == "core" and f.meaning == decision.target_meaning
        ),
        default=0,
    )
    normal = sorted(
        (f for f in forms if not f.recovery_only),
        key=lambda f: (-f.activation, -f.support, f.rule_id or ""),
    )
    recovery = sorted(
        (f for f in forms if f.recovery_only),
        key=lambda f: (f.clause_kind == "question", -f.support, f.rule_id or ""),
    )
    count = 1
    for form in (*normal, *recovery):
        key = (form.meaning, form.hypothesis, form.recovery_only)
        if (
            form.component not in {"core", "reserve"}
            or form.activation <= 0
            or key in seen
            or form.clause_kind == "question"
            and not decision.allow_hook_question
            # An alternative meaning needs independent support too: a one-off observation
            # does not compete with a repeatedly observed committed reaction.
            or not form.recovery_only
            and form.support < 2 <= committed_support
        ):
            continue
        if form.recovery_only:
            if not review.get("failures") or any(c["passed"] for c in review.get("considered", ())):
                review.setdefault("scope_candidates", []).append(
                    {
                        "rule_id": form.rule_id,
                        "meaning": form.meaning,
                        "source_ids": form.evidence_ids,
                        "surface_checked": False,
                        "eligible": False,
                        "reason": "lower_level_not_needed",
                    }
                )
                seen.add(key)
                continue
            review["phase"] = "clarification" if form.clause_kind == "question" else "tone"
            review["reactivation"] = {
                "source": "reader_failure",
                "meaning": form.meaning,
                "level": review["phase"],
                "source_ids": form.evidence_ids,
                "questions": list(review.get("questions", ())),
            }
        elif recovery and count >= limit - 1:
            continue  # reserve one bounded attempt for a supported lower level
        level = form.confidence_level
        if level == "hypothesis" and not hypothesis_allowed:
            continue
        seen.add(key)
        yield replace(
            decision,
            target_meaning=form.meaning,
            rule_id=form.rule_id,
            rule_weight=form.rule_weight,
            rule_source_ids=form.evidence_ids,
            activation=form.activation,
            input_features=form.input_features,
            target_features=form.target_features,
            mixture=form.mixture,
            known_slots=form.known_slots,
            hypothesis=form.hypothesis,
            confidence_level=level,
        )
        count += 1
        if count >= limit:
            return


def response_self_questions(context, forms=(), proposals=()):
    """Inspect missing variables using existing graph/state/receipt evidence only."""
    from debilgi.language.dialogue_learning import ellipsis_variable

    decision = context.response_decision
    if decision is not None and decision.selection_context is not None:
        decision.selection_context["enabled"] = True
        for proposal in proposals:
            if proposal.plan.deliberation is not decision.selection_context:
                proposal.plan.deliberation["control"] = decision.selection_context
        if decision.mode != "free_form" and not context.response_options:
            from time import perf_counter

            from debilgi.language.communication import PendingText, ResponseComposer

            review = decision.selection_context
            composer = ResponseComposer(context.structure)
            started, previews = perf_counter(), []
            playing = any(dict(p.plan.hypothesis).get("kind") == "fantasy" for p in proposals)
            for proposal in proposals:
                if len(previews) >= 4 or perf_counter() - started >= 1.0:
                    break
                # (next to a game the honest limit is previewed too, so that it stays viable)
                if not any(e.kind in {"canonical_claim", "sent_turn", "fantasy_frame"}
                           for e in proposal.premises) and not (
                               playing and proposal.plan.knowledge_limit is not None):
                    continue
                message = replace(proposal.plan, answer_shape="sentence", focus_role=None)
                # Inspect the original form, not a recovery, before choosing
                # among proved facts. A failed first choice cannot hide another.
                surfaces = composer.realize(
                    PendingText(message, (*proposal.patterns, *context.dependency_patterns), 3)
                )
                previews.append(
                    {
                        "plan_key": proposal.plan.key,
                        "passed": bool(surfaces),
                        "surfaces": len(surfaces),
                        "source_ids": proposal.plan.claim.evidence,
                    }
                )
            review["fact_previews"] = previews
            review["viable_plan_keys"] = [p["plan_key"] for p in previews if p["passed"]]
    variable = ellipsis_variable(context.prompt)
    prior = context.prior_bot_turn
    state = context.dialogue_state
    slots = tuple(dict.fromkeys(s for f in forms for s in f.known_slots))
    values = [
        {
            "slot": "addressee",
            "answer": context.turn_form.address_evidence
            if context.turn_form and context.turn_form.addressed
            else None,
            "source_ids": (context.source_event_id,),
        },
        {
            "slot": "evaluation_sign",
            "answer": next((f.event_valence for f in forms if f.event_valence is not None), None),
            "source_ids": tuple(dict.fromkeys(e for f in forms for e in f.activation_source_ids)),
        },
        {
            "slot": "time",
            "answer": tuple(
                dict.fromkeys(
                    (k, v)
                    for t in context.structure.tokens
                    for k, v in t.grammatical_features
                    if k == "tense"
                )
            ),
            "source_ids": (context.source_event_id,),
        },
    ]
    if variable:
        bound = prior and not any(p.operation == "ask_unbound_variable" for p in proposals)
        values.append(
            {
                "slot": variable["role"],
                "features": variable.get("features", {}),
                "answer": prior.get("response_text") if bound else None,
                "source_ids": (prior["receipt_id"],) if bound else (),
                "binding": slots,
                "unresolved_references": tuple(r.kind for r in state.unresolved_references)
                if state
                else (),
            }
        )
    return tuple(values)


def response_act_plan(context, decision, forms=()):
    """A social act or concrete reference question has a plan before its form."""
    from debilgi.language.communication import ClarificationNeed

    goal = reader_goal(context.structure)
    if decision.mode == "clarification":
        if decision.ambiguous_slot is None:
            return None
        # Ask which reference the user intends. Repeating two factual
        # alternatives ("X left or Y left?") would presuppose an unproved event.
        questions = tuple(
            ClauseThought(
                ConceptLexeme("спрашивать", "спрашивать", "verb", (("tense", "present"),)),
                (
                    SemanticDependency(
                        "subject", ConceptArgument(ConceptLexeme("ты", "ты", "pronoun"))
                    ),
                    SemanticDependency(
                        "topic",
                        ConceptArgument(
                            ConceptLexeme(value, value, "noun"),
                            preposition="о",
                            grammatical_case="prepositional",
                        ),
                    ),
                ),
                clause_kind="question",
                confidence=0.95,
                evidence=(context.source_event_id,),
            )
            for value in decision.alternatives
        )
        return DerivedPlan(
            MessagePlan(
                goal,
                questions[0],
                additional_claims=questions[1:],
                clarification=ClarificationNeed(
                    decision.ambiguous_slot,
                    decision.alternatives,
                    decision.expected_information_gain,
                ),
            ),
            "clarify_interlocutor:reference",
            (),
        )
    if decision.mode == "free_form":
        from debilgi.language.affective_stance import reaction_alignment
        from debilgi.language.dialogue_acts import analyze_dialogue_act
        from debilgi.language.dialogue_move_planner import plan_dialogue_moves

        review = decision.selection_context or {}
        forms = tuple(
            replace(f, component="core") if f.recovery_only else f
            for f in forms
            if not f.recovery_only or review.get("phase") in {"tone", "clarification"}
        )
        moves = plan_dialogue_moves(
            analyze_dialogue_act(context.structure),
            prompt=context.prompt,
            observed_forms=tuple(
                f
                for f in forms
                if f.component != "core"
                or decision.target_meaning is None
                or f.meaning == decision.target_meaning
            ),
            reaction_act=decision.act,
            allow_hook_question=decision.allow_hook_question,
            alignment=reaction_alignment(
                form=context.turn_form,
                affect_features=context.affect_features,
                personality=context.personality,
            ),
        )
        if not moves:
            return None
        confidence = decision.activation or moves[0].thought.confidence
        if review.get("enabled"):
            seen, members, residual = set(), [], 1.0
            for f in sorted(forms, key=lambda f: -f.activation):
                if (
                    f.component != "core"
                    or f.meaning != decision.target_meaning
                    or f.recovery_only != (review.get("phase") in {"tone", "clarification"})
                ):
                    continue
                independent = set(f.evidence_ids) - seen
                if not independent:
                    continue
                strength = f.activation * len(independent) / max(1, len(set(f.evidence_ids)))
                residual *= 1 - min(0.95, strength)
                seen.update(independent)
                members.append(
                    {"rule_id": f.rule_id, "source_ids": sorted(independent), "strength": strength}
                )
            confidence = max(confidence, min(0.95, 1 - residual))
            review.setdefault("coalitions", {})[decision.target_meaning] = {
                "members": members,
                "strength": confidence,
                "independent_sources": len(seen),
            }
        thought = replace(moves[0].thought, confidence=confidence)
        return DerivedPlan(
            MessagePlan(
                goal,
                thought,
                speech_act=decision.act,
                mixture=decision.mixture,
                known_slots=decision.known_slots,
                hypothesis=decision.hypothesis,
                deliberation=review,
            ),
            "compose_reaction:" + moves[0].move,
            (),
        )
    return None


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
    counterpart = deictic_counterpart(lemma) if speaker == "user" else None
    if counterpart:
        cell = person_cell(counterpart)
        lemma = surface = counterpart
        if cell and cell[2] == "personal":
            features["person"] = cell[0]
    return ConceptLexeme(
        lemma,
        surface,
        token.part_of_speech,
        tuple(features.items()),
        token.entity_kind,
        token.participant.ref if token.participant else None,
    )


def statement_thought(
    text: str | StructuredUtterance, *, speaker: str | None = None
) -> ClauseThought | None:
    """Conservatively preserve one proposition, including its qualifications.

    Unsupported relations are rejected, never silently removed to obtain a
    simpler but materially different fact.
    """
    structure = (
        text if isinstance(text, StructuredUtterance) else analyze_structured_utterance(text)
    )
    if structure.corrections or any(c.unsupported for c in structure.clauses):
        return None
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
        semantic_time=c.semantic_time,
        scopes=c.scopes,
        parser_version=structure.parser_version,
        source_event_id=structure.source_event_id,
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
    from debilgi.language.semantic_graph import proposition_key

    return proposition_key(thought, signed=False)


def _conflicted(record: SemanticEvidence, records: list[SemanticEvidence]) -> bool:
    from debilgi.language.canonical_claims import bound_participants

    key = _proposition_key(record.thought)
    scope = bound_participants(record.thought)
    return any(
        other.thought.negated != record.thought.negated
        and _proposition_key(other.thought) == key
        and bound_participants(other.thought) == scope
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

    if objective.kind != "polar" or objective.predicate not in _ACTION_BEARING_MODAL_PREDICATES:
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
        and not is_response_control(context.turn_prompt or context.prompt)
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
    focus = focus_request(
        context.turn_prompt or context.prompt,
        context.consultation_messages,
        open_request=(
            active_task.objective.request_text
            if active_task and active_task.goal == "answer_question"
            else None
        ),
    )
    context = replace(
        context,
        prompt=focus.request_text,
        turn_prompt=focus.turn_text,
        response_options=focus.alternatives,
        focus_reason=focus.reason,
        dialogue_state=None if focus.request_text != context.prompt else context.dialogue_state,
    )
    structure = context.structure
    objective = objective_for(structure)
    incomplete_modal = _is_incomplete_modal_request(structure, objective)
    from debilgi.language.dialogue_learning import exchange_analysis

    if (
        context.turn_form is not None
        and context.turn_form.kind == "question"
        and not incomplete_modal
        and not is_response_control(context.turn_prompt or context.prompt)
    ):
        # A new missing slot belongs to this turn. Replacing "which one?" by
        # the earlier action question would turn an honest gap into a repeat.
        return replace(context, focus_reason="current_question_form")
    # Contact starts an episode even when an earlier question was unanswered.
    # Only an actual continuation licenses recovering an older request below.
    if is_contact_request(context.prompt) or exchange_analysis(context.prompt)[1] == "greeting":
        return replace(context, focus_reason="current_contact_new_episode")
    if (
        focus.reason in {"current_question", "answer_control_keeps_request"}
        and not incomplete_modal
    ):
        return context
    contact = is_contact_request(context.prompt)
    if not contact and (
        objective.topics or statement_thought(context.structure, speaker="user") is not None
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
    from debilgi.language.request_focus import is_invention_request

    # A request with its own predicate/slot is independent of earlier requests.
    if is_invention_request(context.turn_prompt or context.prompt, bare=False):
        return context  # «расскажи что-нибудь смешное» asks for play, not the earlier answer
    if (
        objective.kind in {"slot", "polar"}
        and (objective.predicate or objective.known_roles)
        and not incomplete_modal
    ) or (objective.kind == "command" and objective.predicate not in _CONTENT_PREDICATES):
        return context
    from debilgi.language.dialogue_learning import ellipsis_variable

    elliptical_cause = ellipsis_variable(context.prompt)
    if (
        elliptical_cause
        and elliptical_cause.get("relation") == "cause"
        and context.prior_bot_turn is None
    ):
        # "Why?" asks for a cause, not for the earlier question: substituting that question
        # would answer a different kind of request. With no delivered turn to bind to,
        # the variable stays unbound and is asked about.
        return replace(context, focus_reason="elliptical_request_recovery")
    continuation = (
        incomplete_modal
        or is_response_control(context.turn_prompt or context.prompt)
        or objective.kind == "command"
        and objective.predicate in _CONTENT_PREDICATES
        or objective.predicate in _ACTION_BEARING_MODAL_PREDICATES
        and not any(c.action_complement_indices for c in structure.clauses)
        or bool(
            structure.clauses
            and structure.clauses[0].elliptical
            and objective.question_word
            and not objective.topics
        )
    )
    if not continuation:
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
    *,
    explore_meanings: bool = False,
) -> PlanningResult:
    context = resolve_evidence_context(context)
    structure = context.structure
    goal, objective = reader_goal(structure), objective_for(structure)
    state = context.dialogue_state or analyze_dialogue_state(
        context.prompt, context.dialogue_messages, structure=structure
    )
    bindings = dict(state.resolved_bindings)
    if bindings:
        known = tuple((role, bindings.get(role, lemma)) for role, lemma in objective.known_roles)
        objective = replace(objective, known_roles=known)
        goal = replace(goal, known_roles=known)
        if objective.kind == "content" and not objective.topics and "topic" in bindings:
            objective = replace(objective, topics=(bindings["topic"],))
            goal = replace(goal, topics=objective.topics)
    # Self content comes exclusively from the same canonical graph. Corpus and
    # personality may shape the realization, never fill an absent biography.
    from debilgi.language.canonical_claims import self_query

    if context.claim_result is not None and (
        self_query(context.claim_result.query)
        or objective.kind in {"slot", "polar"}
        and context.claim_result.claims
    ):
        if not self_query(context.claim_result.query):
            semantic = _plan_meaning_graph(context, goal)
            if semantic:
                evidence = tuple(dict.fromkeys(p for s in semantic for p in s.premises))
                return PlanningResult(objective, semantic, evidence, ())
        records = tuple(
            SemanticEvidence(
                c.claim_id,
                "canonical_claim",
                c.source_event_id,
                answer_thought(c, context.claim_result.query),
            )
            for c in context.claim_result.claims
        )
        proposals = plans_for_claim_result(context.claim_result, structure)
        if not proposals and (limit := knowledge_limit_plan(context)) is not None:
            proposals = (limit,)
        return PlanningResult(
            objective, proposals, records, () if records else ("query_has_no_supported_fact",)
        )
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
            bool(object_terms & (set(content_lemmas(title)) | {title.casefold().replace("ё", "е")}))
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
                lexeme.lemma in {"я", "мы", "мой", "наш"}
                for dependency in thought.dependencies
                for lexeme in (
                    dependency.argument.head,
                    *dependency.argument.modifiers,
                    *dependency.argument.postmodifiers,
                )
            )
        ):
            records.append(_record("knowledge", text, thought))
    for claim in context.assertion_memory.assertions:
        if claim.speaker != "agent" and context.claim_result is None:
            records.append(_record("dialogue_report", claim.source_event_id, claim.thought))
    if context.claim_result is not None:
        for claim in context.claim_result.claims:
            thought = answer_thought(claim, context.claim_result.query)
            records.append(
                SemanticEvidence(claim.claim_id, "canonical_claim", claim.source_event_id, thought)
            )
    legacy_reports = context.proposition_reports if context.claim_result is None else ()
    for source_event_id, text in legacy_reports:
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
            if record.kind != "canonical_claim" and not _supports(objective, record.thought):
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
    if (
        context.claim_result is None
        and objective.kind in {"content", "slot"}
        and objective.predicate in _REPORT_PREDICATES
    ):
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
        current = statement_thought(context.structure, speaker="user")
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
    if not unique and objective.kind in {"slot", "polar", "content"}:
        limit = knowledge_limit_plan(context)
        if limit is not None:
            unique[limit.plan.key] = limit
    unresolved = (
        ()
        if unique
        else tuple(
            dict.fromkeys(
                s.reason for s in structure.unresolved_slots if s.reason != "requested_slot"
            )
        )
        or (
            "query_has_no_supported_fact"
            if objective.kind in {"slot", "polar", "content"}
            else "no_relevant_continuation",
        )
    )
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
        predicate_features.update({"tense": "present", "number": "singular", "person": "first"})
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
            # Direct persisted facts have one authority: the structural query.
            # Rules may still derive a new fact through the existing proof engine.
            if context.claim_result is not None and not proof.rule_ids and query.kind != "cause":
                continue
            if context.claim_result is not None:
                request = context.claim_result.query
                roles = {d.role: argument_ref(d.argument) for d in proof.thought.dependencies}
                if any(roles.get(role) != ref for role, ref in request.arguments if role != "*"):
                    continue
                time = proof.thought.semantic_time
                if (
                    request.valid_from
                    and request.valid_to
                    and not (
                        time.start
                        and time.end
                        and time.start < request.valid_to
                        and time.end > request.valid_from
                    )
                ):
                    continue
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
                # A directly retained effect can occupy the proof key before
                # forward chaining reaches its causal relation. Retain that
                # relation only when its antecedent is independently supported.
                if not rules:
                    for rule in meanings:
                        if (
                            rule.kind != "cause"
                            or len(rule.premises) != 1
                            or not any(
                                proposition_key(t) == proposition_key(proof.thought)
                                for t in rule.conclusions
                            )
                        ):
                            continue
                        antecedent = next(
                            (
                                p
                                for p in proofs
                                if proposition_key(p.thought) == proposition_key(rule.premises[0])
                            ),
                            None,
                        )
                        if antecedent:
                            rules = [rule]
                            proof = replace(
                                proof,
                                source_event_ids=tuple(
                                    dict.fromkeys(
                                        (
                                            *proof.source_event_ids,
                                            *antecedent.source_event_ids,
                                            rule.source_event_id,
                                        )
                                    )
                                ),
                                rule_ids=(*proof.rule_ids, rule.meaning_id),
                                claim_ids=tuple(
                                    dict.fromkeys((*proof.claim_ids, *antecedent.claim_ids))
                                ),
                            )
                            break
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
            claim = replace(
                proof.thought,
                evidence=tuple(dict.fromkeys((*proof.claim_ids, *proof.source_event_ids))),
            )
            return (
                DerivedPlan(
                    MessagePlan(goal, claim, reason=reason),
                    "infer_from_learned_relations",
                    premises,
                    "почему" if query.kind == "cause" else "polarity",
                ),
            )
    return ()
