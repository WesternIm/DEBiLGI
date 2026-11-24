"""An intended reader update precedes wording and survives surface realization.

The producer supplies propositions, not a reply string. This layer records what
the reader should learn, checks the goal, and checks the rendered roles against
the original proposition. It never creates knowledge from a fluent surface.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

from debilgi.language.canonical_claims import ClaimQuery
from debilgi.language.grammar import SoftGrammar
from debilgi.language.morphology import DEFAULT_MORPHOLOGY, person_cell
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    ProductiveSurface,
    SemanticDependency,
    identity_graph,
    realize_argument,
    realize_thought,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.language.tokenizer import normalize_chat_token, normalize_word, tokenize

_INTERROGATIVES = frozenset(
    {
        "что",
        "кто",
        "где",
        "куда",
        "откуда",
        "когда",
        "как",
        "какой",
        "почему",
        "зачем",
        "сколько",
        "насколько",
    }
)


@dataclass(frozen=True, slots=True)
class ReaderGoal:
    kind: str
    question_word: str | None
    subject: tuple[str, ...]
    predicate: str | None
    addressed: bool
    topics: tuple[str, ...] = ()
    known_roles: tuple[tuple[str, str], ...] = ()


def reader_goal(structure: StructuredUtterance) -> ReaderGoal:
    questions = tuple(c for c in structure.clauses if c.clause_kind == "question")
    clause = questions[-1] if questions else next(iter(structure.clauses), None)
    if clause is None:
        return ReaderGoal("respond", None, (), None, False)
    word = (
        structure.tokens[clause.question_target_index].lemma
        if clause.question_target_index is not None
        else next(
            (
                token.lemma
                for token in structure.tokens[clause.token_start : clause.token_end]
                if token.lemma in _INTERROGATIVES
                and not (
                    clause.predicate_index is None
                    and "rhetorical_challenge" in token.pragmatic_functions
                    and token.semantic_role != "question_target"
                )
            ),
            None,
        )
        if clause.clause_kind == "question" or structure.text.rstrip().endswith("?")
        else None
    )
    from debilgi.language.dialogue_learning import ellipsis_variable

    variable = ellipsis_variable(structure.text)
    if variable and variable.get("relation") == "cause":
        word = variable.get("word", "почему")
    subject = tuple(structure.tokens[i].lemma for i in clause.subject_indices)
    predicate = (
        structure.tokens[clause.predicate_index].lemma
        if clause.predicate_index is not None
        else clause.implicit_predicate
    )
    addressed = clause.implicit_subject == "addressee" or bool(set(subject) & {"ты", "вы"})
    topics = tuple(
        dict.fromkeys(
            t.lemma
            for t in structure.tokens
            if t.part_of_speech == "noun" and t.semantic_role != "vocative"
        )
    )
    explicit_topics = tuple(
        structure.tokens[i].lemma
        for c in structure.clauses
        for comp in c.complements
        if comp.role == "topic"
        for i in comp.object_indices
    )
    topics = explicit_topics or topics
    known_roles = tuple(
        (role, {"ты": "я", "вы": "мы", "я": "ты", "мы": "вы"}.get(token.lemma, token.lemma))
        for role, indices in (
            ("subject", clause.subject_indices),
            ("direct_object", clause.direct_object_indices),
            ("indirect_object", clause.indirect_object_indices),
        )
        for i in indices
        for token in (structure.tokens[i],)
        if i != clause.question_target_index
    )
    return ReaderGoal(
        "answer_slot" if word else "respond",
        word,
        subject,
        predicate,
        addressed,
        topics,
        known_roles,
    )


@dataclass(frozen=True, slots=True)
class KnowledgeLimit:
    """Unknown of a particular graph query, never a technical processing failure."""

    query: ClaimQuery
    question: ClauseThought | None
    question_word: str | None


@dataclass(frozen=True, slots=True)
class ClarificationNeed:
    ambiguous_slot: str
    alternatives: tuple[str, ...]
    expected_information_gain: float


@dataclass(frozen=True, slots=True)
class MessagePlan:
    goal: ReaderGoal
    claim: ClauseThought
    reason: ClauseThought | None = None
    condition: ClauseThought | None = None
    follow_up: ClauseThought | None = None
    answer_shape: str = "sentence"
    focus_role: str | None = None
    additional_claims: tuple[ClauseThought, ...] = ()
    reason_condition: ClauseThought | None = None
    knowledge_limit: KnowledgeLimit | None = None
    speech_act: str | None = None
    clarification: ClarificationNeed | None = None
    mixture: tuple[tuple[str, float], ...] = ()
    known_slots: tuple[tuple[str, str, str], ...] = ()
    hypothesis: tuple[tuple[str, str], ...] = ()
    deliberation: dict = field(default_factory=dict, compare=False)

    @property
    def review(self):
        return self.deliberation.get("control", self.deliberation)

    @property
    def status(self) -> str:
        if self.claim.predicate_class == "dialogue_act":
            return "social"
        if self.knowledge_limit is not None:
            return "express_limit"
        if self.claim.clause_kind == "question":
            return "clarify"
        if (
            self.claim.negated
            and self.claim.predicate.lemma in {"знать", "понимать"}
            and self.goal.predicate != self.claim.predicate.lemma
            and self.claim.response_polarity is None
        ):
            return "express_limit"
        return "inform"

    @property
    def key(self) -> str:
        # The identity is the semantic graph, never wording or a source label.
        data = identity_graph(self)
        data.pop("deliberation", None)  # choice state is not propositional identity
        # Additive optional acts must not orphan existing learned plan keys.
        for name in ("knowledge_limit", "speech_act", "clarification"):
            if data[name] is None:
                data.pop(name)
        for name in ("mixture", "known_slots", "hypothesis"):
            if not data[name]:
                data.pop(name)
        for name in ("claim", "reason", "condition", "follow_up", "reason_condition"):
            if data[name]:
                data[name].pop("confidence", None)
                data[name].pop("evidence", None)
        for claim in data["additional_claims"]:
            claim.pop("confidence", None)
            claim.pop("evidence", None)

        def legacy_graph(value):
            # Stage 1 annotations travel with the plan. Existing policy IDs
            # remain stable until the canonical claim identity in stage 2.
            if isinstance(value, dict):
                return {
                    k: legacy_graph(v)
                    for k, v in value.items()
                    if k
                    not in {
                        "participant_ref",
                        "semantic_time",
                        "scopes",
                        "parser_version",
                        "source_event_id",
                        "exclude_event_id",
                        "snapshot_at",
                        "intent_evidence_ids",
                    }
                }
            if isinstance(value, (tuple, list)):
                return [legacy_graph(v) for v in value]
            return value

        return hashlib.sha256(json.dumps(legacy_graph(data), sort_keys=True).encode()).hexdigest()[
            :24
        ]


# Own states the agent can report from its own operations, by the verb that asks for them
# (as «изучить» projects a learning operation): «что ты думаешь?» asks for what it is
# working out, which is the open questions its sleep will take up.
OWN_STATE_PROJECTIONS = {"думать": "open_questions"}

# Epistemic operators of an opinion; a plan is marked only by one of these.
EPISTEMIC_OPERATORS = frozenset({"наверное", "возможно", "вероятно"})

# The frame verb of a stated opinion (4.45, flag only): «думать» in the first person is what the
# people of the server say to mark a view as theirs, not a fact. The word itself is taken from
# their messages (`learned_stance_marker`), a plan with this kind carries it as its operator.
OPINION_FRAME_VERB = "думать"

# The verb the agent uses to report one of its own operations (as OWN_STATE_PROJECTIONS maps
# «думать» to the open questions): the operation of making something up is reported as a joke.
OWN_OPERATION_VERBS = {"fantasy": "пошутить"}

# Statuses whose wording is a way of not-having-an-answer or a supposition: only these are
# weighed against the construction the agent has just used (a fact is not penalised for being
# said in a familiar clause shape).
CONSTRUCTION_WEIGHED = frozenset({"express_limit"})
CONSTRUCTION_WEIGHED_STATUSES = frozenset({"hypothesis"})


@lru_cache(maxsize=4096)
def surface_construction(text: str) -> tuple:
    """The clause skeleton of a wording: what kind of sentence it is, not which words.

    Per clause: for a main clause its kind, negation, the person of its subject and whether it
    takes an object or a complement; for a subordinate clause only how it is attached and its
    subject person. A quotation is one more feature. Lemmas never enter, so «не знаю, X ли я»
    with any verb is one construction, and so is «Я Y.» with any intransitive verb.
    """
    structure = analyze_structured_utterance(text)
    attached = {
        link.child_clause_index: link.relation.partition(":")[0]
        for link in structure.clause_links
    }
    skeleton = []
    for clause in structure.clauses:
        subject = (structure.tokens[clause.subject_indices[0]]
                   if clause.subject_indices else None)
        cell = person_cell(subject.lemma) if subject is not None else None
        person = (cell[0] if cell else "third") if subject is not None else (
            "elided" if clause.implicit_subject in {"speaker", "addressee"} else "none")
        if clause.clause_index in attached:
            skeleton.append(("sub", attached[clause.clause_index], person))
        else:
            skeleton.append((
                "main", clause.clause_kind, clause.negated, person,
                bool(clause.direct_object_indices or clause.complements
                     or clause.predicate_complement_indices)))
    quoted = any(token.semantic_role == "quoted_content" for token in structure.tokens)
    return tuple(skeleton), quoted


def construction_repetition(text: str, recent: tuple[str, ...]) -> float:
    """1/(distance) of the latest earlier reply built on the same clause skeleton, else 0."""
    mine = surface_construction(text)
    return max((1 / (index + 1) for index, old in enumerate(recent)
                if surface_construction(old) == mine), default=0.0)


def is_amusement_marker(word: str) -> bool:
    """A laughter cue as the pragmatic reader recognises it (not a list of such words)."""
    from debilgi.language.pragmatics import analyze_pragmatics

    return "amusement" in analyze_pragmatics(word).functions


def plan_issues(plan: MessagePlan) -> tuple[str, ...]:
    """Check the intended update before a word has been inflected."""
    opinion = dict(plan.hypothesis)
    marked_opinion = (opinion.get("kind") in {"experience_opinion", "own_evidence_opinion"}
                      and opinion.get("operator") in EPISTEMIC_OPERATORS
                      and opinion.get("operator_source") in plan.claim.evidence
                      and plan.speech_act == "opinion")
    # A stated view (4.45): marked by the observed first-person frame verb, never a bare assertion
    marked_opinion = marked_opinion or (
        opinion.get("kind") == "opinion" and plan.speech_act == "opinion"
        and plan.claim.clause_kind == "statement"
        and DEFAULT_MORPHOLOGY.lemma(opinion.get("operator") or "") == OPINION_FRAME_VERB
        and opinion.get("operator_source") in plan.claim.evidence)
    # A made-up statement about the agent is honest only when it is marked as play: by the
    # grammar of the unreal («бы») or by a laughter marker the agent has observed people use.
    marked_fantasy = (
        opinion.get("kind") == "fantasy"
        and plan.speech_act == "opinion"
        and plan.claim.clause_kind == "statement"
        and any(d.role == "subject" and d.argument.head.lemma == "я"
                for d in plan.claim.dependencies)
        and (dict(plan.claim.predicate.features).get("mood") == "conditional"
             or bool(opinion.get("operator")))
        and (not opinion.get("operator")
             or (opinion.get("operator_source") in plan.claim.evidence
                 and is_amusement_marker(opinion["operator"])))
    )
    if opinion.get("kind") == "fantasy" and not marked_fantasy:
        return ("unmarked_fantasy",)
    if (plan.hypothesis and plan.claim.predicate_class != "dialogue_act"
            and not marked_opinion and not marked_fantasy):
        hypothetical = int(plan.claim.clause_kind != "question")
        known = sum(
            t.clause_kind != "question" and bool(t.evidence) for t in plan.additional_claims
        )
        if hypothetical > known:
            return ("hypothetical_assertion_share_exceeds_half",)
    if plan.claim.predicate_class == "dialogue_act":
        return (
            ()
            if plan.speech_act
            and plan.claim.evidence
            and plan.claim.clause_kind == "fragment"
            and not plan.claim.dependencies
            else ("invalid_dialogue_act_plan",)
        )
    issues = list(
        dict.fromkeys(
            issue
            for thought in (
                plan.claim,
                plan.reason,
                plan.condition,
                plan.follow_up,
                plan.reason_condition,
                *plan.additional_claims,
                *((plan.knowledge_limit.question,) if plan.knowledge_limit else ()),
            )
            if thought
            for issue in (
                *(("missing_predicate",) if not thought.predicate.lemma else ()),
                *(
                    ("missing_subject",)
                    if not any(d.role == "subject" for d in thought.dependencies)
                    else ()
                ),
            )
        )
    )
    claim, goal = plan.claim, plan.goal
    # Control-flow outcomes describe the planner, not the conversation. They
    # must never become a proposition merely because grammar can express them.
    internal_evidence = {
        "answer_not_derived",
        "no_eligible_grounded_explanation",
        "target_in_local_knowledge",
        "target_absent_from_local_knowledge",
        "target_in_dialogue_memory",
        "target_absent_from_dialogue_memory",
        "complete_structural_parse",
        "target_not_grounded_in_dialogue",
    }
    if any(
        internal_evidence.intersection(thought.evidence)
        for thought in (
            plan.claim,
            plan.reason,
            plan.condition,
            plan.follow_up,
            plan.reason_condition,
            *plan.additional_claims,
            *((plan.knowledge_limit.question,) if plan.knowledge_limit else ()),
        )
        if thought
    ):
        issues.append("internal_outcome_is_not_response_content")
    own_state = (claim.predicate.lemma in OWN_STATE_PROJECTIONS and not claim.negated
                 and plan.knowledge_limit is not None
                 and plan.knowledge_limit.question is not None)
    if plan.knowledge_limit and (
        not (claim.predicate.lemma in {"знать", "помнить"} and claim.negated or own_state)
        or not (plan.knowledge_limit.query.predicate or plan.knowledge_limit.query.arguments)
    ):
        issues.append("invalid_knowledge_limit")
    if plan.clarification and (
        not plan.clarification.ambiguous_slot
        or len(plan.clarification.alternatives) < 2
        or plan.clarification.expected_information_gain <= 0
    ):
        issues.append("clarification_without_information_gain")
    # A compact epistemic limit (no embedded question) carries no roles of the
    # asked query; its own subject only reports not knowing, so the asked
    # participant legitimately recurs there (ты не знаешь кто ты → я не знаю).
    compact_limit = plan.knowledge_limit is not None and plan.knowledge_limit.question is None
    if claim.clause_kind == "statement" and goal.kind == "answer_slot" and not compact_limit:
        roles = {(d.role, d.argument.head.lemma) for d in claim.dependencies}
        if goal.predicate == claim.predicate.lemma:
            for role, lemma in goal.known_roles:
                if (role, lemma) not in roles and any(
                    other != role and value == lemma for other, value in roles
                ):
                    issues.append("question_argument_changed_role")
        if goal.predicate in {None, "быть"}:
            # In a nominal property question the topic is the subject of the
            # requested predication. A report of an assessment can embed that
            # subject as its direct object; it cannot turn it into the quality.
            subjects = {lemma for role, lemma in goal.known_roles if role == "subject"}
            targets = {lemma for role, lemma in roles if role in {"subject", "direct_object"}}
            qualities = {lemma for role, lemma in roles if role == "predicate_complement"}
            if subjects.intersection(qualities) and not subjects.intersection(targets):
                issues.append("question_subject_became_property")
    return tuple(dict.fromkeys(issues))


def surface_issues(thought: ClauseThought, text: str) -> tuple[str, ...]:
    """Check role preservation, not just shared words or grammatical polish.

    Reparse only verifies explicit contradictions; an ambiguous morphological
    reading does not justify changing the intended proposition.
    """
    structure = analyze_structured_utterance(text)
    observed = {t.lemma.casefold() for t in structure.tokens}
    observed.update(t.surface.casefold() for t in structure.tokens)
    observed.update(
        r.lemma.casefold()
        for t in structure.tokens
        for r in DEFAULT_MORPHOLOGY.parse_word(t.surface)
    )

    def head_preserved(argument) -> bool:
        if (
            argument.head.lemma.casefold() in observed
            or argument.head.surface.casefold() in observed
        ):
            return True
        if DEFAULT_MORPHOLOGY.lemma(argument.head.lemma).casefold() in observed:
            return True
        if argument.head.part_of_speech in {"adjective", "participle"}:
            form = (
                "short_adjective"
                if argument.head.part_of_speech == "adjective"
                else "short_participle"
            )
            for number, gender in (
                ("singular", "masculine"),
                ("singular", "feminine"),
                ("singular", "neuter"),
                ("plural", None),
            ):
                features = {"form": form, "number": number}
                if gender:
                    features["gender"] = gender
                short = DEFAULT_MORPHOLOGY.inflect(argument.head.surface, **features)
                if short and short in observed:
                    return True
        return False

    missing = tuple(
        f"missing_{d.role}"
        for d in thought.dependencies
        if d.role
        in {
            "subject",
            "direct_object",
            "indirect_object",
            "action_complement",
            "predicate_complement",
        }
        and d.argument.head.lemma.isalpha()
        and not (d.role == "subject" and ("elided", "true") in d.argument.head.features)
        and not head_preserved(d.argument)
    )
    frames = tuple(
        c
        for c in structure.clauses
        if c.predicate_index is not None
        and structure.tokens[c.predicate_index].lemma == thought.predicate.lemma
    )
    if not frames:
        if thought.predicate_class == "lexical" and thought.predicate.lemma not in observed:
            return (*missing, "missing_predicate")
        return missing  # copulas and analytic predicates need a different inverse grammar
    frame = frames[0]
    issues = list(missing)
    if frame.negated != thought.negated:
        issues.append("changed_negation")
    expected = {d.role: d.argument.head.lemma for d in thought.dependencies}
    actual = {
        "subject": {structure.tokens[i].lemma for i in frame.subject_indices},
        "direct_object": {structure.tokens[i].lemma for i in frame.direct_object_indices},
        "indirect_object": {structure.tokens[i].lemma for i in frame.indirect_object_indices},
    }
    # A swapped actor/patient is a meaning error even when every word survives.
    if (
        expected.get("subject") != expected.get("direct_object")
        and expected.get("subject") in actual["direct_object"]
        and expected.get("direct_object") in actual["subject"]
    ):
        issues.append("swapped_actor_and_object")
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class PendingText:
    """A render request. It deliberately contains no generated answer string."""

    plan: MessagePlan
    patterns: tuple[DependencyPattern, ...]
    limit: int
    exchange_forms: tuple = ()
    source_answers: frozenset[str] = frozenset()
    recent_texts: tuple[str, ...] = ()
    variation_key: str = ""


@dataclass(frozen=True, slots=True)
class PlannedSurface:
    """Adapter for producers during migration; add() queues its request."""

    text: PendingText
    role_order: tuple[str, ...]
    confidence: float
    source: str = "planned"
    construction_id: str | None = None


def _particle_li_order(question, words: list[str], nominal: bool) -> list[str] | None:
    """Place "ли" after the focus of a nominal-predicate question ("рад ли я").

    The subject (leading or trailing in the realized text) follows the particle; a negation
    belongs to its focus word ("не рад ли я"). Verbal questions keep the old placement.
    """
    if not nominal:
        return None
    dependency = next((d for d in question.dependencies if d.role == "subject"), None)
    if dependency is None:
        return None
    subject = realize_argument(
        dependency.argument, predicate_lemma=question.predicate.lemma, role="subject"
    ).split()
    n = len(subject)
    if not subject or len(words) <= n:
        return None
    folded = [w.casefold() for w in subject]
    if [w.casefold() for w in words[:n]] == folded:
        rest = words[n:]
    elif [w.casefold() for w in words[-n:]] == folded:
        rest = words[:-n]
    else:
        return None
    focus = 2 if len(rest) > 1 and rest[0].casefold() == "не" else 1
    return [*rest[:focus], "ли", *subject, *rest[focus:]]


def _is_periodic_sound(word: str) -> bool:
    """Laughter or hesitation repeats a short unit ("ахаха", "хехе"); a typo does not."""
    return len(word) >= 4 and any(
        all(word[i] == word[i + period] for i in range(len(word) - period))
        for period in (1, 2, 3)
    )


def _witnessed_word_surface(surface: str, support: int) -> str | None:
    """Prefer the dictionary form of a witnessed word the input reader already corrects.

    A corpus typo reaches a reply as-is unless morphology knows the word only through a
    spelling correction: then the corrected form of the same lexeme is spoken. A word
    morphology neither knows nor corrects, yet that has known neighbours one edit away (a
    likely typo), is spoken only if several exchanges witnessed it; laughter (a repeating
    sound) is not one.
    Expressive stretching ("оооо") is not a spelling mistake and stays as observed.
    """
    if not surface[:1].isalpha():
        return surface
    word = normalize_word(surface)
    readings = DEFAULT_MORPHOLOGY.parse_word(surface)
    if not readings:
        return surface
    if all(r.corrected_form for r in readings) and normalize_chat_token(word).canonical == word:
        return readings[0].corrected_form
    if (
        not any(r.is_known for r in readings)
        and support < 2
        and DEFAULT_MORPHOLOGY.near_known_words(surface)
        and not _is_periodic_sound(word)
    ):
        return None
    return surface


class ResponseComposer:
    """Per-turn lifecycle: plan -> goal check -> wording -> reader check."""

    def __init__(self, structure: StructuredUtterance) -> None:
        self.structure = structure
        self.goal = reader_goal(structure)
        self.records: list[dict] = []
        self._by_surface: dict[str | tuple[str, str], dict] = {}
        self.deferred = False

    def metadata_for(self, text: str, *, plan_key: str | None = None) -> dict:
        key = (plan_key, text) if plan_key is not None else text
        return self._by_surface.get(key, {})

    def express(
        self,
        thought: ClauseThought,
        patterns: tuple[DependencyPattern, ...] = (),
        *,
        limit: int = 3,
    ) -> tuple[ProductiveSurface | PlannedSurface, ...]:
        return self.compose(MessagePlan(self.goal, thought), patterns, limit=limit)

    def compose(
        self,
        plan: MessagePlan,
        patterns: tuple[DependencyPattern, ...] = (),
        *,
        limit: int = 3,
        exchange_forms: tuple = (),
        source_answers: frozenset[str] = frozenset(),
        recent_texts: tuple[str, ...] = (),
        variation_key: str = "",
    ) -> tuple[ProductiveSurface | PlannedSurface, ...]:
        from debilgi.language import grammarian

        if not grammarian.enabled():
            return self._compose(
                plan, patterns, limit=limit, exchange_forms=exchange_forms,
                source_answers=source_answers, recent_texts=recent_texts,
                variation_key=variation_key,
            )
        # DEBILGI_GRAMMARIAN=1: the path every clause of this message took (network or
        # fallback, and why) goes to the trace
        with grammarian.capture() as records:
            surfaces = self._compose(
                plan, patterns, limit=limit, exchange_forms=exchange_forms,
                source_answers=source_answers, recent_texts=recent_texts,
                variation_key=variation_key,
            )
        if records:
            trace = [record.as_dict() for record in records]
            for surface in surfaces:
                for key in (surface.text, (plan.key, surface.text)):
                    known = self._by_surface.get(key, {})
                    self._by_surface[key] = {**known, "grammarian_trace": trace}
        return surfaces

    def _compose(
        self,
        plan: MessagePlan,
        patterns: tuple[DependencyPattern, ...] = (),
        *,
        limit: int = 3,
        exchange_forms: tuple = (),
        source_answers: frozenset[str] = frozenset(),
        recent_texts: tuple[str, ...] = (),
        variation_key: str = "",
    ) -> tuple[ProductiveSurface | PlannedSurface, ...]:
        if self.deferred:
            return (
                PlannedSurface(
                    PendingText(
                        plan,
                        patterns,
                        limit,
                        exchange_forms,
                        source_answers,
                        recent_texts,
                        variation_key,
                    ),
                    (
                        "subject",
                        "predicate",
                        *(d.role for d in plan.claim.dependencies if d.role != "subject"),
                    ),
                    plan.claim.confidence,
                ),
            )
        result = self.realize(
            PendingText(
                plan,
                patterns,
                limit,
                exchange_forms,
                source_answers,
                recent_texts,
                variation_key,
            )
        )
        review = plan.review
        if review.get("enabled"):
            review.setdefault("considered", []).append(
                {
                    "plan_key": plan.key,
                    "meaning": plan.claim.predicate.lemma,
                    "passed": bool(result),
                    "surfaces": len(result),
                }
            )
            if result and review.get("reactivation") and review.get("questions"):
                review["questions"][-1]["answer"] = {
                    "meaning": plan.claim.predicate.lemma,
                    "source_ids": plan.claim.evidence,
                }
            if not result:
                reasons = sorted(
                    {
                        str(issue)
                        for record in self.records[-1:]
                        for check in record.get("surface_checks", ())
                        for issue in check.get("issues", ())
                    }
                ) or ["no_licensed_surface"]
                review.setdefault("failures", []).append({"plan_key": plan.key, "reasons": reasons})
                review.setdefault("questions", []).append(
                    {
                        "slot": "realization",
                        "question": reasons,
                        "source_ids": plan.claim.evidence,
                        "answer": None,
                    }
                )
                # Verify the very same evidenced content if its assertion
                # cannot be worded. This is a question, not an invented Unknown
                # or a fresh world claim. One additional NLG attempt is bounded.
                if (
                    plan.claim.predicate_class != "dialogue_act"
                    and plan.claim.clause_kind == "statement"
                    and plan.claim.evidence
                    and not plan.knowledge_limit
                    and not plan.clarification
                    and not review.get("verification_attempted")
                ):
                    review["verification_attempted"] = True
                    fallback = replace(
                        plan,
                        goal=replace(plan.goal, kind="respond", question_word=None),
                        claim=replace(
                            plan.claim,
                            clause_kind="question",
                            focus_role=None,
                            confidence=min(0.49, plan.claim.confidence),
                        ),
                        reason=None,
                        condition=None,
                        follow_up=None,
                        additional_claims=(),
                        reason_condition=None,
                        hypothesis=(("kind", "verify_known_content"),),
                    )
                    review["reactivation"] = {
                        "source": "reader_failure",
                        "level": "hypothesis",
                        "source_ids": plan.claim.evidence,
                    }
                    result = self.realize(
                        PendingText(
                            fallback,
                            patterns,
                            limit,
                            exchange_forms,
                            source_answers,
                            recent_texts,
                            variation_key,
                        )
                    )
                    review["considered"].append(
                        {
                            "plan_key": fallback.key,
                            "meaning": "verify_known_content",
                            "passed": bool(result),
                            "surfaces": len(result),
                        }
                    )
                    if result:
                        review["questions"][-1]["answer"] = {
                            "meaning": "verify_known_content",
                            "source_ids": plan.claim.evidence,
                        }
        return result

    def realize(self, request: PendingText) -> tuple[ProductiveSurface, ...]:
        plan, patterns, limit = request.plan, request.patterns, request.limit
        issues = plan_issues(plan)
        record = {
            "plan": identity_graph(plan),
            "status": "planned",
            "plan_issues": issues,
            "surface_checks": [],
        }
        self.records.append(record)
        if issues:
            record["status"] = "goal_mismatch"
            return ()
        if plan.claim.predicate_class == "dialogue_act":
            return self._realize_reaction(request, record)
        if plan.answer_shape == "fragment" and plan.focus_role:
            argument = next(
                (item.argument for item in plan.claim.dependencies if item.role == plan.focus_role),
                None,
            )
            if argument is None:
                record["status"] = "missing_focus_argument"
                return ()
            text = realize_argument(
                argument,
                predicate_lemma=plan.claim.predicate.lemma,
                role=plan.focus_role,
            ).strip()
            if not text:
                record["status"] = "empty_focus_argument"
                return ()
            surface = ProductiveSurface(
                text[:1].upper() + text[1:] + ".",
                (plan.focus_role,),
                plan.claim.confidence,
                "message_fragment",
                None,
            )
            self._by_surface[surface.text] = {
                "message_plan": identity_graph(plan),
                "communicative_status": plan.status,
                "reader_checked": True,
                "thought_before_surface": True,
                "message_plan_key": plan.key,
            }
            self._by_surface[(plan.key, surface.text)] = self._by_surface[surface.text]
            record["status"] = "expressed"
            return (surface,)
        claims = realize_thought(plan.claim, patterns, limit=limit)
        embedded = ()
        if plan.knowledge_limit and plan.knowledge_limit.question is not None:
            question = plan.knowledge_limit.question
            if not plan.knowledge_limit.question_word and question.negated:
                # The negation of a polar question is its bias, not the content that is
                # unknown: "не знаю, рад ли я" rather than "не знаю, не рад ли я".
                question = replace(question, negated=False)
            embedded_patterns = ()
            nominal_question = question.predicate.lemma == "быть" and any(
                d.role == "predicate_complement" for d in question.dependencies
            )
            if not plan.knowledge_limit.question_word:
                embedded_patterns = (
                    DependencyPattern(
                        (
                            # A nominal predicate has no surface of its own: the particle
                            # follows the complement ("рад ли я"), not the subject.
                            *(("predicate_complement",) if nominal_question else ()),
                            "predicate",
                            "subject",
                            *(
                                d.role
                                for d in question.dependencies
                                if d.role != "subject"
                                and not (nominal_question and d.role == "predicate_complement")
                            ),
                        ),
                        clause_kind="question",
                        confidence=1.0,
                        evidence_count=8,
                    ),
                )
            embedded = realize_thought(question, embedded_patterns, limit=1)
            if not embedded:
                record["status"] = "unrealized_knowledge_subject"
                return ()
        reason = realize_thought(plan.reason, limit=1) if plan.reason else ()
        condition = realize_thought(plan.condition, limit=1) if plan.condition else ()
        follow_up = realize_thought(plan.follow_up, limit=1) if plan.follow_up else ()
        extras = tuple(realize_thought(t, limit=1) for t in plan.additional_claims)
        reason_condition = (
            realize_thought(plan.reason_condition, limit=1) if plan.reason_condition else ()
        )
        if (
            (plan.reason and not reason)
            or (plan.condition and not condition)
            or (plan.follow_up and not follow_up)
            or any(not surface for surface in extras)
            or (plan.reason_condition and not reason_condition)
        ):
            record["status"] = "unrealized_dependency"
            return ()
        accepted = []
        for claim in claims:
            checked = [(plan.claim, claim)]
            if embedded:
                checked.append((question, embedded[0]))
            if reason:
                checked.append((plan.reason, reason[0]))
            if condition:
                checked.append((plan.condition, condition[0]))
            if follow_up:
                checked.append((plan.follow_up, follow_up[0]))
            checked.extend(
                (t, surface[0]) for t, surface in zip(plan.additional_claims, extras, strict=True)
            )
            if reason_condition:
                checked.append((plan.reason_condition, reason_condition[0]))
            problems = tuple(
                issue
                for thought, surface in checked
                for issue in surface_issues(thought, surface.text)
            )
            record["surface_checks"].append({"text": claim.text, "issues": problems})
            if problems:
                continue
            surface = claim
            if embedded:
                text = embedded[0].text.rstrip(".?!")
                if not plan.knowledge_limit.question_word:
                    words = text.split()
                    ordered = _particle_li_order(question, words, nominal_question)
                    if ordered:
                        words = ordered
                    else:
                        words.insert(1, "ли")
                    text = " ".join(words)
                surface = ProductiveSurface(
                    claim.text.rstrip(".") + ", " + text[:1].lower() + text[1:] + ".",
                    claim.role_order,
                    claim.confidence,
                    "knowledge_limit",
                    None,
                )
            if reason or condition or follow_up or extras:
                text = claim.text.rstrip(".?!")
                if follow_up and surface.source == "knowledge_limit":
                    # The not-knowing keeps what is not known; the follow-up comes after it.
                    text = surface.text.rstrip(".?!")
                for extra in extras:
                    e = extra[0].text.rstrip(".?!")
                    text += (" или " if plan.clarification else " и ") + e[:1].lower() + e[1:]
                if reason:
                    r = reason[0].text.rstrip(".")
                    if reason_condition:
                        c = reason_condition[0].text.rstrip(".")
                        r = "если " + c[:1].lower() + c[1:] + ", " + r[:1].lower() + r[1:]
                    text += ", потому что " + r[:1].lower() + r[1:]
                if condition:
                    c = condition[0].text.rstrip(".")
                    text = "Если " + c[:1].lower() + c[1:] + ", " + text[:1].lower() + text[1:]
                if follow_up:
                    text += ". " + follow_up[0].text.rstrip(".")
                surface = ProductiveSurface(
                    text.rstrip("?") + ("?" if plan.clarification or (
                        plan.follow_up is not None and plan.follow_up.clause_kind == "question"
                    ) else "."),
                    tuple(dict.fromkeys(role for _, part in checked for role in part.role_order)),
                    min(part.confidence for _, part in checked),
                    "message_plan",
                    None,
                )
            opinion = dict(plan.hypothesis)
            if opinion.get("kind") in {"experience_opinion", "own_evidence_opinion", "opinion"}:
                # An observed epistemic operator scopes over the entire new
                # proposition. No unmarked personal assertion leaves composer.
                marker = opinion["operator"]
                surface = replace(surface, text=(marker[:1].upper() + marker[1:] + ", " +
                                                surface.text[:1].lower() + surface.text[1:]))
            if opinion.get("kind") == "fantasy" and opinion.get("operator"):
                # The learned laughter marker closes the made-up statement.
                surface = replace(
                    surface, text=surface.text.rstrip(".!") + " " + opinion["operator"])
            metadata = {
                "message_plan": identity_graph(plan),
                "choice_context": plan.review,
                "communicative_status": plan.status,
                "reader_checked": True,
                "thought_before_surface": True,
                "message_plan_key": plan.key,
            }
            self._by_surface[surface.text] = metadata
            self._by_surface[(plan.key, surface.text)] = metadata
            accepted.append(surface)
        record["status"] = "expressed" if accepted else "unexpressed"
        return tuple(accepted)

    def _realize_reaction(
        self,
        request: PendingText,
        record: dict,
    ) -> tuple[ProductiveSurface, ...]:
        """Realize the committed act by a bounded product of raw observed parts.

        The semantic predicate is already selected. This varies only its
        lexical core, a discourse prefix, a retort payload and expressive marks.
        No output of this method is fed back as a language demonstration.
        """
        from debilgi.language.dialogue_learning import normalized_response

        plan = request.plan
        available = tuple(
            replace(f, component="core") if f.recovery_only else f
            for f in request.exchange_forms
            if not f.recovery_only or plan.review.get("phase") in {"tone", "clarification"}
        )
        forms = tuple(
            f
            for f in available
            if f.component != "core"
            or (
                f.hypothesis == plan.hypothesis
                and (
                    f.recovery_only
                    if plan.review.get("phase") in {"tone", "clarification"}
                    else not f.recovery_only
                )
            )
        )
        # A reaction people repeated outranks a one-off: while the same meaning has cores
        # observed in at least two exchanges, single-witness cores are not used.
        repeated = {
            f.meaning for f in forms if f.component == "core" and f.support >= 2
        }
        forms = tuple(
            f
            for f in forms
            if f.component != "core" or f.support >= 2 or f.meaning not in repeated
        )
        combinations = _exchange_part_combinations(
            forms,
            plan.claim.predicate.lemma,
            seed=request.variation_key,
            recent=request.recent_texts,
        )
        blocked = request.source_answers | frozenset(
            normalized_response(t) for t in request.recent_texts
        )
        accepted, seen = [], set()
        for parts in combinations:
            core = next(f for f in parts if f.component == "core")
            suffix = next((f for f in parts if f.component == "suffix"), None)
            if (
                suffix is None
                and core.clause_kind == "question"
                and any(
                    f.component == "suffix"
                    and f.clause_kind == "question"
                    and any(s["surface"] == "?" for s in f.slots)
                    for f in request.exchange_forms
                )
            ):
                continue
            if suffix is None and set(core.register) & {"smile", "emphatic"}:
                continue
            if "smile" in core.register and suffix and "smile" not in suffix.register:
                continue
            if suffix and suffix.clause_kind != core.clause_kind:
                continue
            surface = self.reaction_surface(plan, parts, self.structure.text)
            if surface is None:
                continue
            from debilgi.language.dialogue_learning import reverse_reaction_check

            reverse_check = reverse_reaction_check(plan, parts, self.structure.text, surface.text)
            if not reverse_check["passed"]:
                record["surface_checks"].append(
                    {"text": surface.text, "issues": reverse_check["issues"]}
                )
                continue
            normalized = normalized_response(surface.text)
            atomic_act = (
                core.meaning in {"return_greeting", "mirror_amusement"}
                and len(core.slots) <= 2
                or core.meaning == "acknowledge"
                and all(s["part"] == "interjection" for s in core.slots)
                or core.meaning == "slot_question"
                and all(s.get("agreement_role") for s in core.slots)
            )
            if (normalized in blocked and not atomic_act) or normalized in seen:
                continue
            seen.add(normalized)
            metadata = {
                "message_plan": identity_graph(plan),
                "message_plan_key": plan.key,
                "communicative_status": plan.status,
                "reader_checked": True,
                "thought_before_surface": True,
                "reaction_meaning": core.meaning,
                "reaction_parts": tuple(asdict(f) for f in parts),
                "event_valence": core.event_valence,
                "condition_report": core.condition_report,
                "part_search": "observed_part_grammar",
                "reverse_check": reverse_check,
                "response_mixture": core.mixture or plan.mixture,
                "known_slots": core.known_slots or plan.known_slots,
                "hypothesis": core.hypothesis or plan.hypothesis,
                "hypothetical_assertions": 0,
                "choice_context": plan.review,
                "coalition": plan.review.get("coalitions", {}).get(core.meaning),
                "borrowed_support": tuple(
                    {"part_id": f.pattern_id, "source_ids": f.evidence_ids, "meaning": f.meaning}
                    for f in parts
                    if f.component != "core" and f.meaning != core.meaning
                ),
                "exchange_pattern_id": surface.construction_id,
                "exchange_clause_kind": core.clause_kind,
                "construction_ids": tuple(f.construction_id for f in parts),
                "corpus_source_event_ids": tuple(
                    dict.fromkeys(e for f in parts for e in f.evidence_ids)
                ),
            }
            self._by_surface[surface.text] = metadata
            self._by_surface[(plan.key, surface.text)] = metadata
            record["surface_checks"].append(
                {"text": surface.text, "issues": (), "parts": tuple(f.pattern_id for f in parts)}
            )
            accepted.append(surface)
            if len(accepted) >= request.limit:
                break
        record["status"] = "expressed" if accepted else "unexpressed"
        return tuple(accepted)

    @staticmethod
    def reaction_surface(plan: MessagePlan, parts: tuple, prompt: str) -> ProductiveSurface | None:
        """Bind lexical/grammatical slots to a ClauseThought, then use morphology."""
        from debilgi.language.dialogue_learning import (
            _reaction_form,
            exchange_analysis,
            projected_slot_features,
            reaction_slot_tokens,
        )

        cores = [f for f in parts if f.component == "core"]
        if len(cores) != 1 or cores[0].meaning != plan.claim.predicate.lemma:
            return None
        if any(not f.evidence_ids or not f.pattern_id for f in parts):
            return None
        structure = exchange_analysis(prompt)[0]
        rendered = []
        for part in parts:
            dependencies = []
            for slot in part.slots:
                if slot["binding"] == "open":
                    return None
                features = {
                    k: v
                    for k, v in slot["features"].items()
                    if k in {"case", "number", "gender", "person", "tense"}
                }
                if slot["binding"] in {"speech_act", "punctuation"}:
                    # A witnessed interjection is already a surface form.
                    # Guessed noun inflection must not erase unfamiliar laughter.
                    features = {}
                lemma, surface = slot["lemma"], slot["surface"]
                if slot["binding"] == "speech_act":
                    surface = _witnessed_word_surface(surface, part.support)
                    if surface is None:
                        return None
                    lemma = surface if slot["lemma"] == slot["surface"] else lemma
                if slot["binding"] == "hypothesis":
                    token = next(
                        (
                            structure.tokens[c.predicate_index]
                            for c in structure.clauses
                            if c.predicate_index is not None
                        ),
                        None,
                    )
                    features = projected_slot_features(structure, slot, token)
                if slot["binding"] == "prompt" or slot.get("agreement_role"):
                    options = reaction_slot_tokens(structure, slot)
                    if len(options) != 1:
                        return None
                    token = options[0]
                    if slot["binding"] == "prompt":
                        lemma, surface = token.lemma, token.surface
                        if {"Name", "Surn"} & set(token.grammemes):
                            lemma, surface = lemma.title(), surface.title()
                    features.update(
                        {
                            k: v
                            for k, v in token.grammatical_features
                            if k
                            in (
                                {"case", "number", "gender"}
                                if slot.get("agreement_role")
                                else {"number", "gender"}
                                if slot["part"] == "noun"
                                else {"number"}
                            )
                        }
                    )
                    if slot.get("source_role") == "unfamiliar_word":
                        features = {}
                    features = projected_slot_features(structure, slot, token)
                    if slot.get("source_role") == "heard_surface":
                        lemma, surface = token.surface, token.surface
                dependencies.append(
                    SemanticDependency(
                        f"act:{len(dependencies)}",
                        ConceptArgument(
                            ConceptLexeme(
                                lemma,
                                surface,
                                slot["part"],
                                tuple(features.items()),
                                entity_kind=(
                                    "quoted_utterance"
                                    if slot.get("source_role") == "heard_surface"
                                    else None
                                ),
                            )
                        ),
                    )
                )
            thought = replace(plan.claim, dependencies=tuple(dependencies))
            pattern = DependencyPattern(
                tuple(part.construction_roles),
                clause_kind=part.clause_kind,
                predicate_class="dialogue_act",
                construction_id=part.construction_id,
            )
            if not part.construction_id:
                return None
            surfaces = realize_thought(thought, (pattern,), limit=1)
            if not surfaces:
                return None
            rendered.append(surfaces[0])
        text = re.sub(r"\s+([,.!?;:)])", r"\1", " ".join(p.text for p in rendered))
        overhead = sum(f.component == "prefix" for f in parts) * 0.08
        surface = ProductiveSurface(
            text,
            tuple(r for p in rendered for r in p.role_order),
            plan.claim.confidence - overhead,
            "dialogue_act",
            None,
        )
        words = re.findall(r"\w+", surface.text.casefold())
        if (
            not words
            or len(words) > 7
            or any(a == b for a, b in zip(words, words[1:], strict=False))
        ):
            return None
        # Revalidate each independently composed clause. This also rejects an
        # injected declarative state even if the tail happens to be a question.
        if cores[0].meaning == "perform_contact":
            from debilgi.cognition.dialogue_tasks import contact_response_binding

            if contact_response_binding(surface.text) < 0.8:
                return None
        elif cores[0].hypothesis and any(s["binding"] == "hypothesis" for s in cores[0].slots):
            from debilgi.language.dialogue_learning import question_surface_preserved

            if not question_surface_preserved(cores[0], prompt, surface.text):
                return None
        elif _reaction_form(prompt, surface.text) is None:
            return None
        from debilgi.language.morphology import DEFAULT_MORPHOLOGY

        if any(
            i.kind == "preposition_case"
            for i in DEFAULT_MORPHOLOGY.analyze_sentence(
                surface.text,
            ).issues
        ):
            return None
        identifier = (
            "exchange-composition:"
            + hashlib.sha256("|".join(f.pattern_id for f in parts).encode()).hexdigest()[:24]
        )
        return replace(surface, construction_id=identifier)

    @staticmethod
    def priority(request: PendingText, metadata: dict, confidence: float) -> tuple:
        """Order content without length, word order, surface scores, or source bias."""
        plan = request.plan
        evidence = bool(
            plan.claim.evidence
            or metadata.get("reasoning_evidence")
            or metadata.get("used_evidence")
        )
        terms = {
            lexeme.lemma
            for d in plan.claim.dependencies
            for lexeme in (d.argument.head, *d.argument.modifiers, *d.argument.postmodifiers)
        }
        terms.add(plan.claim.predicate.lemma)
        bound = len(terms.intersection(plan.goal.topics)) / max(1, len(plan.goal.topics))
        bindings = {(d.role, d.argument.head.lemma) for d in plan.claim.dependencies}
        # Same vocabulary with a different actor is a different proposition.
        # Compare known question arguments in their roles, in reply perspective.
        aligned = sum(binding in bindings for binding in plan.goal.known_roles)
        swapped = sum(
            any(lemma == expected and role != expected_role for role, lemma in bindings)
            for expected_role, expected in plan.goal.known_roles
            if (expected_role, expected) not in bindings
        )
        role_fit = (aligned - swapped) / max(1, len(plan.goal.known_roles))
        recovery = bool(
            metadata.get("semantic_recovery")
            or metadata.get("semantic_scaffold")
            or metadata.get("unresolved_requirement")
            or plan.status == "express_limit"
        )
        return (not recovery, bound, role_fit, evidence, confidence, plan.key)


def _exchange_part_combinations(
    forms: tuple,
    meaning: str,
    *,
    seed: str,
    recent: tuple[str, ...] = (),
    limit: int = 192,
) -> tuple[tuple, ...]:
    """Select demonstrated parts after committing to a response meaning.

    The finite grammar has four roles, never corpus words or character
    transitions. Lexical realization and validation remain in ResponseComposer.
    """
    pools = {}
    for form in forms:
        if form.component == "core" and form.meaning != meaning:
            continue
        if form.component == "payload" and meaning != "retort":
            continue
        key = (
            form.component,
            tuple(
                (s["lemma"], s["binding"], s.get("source_role"), s.get("agreement_role"))
                for s in form.slots
            ),
        )
        old = pools.get(key)
        if old is None or form.relevance > old.relevance:
            pools[key] = form
    parts = tuple(pools.values())
    cores = tuple(f for f in parts if f.component == "core")
    if not cores:
        return ()
    order = {"prefix": 0, "core": 1, "payload": 2, "suffix": 3}
    by_source = {}
    for part in forms:
        for source in part.evidence_ids:
            by_source.setdefault(source, set()).add(part.component)
    paths = [tuple(sorted(roles, key=order.get)) for roles in by_source.values() if "core" in roles]
    # Extracted cores are themselves observed grammatical units. They may
    # start an utterance without inheriting a demonstrator's discourse tic.
    paths.extend(("core",) for _ in cores)
    # The extraction witness of a prefix/suffix includes its adjacent
    # core boundary even when that donor's lexical core is not admissible
    # for the current meaning. Backoff joins these observed boundaries.
    paths.extend(("prefix", "core") for f in parts if f.component == "prefix")
    paths.extend(("core", "suffix") for f in parts if f.component == "suffix")
    # These are grammar types, not lexical popularity votes. Hundreds of
    # one-word greetings must not make the bounded walk miss every observed
    # prefix path before the semantic reader can check a composition.
    paths = tuple(dict.fromkeys(paths))
    grammar = SoftGrammar.learn(paths)
    results, seen = [], set()
    for variant in range(limit):
        layout = _exchange_part_layout(paths, grammar, f"{seed}:parts:{variant}")
        if (
            layout.count("core") != 1
            or len(layout) != len(set(layout))
            or layout != tuple(sorted(layout, key=order.get))
        ):
            continue
        core_weights = Counter({f.pattern_id: max(1, round(f.relevance * 8)) for f in cores})
        core_id = _part_weighted_pick(core_weights, f"{seed}:core:{variant}")
        core = next(f for f in cores if f.pattern_id == core_id)
        endings = [
            f for f in parts if f.component == "suffix" and f.clause_kind == core.clause_kind
        ]
        if "suffix" in layout and not endings:
            layout = tuple(role_name for role_name in layout if role_name != "suffix")
        if (
            "suffix" not in layout
            and endings
            and any(
                path[i : i + 2] == ("core", "suffix")
                for path in paths
                for i in range(len(path) - 1)
            )
        ):
            layout = (*layout, "suffix")
        selected = []
        for role_name in layout:
            options = (
                [core]
                if role_name == "core"
                else endings
                if role_name == "suffix"
                else [f for f in parts if f.component == role_name]
            )
            if not options:
                break
            weights = Counter({f.pattern_id: max(1, round(f.relevance * 8)) for f in options})
            if role_name == "prefix" and recent:
                first = tuple(t for t in tokenize(recent[0]) if t[:1].isalnum())
                for form in options:
                    if tuple(s["surface"] for s in form.slots) == first[: len(form.slots)]:
                        weights[form.pattern_id] = max(1, weights[form.pattern_id] // 3)
            chosen = _part_weighted_pick(weights, f"{seed}:lexeme:{variant}:{role_name}")
            selected.append(next(f for f in options if f.pattern_id == chosen))
        else:
            key = tuple(f.pattern_id for f in selected)
            if key not in seen:
                seen.add(key)
                results.append(tuple(selected))
    return tuple(results)


def _part_number(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def _part_weighted_pick(values: Counter[str], seed: str) -> str:
    position = _part_number(seed) % sum(values.values())
    for value, count in sorted(values.items()):
        if position < count:
            return value
        position -= count
    raise RuntimeError("weighted choice received an empty counter")


def _exchange_part_layout(
    paths: tuple[tuple[str, ...], ...], grammar: SoftGrammar, seed: str
) -> tuple[str, ...]:
    """Apply the existing finite part grammar; no surface token prediction.

    Preserve the demonstrated boundary weights and deterministic variation of
    4.11. Only prefix/core/payload/suffix and the closing boundary are admitted.
    The grammar's POS statistics are features, not a lexical generator.
    """
    starts = Counter(path[0] for path in paths)
    layout = [_part_weighted_pick(grammar.reweight(starts, ()), f"{seed}:start")]
    while len(layout) < 4:
        boundaries = []
        for size in range(min(3, len(layout)), 0, -1):
            ending = tuple(layout[-size:])
            options = Counter()
            for path in paths:
                for index in range(size - 1, len(path)):
                    if path[index - size + 1 : index + 1] == ending:
                        options[path[index + 1] if index + 1 < len(path) else "\0end"] += 1
            if options:
                boundaries.append(options)
        if not boundaries:
            break
        options = boundaries[_part_number(f"{seed}:history:{len(layout)}") % len(boundaries)]
        following = _part_weighted_pick(
            grammar.reweight(options, tuple(layout)), f"{seed}:{len(layout)}"
        )
        if following == "\0end":
            break
        layout.append(following)
    return tuple(layout)
