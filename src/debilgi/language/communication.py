"""An intended reader update precedes wording and survives surface realization.

The producer supplies propositions, not a reply string. This layer records what
the reader should learn, checks the goal, and checks the rendered roles against
the original proposition. It never creates knowledge from a fluent surface.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    DependencyPattern,
    ProductiveSurface,
    realize_thought,
)
from debilgi.language.structured_utterance import (
    StructuredUtterance,
    analyze_structured_utterance,
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
    questions = tuple(c for c in structure.clauses if c.question_target_index is not None)
    clause = questions[-1] if questions else next(iter(structure.clauses), None)
    if clause is None:
        return ReaderGoal("respond", None, (), None, False)
    word = (structure.tokens[clause.question_target_index].lemma
            if clause.question_target_index is not None else None)
    subject = tuple(structure.tokens[i].lemma for i in clause.subject_indices)
    predicate = (structure.tokens[clause.predicate_index].lemma
                 if clause.predicate_index is not None else clause.implicit_predicate)
    addressed = clause.implicit_subject == "addressee" or bool(set(subject) & {"ты", "вы"})
    topics = tuple(dict.fromkeys(t.lemma for t in structure.tokens
                  if t.part_of_speech == "noun" and t.semantic_role != "vocative"))
    known_roles = tuple(
        (role, {"ты": "я", "вы": "мы", "я": "ты", "мы": "вы"}.get(token.lemma, token.lemma))
        for role, indices in (("subject", clause.subject_indices),
                              ("direct_object", clause.direct_object_indices),
                              ("indirect_object", clause.indirect_object_indices))
        for i in indices for token in (structure.tokens[i],)
        if i != clause.question_target_index
    )
    return ReaderGoal("answer_slot" if word else "respond", word, subject,
                      predicate, addressed, topics, known_roles)


@dataclass(frozen=True, slots=True)
class MessagePlan:
    goal: ReaderGoal
    claim: ClauseThought
    reason: ClauseThought | None = None
    condition: ClauseThought | None = None
    follow_up: ClauseThought | None = None

    @property
    def status(self) -> str:
        if self.claim.clause_kind == "question":
            return "clarify"
        if (self.claim.negated and self.claim.predicate.lemma in {"знать", "понимать"}
                and self.goal.predicate != self.claim.predicate.lemma
                and self.claim.response_polarity is None):
            return "express_limit"
        return "inform"

    @property
    def key(self) -> str:
        # The identity is the semantic graph, never wording or a source label.
        data = asdict(self)
        for name in ("claim", "reason", "condition", "follow_up"):
            if data[name]:
                data[name].pop("confidence", None)
                data[name].pop("evidence", None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def plan_issues(plan: MessagePlan) -> tuple[str, ...]:
    """Check the intended update before a word has been inflected."""
    issues = list(dict.fromkeys(
        issue for thought in (plan.claim, plan.reason, plan.condition, plan.follow_up) if thought
        for issue in (
            *(('missing_predicate',) if not thought.predicate.lemma else ()),
            *(('missing_subject',) if not any(d.role == "subject" for d in thought.dependencies)
              else ()),
        )
    ))
    claim, goal = plan.claim, plan.goal
    if claim.clause_kind == "statement" and goal.kind == "answer_slot":
        roles = {(d.role, d.argument.head.lemma) for d in claim.dependencies}
        if goal.predicate == claim.predicate.lemma:
            for role, lemma in goal.known_roles:
                if ((role, lemma) not in roles and any(
                    other != role and value == lemma for other, value in roles
                )):
                    issues.append("question_argument_changed_role")
        if goal.predicate is None:
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
    observed.update(r.lemma.casefold() for t in structure.tokens
                    for r in DEFAULT_MORPHOLOGY.parse_word(t.surface))
    missing = tuple(f"missing_{d.role}" for d in thought.dependencies
                    if d.role in {"subject", "direct_object", "indirect_object"}
                    and d.argument.head.lemma.isalpha()
                    and d.argument.head.lemma.casefold() not in observed
                    and d.argument.head.surface.casefold() not in observed)
    frames = tuple(c for c in structure.clauses if c.predicate_index is not None
                   and structure.tokens[c.predicate_index].lemma == thought.predicate.lemma)
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
    if (expected.get("subject") != expected.get("direct_object")
            and expected.get("subject") in actual["direct_object"]
            and expected.get("direct_object") in actual["subject"]):
        issues.append("swapped_actor_and_object")
    return tuple(issues)


@dataclass(frozen=True, slots=True)
class PendingText:
    """A render request. It deliberately contains no generated answer string."""
    plan: MessagePlan
    patterns: tuple[DependencyPattern, ...]
    limit: int


@dataclass(frozen=True, slots=True)
class PlannedSurface:
    """Adapter for producers during migration; add() queues its request."""
    text: PendingText
    role_order: tuple[str, ...]
    confidence: float
    source: str = "planned"
    construction_id: str | None = None


class ResponseComposer:
    """Per-turn lifecycle: plan -> goal check -> wording -> reader check."""

    def __init__(self, structure: StructuredUtterance) -> None:
        self.goal = reader_goal(structure)
        self.records: list[dict] = []
        self._by_surface: dict[str, dict] = {}
        self.deferred = False

    def metadata_for(self, text: str) -> dict:
        return self._by_surface.get(text, {})

    def express(
        self, thought: ClauseThought, patterns: tuple[DependencyPattern, ...] = (),
        *, limit: int = 3,
    ) -> tuple[ProductiveSurface | PlannedSurface, ...]:
        return self.compose(MessagePlan(self.goal, thought), patterns, limit=limit)

    def compose(
        self, plan: MessagePlan, patterns: tuple[DependencyPattern, ...] = (),
        *, limit: int = 3,
    ) -> tuple[ProductiveSurface | PlannedSurface, ...]:
        if self.deferred:
            return (PlannedSurface(
                PendingText(plan, patterns, limit),
                ("subject", "predicate", *(d.role for d in plan.claim.dependencies
                                           if d.role != "subject")),
                plan.claim.confidence,
            ),)
        return self.realize(PendingText(plan, patterns, limit))

    def realize(self, request: PendingText) -> tuple[ProductiveSurface, ...]:
        plan, patterns, limit = request.plan, request.patterns, request.limit
        issues = plan_issues(plan)
        record = {"plan": asdict(plan), "status": "planned", "plan_issues": issues,
                  "surface_checks": []}
        self.records.append(record)
        if issues:
            record["status"] = "goal_mismatch"
            return ()
        claims = realize_thought(plan.claim, patterns, limit=limit)
        reason = realize_thought(plan.reason, limit=1) if plan.reason else ()
        condition = realize_thought(plan.condition, limit=1) if plan.condition else ()
        follow_up = realize_thought(plan.follow_up, limit=1) if plan.follow_up else ()
        if ((plan.reason and not reason) or (plan.condition and not condition)
                or (plan.follow_up and not follow_up)):
            record["status"] = "unrealized_dependency"
            return ()
        accepted = []
        for claim in claims:
            checked = [(plan.claim, claim)]
            if reason:
                checked.append((plan.reason, reason[0]))
            if condition:
                checked.append((plan.condition, condition[0]))
            if follow_up:
                checked.append((plan.follow_up, follow_up[0]))
            problems = tuple(issue for thought, surface in checked
                             for issue in surface_issues(thought, surface.text))
            record["surface_checks"].append({"text": claim.text, "issues": problems})
            if problems:
                continue
            surface = claim
            if reason or condition or follow_up:
                text = claim.text.rstrip(".")
                if reason:
                    r = reason[0].text.rstrip(".")
                    text += ", потому что " + r[:1].lower() + r[1:]
                if condition:
                    c = condition[0].text.rstrip(".")
                    text = "Если " + c[:1].lower() + c[1:] + ", " + text[:1].lower() + text[1:]
                if follow_up:
                    text += ". " + follow_up[0].text.rstrip(".")
                surface = ProductiveSurface(
                    text + ".", tuple(dict.fromkeys(
                        role for _, part in checked for role in part.role_order
                    )), min(part.confidence for _, part in checked), "message_plan", None,
                )
            metadata = {
                "message_plan": asdict(plan), "communicative_status": plan.status,
                "reader_checked": True, "thought_before_surface": True,
                "message_plan_key": plan.key,
            }
            self._by_surface[surface.text] = metadata
            accepted.append(surface)
        record["status"] = "expressed" if accepted else "unexpressed"
        return tuple(accepted)

    @staticmethod
    def priority(request: PendingText, metadata: dict, confidence: float) -> tuple:
        """Order content without length, word order, surface scores, or source bias."""
        plan = request.plan
        evidence = bool(plan.claim.evidence or metadata.get("reasoning_evidence")
                        or metadata.get("used_evidence"))
        terms = {lexeme.lemma for d in plan.claim.dependencies
                 for lexeme in (d.argument.head, *d.argument.modifiers, *d.argument.postmodifiers)}
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
        recovery = bool(metadata.get("semantic_recovery") or metadata.get("semantic_scaffold")
                        or metadata.get("unresolved_requirement") or plan.status == "express_limit")
        return (not recovery, bound, role_fit, evidence, confidence, plan.key)
