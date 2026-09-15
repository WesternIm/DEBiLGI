"""Task-scoped comprehension: interpret a request before acquiring its evidence.

Definitions are lexical evidence, not proof that a question can be answered.
Only dependencies of the current task are explored; dictionary prose never
recursively creates new lookup obligations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from debilgi.language.clarification_policy import is_contact_request
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.lexical_gaps import nonblocking_event_terms
from debilgi.language.lexicon import UnknownKind, WordAnalysis
from debilgi.language.recovery import is_identity_question, lexical_lookup_target
from debilgi.language.structured_utterance import analyze_structured_utterance


@dataclass(frozen=True, slots=True)
class TaskDependency:
    key: str
    kind: str
    target: str
    dependencies: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    resolved: bool = False


@dataclass(frozen=True, slots=True)
class DependencyFrontier:
    pending: tuple[TaskDependency, ...]
    cycles: tuple[tuple[str, ...], ...] = ()
    limited: bool = False


def dependency_frontier(
    nodes: tuple[TaskDependency, ...], root: str, *, max_depth: int = 8, max_nodes: int = 64,
) -> DependencyFrontier:
    """Return actionable leaves; neither a cycle nor a limit proves completion."""
    indexed = {node.key: node for node in nodes}
    pending: dict[str, TaskDependency] = {}
    visited: set[str] = set()
    cycles: list[tuple[str, ...]] = []
    limited = False

    def visit(key: str, path: tuple[str, ...]) -> None:
        nonlocal limited
        if key in path:
            cycles.append((*path[path.index(key):], key))
            return
        if key in visited:
            return
        if len(path) > max_depth or len(visited) >= max_nodes:
            limited = True
            return
        visited.add(key)
        node = indexed.get(key, TaskDependency(key, "missing_dependency", key))
        if node.resolved:
            return
        unmet = tuple(d for d in node.dependencies if not (
            d in indexed and indexed[d].resolved
        ))
        if unmet:
            for dependency in unmet:
                visit(dependency, (*path, key))
        else:
            pending[key] = node

    visit(root, ())
    return DependencyFrontier(tuple(pending.values()), tuple(cycles), limited)


@dataclass(frozen=True, slots=True)
class ComprehensionTask:
    goal: str
    answer_slot: str | None
    nodes: tuple[TaskDependency, ...]
    lexical_states: tuple[tuple[str, str], ...]
    incidental_terms: tuple[str, ...]
    antecedent: str | None = None

    @property
    def frontier(self) -> DependencyFrontier:
        return dependency_frontier(self.nodes, "answer")

    @property
    def interpretation_ready(self) -> bool:
        return all(n.resolved for n in self.nodes if n.kind not in {"answer", "evidence"})

    @property
    def answer_ready(self) -> bool:
        frontier = self.frontier
        return not frontier.pending and not frontier.cycles and not frontier.limited

    @property
    def comprehension_score(self) -> float:
        requirements = tuple(n for n in self.nodes if n.kind not in {"answer", "evidence"})
        return sum(n.resolved for n in requirements) / max(1, len(requirements))

    @property
    def lookup_terms(self) -> tuple[str, ...]:
        return tuple(n.target for n in self.frontier.pending if n.kind == "lexical_meaning")

    @property
    def research_allowed(self) -> bool:
        frontier = self.frontier
        return self.goal != "explain_previous" and not (
            frontier.cycles or frontier.limited
            or any(n.kind in {"referent", "dialogue_goal", "sense_selection"}
                   for n in frontier.pending)
        )

    def with_answer_evidence(self, evidence: tuple[str, ...]) -> ComprehensionTask:
        """Only a caller with an applicable, grounded result can finish an answer."""
        if not evidence or not self.interpretation_ready:
            return self
        return replace(self, nodes=tuple(
            replace(n, resolved=True, evidence=evidence)
            if n.kind in {"evidence", "answer"} else n for n in self.nodes
        ))


def lexical_state(word: WordAnalysis) -> str:
    if word.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT:
        return "ambiguous"
    if word.semantic_relations or word.typed_relations:
        return "related"
    if word.has_usable_meaning:
        return "defined"
    return "form_only" if word.part_of_speech else "unrecognized"


def plan_comprehension(
    prompt: str, analyses: tuple[WordAnalysis, ...], context_messages: tuple[str, ...] = (),
    resolved_bindings: tuple[tuple[str, str], ...] = (),
) -> ComprehensionTask:
    structure = analyze_structured_utterance(prompt)
    words = {t.lemma for t in structure.tokens}
    frame = structure.clauses[0] if structure.clauses else None
    slot = next((structure.tokens[c.question_target_index].lemma for c in structure.clauses
                 if c.question_target_index is not None), None)
    target = lexical_lookup_target(prompt)
    discourse = analyze_dialogue_state(prompt, context_messages)
    antecedent = context_messages[-1] if context_messages else None
    prior_explanation = bool(
        not target and words & {"понимать", "понять", "значить", "означать"}
        and words & {"это", "то"} and slot in {"как", "что"}
    )
    literal_terms = nonblocking_event_terms(
        structure, tuple(dict.fromkeys(
            form for word in analyses for form in (word.lemma or word.normalized_form,
                                                   word.normalized_form)
        )),
    )
    goal = (
        "define" if target else "engage" if is_contact_request(prompt)
        else "identify_participant" if is_identity_question(prompt)
        else "explain_previous" if prior_explanation
        else "explain_reason" if slot in {"почему", "отчего", "зачем"}
        else "recall_slot" if slot in {"когда", "где", "куда", "кто", "что", "какой"}
        and frame is not None and frame.predicate_index is not None
        else "answer_question" if slot or "?" in prompt
        else "execute_request" if frame and frame.clause_kind == "command"
        else "record_statement" if frame and frame.predicate_index is not None
        else "interpret_fragment"
    )
    nodes = [TaskDependency(
        "goal", "dialogue_goal", goal, resolved=goal != "interpret_fragment",
        evidence=(f"structure:{goal}",) if goal != "interpret_fragment" else (),
    )]
    references: list[str] = []
    if prior_explanation or (slot in {"почему", "зачем", "как", "когда", "где", "куда"}
                             and frame is not None and frame.predicate_index is None):
        references.append("antecedent")
        nodes.append(TaskDependency("antecedent", "referent", "previous_proposition",
                                    resolved=antecedent is not None,
                                    evidence=(antecedent,) if antecedent else ()))
    elif goal not in {"define", "engage", "identify_participant"}:
        for reference in discourse.unresolved_references:
            if reference.semantic_role in dict(resolved_bindings):
                continue
            # Demonstratives modifying an overt noun do not need a separate entity.
            if reference.kind != "explicit_pronoun":
                continue
            key = f"referent:{reference.token_index}"
            references.append(key)
            nodes.append(TaskDependency(key, "referent", reference.surface))

    states = tuple((w.lemma or w.normalized_form, lexical_state(w)) for w in analyses)
    incidental: list[str] = []
    required: list[str] = ["goal", *references]
    content_pos = {"NOUN", "VERB", "INFN", "ADJF", "noun", "verb", "adjective"}
    for word in analyses:
        term = word.lemma or word.normalized_form
        tokens = tuple(t for t in structure.tokens
                       if t.lemma == term or t.normalized == word.normalized_form)
        if tokens and all(t.part_of_speech in {
            "pronoun", "preposition", "particle", "conjunction", "punctuation", "numeral",
        } for t in tokens):
            continue
        if word.part_of_speech not in content_pos and not word.requires_lexical_study:
            continue
        relevant = (
            term == target or word.normalized_form == target if target
            else goal not in {"engage", "identify_participant", "explain_previous"}
            and term not in literal_terms and word.normalized_form not in literal_terms
        )
        if not relevant:
            incidental.append(term)
            continue
        # Literal identification needs the participant, not a selected dictionary sense.
        ambiguous = word.unknown_kind == UnknownKind.AMBIGUOUS_CONTEXT
        kind = "sense_selection" if ambiguous and goal != "recall_slot" else "lexical_meaning"
        key = f"{kind}:{term}"
        if key in required:
            continue
        required.append(key)
        nodes.append(TaskDependency(
            key, kind, term, dependencies=("goal", *references),
            resolved=word.has_usable_meaning and kind != "sense_selection",
            evidence=(f"lexicon:{word.source}:{term}",) if word.has_usable_meaning else (),
        ))
    needs_answer = goal not in {"record_statement", "engage"}
    nodes.append(TaskDependency("evidence", "evidence", slot or goal,
                                dependencies=tuple(required), resolved=not needs_answer,
                                evidence=("current_message",) if not needs_answer else ()))
    nodes.append(TaskDependency("answer", "answer", goal,
                                dependencies=(*required, "evidence"),
                                resolved=not needs_answer and all(n.resolved for n in nodes)))
    return ComprehensionTask(goal, slot, tuple(nodes), states,
                             tuple(dict.fromkeys(incidental)), antecedent)
