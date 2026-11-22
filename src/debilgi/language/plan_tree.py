"""The plan as a tree for a surface layer: ``thought_to_tree`` (stage 4.24, «Этап A грамматиста»).

One JSON-safe format feeds the future grammarian — at training time (from corpus parses) and at
run time (from ``ClauseThought`` / ``MessagePlan``). Format of LAST_MESSAGE39 §2:

* **nodes** — one per word the plan names: the predicate (id 0) and, per argument, its head
  word, modifiers and post-modifiers. Each has ``lemma``, ``pos``, the *lexical* grammemes of
  its morphology (gender and animacy of a noun, aspect and transitivity of a verb; the number of
  a noun is a meaning of the plan), the ``role`` toward its ``head`` node, and the closed-class
  words the plan itself licenses (``preposition``, the likeness ``marker``). Nodes are listed in
  a canonical role order, so the list carries no word order of any learned pattern.
* **clause** — ``move`` (statement / polar_question / slot_question / command), the ``focus``
  node, ``negated`` and its node, ``mood``, ``tense``, ``response_polarity``, ``subject_elided``
  (pro-drop), ``predicate_class``.
* **link** (a message, not a lone clause) — how a clause hangs on the main one: ``cause``,
  ``condition``, ``coordination``, ``embedded_polar``, ``embedded_wh``.

**Not in the input** (they are what a grammarian learns to produce): case, the person and number of
a verb, the gender of an adjective, the word order. The only exception is a case the plan itself
fixed (``grammatical_case`` of an argument), given as ``case_constraint``.

The function reads the plan only; it never calls a realizer and never adds a word.
"""

from __future__ import annotations

from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)

TREE_VERSION = 1

# Canonical listing order of the roles: independent of any observed or learned word order.
_ROLE_RANK = {
    role: rank
    for rank, role in enumerate(
        (
            "subject",
            "indirect_object",
            "direct_object",
            "predicate_complement",
            "action_complement",
            "manner",
            "cause",
            "location",
            "topic",
            "comparison",
        )
    )
}
_LEXICAL_NOUN = ("gender", "animacy")
_LEXICAL_VERB = ("aspect", "transitivity")


def _lexical(lexeme: ConceptLexeme) -> tuple[str, dict[str, str]]:
    """Part of speech and the lexical grammemes of a lemma (never case or agreement)."""
    reading = DEFAULT_MORPHOLOGY.best(lexeme.lemma)
    pos = lexeme.part_of_speech or (reading.part_of_speech if reading is not None else None)
    features = dict(reading.features()) if reading is not None else {}
    features.update(dict(lexeme.features))
    lexical: dict[str, str] = {}
    if pos == "noun":
        for name in _LEXICAL_NOUN:
            if features.get(name):
                lexical[name] = features[name]
        # The number of a noun is chosen by the plan (one cat or several), not by grammar.
        if features.get("number"):
            lexical["number"] = features["number"]
    elif pos in {"verb", "participle"}:
        for name in _LEXICAL_VERB:
            if features.get(name):
                lexical[name] = features[name]
    return pos or "unknown", lexical


def _node(nodes: list[dict], lexeme: ConceptLexeme, role: str, head: int | None) -> int:
    pos, lexical = _lexical(lexeme)
    node = {
        "id": len(nodes),
        "lemma": lexeme.lemma,
        "pos": pos,
        "lex": lexical,
        "role": role,
        "head": head,
    }
    if ("elided", "true") in lexeme.features:
        node["elided"] = True
    nodes.append(node)
    return node["id"]


def _argument_nodes(nodes: list[dict], dependency: SemanticDependency, head: int | None) -> int:
    argument: ConceptArgument = dependency.argument
    index = _node(nodes, argument.head, dependency.role, head)
    node = nodes[index]
    if argument.grammatical_case:
        node["case_constraint"] = argument.grammatical_case
    if argument.preposition:
        # «как» of a comparison is the plan's marker; any other word is a preposition.
        node["marker" if dependency.role == "comparison" else "preposition"] = argument.preposition
    for modifier in argument.modifiers:
        _node(nodes, modifier, "modifier", index)
    for modifier in argument.postmodifiers:
        _node(nodes, modifier, "postmodifier", index)
    return index


def _clause_tree(thought: ClauseThought) -> dict:
    nodes: list[dict] = []
    predicate = _node(nodes, thought.predicate, "predicate", None)
    attached: dict[str, int] = {}
    ordered = sorted(
        enumerate(thought.dependencies),
        key=lambda item: (item[1].head is not None, _ROLE_RANK.get(item[1].role, 99), item[0]),
    )
    # Heads before their dependents: an argument of an embedded verb needs that verb's node.
    for _, dependency in ordered:
        head = attached.get(dependency.head, predicate) if dependency.head else predicate
        attached[dependency.role] = _argument_nodes(nodes, dependency, head)
    features = dict(thought.predicate.features)
    mood = features.get("mood", "indicative")
    if thought.clause_kind == "command":
        mood = "imperative"
    focus = attached.get(thought.focus_role) if thought.focus_role else None
    move = {"statement": "statement", "command": "command"}.get(thought.clause_kind)
    if move is None:
        move = "slot_question" if focus is not None else "polar_question"
    subject = next((d for d in thought.dependencies if d.role == "subject"), None)
    return {
        "nodes": nodes,
        "clause": {
            "move": move,
            "focus": focus,
            "negated": thought.negated,
            "negation_node": predicate if thought.negated else None,
            "mood": mood,
            "tense": features.get("tense") or thought.semantic_time.tense or "present",
            "response_polarity": (
                None
                if thought.response_polarity is None
                else ("yes" if thought.response_polarity else "no")
            ),
            "subject_elided": bool(
                subject is not None and ("elided", "true") in subject.argument.head.features
            ),
            "predicate_class": thought.predicate_class,
        },
    }


def thought_to_tree(plan) -> dict:
    """Serialize a ``ClauseThought`` or ``MessagePlan`` into the grammarian input (JSON-safe)."""
    if isinstance(plan, ClauseThought):
        return {"version": TREE_VERSION, "kind": "clause", **_clause_tree(plan)}
    clauses = [{"link": {"type": "main", "head_clause": None}, **_clause_tree(plan.claim)}]

    def add(thought: ClauseThought | None, link: str) -> None:
        if thought is not None:
            clauses.append({"link": {"type": link, "head_clause": 0}, **_clause_tree(thought)})

    add(plan.reason, "cause")
    add(plan.reason_condition, "condition")
    add(plan.condition, "condition")
    for extra in plan.additional_claims:
        add(extra, "coordination")
    add(plan.follow_up, "coordination")
    limit = plan.knowledge_limit
    if limit is not None and limit.question is not None:
        add(limit.question, "embedded_wh" if limit.question_word else "embedded_polar")
    for index, clause in enumerate(clauses):
        clause["id"] = index
    return {
        "version": TREE_VERSION,
        "kind": "message",
        "speech_act": plan.speech_act,
        "clauses": clauses,
    }
