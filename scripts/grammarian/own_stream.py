# ruff: noqa: E501
"""The «own readers» stream: a corpus phrase read by the bot's own readers, aligned to a target.

Phrase -> ``analyze_structured_utterance`` -> ``parse_proposition(tree=True)`` -> ``ClauseThought`` ->
``thought_to_tree`` (the runtime path itself, not a re-implementation) + the surface target built from
the phrase's own tokens. It is exactly what the network will see in the bot, with the readers' holes
(a phrase they cannot read is simply absent). Strict completeness as in ``build_bench.py``: every word of
the phrase is a node of the plan or a closed-class word the plan licenses.

The tag of a word comes from the pymorphy grammemes of the reader's token (``tag_from_grammemes``) and is
verified by the round trip ``inflect(lemma, tag) == surface`` — a word that does not round-trip rejects the
phrase, so the target never contains a form the realization side could not produce.
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace

from ud_to_tree import (
    LONG_CASE,
    Converted,
    Reject,
    annotate_from_target,
    check_consistency,
    finalize_target,
    inflect,
    norm,
)

from debilgi.language.plan_tree import thought_to_tree
from debilgi.language.productive_nlg import identity_graph
from debilgi.language.semantic_graph import parse_proposition
from debilgi.language.structured_utterance import analyze_structured_utterance

_CASE = {
    "nomn": "nomn",
    "gent": "gent",
    "gen1": "gent",
    "gen2": "gent",
    "datv": "datv",
    "accs": "accs",
    "acc2": "accs",
    "ablt": "ablt",
    "loct": "loct",
    "loc1": "loct",
    "loc2": "loct",
    "voct": "voct",
}
_UPOS = {
    "noun": "NOUN",
    "verb": "VERB",
    "adjective": "ADJ",
    "participle": "VERB",
    "adverb": "ADV",
    "number": "NUM",
    "pronoun": "PRON",
    "gerund": "VERB",
}
_PUNCT = frozenset(".,?!-—:;…")


def tag_from_grammemes(grammemes: frozenset[str]) -> str:
    """The same tag vocabulary as ``ud_to_tree.tag_of``, from pymorphy grammemes."""
    g = set(grammemes)
    case = next((_CASE[x] for x in g if x in _CASE), None)
    number = "sing" if "sing" in g else "plur" if "plur" in g else None
    gender = next((x for x in ("masc", "femn", "neut") if x in g), None)
    animacy = next((x for x in ("anim", "inan") if x in g), None)
    out: set[str] = set()
    if "INFN" in g:
        out = {"INFN"}
    elif "GRND" in g:
        out = {"GRND"}
    elif "COMP" in g:
        out = {"COMP"}
    elif "VERB" in g:
        if "impr" in g:
            out = {"impr", number or ""}
        else:
            tense = "past" if "past" in g else "futr" if "futr" in g else "pres"
            out = {tense, number or ""}
            out.add(
                (gender or "")
                if tense == "past" and number == "sing"
                else next((x for x in ("1per", "2per", "3per") if x in g), "")
                if tense != "past"
                else ""
            )
    elif "ADJS" in g or "PRTS" in g:
        out = {
            "ADJS" if "ADJS" in g else "PRTS",
            number or "",
            (gender or "") if number == "sing" else "",
        }
    elif "ADJF" in g or "PRTF" in g or "ADJ" in g:
        out = {"PRTF"} if "PRTF" in g else set()
        out |= {case or "", number or "", (gender or "") if number == "sing" else ""}
        if case == "accs" and (number == "plur" or gender == "masc"):
            out.add(animacy or "")
        if "PRTF" in g:
            out |= {
                "pres" if "pres" in g else "past" if "past" in g else "",
                "actv" if "actv" in g else "pssv" if "pssv" in g else "",
            }
    elif "NOUN" in g:
        out = {case or "", number or ""}
    elif "NPRO" in g or "NUMR" in g:
        out = {case or ""}
        if "NUMR" in g and gender:
            out.add(gender)
    out.discard("")
    return ",".join(sorted(out)) if out else "-"


_CASE_OF_LONG = {long: short for short, long in LONG_CASE.items()}


def _fixed_case_tag(node: dict, upos: str, tag: str, surface: str) -> str:
    """The reader's token carries ONE of the cases a syncretic form has («очередь» is a nominative and an accusative):
    when the plan fixed another case that gives the very same word, the tag is the plan's (the word is the same,
    the tag must not contradict the plan's constraint)."""
    wanted = _CASE_OF_LONG.get(node.get("case_constraint") or "")
    grams = tag.split(",") if tag != "-" else []
    current = next((g for g in grams if g in _CASE_OF_LONG.values()), None)
    if wanted is None or current is None or current == wanted:
        return tag
    swapped = ",".join(sorted(wanted if g == current else g for g in grams))
    if inflect(node["lemma"], upos, swapped, bool(node.get("preposition"))) == surface:
        return swapped
    return tag


def build_item(text: str, *, max_words: int = 12) -> Converted:
    """Phrase -> tree (by the bot's readers) + target; ``Reject(reason)`` when unreadable/incomplete."""
    text = text.strip()
    if not 2 <= len(text.split()) <= max_words or re.search(
        r"[A-Za-z0-9@#:/,;()\U0001F300-\U0001FAFF]|\.\.|[!?]{2}", text
    ):
        raise Reject("own:shape")
    structure = analyze_structured_utterance(text)
    if len(structure.clauses) != 1 or structure.clause_links:
        raise Reject("own:not_one_clause")
    clause = structure.clauses[0]
    thought = parse_proposition(
        structure, perspective="agent", allow_question=True, irrealis=True, tree=True
    )
    if thought is None or clause.clause_kind not in {"statement", "question"}:
        raise Reject("own:unread")
    licensed = {"не": "negation", "бы": "conditional", "ли": "polar"}
    for dep in thought.dependencies:
        if dep.argument.preposition:
            licensed[dep.argument.preposition] = "preposition"
    lemmas = {thought.predicate.lemma}
    for dep in thought.dependencies:
        lemmas.update(lex.lemma for lex in (dep.argument.head, *dep.argument.modifiers))
    for token in structure.tokens:
        if not token.surface[:1].isalnum():
            continue
        if token.semantic_role == "question_target" and not (
            token.lemma == "как" and any(d.role == "comparison" for d in thought.dependencies)
        ):
            raise Reject("own:wh_word")
        if token.lemma not in lemmas and token.lemma not in licensed:
            raise Reject("own:incomplete")
    subject = next(d for d in thought.dependencies if d.role == "subject")
    if not clause.subject_indices and "elided" not in dict(subject.argument.head.features):
        head = replace(
            subject.argument.head, features=(*subject.argument.head.features, ("elided", "true"))
        )
        thought = replace(
            thought,
            dependencies=tuple(
                replace(d, argument=replace(d.argument, head=head)) if d is subject else d
                for d in thought.dependencies
            ),
        )
    thought = replace(thought, clause_kind=clause.clause_kind)
    tree = thought_to_tree(thought)
    nodes = tree["nodes"]
    by_lemma: dict[str, int] = {}
    for node in nodes:
        if node.get("elided"):
            continue
        if node["lemma"] in by_lemma:
            raise Reject("own:repeated_lemma")
        by_lemma[node["lemma"]] = node["id"]
    # the nodes that carry each preposition / marker, in plan order: two «в» license two different nodes
    prep_node: dict[str, list[int]] = {}
    marker_node: dict[str, list[int]] = {}
    for node in nodes:
        if node.get("preposition"):
            prep_node.setdefault(node["preposition"], []).append(node["id"])
        if node.get("marker"):
            marker_node.setdefault(node["marker"], []).append(node["id"])
    target: list = []
    used: set[int] = set()
    for token in structure.tokens:
        if not token.surface[:1].isalnum():
            if token.surface in _PUNCT:
                target.append(["P", token.surface])
            continue
        lemma = norm(token.lemma)
        if lemma in by_lemma and by_lemma[lemma] not in used:
            node = nodes[by_lemma[lemma]]
            tag = tag_from_grammemes(frozenset(token.grammemes))
            upos = _UPOS.get(node["pos"], "NOUN")
            tag = _fixed_case_tag(node, upos, tag, norm(token.surface))
            form = inflect(node["lemma"], upos, tag, bool(node.get("preposition")))
            if form != norm(token.surface):
                raise Reject(f"own:roundtrip:{node['pos']}")
            used.add(node["id"])
            target.append(["C", node["id"], tag])
        elif marker_node.get(lemma):
            target.append(["F", norm(token.surface), f"marker:{marker_node[lemma].pop(0)}"])
        elif prep_node.get(lemma):
            target.append(["F", norm(token.surface), f"preposition:{prep_node[lemma].pop(0)}"])
        elif lemma in licensed:
            target.append(["F", norm(token.surface), f"particle_{licensed[lemma]}:0"])
        else:
            raise Reject("own:unplaced")
    silent_copula = (
        tree["clause"]["predicate_class"] == "copular" and tree["clause"]["tense"] == "present"
    )
    for node in nodes:
        if node["id"] in used or node.get("elided") or (node["id"] == 0 and silent_copula):
            continue
        raise Reject("own:node_not_in_surface")
    from ud_to_tree import _meta

    # the stand's convention: the terminal mark is the runtime's (a question «?», else «.»), not the chat's noise
    end = "?" if tree["clause"]["move"].endswith("question") else "."
    target = finalize_target(tree, target, end=end)
    runtime_tree = copy.deepcopy(tree)  # what the runtime plan alone gives (no features read off the surface)
    annotate_from_target(tree, target)
    check_consistency(tree, target)
    from_text = type("S", (), {"sent_id": "", "text": text})()
    meta = _meta(tree, target, from_text)  # type: ignore[arg-type]
    meta["thought"] = identity_graph(thought)
    meta["tree_runtime"] = runtime_tree
    return Converted("", text, tree, target, meta)
