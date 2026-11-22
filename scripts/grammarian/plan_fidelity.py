# ruff: noqa: E501
"""«Plan fidelity» of a realized phrase (stage 4.29, from the prototype of stage 4.28): measurable on EVERY row.

The output is checked against what the PLAN fixed, not against a corpus reference and not through the bot's reader.
A violation = the realized phrase says something the plan did not say:

* ``case``   — the tag of a node lacks the case the plan fixed (``case_constraint``);
* ``lex``    — a fixed lexical grammeme is lost (noun number, ``degree=comparative`` -> COMP, ``form=short`` -> ADJS/PRTS,
               the gender of a speaker pronoun for a past-tense predicate);
* ``agree``  — the finite predicate disagrees with its subject (number; gender in the past singular; person), or an
               adjectival modifier disagrees with its head noun (case / number / gender in the singular);
* ``neg``    — «не» is not immediately before a word of the negated node's phrase;
* ``np``     — a nominal phrase is broken (its words are not contiguous: another reading of the attachment);
* ``cc``     — the closed-class words differ from what the plan gave (conjunctions, links, «ли», «бы», «не»).
               v3 trees carry the words themselves (``func``), a v2 tree does not: then the reference target is used.

``check`` works on a tree + a target in the format of ``ud_to_tree`` (actions ``["C", node, tag]`` / ``["F", word, licence]``
/ ``["P", mark]``). Pure Python: no morphology is needed, the tags are strings. The gate use of the measure is allowed
only when ``check`` gives no violation on the REFERENCE targets of the evaluation rows (see ``explain`` and the
stage report).
"""

from __future__ import annotations

import collections

CASE = {
    "nominative": "nomn",
    "genitive": "gent",
    "dative": "datv",
    "accusative": "accs",
    "instrumental": "ablt",
    "prepositional": "loct",
    "vocative": "voct",
}
CASES = {"nomn", "gent", "datv", "accs", "ablt", "loct", "voct", "gen2", "loc2", "acc2"}
GEN = {"masculine": "masc", "feminine": "femn", "neuter": "neut"}
NUM = {"singular": "sing", "plural": "plur"}
PLURAL_LEMMAS = frozenset({"мы", "вы", "они"})
# a subject whose agreement is not decided by its own number: the verb goes by the sense, not by the form
COLLECTIVE_OR_FREE = frozenset({"это", "что", "кто", "все", "никто", "ничто", "что-то", "кто-то", "нечто", "некто", "сколько", "столько"})
COMITATIVE = frozenset({"с", "со"})
SINGULAR_LEMMAS = frozenset({"я", "ты", "он", "она", "оно"})


def flat_view(tree: dict) -> tuple[list[dict], list[dict], list[int | None], list[int]]:
    clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
    nodes: list[dict] = []
    heads: list[int | None] = []
    clause_of: list[int] = []
    base = 0
    for ci, clause in enumerate(clauses):
        for node in clause["nodes"]:
            nodes.append(node)
            heads.append(None if node["head"] is None else base + node["head"])
            clause_of.append(ci)
        base += len(clause["nodes"])
    return clauses, nodes, heads, clause_of


def under(j: int, root: int, heads: list[int | None]) -> bool:
    k, steps = heads[j], 0
    while k is not None and steps < 60:
        if k == root:
            return True
        k, steps = heads[k], steps + 1
    return False


def func_phrases(tree: dict) -> collections.Counter | None:
    """The closed-class phrases (``kind|word``) the v3 plan demands; ``None`` for a tree without ``func`` (v2)."""
    clauses, nodes, _heads, _clause_of = flat_view(tree)
    if not any("func" in c["clause"] for c in clauses):
        return None
    out: collections.Counter = collections.Counter()
    for c in clauses:
        for kind, word in c["clause"].get("func", []):
            out[f"{kind}|{word}"] += 1
    for node in nodes:
        for kind, word in node.get("func", []):
            out[f"{kind}|{word}"] += 1
    return out


def _same_word(surface, node: dict, tag: set[str], other_case: str) -> bool:
    """The word of ``node`` in ``tag`` is also the word of the tag with another case (syncretism: «очередь», «любые» are a
    nominative and an accusative at once): a wrong tag with the same word is not a wrong word."""
    if surface is None:
        return False
    swapped = ",".join(sorted((tag - CASES) | {other_case}))
    a, b = surface(node, ",".join(sorted(tag))), surface(node, swapped)
    return a is not None and a == b


def check(
    tree: dict,
    target: list,
    reference_target: list | None = None,
    explain: dict | None = None,
    *,
    fixed_case: bool = True,
    surface=None,  # noqa: ANN001 - optional callable (node, tag) -> word; without it the tags are compared as they are
) -> set[str]:
    """The set of violations of ``target`` against the plan ``tree``; ``explain`` collects (kind -> details).

    ``fixed_case=False`` is the honest mode of a new thought: the plan fixed no case, so no case constraint is checked.
    ``surface`` (the realizer) lets the check see that two case tags are one and the same word."""
    clauses, nodes, heads, clause_of = flat_view(tree)
    tags = {a[1]: set(a[2].split(",")) for a in target if a[0] == "C"}
    order = [a[1] for a in target if a[0] == "C"]
    found: set[str] = set()

    def flag(kind: str, why: str) -> None:
        found.add(kind)
        if explain is not None:
            explain.setdefault(kind, []).append(why)

    for i, n in enumerate(nodes):
        t = tags.get(i)
        if t is None:
            continue
        cc = n.get("case_constraint") if fixed_case else None
        if (
            cc
            and CASE.get(cc)
            and t & CASES
            and CASE[cc] not in t
            and not (CASE[cc] == "gent" and "gen2" in t)
            and not (CASE[cc] == "loct" and "loc2" in t)
            and not _same_word(surface, n, t, CASE[cc])
        ):
            flag("case", f"{n['lemma']}:{cc}!{sorted(t)}")
        lx = n["lex"]
        if n["pos"] == "noun" and lx.get("number") in NUM and t & {"sing", "plur"} and NUM[lx["number"]] not in t:
            flag("lex", f"{n['lemma']}:number={lx['number']}!{sorted(t)}")
        if lx.get("degree") == "comparative" and "COMP" not in t:
            flag("lex", f"{n['lemma']}:degree")
        if lx.get("form") == "short" and not t & {"ADJS", "PRTS"}:
            flag("lex", f"{n['lemma']}:short!{sorted(t)}")
        h = heads[i]
        has_numeral = lambda head: any(nodes[j]["pos"] == "number" and heads[j] == head for j in range(len(nodes)))  # noqa: E731
        # agreement of an ADJECTIVAL modifier (a quantity noun or a numeral in front of a noun governs the case itself: «тысячи раз»)
        if n["role"] == "modifier" and n["pos"] in {"adjective", "participle", "pronoun"} and h is not None and nodes[h]["pos"] == "noun" and h in tags and t & CASES:
            ht = tags[h]
            hc, mc = ht & CASES, t & CASES
            if hc and mc and not (hc & mc) and not has_numeral(h) and not any(_same_word(surface, n, t, c) for c in hc):
                flag("agree", f"mod-case {n['lemma']}{sorted(t)}<{nodes[h]['lemma']}{sorted(ht)}")
            hn = nodes[h]["lex"].get("number")
            if hn in NUM and t & {"sing", "plur"} and NUM[hn] not in t and not has_numeral(h):
                flag("agree", f"mod-number {n['lemma']}{sorted(t)}<{nodes[h]['lemma']}:{hn}")
            hg = nodes[h]["lex"].get("gender")
            if hg in GEN and "sing" in t and t & {"masc", "femn", "neut"} and GEN[hg] not in t and not has_numeral(h):
                flag("agree", f"mod-gender {n['lemma']}{sorted(t)}<{nodes[h]['lemma']}:{hg}")
        if n["role"] == "subject" and h is not None and h in tags and n["pos"] in {"noun", "pronoun"} and ("nomn" in t or not t & CASES):
            pt = tags[h]
            if pt & {"past", "pres", "futr"} and nodes[h]["lemma"] != "быть":
                kid_ids = [j for j in range(len(nodes)) if heads[j] == i]
                kids = [nodes[j] for j in kid_ids]
                # the verb goes by the sense, not by the form: coordination, a numeral/quantity, a comitative «Дядя с Иваном пошли»
                # a quantity noun with a genitive plural («большинство пилотов используют», «ряд авторов признали») has a plural verb
                quantity = any(nodes[j]["role"] == "postmodifier" and {"gent", "plur"} <= tags.get(j, set()) for j in kid_ids)
                free = (
                    n["lemma"] in COLLECTIVE_OR_FREE
                    or quantity
                    or any(k["role"] == "conjunct" or k["pos"] == "number" or k.get("preposition") in COMITATIVE for k in kids)
                )
                number = "plur" if "plur" in t else "sing" if "sing" in t else "plur" if n["lemma"] in PLURAL_LEMMAS else "sing" if n["lemma"] in SINGULAR_LEMMAS else None
                if not free and number and pt & {"sing", "plur"} and number not in pt:
                    flag("agree", f"subj-number {n['lemma']}{sorted(t)}>{nodes[h]['lemma']}{sorted(pt)}")
                g = GEN.get(n["lex"].get("gender"))
                if g and not free and "past" in pt and "sing" in pt and pt & {"masc", "femn", "neut"} and g not in pt:
                    flag("agree", f"subj-gender {n['lemma']}:{g}>{sorted(pt)}")
                person = {"я": "1per", "мы": "1per", "ты": "2per", "вы": "2per"}.get(n["lemma"], "3per")
                if pt & {"1per", "2per", "3per"} and person not in pt and not free:
                    flag("agree", f"subj-person {n['lemma']}>{sorted(pt)}")
    # «не» right before a word of the negated node's phrase
    seq = [a for a in target if a[0] in "CF"]
    negated_nodes: list[int] = []
    base = 0
    for c in clauses:
        if c["clause"].get("negated") and c["clause"].get("negation_node") is not None:
            negated_nodes.append(base + c["clause"]["negation_node"])
        base += len(c["nodes"])
    for k, a in enumerate(seq):
        if a[0] == "F" and a[1] in {"не", "ни"}:
            nxt = next((b for b in seq[k + 1 :] if b[0] == "C"), None)
            if nxt is None or not negated_nodes:
                continue
            chain, j = set(), nxt[1]
            while j is not None:
                chain.add(j)
                j = heads[j]
            if not any(x in chain for x in negated_nodes) and nodes[nxt[1]]["pos"] not in {"particle", "adverb"}:
                flag("neg", f"не before {nodes[nxt[1]]['lemma']}")
    # nominal phrases contiguous (the verbal ones are not: a nominal predicate is moved away from its infinitive)
    pos = {x: p for p, x in enumerate(order)}
    for hd in range(len(nodes)):
        if nodes[hd]["pos"] == "verb" or nodes[hd]["role"] in {"predicate", "predicate_complement", "action_complement"}:
            continue
        sub = {hd} | {j for j in range(len(nodes)) if under(j, hd, heads)}
        p = sorted(pos[x] for x in sub if x in pos)
        if len(p) > 1 and p[-1] - p[0] + 1 != len(p):
            flag("np", f"{nodes[hd]['lemma']}")
    expected = func_phrases(tree)
    if expected is not None:
        actual = collections.Counter(f"{a[2].partition(':')[0]}|{a[1]}" for a in target if a[0] == "F")
        if actual != expected:
            flag("cc", f"{dict(actual)} != {dict(expected)}")
    elif reference_target is not None and collections.Counter(a[1] for a in target if a[0] == "F" and a[1] != "не") != collections.Counter(
        a[1] for a in reference_target if a[0] == "F" and a[1] != "не"
    ):
        flag("cc", "closed-class words differ from the reference")
    return found
