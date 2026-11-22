# ruff: noqa: E501
"""CoNLL-U (Universal Dependencies, Russian) -> ``thought_to_tree`` v1 tree + surface target.

Stage 4.25, «Этап B грамматиста». The converter is the riskiest piece of the data path: it must
produce the SAME format the bot produces at run time with ``debilgi.language.plan_tree.thought_to_tree``
(see that module's docstring and LAST_MESSAGE39 §2), otherwise the network would be good on corpora and
bad in the bot. Everything it cannot express in that format is *rejected with a counted reason*, never
guessed: strict completeness (every token of the sentence is a node, an attribute of a node, a licensed
closed-class word or punctuation).

Mapping UD -> plan (see ``SPEC`` below and ``scripts/grammarian/README.md``):

  nsubj(:pass)->subject            obj->direct_object            iobj->indirect_object
  obl + ADP of the project's location set->location    every other prepositional obl->topic (runtime parity:
  ``parse_proposition`` does the same); bare-case obl->oblique (extension)
  advmod ADV->manner            xcomp VERB (infinitive)->action_complement
  xcomp/root + cop(быть)->predicate_complement (predicate = быть; under an infinitive: action_complement
  быть + predicate_complement with head=action_complement)       «как» + nominative NP->comparison
  amod/det/nummod->modifier        nmod/appos/flat->postmodifier  ADP (case)->node.preposition
  PART не->negated (+negation_node)   PART бы->mood=conditional   PART ли->move=polar_question
  PronType=Int word->focus + move=slot_question                  conj/advcl/ccomp->clause links
  (coordination / cause / condition / embedded_polar / embedded_wh / complement).

Nodes that are NOT in the surface (elided subject of a 1st/2nd person verb, the zero copula) carry
``elided`` / ``zero`` and have no action in the target. Case, person, number of a verb, gender of an
adjective and the order are NOT in the input — they are the target:

  target = [["C", ptr, tag], ["F", word], ["P", char], ...] in surface order; ``ptr`` is the global node
  index (clauses flattened in order), ``tag`` the sorted pymorphy grammemes (``"nomn,sing"``), ``"-"`` for
  an invariant word. Words are lower-cased, ё folded to е.
"""

from __future__ import annotations

import functools
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- vocabulary of the format
TREE_VERSION = 1
CASES = {
    "Nom": "nomn",
    "Gen": "gent",
    "Par": "gent",
    "Dat": "datv",
    "Acc": "accs",
    "Ins": "ablt",
    "Loc": "loct",
    "Voc": "voct",
}
NUMBERS = {"Sing": "sing", "Plur": "plur"}
GENDERS = {"Masc": "masc", "Fem": "femn", "Neut": "neut"}
PERSONS = {"1": "1per", "2": "2per", "3": "3per"}
POS_NAMES = {
    "NOUN": "noun",
    "PROPN": "noun",
    "VERB": "verb",
    "AUX": "verb",
    "ADJ": "adjective",
    "ADV": "adverb",
    "NUM": "number",
    "PRON": "pronoun",
    "DET": "pronoun",
}
PUNCT_OK = frozenset(".,?!-—:;…")
TERMINALS = frozenset(".?!…")
# Long names of the cases: the format of the runtime plan (``ConceptArgument.grammatical_case``).
LONG_CASE = {
    "nomn": "nominative",
    "gent": "genitive",
    "datv": "dative",
    "accs": "accusative",
    "ablt": "instrumental",
    "loct": "prepositional",
    "voct": "vocative",
}
LONG_GENDER = {"masc": "masculine", "femn": "feminine", "neut": "neuter"}
SPEAKER_PRONOUNS = frozenset({"я", "ты", "вы"})
# Closed classes that may appear as FUNC actions (the net's output vocabulary beyond word forms).
FUNC_UPOS = frozenset({"ADP", "PART", "SCONJ", "CCONJ"})
LICENSED_PARTICLES = {
    "не": "negation",
    "ни": "negation",
    "бы": "conditional",
    "б": "conditional",
    "ли": "polar",
}
# Lemmas that make a «what a pronoun-adverb asks» role (closed list of the format, not of the bot).
WH_ROLE = {
    "почему": "cause",
    "зачем": "cause",
    "где": "location",
    "куда": "location",
    "откуда": "location",
}
COPULA_LIKE = frozenset(
    {"быть", "стать", "казаться", "являться", "оказаться", "остаться", "считаться", "выглядеть"}
)
THIRD = frozenset({"он", "она", "оно", "они"})
ROLE_RANK = {
    r: i
    for i, r in enumerate(
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
            "oblique",
            "particle",
            "conjunct",
        )
    )
}
SKIP_RELS = {"punct"}


def _project_sets() -> tuple[frozenset, frozenset, frozenset, frozenset]:
    """The project's own closed sets of prepositions and modals (single source of truth)."""
    from debilgi.language import structured_utterance as su
    from debilgi.language.semantic_graph import _MODALS

    return (
        frozenset(su._TOPIC_PREPOSITIONS),
        frozenset(su._LOCATION_PREPOSITIONS),
        frozenset(_MODALS),
        frozenset(su._MODAL_LEMMAS),
    )


TOPIC_PREP, LOCATION_PREP, MODALS, _MODAL_LEMMAS = _project_sets()


# --------------------------------------------------------------------------- CoNLL-U reading
@dataclass(slots=True)
class Tok:
    i: int
    form: str
    lemma: str
    upos: str
    feats: dict[str, str]
    head: int
    rel: str
    children: list[int] = field(default_factory=list)

    @property
    def base_rel(self) -> str:
        return self.rel.split(":")[0]


@dataclass(slots=True)
class Sentence:
    sent_id: str
    text: str
    toks: list[Tok]  # index = id-1
    # lemmas of the previous sentences (at most ``CONTEXT_SENTENCES``) of the SAME document, or ``None`` when the
    # corpus does not tell the document of a sentence (the givenness of a node is then unknown, not «new»)
    context: frozenset[str] | None = None


CONTEXT_SENTENCES = 3
_DOC_ID = re.compile(r"^(?P<doc>.+?)[_-](?P<n>\d+)$")
_NO_DOC_PREFIXES = ("taiga-", "train-", "dev-", "test-")  # ids that number sentences of a whole file, not of a document


def _document_key(sent_id: str) -> str | None:
    if sent_id.startswith(_NO_DOC_PREFIXES):
        return None
    found = _DOC_ID.match(sent_id)
    return found["doc"] if found else None


def read_conllu(path: Path) -> Iterator[Sentence]:
    meta: dict[str, str] = {}
    rows: list[Tok] = []
    recent: list[frozenset[str]] = []
    state = {"doc": None}

    def flush() -> Sentence | None:
        if not rows:
            meta.clear()
            return None
        for tok in rows:
            if tok.head:
                rows[tok.head - 1].children.append(tok.i)
        sent_id = meta.get("sent_id", "")
        doc = meta.get("newdoc id") or meta.get("newdoc_id") or _document_key(sent_id)
        if "newdoc" in meta or doc != state["doc"]:
            recent.clear()
        state["doc"] = doc
        context = frozenset().union(*recent) if doc is not None and recent else (frozenset() if doc is not None else None)
        sentence = Sentence(sent_id, meta.get("text", ""), list(rows), context)
        recent.append(frozenset(norm(tok.lemma) for tok in rows))
        del recent[:-CONTEXT_SENTENCES]
        rows.clear()
        meta.clear()
        return sentence

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                if (s := flush()) is not None:
                    yield s
            elif line.startswith("#"):
                if line.startswith("# newdoc"):
                    meta["newdoc"] = "1"  # a document starts here: the context of givenness is reset
                if " = " in line:
                    key, _, value = line[1:].partition(" = ")
                    meta[key.strip()] = value
            else:
                cols = line.split("\t")
                if len(cols) < 8 or not cols[0].isdigit():
                    continue  # multiword range / empty node
                feats = {}
                if cols[5] != "_":
                    feats = dict(item.split("=", 1) for item in cols[5].split("|") if "=" in item)
                rows.append(
                    Tok(int(cols[0]), cols[1], cols[2], cols[3], feats, int(cols[6]), cols[7])
                )
    if (s := flush()) is not None:
        yield s


def norm(text: str) -> str:
    return text.casefold().replace("ё", "е")


# --------------------------------------------------------------------------- morphology (tag <-> form)
class Reject(Exception):
    """A sentence the format cannot express; ``args[0]`` is the counted reason."""


@functools.lru_cache(maxsize=1)
def _analyzer():
    from debilgi.language.morphology import DEFAULT_MORPHOLOGY

    return DEFAULT_MORPHOLOGY._analyzer


_POS_OK = {
    "NOUN": {"NOUN"},
    "PROPN": {"NOUN"},
    "ADJ": {"ADJF", "ADJS", "COMP", "PRTF", "PRTS", "NPRO"},
    "DET": {"ADJF", "NPRO", "NUMR", "ADJS"},
    "PRON": {"NPRO", "ADJF", "NOUN"},
    "NUM": {"NUMR", "ADJF", "NOUN"},
    "VERB": {"VERB", "INFN", "PRTF", "PRTS", "GRND"},
    "AUX": {"VERB", "INFN"},
    "ADV": {"ADVB", "COMP", "PRED", "NPRO"},
}


def tag_of(tok: Tok) -> str:
    """UD morphology of a content token as a sorted set of pymorphy grammemes (``-`` = invariant)."""
    u, f = tok.upos, tok.feats
    case, number = CASES.get(f.get("Case", "")), NUMBERS.get(f.get("Number", ""))
    gender = GENDERS.get(f.get("Gender", ""))
    grams: set[str] = set()
    if u in {"NOUN", "PROPN"}:
        grams = {g for g in (case, number) if g}
        if not case or not number:
            raise Reject("tag:noun_incomplete")
    elif f.get("VerbForm") == "Part" or u in {"ADJ", "DET"}:
        part = f.get("VerbForm") == "Part"
        if f.get("Variant") == "Short":
            grams = {"PRTS" if part else "ADJS", number or "", gender if number == "sing" else ""}
        elif f.get("Degree") == "Cmp":
            grams = {"COMP"}
        else:
            grams = {"PRTF"} if part else set()
            grams |= {case or "", number or ""}
            if number == "sing":
                grams.add(gender or "")
            if case == "accs" and (number == "plur" or gender == "masc"):
                grams.add({"Anim": "anim", "Inan": "inan"}.get(f.get("Animacy", ""), ""))
        if part:
            grams |= {
                {"Pres": "pres", "Past": "past"}.get(f.get("Tense", ""), ""),
                {"Act": "actv", "Pass": "pssv"}.get(f.get("Voice", ""), ""),
            }
        grams.discard("")
        if not grams or (
            not case and "ADJS" not in grams and "PRTS" not in grams and "COMP" not in grams
        ):
            raise Reject("tag:adjective_incomplete")
    elif u in {"VERB", "AUX"}:
        form = f.get("VerbForm")
        if form == "Inf":
            grams = {"INFN"}
        elif form == "Conv":
            grams = {"GRND"}
        elif f.get("Mood") == "Imp":
            grams = {"impr", number or ""}
        else:
            tense = {"Pres": "pres", "Fut": "futr", "Past": "past"}.get(f.get("Tense", ""))
            if not tense:
                raise Reject("tag:verb_no_tense")
            grams = {tense, number or ""}
            if tense == "past":
                grams.add(gender if number == "sing" else "")
            else:
                grams.add(PERSONS.get(f.get("Person", ""), ""))
        grams.discard("")
    elif u == "PRON":
        if case:
            grams = {case}
            if f.get("Person") and not f.get("Number"):
                pass
        elif f.get("Case") is None:
            grams = set()
    elif u == "NUM":
        grams = {g for g in (case, gender if gender else "") if g}
        if case == "accs" and f.get("Animacy"):
            grams.add({"Anim": "anim", "Inan": "inan"}[f["Animacy"]])
    elif u == "ADV" and f.get("Degree") == "Cmp":
        grams = {"COMP"}
    grams = {g for g in grams if g}
    return ",".join(sorted(grams)) if grams else "-"


@functools.lru_cache(maxsize=400_000)
def inflect(lemma: str, upos: str, tag: str, prosthetic: bool = False) -> str | None:
    """The pymorphy form of ``lemma`` for ``tag`` (the realization side of the target)."""
    lemma_n = norm(lemma)
    if tag == "-":
        return lemma_n
    wanted = _POS_OK.get(upos)
    grams = set(tag.split(","))
    readings = _analyzer().parse(lemma_n)
    own = [r for r in readings if r.normal_form == lemma_n] or list(readings)
    own = [r for r in own if wanted is None or str(r.tag.POS) in wanted] or own
    for reading in own:
        try:
            inflected = reading.inflect(grams)
        except Exception:  # noqa: BLE001 - pymorphy rejects unknown grammemes for a POS
            continue
        if inflected is not None:
            word = norm(str(inflected.word))
            if prosthetic and lemma_n in THIRD and word[:1] in "еёи":
                word = "н" + word
            return word
    return None


# --------------------------------------------------------------------------- tree building
@functools.lru_cache(maxsize=200_000)
def _lexical_cached(lemma: str, pos: str, number: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    from debilgi.language.plan_tree import _lexical
    from debilgi.language.productive_nlg import ConceptLexeme

    features = (("number", number),) if number else ()
    pos_name, lexical = _lexical(ConceptLexeme(lemma, lemma, pos, features))
    return pos_name, tuple(sorted(lexical.items()))


COMPARATORS = frozenset({"как", "словно", "будто", "точно"})
NOMINAL = frozenset({"NOUN", "PROPN", "PRON"})
VERBAL = frozenset({"VERB", "AUX"})
UNSTRUCTURED_RELS = {
    "acl",
    "orphan",
    "vocative",
    "discourse",
    "parataxis",
    "dep",
    "expl",
    "csubj",
    "list",
    "reparandum",
    "goeswith",
    "dislocated",
}
SUBJECT_PRONOUN = {
    ("1", "Sing"): "я",
    ("2", "Sing"): "ты",
    ("1", "Plur"): "мы",
    ("2", "Plur"): "вы",
}
# Dependents that belong to the clause, not to the nominal predicate they hang on.
CLAUSE_LEVEL = frozenset(
    {
        "nsubj",
        "obj",
        "iobj",
        "obl",
        "xcomp",
        "cop",
        "mark",
        "cc",
        "conj",
        "advcl",
        "ccomp",
        "aux",
        "punct",
    }
)


@dataclass(slots=True)
class Clause:
    nodes: list[dict] = field(default_factory=list)
    toks: dict[int, int] = field(default_factory=dict)  # token id -> node id
    link: dict = field(default_factory=dict)
    clause: dict = field(
        default_factory=lambda: {
            "move": "statement",
            "focus": None,
            "negated": False,
            "negation_node": None,
            "mood": "indicative",
            "tense": "present",
            "response_polarity": None,
            "subject_elided": False,
            "predicate_class": "lexical",
        }
    )
    predicate_tok: Tok | None = None
    silent_copula: bool = False


def canonicalize(c: Clause) -> dict[int, int]:
    """List the nodes in the canonical role order of ``thought_to_tree`` (the predicate, then the
    arguments of the predicate by role, then the arguments of embedded words; each argument is
    followed by its modifiers and post-modifiers), so the node list carries no word order."""
    nodes = c.nodes
    children: dict[int, list[dict]] = {}
    for node in nodes:
        if node["head"] is not None:
            children.setdefault(node["head"], []).append(node)
    order: list[dict] = [nodes[0]]
    emitted = {0}
    modifier_roles = {"modifier", "postmodifier"}

    def subtree(node: dict) -> None:
        order.append(node)
        emitted.add(node["id"])
        for kid in sorted(
            (k for k in children.get(node["id"], ()) if k["role"] in modifier_roles),
            key=lambda k: (k["role"] == "postmodifier", k["id"]),
        ):
            subtree(kid)

    while len(order) < len(nodes):  # level by level: arguments of the predicate first
        candidates = [n for n in nodes if n["id"] not in emitted and n["head"] in emitted]
        arguments = [n for n in candidates if n["role"] not in modifier_roles]
        for node in sorted(
            arguments or candidates, key=lambda k: (ROLE_RANK.get(k["role"], 99), k["id"])
        ):
            if node["id"] not in emitted:
                subtree(node)
    new_id = {node["id"]: i for i, node in enumerate(order)}
    for node in order:
        node["head"] = None if node["head"] is None else new_id[node["head"]]
    for i, node in enumerate(order):
        node["id"] = i
    c.nodes = order
    c.toks = {t: new_id[n] for t, n in c.toks.items()}
    if c.clause["focus"] is not None:
        c.clause["focus"] = new_id[c.clause["focus"]]
    if c.clause["negation_node"] is not None:
        c.clause["negation_node"] = new_id[c.clause["negation_node"]]
    return new_id


class Builder:
    """Builds the clauses of one sentence; every token must end up accounted for."""

    def __init__(self, sentence: Sentence):
        self.s = sentence
        self.consumed: set[int] = set()
        self.func: dict[int, tuple[str, int, int | None]] = {}  # tok -> (kind, clause, node)
        self.puncts: dict[int, str] = {}
        self.clauses: list[Clause] = []
        self.merged: dict[int, int] = {}  # a fixed-expression token -> the head token whose node stands for both («как раз»)
        self.merged_forms: dict[int, str] = {}  # head token -> the surface of the whole expression

    # ---- bookkeeping
    def t(self, i: int) -> Tok:
        return self.s.toks[i - 1]

    def kids(self, t: Tok, *rels: str) -> list[Tok]:
        return [
            c
            for c in (self.t(i) for i in t.children)
            if c.base_rel in rels and c.i not in self.consumed
        ]

    def take(self, t: Tok) -> None:
        if t.i in self.consumed:
            raise Reject("internal:twice")
        self.consumed.add(t.i)

    def licence(self, c: Clause, t: Tok, kind: str, node: int | None = None) -> None:
        self.take(t)
        self.func[t.i] = (kind, self.clauses.index(c), node)

    def punct(self, t: Tok) -> None:
        if t.form not in PUNCT_OK:
            raise Reject("punct:" + t.form)
        self.take(t)
        self.puncts[t.i] = t.form

    def node(
        self,
        c: Clause,
        t: Tok | None,
        role: str,
        head: int | None,
        *,
        lemma: str | None = None,
        pos: str | None = None,
        elided: bool = False,
        merge_fixed: bool = False,
    ) -> int:
        fixed = sorted(self.kids(t, "fixed"), key=lambda k: k.i) if merge_fixed and t is not None else []
        if fixed:  # a fixed expression of an adverb / particle («как раз», «как бы») is ONE node of several words
            lemma = " ".join(norm(x.lemma) for x in (t, *fixed))
            self.merged_forms[t.i] = " ".join(norm(x.form) for x in (t, *fixed))
            for extra in fixed:
                self.take(extra)
                self.merged[extra.i] = t.i
        lemma = norm(lemma or (t.lemma if t else ""))
        upos = t.upos if t else "PRON"
        pos_name = pos or POS_NAMES.get(upos)
        if pos_name is None or not lemma:
            raise Reject(f"pos:{upos}")
        if t is not None and t.feats.get("VerbForm") == "Part":
            pos_name = "participle"
        number = ""
        if t is not None and upos in {"NOUN", "PROPN"}:
            number = {"Sing": "singular", "Plur": "plural"}.get(t.feats.get("Number", ""), "")
        pos_final, lex = _lexical_cached(lemma, pos_name, number)
        node = {
            "id": len(c.nodes),
            "lemma": lemma,
            "pos": pos_final,
            "lex": dict(lex),
            "role": role,
            "head": head,
        }
        gender = GENDERS.get(t.feats.get("Gender", "")) if t is not None and upos in {"NOUN", "PROPN"} else None
        if gender:  # the gender of the noun in THIS text (a dictionary gives «Люда», «Александра» the other one)
            node["lex"]["gender"] = LONG_GENDER[gender]
        if elided:
            node["elided"] = True
        elif self.s.context is not None:
            node["given"] = lemma in self.s.context  # mentioned in the previous sentences of the document
        c.nodes.append(node)
        if t is not None:
            c.toks[t.i] = node["id"]
            self.take(t)
        return node["id"]

    # ---- noun phrases and modifiers
    def comparator(self, t: Tok) -> Tok | None:
        for rel in ("mark", "case", "cc", "advmod"):
            for k in self.kids(t, rel):
                if (
                    norm(k.lemma) in COMPARATORS
                    and k.upos in {"SCONJ", "ADP", "CCONJ", "ADV", "PART"}
                    and not self.kids(k, "fixed")  # «как раз», «как бы», «как и» are not a comparison
                ):
                    return k
        return None

    def fixed_comparator(self, k: Tok) -> bool:
        """«как раз», «как бы», «как и», «как будто»: a fixed expression that begins like a comparison marker and is not one."""
        return norm(k.lemma) in COMPARATORS and bool(self.kids(k, "fixed"))

    def marker_kind(self, head: Tok, marker: Tok) -> str:
        """``comparison`` when the marked phrase is set off by commas («Сидели тихо, как мыши»), ``as_role`` when
        it is not («воспринял это как комплимент»): the comma of such a phrase depends on the sense."""
        before = self.s.toks[marker.i - 2] if marker.i >= 2 else None
        subtree, stack = {head.i}, [head.i]
        while stack:
            for child in self.t(stack.pop()).children:
                if child not in subtree:
                    subtree.add(child)
                    stack.append(child)
        last = max(subtree)
        after = self.s.toks[last] if last < len(self.s.toks) else None
        return "comparison" if (before is not None and before.form == ",") or (after is not None and after.form == ",") else "as_role"

    def np(
        self,
        c: Clause,
        t: Tok,
        role: str,
        head: int,
        *,
        clause_level: bool = False,
        marker: Tok | None = None,
    ) -> int:
        """A noun phrase: ``t`` is its word; with ``clause_level`` the dependents that belong to
        the clause (subject, objects, negation, ...) are left to ``fill``."""
        idx = self.node(c, t, role, head)
        node = c.nodes[idx]
        case = CASES.get(t.feats.get("Case", ""))
        adjectival_complement = role == "predicate_complement" and t.upos in {"ADJ", "VERB", "DET"}
        if case and (t.upos in {"NOUN", "PROPN", "PRON", "NUM"} or adjectival_complement):
            node["case_constraint"] = LONG_CASE[case]  # the nominal group's case is given (dropped at training)
        preps = [k for k in self.kids(t, "case") if k is not marker and not self.fixed_comparator(k)]
        if len(preps) > 1:
            raise Reject("np:two_prepositions")
        for prep in preps:
            if prep.upos != "ADP":
                raise Reject("np:case_not_adp")
            node["preposition"] = norm(prep.lemma)
            self.licence(c, prep, "preposition", idx)
            if self.kids(prep, "fixed"):
                raise Reject("np:multiword_preposition")
        if marker is not None:
            node["marker"] = norm(marker.lemma)
            node["marker_kind"] = self.marker_kind(t, marker)
            self.licence(c, marker, "marker", idx)
        for child in (self.t(i) for i in list(t.children)):
            rel = child.base_rel
            if child.i in self.consumed:
                continue
            if self.fixed_comparator(child) and rel in {"advmod", "mark", "cc", "case", "discourse"}:
                self.node(c, child, "particle", idx, pos="particle", merge_fixed=True)  # ONE node of several words
                continue
            if clause_level and (rel in CLAUSE_LEVEL or (rel == "advmod" and child.upos == "PART")):
                continue
            if rel in {"advmod", "discourse"} and child.upos == "PART":
                self.particle(c, child, idx)
                continue
            if rel in {"amod", "det", "nummod", "advmod"} and child.upos not in {"PART", "SCONJ"}:
                self.modifier(c, child, idx)
            elif rel in {"nmod", "appos", "flat", "compound"} and child.upos in {
                "NOUN",
                "PROPN",
                "PRON",
                "ADJ",
                "NUM",
                "DET",
            }:
                self.np(c, child, "postmodifier", idx)
            elif rel == "punct":
                self.punct(child)
            elif rel == "conj" and child.upos == t.upos and role != "predicate_complement":
                self.np(c, child, "conjunct", idx)
            elif rel == "cc" and role == "conjunct" and child.upos == "CCONJ":
                self.licence(c, child, "conjunction", idx)
            elif rel in {"conj", "cc"}:
                raise Reject("np:coordination")
            else:
                raise Reject(f"np:rel:{child.rel}")
        return idx

    def modifier(self, c: Clause, t: Tok, head: int) -> None:
        if t.upos not in {"ADJ", "DET", "NUM", "ADV", "PRON", "VERB", "NOUN", "PROPN"}:
            raise Reject(f"modifier:{t.upos}")
        if t.upos == "VERB" and t.feats.get("VerbForm") != "Part":
            raise Reject("modifier:finite_verb")
        idx = self.node(c, t, "modifier", head, merge_fixed=t.upos in {"ADV", "PART"})
        if t.feats.get("PronType") == "Int":
            self.focus(c, idx)
        for child in (self.t(i) for i in list(t.children)):
            if child.i in self.consumed:
                continue
            rel = child.base_rel
            if rel in {"advmod", "amod", "det"} and child.upos in {"ADV", "ADJ", "DET"}:
                self.modifier(c, child, idx)
            elif rel in {"nmod", "flat"} and child.upos in {"NOUN", "PROPN"}:
                self.np(c, child, "postmodifier", idx)
            elif rel == "punct":
                self.punct(child)
            else:
                raise Reject(f"modifier:rel:{child.rel}")

    def focus(self, c: Clause, idx: int) -> None:
        """The asked slot: the wh-word itself, or the head noun of «какой N»."""
        if c.clause["focus"] is not None:
            raise Reject("wh:two_focus")
        node = c.nodes[idx]
        c.clause["move"] = "slot_question"
        c.clause["focus"] = node["head"] if node["role"] == "modifier" else idx

    # ---- the clause
    def argument_role(self, t: Tok, rel: str) -> str:
        if rel == "nsubj":
            return "subject"
        if rel == "obj":
            return "direct_object"
        if rel == "iobj":
            return "indirect_object"
        preps = [norm(k.lemma) for k in self.kids(t, "case") if k.upos == "ADP"]
        # the runtime reader (``parse_proposition``) calls a prepositional complement ``location`` only for
        # the location prepositions and every other one ``topic``; the converter keeps that, for format
        # parity. ``oblique`` (an extension) is a complement without a preposition.
        if preps and preps[0] in LOCATION_PREP:
            return "location"
        return "topic" if preps else "oblique"

    def pred_arg(self, c: Clause, t: Tok, role: str, head: int) -> None:
        marker = self.comparator(t)
        if marker is not None:
            if t.upos not in NOMINAL or role in {"subject", "direct_object", "indirect_object"}:
                raise Reject("comparator:unsupported")
            role = "comparison"
        idx = self.np(c, t, role, head, marker=marker)
        if t.feats.get("PronType") == "Int":
            self.focus(c, idx)

    def fill(self, c: Clause, h: Tok, node_id: int, *, embedded: bool) -> None:
        """Attach the clause-level dependents of ``h`` (a verb or a nominal predicate)."""
        for child in (self.t(i) for i in list(h.children)):
            if child.i in self.consumed:
                continue
            rel, u = child.base_rel, child.upos
            if rel in {"nsubj", "obj", "iobj", "obl"} and u in NOMINAL:
                if rel == "nsubj" and embedded:
                    raise Reject("embedded:subject")
                self.pred_arg(c, child, self.argument_role(child, rel), node_id)
            elif rel == "advmod" and u == "ADV":
                wh = child.feats.get("PronType") == "Int"
                if norm(child.lemma) in COMPARATORS and not wh and not self.kids(child, "fixed"):
                    raise Reject("advmod:comparator")
                idx = self.node(
                    c, child, WH_ROLE.get(norm(child.lemma), "manner") if wh else "manner", node_id,
                    merge_fixed=True,
                )
                if wh:
                    self.focus(c, idx)
                for sub in (self.t(i) for i in list(child.children)):
                    if sub.i in self.consumed:
                        continue
                    if sub.base_rel == "punct":
                        self.punct(sub)
                    elif sub.base_rel == "advmod" and sub.upos == "ADV":
                        self.modifier(c, sub, idx)
                    else:
                        raise Reject(f"manner:rel:{sub.rel}")
            elif rel in {"advmod", "aux", "discourse"} and u in {"PART", "AUX"}:
                self.particle(c, child, node_id)
            elif rel == "xcomp":
                self.xcomp(c, child, node_id)
            elif rel == "punct":
                self.punct(child)
            elif rel in {"mark", "cc"} and u in {"SCONJ", "ADP", "PART", "CCONJ"}:
                if c.link.get("type") == "main":
                    # a connective opening the main clause refers to earlier discourse: no plan link licenses it
                    raise Reject("unlicensed_func:connective_on_main")
                self.licence(c, child, "link")
                for fixed in self.kids(child, "fixed"):
                    self.licence(c, fixed, "link")
            elif rel in {"conj", "advcl", "ccomp"} and not embedded:
                self.subclause(c, child, rel)
            elif rel in UNSTRUCTURED_RELS or rel in {"conj", "advcl", "ccomp"}:
                raise Reject(f"clause:{rel}")
            else:
                raise Reject(f"clause:rel:{child.rel}:{u}")

    def particle(self, c: Clause, t: Tok, node_id: int) -> None:
        lemma, form = norm(t.lemma), norm(t.form)
        kind = LICENSED_PARTICLES.get(lemma) or LICENSED_PARTICLES.get(form)
        if kind is None:
            if t.upos != "PART":
                raise Reject("unlicensed_func:" + lemma)
            # «тоже», «только», «даже», «вот»: a meaning-bearing particle is a node to copy, not a
            # licensed closed-class word (an extension of the format: role ``particle``)
            self.node(c, t, "particle", node_id, pos="particle", merge_fixed=True)
            return
        self.licence(c, t, "particle_" + kind)
        if kind == "negation":
            if c.clause["negated"]:
                raise Reject("double_negation")
            c.clause["negated"] = True
            c.clause["negation_node"] = node_id
        elif kind == "conditional":
            c.clause["mood"] = "conditional"
        else:
            c.clause["move"] = "polar_question"

    def xcomp(self, c: Clause, child: Tok, node_id: int) -> None:
        cops = self.kids(child, "cop")
        if child.upos == "VERB" and child.feats.get("VerbForm") == "Inf" and not cops:
            idx = self.node(c, child, "action_complement", node_id)
            self.fill(c, child, idx, embedded=True)
        elif cops and norm(cops[0].lemma) == "быть" and child.upos in NOMINAL | {"ADJ", "NUM"}:
            idx = self.node(c, cops[0], "action_complement", node_id, lemma="быть", pos="verb")
            self.np(c, child, "predicate_complement", idx, clause_level=True)
            self.fill(c, child, idx, embedded=True)
        elif child.upos in NOMINAL | {"ADJ"} and not cops:
            self.np(c, child, "predicate_complement", node_id, clause_level=True)
            self.fill(c, child, node_id, embedded=True)
        else:
            raise Reject(f"xcomp:{child.upos}")

    def subclause(self, parent: Clause, t: Tok, rel: str) -> None:
        words = {norm(m.lemma) for m in self.kids(t, "mark")}
        if rel == "conj":
            link = "coordination"
            if not (self.kids(t, "nsubj") or t.upos in VERBAL):
                raise Reject("np:coordination")
        elif rel == "ccomp":
            link = "complement"
        elif words & {"если", "коли"}:
            link = "condition"
        elif words & {"потому", "поскольку", "ибо"} or "так" in words:
            link = "cause"
        else:
            link = "adverbial"  # temporal / purpose / concession: the connective word is licensed
        self.clause(t, link, self.clauses.index(parent))

    def clause(self, h: Tok, link: str, head_clause: int | None) -> Clause:
        c = Clause(link={"type": link, "head_clause": head_clause})
        self.clauses.append(c)
        cops = self.kids(h, "cop")
        if cops or (h.upos not in VERBAL and self.kids(h, "nsubj")):
            if cops:
                cop = cops[0]
                if norm(cop.lemma) != "быть":
                    raise Reject("cop:not_byt")
                if cop.feats.get("Tense") == "Pres" or cop.feats.get("VerbForm") != "Fin":
                    raise Reject("cop:explicit_present")
                pred = self.node(c, cop, "predicate", None, pos="verb")
                c.predicate_tok = cop
            else:
                pred = self.node(c, None, "predicate", None, lemma="быть", pos="verb")
                c.silent_copula = True
            if h.upos not in NOMINAL | {"ADJ", "NUM", "VERB", "DET"}:
                raise Reject(f"copular:{h.upos}")
            c.clause["predicate_class"] = "copular"
            marker = self.comparator(h)
            self.np(c, h, "predicate_complement", pred, clause_level=True, marker=marker)
            self.fill(c, h, pred, embedded=False)
        elif h.upos in VERBAL:
            pred = self.node(c, h, "predicate", None, pos="verb")
            c.predicate_tok = h
            c.clause["predicate_class"] = "modal" if norm(h.lemma) in MODALS else "lexical"
            self.fill(c, h, pred, embedded=False)
        else:
            raise Reject("fragment")
        self.finish_clause(c, pred)
        return c

    def finish_clause(self, c: Clause, pred: int) -> None:
        features = c.predicate_tok.feats if c.predicate_tok is not None else {}
        tense = {"Past": "past", "Pres": "present", "Fut": "future"}.get(features.get("Tense", ""))
        c.clause["tense"] = "present" if c.silent_copula else (tense or "present")
        imperative = features.get("Mood") == "Imp"
        if imperative:
            c.clause["mood"] = "imperative"
            if c.clause["move"] == "statement":
                c.clause["move"] = "command"
        has_subject = any(n["role"] == "subject" for n in c.nodes)
        if not has_subject:
            if c.silent_copula:
                raise Reject("fragment")
            if features.get("VerbForm") == "Fin" and not imperative:
                pronoun = SUBJECT_PRONOUN.get(
                    (features.get("Person", ""), features.get("Number", ""))
                )
                if not pronoun:
                    raise Reject("verb_without_subject")
                self.node(c, None, "subject", pred, lemma=pronoun, pos="pronoun", elided=True)
                c.clause["subject_elided"] = True

    def build(self) -> list[Clause]:
        roots = [t for t in self.s.toks if t.rel == "root"]
        if len(roots) != 1:
            raise Reject("roots")
        self.clause(roots[0], "main", None)
        for t in self.s.toks:
            if t.i not in self.consumed:
                raise Reject(f"unconsumed:{t.rel}:{t.upos}")
        for ci, c in enumerate(self.clauses):
            remap = canonicalize(c)
            for tok_id, (kind, clause_index, node) in list(self.func.items()):
                if clause_index == ci and node is not None:
                    self.func[tok_id] = (kind, ci, remap[node])
        # a sub-clause of a question is asked, not asserted
        for c in self.clauses[1:]:
            if c.link["type"] == "complement":
                if c.clause["focus"] is not None:
                    c.link["type"] = "embedded_wh"
                elif c.clause["move"] == "polar_question":
                    c.link["type"] = "embedded_polar"
        return self.clauses


# --------------------------------------------------------------------------- v2 format: finishing a target
UPOS_OF_POS = {
    "noun": "NOUN",
    "verb": "VERB",
    "adjective": "ADJ",
    "participle": "VERB",
    "adverb": "ADV",
    "number": "NUM",
    "pronoun": "PRON",
    "gerund": "VERB",
    "particle": "PART",
}
PREP_FORMS_PATH = Path(__file__).resolve().parent / "prep_forms.json"


@functools.lru_cache(maxsize=1)
def _prep_forms() -> dict:
    import json

    return json.loads(PREP_FORMS_PATH.read_text(encoding="utf-8")) if PREP_FORMS_PATH.exists() else {}


def preposition_form(preposition: str, next_word: str) -> str:
    """Surface of a preposition before ``next_word`` (в/во, с/со, к/ко ...): a decision list over the
    first letters of the next word, read from the corpus (``prep_forms.py``), never typed by hand."""
    rules = _prep_forms().get(preposition)
    if not rules:
        return preposition
    word = norm(next_word)
    for prefix, form in rules:
        if word.startswith(prefix):
            return form
    return preposition


def flat_view(tree: dict) -> tuple[list[dict], list[dict], list[int | None]]:
    """(clauses, flat node list, global head index per node) of a clause or message tree."""
    clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
    flat: list[dict] = []
    heads: list[int | None] = []
    base = 0
    for clause in clauses:
        for node in clause["nodes"]:
            flat.append(node)
            heads.append(None if node["head"] is None else base + node["head"])
        base += len(clause["nodes"])
    return clauses, flat, heads


def phrase_members(flat: list[dict], heads: list[int | None], k: int) -> set[int]:
    """The words a preposition of node ``k`` may stand in front of: the node and its modifiers /
    post-modifiers, not crossing into a dependent phrase that has its own preposition or marker."""
    members, stack = {k}, [k]
    while stack:
        x = stack.pop()
        for j, h in enumerate(heads):
            if (
                h == x
                and flat[j]["role"] in {"modifier", "postmodifier"}
                and not flat[j].get("preposition")
                and not flat[j].get("marker")
            ):
                members.add(j)
                stack.append(j)
    return members


def punct_capacity(tree: dict) -> dict[str, int]:
    """How many of each inner mark the input allows: commas by clauses, coordinated members and
    runs of modifiers; a pair of dashes; one colon. Terminal marks are not inner marks."""
    clauses, flat, heads = flat_view(tree)
    siblings: dict[int, int] = {}
    for j, h in enumerate(heads):
        if h is not None and flat[j]["role"] == "modifier":
            siblings[h] = siblings.get(h, 0) + 1
    runs = sum(max(0, v - 1) for v in siblings.values())
    conjuncts = sum(1 for n in flat if n["role"] == "conjunct")
    inner = len(clauses) - 1
    return {",": 2 * inner + conjuncts + runs + 1, "-": 2, "—": 2, ":": 1, ";": inner + 1}


def _surface(flat: list[dict], action: list) -> str:
    node = flat[action[1]]
    word = inflect(
        node["lemma"],
        UPOS_OF_POS.get(node["pos"], "NOUN"),
        action[2],
        bool(node.get("preposition")),
    )
    return word or "?"


def finalize_target(tree: dict, target: list, *, end: str | None = None) -> list:
    """Bring a surface target to the v2 format (mutates ``tree``: ``clause.end`` of the main clause).

    * a leading mark (the dash of a chat line) is dropped; the terminal mark leaves the target and becomes
      the clause feature ``end`` (``.``/``!``/``?``/``…``/``none``; ``end`` overrides what the text had);
    * inner marks: never repeated, never terminal, within ``punct_capacity``;
    * a preposition is a property of its node: its action leaves the target, it must have stood directly in
      front of the first word of the node's phrase and be the form ``preposition_form`` gives;
    * consecutive closed-class words of one licence become ONE action (a connective phrase).
    """
    clauses, flat, heads = flat_view(tree)
    actions = [list(a) for a in target]
    while actions and actions[0][0] == "P":
        actions.pop(0)
    final = None
    if actions and actions[-1][0] == "P":
        if actions[-1][1] not in TERMINALS:
            raise Reject("punct:dangling")
        final = actions.pop()[1]
    used: dict[str, int] = {}
    for i, a in enumerate(actions):
        if a[0] != "P":
            continue
        if a[1] in TERMINALS:
            raise Reject("punct:internal_terminal")
        if i and actions[i - 1][0] == "P":
            raise Reject("punct:repeat")
        used[a[1]] = used.get(a[1], 0) + 1
    capacity = punct_capacity(tree)
    for mark, count in used.items():
        if count > capacity.get(mark, 0):
            raise Reject("punct:capacity")
    position = {a[1]: i for i, a in enumerate(actions) if a[0] == "C"}
    kept: list[list] = []
    for i, a in enumerate(actions):
        if a[0] == "F" and a[2].startswith("marker:"):
            # the marker of a comparison is a property of its node (like a preposition): it stands directly in front
            # of the first word of the node's phrase (in front of the node's preposition, when it has one)
            k = int(a[2].partition(":")[2])
            members = phrase_members(flat, heads, k)
            j = i + 1
            if j < len(actions) and actions[j][0] == "F" and actions[j][2] == f"preposition:{k}":
                j += 1
            first = min((position[m] for m in members if m in position), default=None)
            if j >= len(actions) or actions[j][0] != "C" or first != j or actions[j][1] not in members:
                raise Reject("marker:not_before_phrase")
            if a[1] != flat[k].get("marker"):
                raise Reject("marker:form")
            continue
        if a[0] == "F" and a[2].startswith("preposition:"):
            k = int(a[2].partition(":")[2])
            members = phrase_members(flat, heads, k)
            nxt = actions[i + 1] if i + 1 < len(actions) else None
            first = min((position[m] for m in members if m in position), default=None)
            if nxt is None or nxt[0] != "C" or first != i + 1 or nxt[1] not in members:
                raise Reject("preposition:not_before_phrase")
            if preposition_form(flat[k]["preposition"], _surface(flat, nxt)) != a[1]:
                raise Reject("preposition:allomorph")
            continue
        if a[0] == "F" and kept and kept[-1][0] == "F" and kept[-1][2] == a[2]:
            kept[-1][1] += " " + a[1]
            continue
        kept.append(a)
    root = clauses[0]["clause"]
    root["end"] = end or final or "none"
    # the closed-class words are an INPUT of the network (stage 4.29): which conjunction, which link, «не»/«бы»/«ли» —
    # the plan decides them (they carry meaning), the network only places them
    for clause in clauses:
        clause["clause"]["func"] = []
    for node in flat:
        node.pop("func", None)
    for a in kept:
        if a[0] != "F":
            continue
        kind, _, where = a[2].partition(":")
        if kind == "conjunction":
            flat[int(where)].setdefault("func", []).append([kind, a[1]])
        else:
            clauses[int(where)]["clause"]["func"].append([kind, a[1]])
    return kept


def annotate_from_target(tree: dict, target: list) -> None:
    """Input features of the v2 format that the SURFACE fixes (so a corpus can supply them; the runtime
    plan has to know them itself — see LAST_MESSAGE43):

    * ``lex.degree = comparative`` / ``lex.form = short`` of a word that stood as a comparative / a short form;
    * the gender of я/ты/вы, from the agreeing word of the same clause (past verb, short adjective).
    """
    clauses, flat_nodes, _heads = flat_view(tree)
    tags = {a[1]: a[2] for a in target if a[0] == "C"}
    for index, tag in tags.items():
        grams = set(tag.split(","))
        node = flat_nodes[index]
        if node["role"] == "predicate_complement" and node["pos"] in {"adjective", "participle"}:
            case = next((LONG_CASE[g] for g in LONG_CASE if g in grams), None)
            if case:  # the case of an adjectival nominal part is the text's (the plan's rule gives another one)
                node["case_constraint"] = case
        if "COMP" in grams:
            flat_nodes[index]["lex"]["degree"] = "comparative"
        if grams & {"ADJS", "PRTS"}:
            flat_nodes[index]["lex"]["form"] = "short"
    base = 0
    for clause in clauses:
        nodes = clause["nodes"]
        for node in nodes:
            if node["role"] != "subject" or node["lemma"] not in SPEAKER_PRONOUNS:
                continue
            for other in nodes:
                tag = tags.get(base + other["id"])
                if (
                    tag is None
                    or other["pos"] not in {"verb", "adjective", "participle"}
                    or other["role"] not in {"predicate", "predicate_complement"}
                ):
                    continue
                grams = set(tag.split(","))
                gender = next((LONG_GENDER[g] for g in LONG_GENDER if g in grams), None)
                if "sing" in grams and gender:
                    node["lex"]["gender"] = gender
                    break
        base += len(nodes)


def rank_conjuncts(tree: dict, target: list) -> None:
    """``conj_rank`` of a coordinated member: its place in the enumeration (the head member is the 0-th). The plan
    knows the order of an enumeration; the node list does not carry it."""
    _clauses, flat, heads = flat_view(tree)
    counter: dict[int, int] = {}
    for action in target:
        if action[0] != "C" or flat[action[1]]["role"] != "conjunct":
            continue
        root = action[1]
        while flat[root]["role"] == "conjunct" and heads[root] is not None:
            root = heads[root]
        counter[root] = counter.get(root, 0) + 1
        flat[action[1]]["conj_rank"] = counter[root]


def check_consistency(tree: dict, target: list) -> None:
    """A row whose reference contradicts its own plan (a gender or a case the input fixes and the reference does not
    follow, a broken nominal phrase, an agreement the plan cannot produce) is not a row to learn from or to be
    measured on: ``Reject('inconsistent:<kinds>')``. The same function is the measure of plan fidelity."""
    import plan_fidelity

    found = plan_fidelity.check(tree, target) - {"cc"}
    if found:
        raise Reject("inconsistent:" + "+".join(sorted(found)))


def realize(tree: dict, target: list) -> list[str]:
    """Tokens (words and marks) a v2 target stands for: the prepositions are put back in front of the
    first word of their node's phrase, the terminal mark is appended from ``clause.end``."""
    clauses, flat, heads = flat_view(tree)
    words: list[str | None] = []
    owner: list[int | None] = []
    for a in target:
        if a[0] == "C":
            words.append(_surface(flat, a))
            owner.append(a[1])
        else:
            words.append(a[1])
            owner.append(None)
    inserts: dict[int, list[tuple[str, str]]] = {}
    # a v2 target (the evaluation of a v2 export) still carries the marker as an action: it must not be inserted twice
    marker_is_action = any(a[0] == "F" and str(a[2]).startswith("marker") for a in target)
    for k, node in enumerate(flat):
        if not node.get("preposition") and not node.get("marker"):
            continue
        members = phrase_members(flat, heads, k)
        at = next((i for i, o in enumerate(owner) if o in members), None)
        if at is not None:
            if node.get("marker") and not marker_is_action:
                inserts.setdefault(at, []).append(("marker", node["marker"]))
            if node.get("preposition"):
                inserts.setdefault(at, []).append(("preposition", node["preposition"]))
    out: list[str] = []
    for i, word in enumerate(words):
        for kind, text in inserts.get(i, ()):
            out.append(preposition_form(text, word or "") if kind == "preposition" else text)
        out.append(word or "")
    end = clauses[0]["clause"].get("end")
    if end and end != "none":
        out.append(end)
    return out


# --------------------------------------------------------------------------- conversion result
@dataclass(slots=True)
class Converted:
    sent_id: str
    text: str
    tree: dict
    target: list
    meta: dict


def _skeleton(tree: dict, target: list) -> tuple[str, str]:
    """Delexicalised skeleton: (coarse, order). The coarse one is the stand's skeleton of
    ``build_bench.py`` (predicate class, move, negation, mood, sorted roles; ``^`` = bound to an embedded
    word) plus the clause links — what the deduplication caps. The order one is the sequence of
    roles+tags and closed-class words of the target (finer; for statistics and ``train_wide``)."""
    clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
    first = clauses[0]
    roles: list[str] = []
    ptr_role: dict[int, str] = {}
    base = 0
    for clause in clauses:
        for node in clause["nodes"]:
            mark = "^" if node["head"] not in (None, 0) else ""
            ptr_role[base + node["id"]] = f"{node['role']}{mark}"
            if node["role"] != "modifier":
                roles.append(f"{node['role']}{mark}")
        base += len(clause["nodes"])
    c = first["clause"]
    head = "|".join(
        (
            c["predicate_class"],
            "question" if c["move"].endswith("question") else c["move"],
            str(c["negated"]),
            c["mood"],
            *sorted(roles),
        )
    )
    order = " ".join(f"{ptr_role[a[1]]}/{a[2]}" if a[0] == "C" else a[1] for a in target)
    links = ",".join(sorted(cl["link"]["type"] for cl in clauses[1:]))
    return f"{head}||{links}", order


def convert(sentence: Sentence, *, max_words: int = 25) -> Converted:
    """One UD sentence -> tree + target, or ``Reject(reason)``."""
    words = [t for t in sentence.toks if t.upos != "PUNCT"]
    if not 2 <= len(words) <= max_words:
        raise Reject("length")
    if any(
        re.search(r"[A-Za-z0-9]", t.form) or t.upos in {"X", "SYM", "INTJ"} for t in sentence.toks
    ):
        raise Reject("latin_digit_symbol_intj")
    builder = Builder(sentence)
    clauses = builder.build()
    last = sentence.toks[-1]
    root = clauses[0]
    if (
        root.clause["move"] == "statement"
        and last.form == "?"
        and root.clause["mood"] != "imperative"
    ):
        root.clause["move"] = "polar_question"
    offsets, total = [], 0
    for c in clauses:
        offsets.append(total)
        total += len(c.nodes)
    owner: dict[int, tuple[int, int]] = {}
    for ci, c in enumerate(clauses):
        for tok_id, node_id in c.toks.items():
            owner[tok_id] = (ci, node_id)
    target: list = []
    for t in sentence.toks:
        if t.i in builder.merged:
            continue  # the second word of a fixed expression: its node's one action stands for it
        if t.i in builder.puncts:
            target.append(["P", t.form])
        elif t.i in owner:
            ci, node_id = owner[t.i]
            node = clauses[ci].nodes[node_id]
            tag = tag_of(t)
            form = inflect(node["lemma"], t.upos, tag, bool(node.get("preposition")))
            if form != builder.merged_forms.get(t.i, norm(t.form)):
                raise Reject(f"roundtrip:{t.upos}")
            target.append(["C", offsets[ci] + node_id, tag])
        elif t.i in builder.func:
            kind, ci, node_id = builder.func[t.i]
            where = offsets[ci] + node_id if node_id is not None else ci
            target.append(["F", norm(t.form), f"{kind}:{where}"])
        else:
            raise Reject("internal:unplaced")
    if len(clauses) == 1:
        tree = {
            "version": TREE_VERSION,
            "kind": "clause",
            "nodes": clauses[0].nodes,
            "clause": clauses[0].clause,
        }
    else:
        entries = []
        for ci, c in enumerate(clauses):
            entries.append({"link": c.link, "id": ci, "nodes": c.nodes, "clause": c.clause})
        tree = {"version": TREE_VERSION, "kind": "message", "speech_act": None, "clauses": entries}
    target = finalize_target(tree, target)
    annotate_from_target(tree, target)
    rank_conjuncts(tree, target)
    check_consistency(tree, target)
    meta = _meta(tree, target, sentence)
    return Converted(sentence.sent_id, sentence.text, tree, target, meta)


def _meta(tree: dict, target: list, sentence: Sentence) -> dict:
    clauses = tree["clauses"] if tree["kind"] == "message" else [tree]
    nodes = [n for c in clauses for n in c["nodes"]]
    constructions = set()
    roles = {n["role"] for n in nodes}
    first = clauses[0]["clause"]
    if "comparison" in roles:
        constructions.add("comparison")
    if any(
        n["head"] not in (None, 0) and n["role"] not in {"modifier", "postmodifier"} for n in nodes
    ):
        constructions.add("nested_binding")
    if (
        any(n["role"] == "action_complement" and n["lemma"] in {"быть", "стать"} for n in nodes)
        and "predicate_complement" in roles
    ):
        constructions.add("modal_be_predicate")
    if any(c["clause"]["focus"] is not None for c in clauses):
        constructions.add("wh_question")
    if first["move"] == "polar_question":
        constructions.add("polar_question")
    if any(c["clause"]["mood"] == "conditional" for c in clauses):
        constructions.add("conditional")
    if any(c["clause"]["negated"] for c in clauses):
        constructions.add("negation")
    if first["predicate_class"] == "copular":
        constructions.add("copular")
    if first["subject_elided"]:
        constructions.add("pro_drop")
    if len(clauses) > 1:
        constructions.add("link_" + "_".join(sorted({c["link"]["type"] for c in clauses[1:]})))
    coarse, order = _skeleton(tree, target)
    return {
        "skeleton": coarse,
        "skeleton_order": order,
        "lemmas": sorted({n["lemma"] for n in nodes if not n.get("elided")}),
        "roles": sorted(roles),
        "constructions": sorted(constructions),
        "n_nodes": len(nodes),
        "n_words": sum(1 if a[0] == "C" else len(a[1].split()) for a in target if a[0] != "P")
        + sum(1 for n in nodes if n.get("preposition"))
        + sum(1 for n in nodes if n.get("marker")),
    }


def target_text(target: list, tree: dict) -> str:
    """The sentence the target realizes (for the round-trip check of the data and the local evaluation)."""
    return " ".join(realize(tree, target))


def main(argv: list[str]) -> int:
    """Smoke CLI: convert files, print reject statistics and a few examples."""
    from collections import Counter

    paths = [Path(a) for a in argv[1:]]
    limit = 5000
    ok, rejects, shown = 0, Counter(), 0
    for path in paths:
        for sentence in read_conllu(path):
            if ok + sum(rejects.values()) >= limit:
                break
            try:
                result = convert(sentence)
            except Reject as reason:
                rejects[str(reason)] += 1
                continue
            ok += 1
            if shown < 8:
                shown += 1
                print(sentence.text, "->", result.meta["constructions"])
    print("ok", ok, "rejects", sum(rejects.values()))
    for reason, n in rejects.most_common(25):
        print(f"  {n:5d} {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
