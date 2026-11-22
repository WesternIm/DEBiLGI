# ruff: noqa: E501
"""Stage 4.25: the UD -> ``thought_to_tree`` converter of the grammarian data path.

The CoNLL-U below is annotated by hand for the test (UD v2 Russian conventions), no corpus text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "grammarian"))

import ud_to_tree as u  # noqa: E402


def sentence(text: str, rows: list[str]) -> u.Sentence:
    """Rows are ``form lemma upos feats head rel`` (space separated, ``-`` = no feats)."""
    toks = []
    for number, row in enumerate(rows, 1):
        form, lemma, upos, feats, head, rel = row.split()
        parsed = {} if feats == "-" else dict(item.split("=") for item in feats.split("|"))
        toks.append(u.Tok(number, form, lemma, upos, parsed, int(head), rel))
    for tok in toks:
        if tok.head:
            toks[tok.head - 1].children.append(tok.i)
    return u.Sentence("t", text, toks)


LOVE = sentence(
    "Я люблю кошек.",
    [
        "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
        "люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
        "кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 2 obj",
        ". . PUNCT - 2 punct",
    ],
)


def nodes_of(result: u.Converted) -> dict[str, dict]:
    clause = result.tree["clauses"][0] if result.tree["kind"] == "message" else result.tree
    return {n["lemma"]: n for n in clause["nodes"]}


def test_simple_clause_roles_and_target():
    result = u.convert(LOVE)
    nodes = nodes_of(result)
    assert result.tree["kind"] == "clause" and result.tree["version"] == 1
    assert nodes["любить"]["role"] == "predicate" and nodes["любить"]["head"] is None
    assert nodes["я"]["role"] == "subject" and nodes["кошка"]["role"] == "direct_object"
    assert all(a[0] == "C" for a in result.target)  # v2: the terminal mark is the clause feature ``end``, not an action
    assert result.tree["clause"]["end"] == "."
    words = [a for a in result.target if a[0] == "C"]
    assert len(words) == 3 and {a[2] for a in words} == {"nomn", "1per,pres,sing", "accs,plur"}
    assert u.target_text(result.target, result.tree) == "я люблю кошек ."


def test_node_list_carries_no_word_order():
    """The same plan from a different surface order lists the nodes identically (canonical roles)."""
    swapped = sentence(
        "Кошек люблю я.",
        [
            "Кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 2 obj",
            "люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            ". . PUNCT - 2 punct",
        ],
    )
    assert u.convert(swapped).tree == u.convert(LOVE).tree
    assert u.convert(swapped).target != u.convert(LOVE).target  # the order is the target


def test_modal_with_infinitive_binds_the_object_to_the_embedded_verb():
    modal = sentence(
        "Я хочу найти воробья.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "хочу хотеть VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "найти найти VERB Aspect=Perf|VerbForm=Inf 2 xcomp",
            "воробья воробей NOUN Case=Acc|Gender=Masc|Number=Sing|Animacy=Anim 3 obj",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(modal)
    tree = result.tree
    by_lemma = {n["lemma"]: n for n in tree["nodes"]}
    assert tree["clause"]["predicate_class"] == "modal"
    assert by_lemma["найти"]["role"] == "action_complement" and by_lemma["найти"]["head"] == 0
    assert by_lemma["воробей"]["role"] == "direct_object"
    assert by_lemma["воробей"]["head"] == by_lemma["найти"]["id"]  # ``head`` of 4.24


def test_to_be_with_a_nominal_is_action_complement_plus_predicate_complement():
    sample = sentence(
        "Я хочу быть врачом.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "хочу хотеть VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "быть быть AUX Aspect=Imp|VerbForm=Inf 4 cop",
            "врачом врач NOUN Case=Ins|Gender=Masc|Number=Sing|Animacy=Anim 2 xcomp",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(sample)
    by_lemma = {n["lemma"]: n for n in result.tree["nodes"]}
    assert by_lemma["быть"]["role"] == "action_complement"
    assert by_lemma["врач"]["role"] == "predicate_complement"
    assert by_lemma["врач"]["head"] == by_lemma["быть"]["id"]
    assert ["C", by_lemma["врач"]["id"], "ablt,sing"] in result.target
    assert "modal_be_predicate" in result.meta["constructions"]


def test_comparison_keeps_the_marker_and_the_nominative():
    sample = sentence(
        "Я хочу быть как пилот.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "хочу хотеть VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "быть быть VERB Aspect=Imp|VerbForm=Inf 2 xcomp",
            "как как SCONJ - 5 mark",
            "пилот пилот NOUN Case=Nom|Gender=Masc|Number=Sing|Animacy=Anim 3 obl",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(sample)
    comparison = next(n for n in result.tree["nodes"] if n["role"] == "comparison")
    assert comparison["lemma"] == "пилот" and comparison["marker"] == "как"
    assert ["C", comparison["id"], "nomn,sing"] in result.target
    # stage 4.29: the marker is a property of its node (like a preposition), not an action of the target
    assert not any(a[0] == "F" and a[2].startswith("marker:") for a in result.target)
    assert "как пилот" in u.target_text(result.target, result.tree)
    assert "comparison" in result.meta["constructions"]


def test_particles_are_features_not_nodes():
    sample = sentence(
        "Ты бы не знал?",
        [
            "Ты ты PRON Case=Nom|Number=Sing|Person=2 4 nsubj",
            "бы бы PART Mood=Cnd 4 advmod",
            "не не PART Polarity=Neg 4 advmod",
            "знал знать VERB Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            "? ? PUNCT - 4 punct",
        ],
    )
    result = u.convert(sample)
    clause = result.tree["clause"]
    assert clause["mood"] == "conditional" and clause["negated"] and clause["negation_node"] == 0
    assert clause["move"] == "polar_question" and clause["tense"] == "past"
    licences = {a[2].split(":")[0] for a in result.target if a[0] == "F"}
    assert licences == {"particle_conditional", "particle_negation"}
    assert all(n["lemma"] not in {"бы", "не"} for n in result.tree["nodes"])


def test_wh_word_is_the_focus():
    sample = sentence(
        "Что ты любишь?",
        [
            "Что что PRON Case=Acc|PronType=Int 3 obj",
            "ты ты PRON Case=Nom|Number=Sing|Person=2 3 nsubj",
            "любишь любить VERB Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin 0 root",
            "? ? PUNCT - 3 punct",
        ],
    )
    result = u.convert(sample)
    focus = result.tree["clause"]["focus"]
    assert result.tree["clause"]["move"] == "slot_question"
    assert result.tree["nodes"][focus]["lemma"] == "что"
    assert "wh_question" in result.meta["constructions"]


def test_pro_drop_adds_an_elided_subject_without_a_surface_action():
    sample = sentence(
        "Люблю кошек.",
        [
            "Люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 1 obj",
            ". . PUNCT - 1 punct",
        ],
    )
    result = u.convert(sample)
    subject = next(n for n in result.tree["nodes"] if n["role"] == "subject")
    assert subject["lemma"] == "я" and subject["elided"] and result.tree["clause"]["subject_elided"]
    assert all(a[1] != subject["id"] for a in result.target if a[0] == "C")


def test_zero_copula_clause_is_copular_with_a_silent_predicate():
    sample = sentence(
        "Кошка умная.",
        [
            "Кошка кошка NOUN Case=Nom|Gender=Fem|Number=Sing|Animacy=Anim 2 nsubj",
            "умная умный ADJ Case=Nom|Degree=Pos|Gender=Fem|Number=Sing 0 root",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(sample)
    assert result.tree["clause"]["predicate_class"] == "copular"
    assert result.tree["clause"]["tense"] == "present"
    assert result.tree["nodes"][0]["lemma"] == "быть"
    assert all(a[1] != 0 for a in result.target if a[0] == "C")  # nothing is said for the copula


@pytest.mark.parametrize(
    "rows,reason",
    [
        (["Hello hello X - 0 root", "world world X - 1 flat"], "latin_digit_symbol_intj"),
        (
            [
                "Он он PRON Case=Nom|Number=Sing 4 nsubj",
                "не не PART Polarity=Neg 4 advmod",
                "не не PART Polarity=Neg 4 advmod",
                "знал знать VERB Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            ],
            "double_negation",
        ),
        (
            [
                "Знал знать VERB Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
                "кошку кошка NOUN Case=Acc|Gender=Fem|Number=Sing|Animacy=Anim 1 obj",
            ],
            "verb_without_subject",
        ),
    ],
)
def test_unrepresentable_sentences_are_rejected_with_a_reason(rows, reason):
    text = " ".join(r.split()[0] for r in rows)
    with pytest.raises(u.Reject) as caught:
        u.convert(sentence(text, rows))
    assert str(caught.value).startswith(reason)


def test_a_form_the_paradigm_cannot_reproduce_rejects_the_sentence():
    wrong = sentence(
        "Я люблю кошкам.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "кошкам кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 2 obj",
        ],
    )
    with pytest.raises(u.Reject, match="roundtrip"):
        u.convert(wrong)


def test_the_converter_agrees_with_the_runtime_path_on_a_shared_phrase():
    """The same text through the bot's own readers gives the same tree signature (format parity)."""
    import compare_paths
    import own_stream

    ud = u.convert(LOVE)
    runtime = own_stream.build_item("Я люблю кошек.")
    assert compare_paths.signature(ud.tree) == compare_paths.signature(runtime.tree)
    assert [a[0] for a in ud.target] == [a[0] for a in runtime.target]


def test_tag_vocabulary_round_trips_through_inflect():
    assert u.inflect("кошка", "NOUN", "gent,plur") == "кошек"
    assert u.inflect("хотеть", "VERB", "1per,pres,sing") == "хочу"
    assert u.inflect("он", "PRON", "gent", prosthetic=True) == "него"
    assert u.inflect("быстро", "ADV", "-") == "быстро"


def test_dataset_rules_heldout_lemma_cap_and_bench_exclusion():
    import hashlib

    import build_dataset as b

    tree = u.convert(LOVE).tree
    expected = any(
        int(hashlib.sha256(("L:" + n["lemma"]).encode()).hexdigest()[:12], 16) % 10 == 0
        for n in tree["nodes"]
        if n["pos"] in {"noun", "verb"} and not n.get("elided")
    )
    assert b.heldout_lemma(tree) is expected
    rows = [{"text": f"t{i}", "skeleton": "s"} for i in range(50)]
    kept = b.cap_by_skeleton(rows, 30, b.Counter())
    assert len(kept) == 30
    assert b.normalize("Ёж, и ёлка!") == "еж и елка"


# --------------------------------------------------------------------------- stage 4.27: the v2 format
IN_THE_HOUSE = sentence(
    "Она жила в большом доме!",
    [
        "Она она PRON Case=Nom|Gender=Fem|Number=Sing|Person=3 2 nsubj",
        "жила жить VERB Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
        "в в ADP - 5 case",
        "большом большой ADJ Case=Loc|Degree=Pos|Gender=Masc|Number=Sing 5 amod",
        "доме дом NOUN Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing 2 obl",
        "! ! PUNCT - 2 punct",
    ],
)


def test_v2_preposition_is_a_node_property_not_an_action():
    result = u.convert(IN_THE_HOUSE)
    assert result.tree["clause"]["end"] == "!"
    house = nodes_of(result)["дом"]
    assert house["preposition"] == "в"
    assert not any(a[0] == "F" and a[2].startswith("preposition") for a in result.target)
    assert "в большом доме" in u.target_text(result.target, result.tree)


def test_v2_case_of_a_nominal_group_is_given_in_the_runtime_long_names():
    nodes = nodes_of(u.convert(LOVE))
    assert nodes["кошка"]["case_constraint"] == "accusative" and nodes["я"]["case_constraint"] == "nominative"
    assert "case_constraint" not in nodes["любить"]


def test_v2_end_mark_is_a_feature_and_a_leading_dash_is_dropped():
    asked = sentence(
        "— Я люблю кошек?",
        [
            "— — PUNCT - 3 punct",
            "Я я PRON Case=Nom|Number=Sing|Person=1 3 nsubj",
            "люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 3 obj",
            "? ? PUNCT - 3 punct",
        ],
    )
    result = u.convert(asked)
    assert result.tree["clause"]["end"] == "?" and result.tree["clause"]["move"] == "polar_question"
    assert result.target[0][0] == "C"  # no leading dash
    assert all(a[0] != "P" for a in result.target)


def test_v2_speaker_gender_comes_from_the_agreeing_word():
    past = sentence(
        "Я любила кошек.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "любила любить VERB Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            "кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 2 obj",
            ". . PUNCT - 2 punct",
        ],
    )
    assert nodes_of(u.convert(past))["я"]["lex"].get("gender") == "feminine"
    assert "gender" not in nodes_of(u.convert(LOVE))["я"]["lex"]  # a present verb says nothing about the speaker


def test_v2_comparative_and_short_form_are_lexical_features():
    tree = {
        "kind": "clause",
        "nodes": [
            {"id": 0, "lemma": "быть", "pos": "verb", "lex": {}, "role": "predicate", "head": None},
            {"id": 1, "lemma": "маленький", "pos": "adjective", "lex": {}, "role": "predicate_complement", "head": 0},
            {"id": 2, "lemma": "рад", "pos": "adjective", "lex": {}, "role": "modifier", "head": 1},
        ],
        "clause": {"move": "statement", "end": "."},
    }
    u.annotate_from_target(tree, [["C", 1, "COMP"], ["C", 2, "ADJS,masc,sing"]])
    assert tree["nodes"][1]["lex"] == {"degree": "comparative"}
    assert tree["nodes"][2]["lex"] == {"form": "short"}


def _clause_tree(nodes: list[dict]) -> dict:
    return {
        "kind": "clause",
        "nodes": nodes,
        "clause": {"move": "statement", "negated": False, "mood": "indicative"},
    }


def _node(i: int, lemma: str, pos: str, role: str, head: int | None, **extra) -> dict:
    return {"id": i, "lemma": lemma, "pos": pos, "lex": {}, "role": role, "head": head, **extra}


def test_v2_finalize_preposition_must_stand_before_its_phrase_and_leaves_the_target():
    tree = _clause_tree(
        [
            _node(0, "жить", "verb", "predicate", None),
            _node(1, "дом", "noun", "location", 0, preposition="в"),
            _node(2, "большой", "adjective", "modifier", 1),
        ]
    )
    target = [["C", 0, "past,masc,sing"], ["F", "в", "preposition:1"], ["C", 2, "loct,masc,sing"], ["C", 1, "loct,sing"]]
    kept = u.finalize_target(tree, target)
    assert kept == [["C", 0, "past,masc,sing"], ["C", 2, "loct,masc,sing"], ["C", 1, "loct,sing"]]
    assert tree["clause"]["end"] == "none"
    assert " ".join(u.realize(tree, kept)) == "жил в большом доме"
    misplaced = [["C", 0, "past,masc,sing"], ["C", 2, "loct,masc,sing"], ["F", "в", "preposition:1"], ["C", 1, "loct,sing"]]
    with pytest.raises(u.Reject, match="preposition"):
        u.finalize_target(tree, misplaced)


def test_v2_finalize_merges_a_connective_phrase_into_one_action():
    tree = _clause_tree([_node(0, "жить", "verb", "predicate", None)])
    kept = u.finalize_target(tree, [["C", 0, "INFN"], ["F", "потому", "link:0"], ["F", "что", "link:0"]])
    assert kept[-1] == ["F", "потому что", "link:0"]


def test_v2_finalize_marks_never_repeat_never_end_and_respect_capacity():
    tree = _clause_tree([_node(0, "жить", "verb", "predicate", None), _node(1, "дом", "noun", "location", 0)])
    base = [["C", 0, "INFN"], ["P", ","], ["C", 1, "loct,sing"]]
    assert u.finalize_target(tree, base)[1] == ["P", ","]
    with pytest.raises(u.Reject, match="punct:repeat"):
        u.finalize_target(tree, [["C", 0, "INFN"], ["P", ","], ["P", "-"], ["C", 1, "loct,sing"]])
    with pytest.raises(u.Reject, match="punct:dangling"):
        u.finalize_target(tree, [["C", 0, "INFN"], ["C", 1, "loct,sing"], ["P", ","]])
    with pytest.raises(u.Reject, match="punct:capacity"):
        u.finalize_target(tree, [["C", 0, "INFN"], ["P", ":"], ["C", 1, "loct,sing"], ["P", ":"], ["C", 0, "INFN"]])
    assert u.punct_capacity(tree)[","] >= 1


def test_v2_preposition_form_is_read_from_the_corpus_table():
    assert u.preposition_form("в", "доме") == "в"
    assert u.preposition_form("в", "всем") == "во"  # a corpus-learned decision list, not a typed word list
    assert u.preposition_form("на", "столе") == "на"
