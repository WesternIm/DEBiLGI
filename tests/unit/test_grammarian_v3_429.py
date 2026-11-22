# ruff: noqa: E501
"""Stage 4.29: the v3 format of the grammarian data (closed-class words as input, the marker as a node property, conjunct
rank, givenness, the case of an adjectival nominal part), the fixed ``words()`` and the plan fidelity measure.

The CoNLL-U below is annotated by hand for the test, no corpus text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "grammarian"))

import alt_orders  # noqa: E402
import plan_fidelity as pf  # noqa: E402
import ud_to_tree as u  # noqa: E402
from test_ud_to_tree_425 import LOVE, nodes_of, sentence  # noqa: E402


def clauses(result: u.Converted) -> list[dict]:
    return result.tree["clauses"] if result.tree["kind"] == "message" else [result.tree]


def test_words_do_not_count_a_dash_as_a_word_and_compare_hyphenated_parts():
    assert alt_orders.words("-- Я пробовал.") == alt_orders.words("Я пробовал.") == ["я", "пробовал"]
    assert alt_orders.words("бизнес - сообщества") == alt_orders.words("бизнес-сообщества") == ["бизнес", "сообщества"]
    assert alt_orders.words("Москва — столица") == ["москва", "столица"]


def test_closed_class_words_are_input_of_the_clause_and_the_node():
    sample = sentence(
        "Если ты придёшь, я не уйду.",
        [
            "Если если SCONJ - 3 mark",
            "ты ты PRON Case=Nom|Number=Sing|Person=2 3 nsubj",
            "придёшь прийти VERB Mood=Ind|Number=Sing|Person=2|Tense=Fut|VerbForm=Fin 7 advcl",
            ", , PUNCT - 7 punct",
            "я я PRON Case=Nom|Number=Sing|Person=1 7 nsubj",
            "не не PART Polarity=Neg 7 advmod",
            "уйду уйти VERB Mood=Ind|Number=Sing|Person=1|Tense=Fut|VerbForm=Fin 0 root",
            ". . PUNCT - 7 punct",
        ],
    )
    result = u.convert(sample)
    main, sub = clauses(result)
    assert main["clause"]["func"] == [["particle_negation", "не"]]
    assert sub["clause"]["func"] == [["link", "если"]] and sub["link"]["type"] == "condition"
    assert pf.func_phrases(result.tree) == {"particle_negation|не": 1, "link|если": 1}
    assert pf.check(result.tree, result.target) == set()
    dropped = [a for a in result.target if not (a[0] == "F" and a[1] == "если")]
    assert "cc" in pf.check(result.tree, dropped)  # a missing closed-class word of the plan is a violation


def test_the_marker_of_a_comparison_is_a_node_property_and_a_fixed_expression_is_not_one():
    sample = sentence(
        "Он пришёл как раз.",
        [
            "Он он PRON Case=Nom|Gender=Masc|Number=Sing|Person=3 2 nsubj",
            "пришёл прийти VERB Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            "как как ADV - 2 advmod",
            "раз раз ADV - 3 fixed",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(sample)
    assert "как раз" in nodes_of(result) and nodes_of(result)["как раз"]["role"] == "manner"
    assert not any(n.get("marker") for n in clauses(result)[0]["nodes"])
    assert u.target_text(result.target, result.tree) == "он пришел как раз ."


def test_conjunct_rank_is_the_place_in_the_enumeration():
    sample = sentence(
        "Я люблю кошек, собак и птиц.",
        [
            "Я я PRON Case=Nom|Number=Sing|Person=1 2 nsubj",
            "люблю любить VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root",
            "кошек кошка NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 2 obj",
            ", , PUNCT - 5 punct",
            "собак собака NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 3 conj",
            "и и CCONJ - 7 cc",
            "птиц птица NOUN Case=Acc|Gender=Fem|Number=Plur|Animacy=Anim 5 conj",
            ". . PUNCT - 2 punct",
        ],
    )
    result = u.convert(sample)
    nodes = nodes_of(result)
    assert (nodes["собака"]["conj_rank"], nodes["птица"]["conj_rank"]) == (1, 2)
    assert "conj_rank" not in nodes["кошка"] and nodes["птица"]["func"] == [["conjunction", "и"]]


def test_givenness_is_known_only_with_a_document_context():
    assert all("given" not in n for n in nodes_of(u.convert(LOVE)).values())
    sample = sentence(LOVE.text, [])  # same shape, a context is attached below
    sample.toks, sample.context = LOVE.toks, frozenset({"кошка", "любить"})
    nodes = nodes_of(u.convert(sample))
    assert nodes["кошка"]["given"] is True and nodes["любить"]["given"] is True and nodes["я"]["given"] is False


def test_an_adjectival_nominal_part_carries_its_case_and_the_noun_its_gender_in_the_text():
    sample = sentence(
        "Дом был красивым.",
        [
            "Дом дом NOUN Case=Nom|Gender=Masc|Number=Sing|Animacy=Inan 3 nsubj",
            "был быть AUX Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 3 cop",
            "красивым красивый ADJ Case=Ins|Degree=Pos|Gender=Masc|Number=Sing 0 root",
            ". . PUNCT - 3 punct",
        ],
    )
    nodes = nodes_of(u.convert(sample))
    assert nodes["красивый"]["case_constraint"] == "instrumental"
    # a name the dictionary gives the other gender: the gender of THIS text is the input («Люда шла»)
    name = sentence(
        "Люда шла.",
        [
            "Люда люда NOUN Case=Nom|Gender=Fem|Number=Sing|Animacy=Anim 2 nsubj",
            "шла идти VERB Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            ". . PUNCT - 2 punct",
        ],
    )
    assert nodes_of(u.convert(name))["люда"]["lex"]["gender"] == "feminine"


def test_a_row_that_contradicts_its_plan_is_rejected():
    split_group = sentence(
        "Книгу я читал интересную.",
        [
            "Книгу книга NOUN Case=Acc|Gender=Fem|Number=Sing|Animacy=Inan 3 obj",
            "я я PRON Case=Nom|Number=Sing|Person=1 3 nsubj",
            "читал читать VERB Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin 0 root",
            "интересную интересный ADJ Case=Acc|Degree=Pos|Gender=Fem|Number=Sing 1 amod",
            ". . PUNCT - 3 punct",
        ],
    )
    with pytest.raises(u.Reject, match="inconsistent:np"):
        u.convert(split_group)


def test_plan_fidelity_flags_what_the_plan_did_not_say():
    result = u.convert(LOVE)
    assert pf.check(result.tree, result.target) == set()
    wrong_case = [[*a[:2], "gent,plur"] if a[0] == "C" and a[2] == "accs,plur" else a for a in result.target]
    assert "case" in pf.check(result.tree, wrong_case)
    wrong_person = [[*a[:2], "3per,pres,sing"] if a[0] == "C" and a[2].startswith("1per") else a for a in result.target]
    assert "agree" in pf.check(result.tree, wrong_person)
