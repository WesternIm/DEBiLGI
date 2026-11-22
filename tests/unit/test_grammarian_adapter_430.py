# ruff: noqa: E501
"""Stage 4.30 (D): the grammarian wired into ``realize_thought`` behind ``DEBILGI_GRAMMARIAN`` (numpy inference, fallback, trace)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from debilgi.language import grammarian
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    _realize_thought_deterministic,
    productive_pattern_curriculum,
    realize_thought,
)

ARTIFACTS = grammarian.artifact_dir()
needs_model = pytest.mark.skipif(
    not (ARTIFACTS / "grammarian_v3.npz").exists(), reason="grammarian artifacts are not installed (data/models/grammarian)"
)
ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / ".codex_work/grammarian_data/dev.jsonl"


def lex(lemma, pos="noun", surface=None, **features):
    return ConceptLexeme(lemma, surface or lemma, pos, tuple(sorted(features.items())))


def verb(lemma, **features):
    return lex(lemma, "verb", **features)


def dep(role, head, *, case=None, preposition=None, modifiers=(), posts=(), at=None):
    return SemanticDependency(
        role, ConceptArgument(head, tuple(modifiers), preposition, case, tuple(posts)), head=at
    )


def cat_thought(**kwargs):
    return ClauseThought(
        verb("видеть", tense="present"),
        (dep("subject", lex("я", "pronoun")), dep("direct_object", lex("кошка", gender="feminine", number="singular"))),
        confidence=0.9,
        **kwargs,
    )


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("DEBILGI_GRAMMARIAN", "1")
    monkeypatch.delenv("DEBILGI_SELF_GENDER", raising=False)
    monkeypatch.delenv("DEBILGI_GRAMMARIAN_THRESHOLD", raising=False)
    grammarian.reset_stats()


def test_flag_off_is_the_deterministic_realizer(monkeypatch):
    monkeypatch.delenv("DEBILGI_GRAMMARIAN", raising=False)
    grammarian.reset_stats()
    thought = cat_thought()
    patterns = productive_pattern_curriculum()
    assert realize_thought(thought, patterns) == _realize_thought_deterministic(thought, patterns)
    assert not grammarian.STATS  # not even a journal entry: the path is untouched


@needs_model
def test_numpy_logits_match_the_reference_of_the_export():
    model = grammarian.load_model()
    ref = np.load(ARTIFACTS / "reference_logits.npz")
    worst = 0.0
    for b in range(ref["in_cat"].shape[0]):
        n = int(ref["in_node_mask"][b].sum())
        e = {k: ref[f"in_{k}"][b, :n] for k in ("cat", "flags", "lex", "chars", "verb")}
        e["rel"], e["N"] = ref["in_rel"][b, :n, :n], n
        enc = model.encode_nodes(e)
        silent, para, grp = ref["in_silent"][b, :n], ref["in_para"][b, :n], ref["in_grp"][b, :n, :n]
        need, punct = ref["in_need"][b].copy(), ref["in_punct_cap"][b].copy()
        st, si = ref["in_step_type"][b], ref["in_step_idx"][b]
        state = np.tanh(enc.mean(0) @ model.w["init_h.weight"].T + model.w["init_h.bias"]).astype(np.float32)
        prev_type, prev_vec, prev_other, used = 4, np.zeros(model.d, np.float32), 0, np.zeros(n, bool)
        for t in range(len(st)):
            if st[t] < 0:
                break
            state, comb, scores = model.step_logits(state, enc, prev_type, prev_vec, prev_other)
            logits = model.mask_actions(scores, ~silent & ~used, punct, need, prev_type == 1, t > 0, used, grp)
            expected = ref["logits"][b, t][: len(logits)]
            keep = expected > -5e3
            worst = max(worst, float(np.abs(logits[keep] - expected[keep]).max()))
            kind, idx = int(st[t]), int(si[t])
            if kind == 3:
                tags = model.tag_logits(comb, enc[idx], para[idx])
                expected_tag = ref["tag_logits"][b, t]
                worst = max(worst, float(np.abs(tags[expected_tag > -5e3] - expected_tag[expected_tag > -5e3]).max()))
                used[idx], prev_type, prev_vec, prev_other = True, 3, enc[idx], 0
            elif kind == 1:
                punct[idx] -= 1
                prev_type, prev_other = 1, 2 + idx
            elif kind == 2:
                need[idx] -= 1
                prev_type, prev_other = 2, 2 + model.n_punct + idx
            else:
                prev_type, prev_other = 0, 1
    assert worst < 1e-4


@needs_model
@pytest.mark.skipif(not DEV.exists(), reason="the grammarian dataset is private and not installed")
def test_features_of_the_runtime_encoder_equal_the_notebook_arrays():
    model = grammarian.load_model()
    ref = np.load(ARTIFACTS / "reference_logits.npz")
    rows = [json.loads(line) for _, line in zip(range(60), DEV.open(encoding="utf-8"), strict=False)]
    encoded = [model.encode(r["tree"]) for r in rows]
    for b in range(4):
        n = int(ref["in_node_mask"][b].sum())
        match = next(e for e in encoded if e and e["N"] == n and np.array_equal(e["cat"], ref["in_cat"][b, :n]))
        for key in ("flags", "lex", "chars", "verb", "silent", "para", "grp"):
            assert np.array_equal(match[key], ref[f"in_{key}"][b][:n] if key != "grp" else ref["in_grp"][b, :n, :n]), key
        assert np.array_equal(match["need"], ref["in_need"][b])
        assert np.array_equal(match["punct_cap"], ref["in_punct_cap"][b])


@needs_model
def test_the_network_words_a_plan_and_the_trace_says_so(on):
    with grammarian.capture() as records:
        surfaces = realize_thought(cat_thought(), productive_pattern_curriculum())
    assert surfaces[0].source == "grammarian"
    assert surfaces[0].text[0].isupper() and surfaces[0].text.endswith(".")
    assert "subject" in surfaces[0].role_order and "predicate" in surfaces[0].role_order
    assert [r.path for r in records] == ["network"]
    assert records[0].confidence >= records[0].threshold
    assert grammarian.STATS["network"] == 1
    # the deterministic surfaces stay behind it as alternatives for the reader check
    assert any(s.source != "grammarian" for s in surfaces[1:]) or len(surfaces) == 1


@needs_model
def test_low_confidence_falls_back_to_the_deterministic_realizer(on, monkeypatch):
    monkeypatch.setenv("DEBILGI_GRAMMARIAN_THRESHOLD", "1.1")
    grammarian._model_cache.clear()
    try:
        patterns = productive_pattern_curriculum()
        with grammarian.capture() as records:
            assert realize_thought(cat_thought(), patterns) == _realize_thought_deterministic(cat_thought(), patterns)
        assert records[0].path == "fallback" and records[0].reason == "low_confidence"
    finally:
        grammarian._model_cache.clear()


def test_missing_artifacts_fall_back_without_an_error(on, monkeypatch, tmp_path):
    monkeypatch.setenv("DEBILGI_GRAMMARIAN_DIR", str(tmp_path))
    patterns = productive_pattern_curriculum()
    with grammarian.capture() as records:
        assert realize_thought(cat_thought(), patterns) == _realize_thought_deterministic(cat_thought(), patterns)
    assert records[0].reason == "model_unavailable" and records[0].path == "fallback"


@needs_model
def test_one_pattern_is_an_order_the_caller_demands(on):
    pattern = productive_pattern_curriculum()[0]
    with grammarian.capture():
        surfaces = realize_thought(cat_thought(), (pattern,), limit=1)
    assert all(s.source != "grammarian" for s in surfaces)
    assert grammarian.STATS["fallback:order_fixed_by_caller"] == 1


@needs_model
def test_the_gender_of_i_is_the_bots_own_and_an_unknown_gender_is_incomplete(on, monkeypatch):
    past = ClauseThought(
        verb("видеть", tense="past"),
        (dep("subject", lex("я", "pronoun")), dep("direct_object", lex("кошка", gender="feminine", number="singular"))),
        confidence=0.9,
    )
    surfaces = realize_thought(past, productive_pattern_curriculum())
    assert surfaces[0].source == "grammarian" and " видел" in f" {surfaces[0].text.lower()} "
    monkeypatch.setenv("DEBILGI_SELF_GENDER", "feminine")
    assert " видела" in f" {realize_thought(past, productive_pattern_curriculum())[0].text.lower()} "
    # «ты»: the interlocutor's gender is not a fact of the plan: refusal, never a guess
    you = replace(past, dependencies=(dep("subject", lex("ты", "pronoun")), past.dependencies[1]))
    with grammarian.capture() as records:
        out = realize_thought(you, productive_pattern_curriculum())
    assert records[0].reason == "speaker_gender_unknown" and out[0].source != "grammarian"
    # ... unless the lexeme (a personal fact the plan carries) names it
    known = replace(you, dependencies=(dep("subject", lex("ты", "pronoun", gender="feminine")), past.dependencies[1]))
    assert " видела" in f" {realize_thought(known, productive_pattern_curriculum())[0].text.lower()} "


@needs_model
def test_present_tense_does_not_need_a_gender(on):
    with grammarian.capture() as records:
        realize_thought(cat_thought(), productive_pattern_curriculum())
    assert records[0].reason is None


@needs_model
def test_input_contract_case_comparison_nominal_part_and_closed_class(on):
    model = grammarian.load_model()
    context = grammarian.GrammarianContext(self_gender="masculine")
    thought = ClauseThought(
        verb("быть", tense="present"),
        (
            dep("subject", lex("кот", gender="masculine", number="singular")),
            dep("predicate_complement", lex("добрый", "adjective")),
            dep("comparison", lex("собака", gender="feminine", number="singular")),
        ),
        negated=True,
        predicate_class="copular",
        confidence=0.9,
    )
    tree, _ = grammarian.plan_to_tree(thought, context, model)
    by_role = {n["role"]: n for n in tree["nodes"]}
    assert "case_constraint" not in by_role["subject"]  # the plan fixed none: the network is not given one
    assert by_role["predicate_complement"]["case_constraint"] == "nominative"  # present «быть»
    assert by_role["comparison"]["marker_kind"] == "comparison" and by_role["comparison"]["marker"]
    assert tree["clause"]["func"] == [["particle_negation", "не"]] and tree["clause"]["end"] == "."
    past = replace(thought, predicate=verb("быть", tense="past"))
    tree, _ = grammarian.plan_to_tree(past, context, model)
    assert {n["role"]: n for n in tree["nodes"]}["predicate_complement"]["case_constraint"] == "instrumental"
    fixed = replace(
        thought,
        dependencies=(dep("subject", lex("кот", gender="masculine"), case="nominative"), *thought.dependencies[1:]),
    )
    tree, _ = grammarian.plan_to_tree(fixed, context, model)
    assert {n["role"]: n for n in tree["nodes"]}["subject"]["case_constraint"] == "nominative"


@needs_model
def test_a_copular_predicate_other_than_be_is_not_given_to_the_network(on):
    model = grammarian.load_model()
    thought = ClauseThought(
        verb("казаться", tense="present"),
        (dep("subject", lex("кот", gender="masculine")), dep("predicate_complement", lex("добрый", "adjective"))),
        predicate_class="copular",
    )
    with pytest.raises(grammarian.Refusal, match="copular_not_be"):
        grammarian.plan_to_tree(thought, grammarian.GrammarianContext(self_gender="masculine"), model)


@needs_model
def test_a_postmodifier_is_copied_as_the_plan_inflected_it(on):
    thought = ClauseThought(
        verb("видеть", tense="present"),
        (
            dep("subject", lex("я", "pronoun")),
            dep("direct_object", lex("кошка", gender="feminine", number="singular"), posts=(lex("Иванова", "noun", "Ивановой"),)),
        ),
        confidence=0.9,
    )
    surfaces = realize_thought(thought, productive_pattern_curriculum())
    assert surfaces[0].source == "grammarian" and "Ивановой" in surfaces[0].text


def test_join_tokens_glues_commas_and_keeps_dashes_apart():
    assert grammarian.join_tokens(["я", ",", "как", "кот", "-", "спал"]) == "я, как кот - спал"


@needs_model
def test_a_repeated_plan_is_answered_from_the_cache_and_still_journaled(on):
    patterns = productive_pattern_curriculum()
    with grammarian.capture() as records:
        first = realize_thought(cat_thought(), patterns)
        second = realize_thought(cat_thought(), patterns)
    assert first == second
    assert [r.details.get("cache") for r in records] == [None, True]
    assert records[1].milliseconds == 0.0 and records[1].text == records[0].text
    assert grammarian.STATS["cache_hit"] == 1


@needs_model
def test_the_preposition_o_takes_its_vowel_allomorph():
    model = grammarian.load_model()
    assert model.preposition_form("о", "Оле") == "об" and model.preposition_form("о", "Маше") == "о"
    assert model.preposition_form("в", "вторник") == "во"  # a rule read from the corpus


@needs_model
def test_a_question_whose_focus_is_a_wh_word_keeps_it_in_front(on):
    thought = ClauseThought(
        verb("читать", tense="present"),
        (dep("subject", lex("ты", "pronoun")), dep("direct_object", lex("что", "pronoun"))),
        clause_kind="question",
        focus_role="direct_object",
        confidence=0.9,
    )
    surfaces = realize_thought(thought, ())
    assert surfaces and surfaces[0].text.lower().startswith("что ")


@needs_model
def test_the_noun_of_an_embedded_be_is_given_as_a_nominal_part_in_the_instrumental(on):
    model = grammarian.load_model()
    thought = ClauseThought(
        verb("хотеть", tense="past", mood="conditional"),
        (
            dep("subject", lex("я", "pronoun")),
            dep("action_complement", lex("быть", "verb")),
            dep("direct_object", lex("животное", gender="neuter", number="singular"), at="action_complement"),
        ),
        predicate_class="modal",
        confidence=0.9,
    )
    tree, _ = grammarian.plan_to_tree(thought, grammarian.GrammarianContext(self_gender="masculine"), model)
    by_lemma = {n["lemma"]: n for n in tree["nodes"]}
    assert by_lemma["животное"]["role"] == "predicate_complement"
    assert by_lemma["животное"]["case_constraint"] == "instrumental"
    assert by_lemma["животное"]["head"] == by_lemma["быть"]["id"]
    text = realize_thought(thought, productive_pattern_curriculum())[0].text.lower()
    assert "животным" in text and "бы" in text


@needs_model
def test_a_gender_that_disagrees_with_the_speaker_the_plan_named_is_refused(on):
    model = grammarian.load_model()
    tree = {"nodes": [
        {"id": 0, "role": "predicate", "pos": "verb", "lemma": "быть", "lex": {}},
        {"id": 1, "role": "subject", "pos": "pronoun", "lemma": "я", "lex": {"gender": "masculine"}},
        {"id": 2, "role": "predicate_complement", "pos": "adjective", "lemma": "рад", "lex": {}},
    ]}
    wrong, right = model.tag_index["ADJS,neut,sing"], model.tag_index["ADJS,masc,sing"]
    assert grammarian._speaker_agreement_broken(tree, [("N", 2, wrong)], model)
    assert not grammarian._speaker_agreement_broken(tree, [("N", 2, right)], model)
