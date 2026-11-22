"""Independent annotated constructions and contrastive source invariants."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.communication import MessagePlan, reader_goal
from debilgi.language.dialogue_obligations import thought_from_data
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.generative_planner import statement_thought
from debilgi.language.planning import ReplyContext
from debilgi.language.semantic_graph import parse_proposition
from debilgi.language.structured_utterance import (
    PARSER_VERSION,
    _cached_structure,
    analyze_structured_utterance,
)

CASES = json.loads((Path(__file__).parents[1] / "fixtures/meaning_ir.json").read_text())
NOW = datetime(2026, 11, 14, 23, 30, tzinfo=UTC)


def event(text, *, actor=11, target=22, bot=22, at=NOW, mentioned=True, **kwargs):
    return normalize_message_created(
        message_id=100,
        guild_id=1,
        channel_id=2,
        author_id=actor,
        created_at=at,
        content=text,
        author_is_bot=False,
        bot_user_id=bot,
        bot_mentioned=mentioned,
        reply_to_author_id=target,
        **kwargs,
    )


def roles(structure, clause):
    return {
        role: [structure.tokens[i].lemma for i in indices]
        for role, indices in (
            ("subject", clause.subject_indices),
            ("predicate", () if clause.predicate_index is None else (clause.predicate_index,)),
            ("direct_object", clause.direct_object_indices),
            ("predicate_complement", clause.predicate_complement_indices),
            ("action_complement", clause.action_complement_indices),
            ("duration", clause.duration_indices),
            ("experiencer", clause.experiencer_indices),
        )
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["text"])
def test_annotated_constructions(case):
    structure = event(case["text"]).meaning
    assert structure.parser_version == PARSER_VERSION
    assert structure.supported == case["supported"]
    if not case["supported"]:
        assert case["reason"] in {r for c in structure.clauses for r in c.unsupported}
        assert parse_proposition(structure) is None
        assert statement_thought(structure) is None
        return
    assert len(structure.clauses) == len(case["clauses"])
    for clause, expected in zip(structure.clauses, case["clauses"], strict=True):
        actual = roles(structure, clause)
        for role, lemmas in expected["roles"].items():
            assert actual[role] == lemmas
        assert clause.clause_kind == expected["kind"]
        assert clause.negated == expected["negated"]
        assert clause.semantic_time.tense == expected["tense"]
        if clause.duration_indices:
            assert not set(clause.direct_object_indices) & set(clause.duration_indices)
    if "construction" in case:
        assert structure.clauses[0].construction == case["construction"]
    if "link" in case:
        assert [link.relation for link in structure.clause_links] == [case["link"]]
    if case.get("correction"):
        assert structure.corrections[0].target == "previous_fact"
        assert not structure.clauses[0].negated
        # Stage 1 must not quietly update a fact through a legacy projection.
        assert statement_thought(structure, speaker="user") is None


@pytest.mark.parametrize(
    "word,days",
    [("позавчера", -2), ("вчера", -1), ("сегодня", 0), ("завтра", 1), ("послезавтра", 2)],
)
@pytest.mark.parametrize("actor", [11, 33])
def test_source_relative_time_and_participants(word, days, actor):
    source = event(f"Я {word} читал газету.", actor=actor)
    structure = source.meaning
    time = structure.clauses[0].semantic_time
    assert time.anchor == NOW.isoformat()
    expected = NOW.replace(hour=0, minute=0) + timedelta(days=days)
    assert time.start == expected.isoformat()
    assert time.end == (expected + timedelta(days=1)).isoformat()
    thought = parse_proposition(structure)
    assert thought.semantic_time == time
    head = thought.dependencies[0].argument.head
    assert head.lemma == "ты"
    assert head.participant_ref == f"discord:user:{actor}"
    assert thought_from_data(json.loads(json.dumps(asdict(thought)))) == thought


@pytest.mark.parametrize(
    "text",
    ["Я не хочу читать.", "Я хочу не читать.", "Ты не можешь рисовать.", "Ты можешь не рисовать."],
)
def test_negation_and_modality_scopes_are_distinct(text):
    clause = event(text).meaning.clauses[0]
    negation = next(s for s in clause.scopes if s.operator == "negation")
    modal = next(s for s in clause.scopes if s.operator in {"мочь", "хотеть"})
    assert modal.target_indices == clause.action_complement_indices
    target = clause.predicate_index if clause.negated else clause.action_complement_indices[0]
    assert negation.target_indices == (target,)
    if not clause.negated:
        assert parse_proposition(event(text).meaning) is None


@pytest.mark.parametrize(
    "actor,target,bot,mentioned",
    [(11, 22, 22, True), (22, 11, 22, False), (11, 33, 22, False), (11, None, 22, False)],
)
def test_addressee_is_not_assumed_to_be_agent(actor, target, bot, mentioned):
    structure = event(
        "Я читаю, ты рисуешь.", actor=actor, target=target, bot=bot, mentioned=mentioned
    ).meaning
    participants = dict(structure.participants)
    refs = {t.lemma: t.participant for t in structure.tokens if t.lemma in {"я", "ты"}}
    assert refs["я"].ref == f"discord:user:{actor}"
    assert refs["ты"] == participants["addressee"]
    assert refs["ты"].resolved == (target is not None)
    if target != bot:
        assert refs["ты"] != participants["agent"]


def test_third_person_identity_and_unresolved_reference():
    source = event(
        "Ольга читает, она рисует.", mentioned_users=[{"id": 33, "display_name": "Ольга"}]
    )
    tokens = {t.lemma: t for t in source.meaning.tokens}
    assert tokens["ольга"].participant.ref == "discord:user:33"
    assert tokens["она"].participant.resolved is False
    assert any(s.reason == "unresolved_participant" for s in source.meaning.unresolved_slots)


def test_correction_keeps_reply_edge_without_mutating_prior_fact():
    structure = event("Нет, я читаю журнал.", reply_to_message_id=99).meaning
    assert structure.corrections[0].target_event_id == "discord:message:99"
    assert parse_proposition(structure) is None


def test_shared_parse_is_transported_and_projections_do_not_reparse():
    _cached_structure.cache_clear()
    source = event("Я читаю газету.")
    with patch(
        "debilgi.language.structured_utterance._analyze_structure",
        wraps=__import__(
            "debilgi.language.structured_utterance", fromlist=["_analyze_structure"]
        )._analyze_structure,
    ) as parser:
        structure = source.meaning
        context = ReplyContext(prompt=structure.text, input_meaning=structure)
        assert source.meaning is context.structure is structure
        assert parse_proposition(structure)
        assert statement_thought(structure, speaker="user")
        analyze_dialogue_state(structure.text, structure=structure)
        assert (
            analyze_structured_utterance(structure.text).clauses[0].subject_indices
            == structure.clauses[0].subject_indices
        )
        assert parser.call_count == 1
    changed = replace(context, prompt="Ты рисуешь карту.")
    assert changed.structure.text == changed.prompt
    assert changed.structure is not structure


def test_annotations_do_not_change_legacy_plan_policy_identity():
    raw = analyze_structured_utterance("Я читаю газету.")
    bound = event(raw.text).meaning
    old = MessagePlan(reader_goal(raw), parse_proposition(raw))
    new = MessagePlan(reader_goal(bound), parse_proposition(bound))
    assert old.key == new.key


def test_duration_is_semantic_data_and_missing_arguments_are_explicit():
    structure = event("Я гулял два часа.").meaning
    time = structure.clauses[0].semantic_time
    assert time.duration_unit == "час"
    assert time.duration_quantity == "два"
    assert any(
        s.role == "direct_object" and s.reason == "missing_argument"
        for s in event("Я люблю.").meaning.unresolved_slots
    )


def test_reflexive_binds_to_command_addressee_and_coordination_keeps_subject():
    structure = event("Расскажи о себе.").meaning
    assert (
        next(t for t in structure.tokens if t.lemma == "себя").participant.ref == "discord:user:22"
    )
    coordinated = event("Я читаю и рисую.").meaning
    assert coordinated.clauses[1].subject_ref == coordinated.clauses[0].subject_ref


def test_unknown_addressee_and_time_constraints_do_not_become_facts():
    structure = event("Ты сегодня завтра читаешь.", target=None, mentioned=False).meaning
    assert {s.reason for s in structure.unresolved_slots} >= {
        "unresolved_participant",
        "conflicting_deictics",
    }
    assert structure.clauses[0].semantic_time.start is None


def test_source_calendar_uses_timezone_across_utc_midnight():
    from datetime import timezone

    at = datetime(2026, 11, 15, 0, 30, tzinfo=timezone(timedelta(hours=5)))
    time = event("Я сегодня работал весь день.", at=at).meaning.clauses[0].semantic_time
    assert time.start == "2026-11-15T00:00:00+05:00"
    assert time.end == "2026-11-16T00:00:00+05:00"


def test_reply_context_retains_reflexive_topic_instead_of_greeting():
    from debilgi.language.generative_planner import objective_for

    structure = event("Привет, расскажи о себе.").meaning
    assert objective_for(structure).topics == ("себя",)


def test_raw_reloaded_event_reuses_same_versioned_meaning():
    source = event("Я вчера читал газету.")
    reloaded = replace(source)
    assert reloaded.meaning is source.meaning


@pytest.mark.parametrize("text", ["Мы читаем.", "Вы рисуете."])
def test_plural_participant_does_not_collapse_into_one_person(text):
    structure = event(text).meaning
    token = structure.tokens[0]
    assert token.participant.kind == "group"
    assert token.participant.resolved is False
    assert token.participant not in dict(structure.participants).values()
