from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.pipeline import EventPipeline
from debilgi.events.repository import EventRepository
from debilgi.language.dialogue_learning import (
    ObservedDialogueForm,
    realize_observed_form,
    response_forms,
)
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.storage.database import Database

NOW = datetime(2026, 9, 20, tzinfo=UTC)


@pytest.mark.parametrize("missing", ["turn:self_report", "valence:negative", "tone:coarse"])
def test_transformation_conditions_cannot_be_outvoted_by_other_features(missing):
    from debilgi.language.dialogue_learning import activate_transformation

    required = {"turn:self_report", "valence:negative", "tone:coarse", "person:first"}
    rule = {"input": sorted(required), "rule_id": "test-rule", "weight": 0.95}
    assert activate_transformation(rule, required) > 0
    assert activate_transformation(rule, required - {missing}) == 0


def test_expressive_derivation_is_bounded_and_not_factual_identity():
    from debilgi.language.lexical_matching import expressive_family, predicate_family

    assert expressive_family("скучно", "скука")
    assert expressive_family("лошара", "лох")
    assert not expressive_family("", "скука")
    assert predicate_family("болеть", "заболеть")
    assert not predicate_family("болеть", "отдыхать")


@pytest.fixture
async def learning(tmp_path):
    database = Database(tmp_path / "exchanges.sqlite3", Path("migrations"))
    await database.initialize()
    repository = EventRepository(database)
    assimilation = SemanticAssimilationService(database, SelfDatasetRepository(database))
    pipeline = EventPipeline(
        repository,
        [
            DiscourseMemoryRepository(database).observe,
            assimilation.dialogue_learning.observe,
        ],
    )
    await pipeline.start()
    yield database, assimilation, pipeline
    await pipeline.stop()
    await database.close()


def turn(number, text, *, actor=100, channel=10, seconds=None, parent=None):
    return normalize_message_created(
        message_id=number,
        guild_id=1,
        channel_id=channel,
        author_id=actor,
        created_at=NOW + timedelta(seconds=number if seconds is None else seconds),
        content=text,
        reply_to_message_id=parent,
        author_is_bot=False,
    )


async def test_shared_observer_learns_acts_slots_and_topics_without_bot_replies(learning):
    database, assimilation, pipeline = learning
    prompt = turn(1, "Ты читаешь книги?")
    answer = turn(2, "Я читаю книги.", actor=200, parent=1)
    for event in (prompt, answer):
        await pipeline.publish(event)
    await pipeline.join()

    exchange = await database.fetchone("SELECT * FROM language_observed_exchanges")
    assert exchange["prompt_event_id"] == prompt.event_id
    assert exchange["answer_event_id"] == answer.event_id
    assert exchange["prompt_act"] == "information_question"
    assert '"binding": "prompt"' in exchange["response_forms_json"]
    assert '"features"' in exchange["response_forms_json"]
    assert "книга" not in exchange["response_forms_json"]
    assert "direct_object" in exchange["topic_links_json"]
    assert await database.fetchone("SELECT 1 FROM language_productive_constructions")
    assert await database.fetchone("SELECT 1 FROM language_semantic_lessons WHERE status='active'")
    guidance = await assimilation.guidance_for_prompt(1, "Ты изучаешь звёзды?")
    assert "statement" in guidance.expected_response_acts
    # Demonstrations are not rewards or first-person goals of this agent.
    assert not await database.fetchone("SELECT 1 FROM language_meaning_intentions")

    await assimilation.dialogue_learning.observe(answer)
    count = await database.fetchone(
        "SELECT evidence_count FROM language_dialogue_act_patterns WHERE source='local_exchange'"
    )
    assert count["evidence_count"] == 1


async def test_adjacent_speakers_pair_but_same_speaker_channel_and_stale_turns_do_not(learning):
    database, _, pipeline = learning
    events = [
        turn(1, "Привет!"),
        turn(2, "Я читаю книгу."),
        turn(3, "Ты читаешь книгу?", actor=200),
        turn(4, "Привет!", actor=300, channel=20),
        turn(5, "Я рисую.", actor=400, seconds=900),
        turn(6, "Привет!", actor=500, parent=1, seconds=901),
    ]
    for event in events:
        await pipeline.publish(event)
    await pipeline.join()
    rows = await database.fetchall("SELECT * FROM language_observed_exchanges")
    assert len(rows) == 1
    assert rows[0]["answer_event_id"] == events[2].event_id


def test_morphological_slots_transfer_content_instead_of_corpus_persona():
    forms, _ = response_forms("Ты читаешь книги?", "Ты читаешь книги?")
    form = ObservedDialogueForm(
        tuple(forms[0]["slots"]),
        "question",
        "information_question",
        "information_question",
        ("demo",),
        2,
        1,
    )
    text = realize_observed_form(form, "Ты изучаешь звёзды?")
    assert text == "Ты изучаешь звёзды?"
    forms, _ = response_forms("Чем занимаешься?", "Я выращиваю собак.")
    form = ObservedDialogueForm(
        tuple(forms[0]["slots"]),
        "statement",
        "information_question",
        "statement",
        ("demo",),
        100,
        1,
    )
    assert realize_observed_form(form, "Чем занимаешься?") is None


async def test_social_act_is_learned_and_cold_start_has_no_surface(learning):
    _, assimilation, pipeline = learning
    query = turn(10, "ку бро", actor=300)
    assert not await assimilation.dialogue_learning.forms_for(query, "ку бро")
    for event in (turn(1, "Привет!"), turn(2, "Здравствуй!", actor=200)):
        await pipeline.publish(event)
    await pipeline.join()
    forms = await assimilation.dialogue_learning.forms_for(query, "ку бро")
    assert forms
    assert any(f.meaning == "return_greeting" and f.component == "core" for f in forms)
    assert all(realize_observed_form(form, "ку бро") is None for form in forms)
    assert await assimilation._database.fetchone(
        "SELECT 1 FROM language_productive_constructions WHERE predicate_class='dialogue_act'"
    )


def test_leading_reaction_is_separated_from_biography_with_original_punctuation():
    forms, _ = response_forms("я читаю книгу", "Здорово! Я работаю врачом.")
    reactions = [f for f in forms if f.get("reactive_act")]
    assert [f["component"] for f in reactions] == ["core", "suffix"]
    assert [s["surface"] for s in reactions[0]["slots"]] == ["здорово"]
    assert [s["surface"] for s in reactions[1]["slots"]] == ["!"]
    # A later appraisal must not be mined out of an unrelated personal story.
    forms, _ = response_forms("я читаю книгу", "Я работаю врачом. Здорово!")
    assert not any(f.get("reactive_act") for f in forms)


async def test_parts_backfill_restart_and_observation_are_idempotent(learning):
    import json

    database, assimilation, pipeline = learning
    for event in (turn(1, "красавчик"), turn(2, "ну а то))", actor=200)):
        await pipeline.publish(event)
    await pipeline.join()
    answer = turn(2, "ну а то))", actor=200)
    query = turn(50, "ты крутой", actor=300)
    # Simulate a pre-4.6 exchange projection while preserving its raw events.
    await database.execute("UPDATE language_observed_exchanges SET response_forms_json='[]'")
    first = await assimilation.dialogue_learning.forms_for(query, "ты крутой")
    before = [dict(r) for r in await database.fetchall(
        "SELECT * FROM language_productive_constructions ORDER BY construction_id")]
    await assimilation.dialogue_learning.observe(answer)
    second = await assimilation.dialogue_learning.forms_for(query, "ты крутой")
    from debilgi.language.dialogue_learning import DialogueStructureLearning

    restarted = DialogueStructureLearning(database, assimilation)
    assert first == second == await restarted.forms_for(query, "ты крутой")
    after = [dict(r) for r in await database.fetchall(
        "SELECT * FROM language_productive_constructions ORDER BY construction_id")]
    assert before == after
    row = await database.fetchone("SELECT response_forms_json FROM language_observed_exchanges")
    assert all(f.get("projection_version") == 10 for f in json.loads(row[0]))


async def test_bot_surfaces_and_replies_to_them_cannot_become_exchange_parts(learning):
    from dataclasses import replace

    database, _, pipeline = learning
    for observed in (turn(1, "ты тупой"), turn(2, "сам такой долбоеб))", actor=200)):
        await pipeline.publish(observed)
    await pipeline.join()
    before = [dict(r) for r in await database.fetchall(
        "SELECT * FROM language_productive_constructions WHERE predicate_class='dialogue_act' "
        "ORDER BY construction_id")]
    bot = turn(3, "ну сам такой долбоеб))", actor=999)
    bot = replace(bot, payload={**bot.payload, "author_is_bot": True})
    for observed in (bot, turn(4, "ну а то))", actor=200, parent=3)):
        await pipeline.publish(observed)
    await pipeline.join()
    after = [dict(r) for r in await database.fetchall(
        "SELECT * FROM language_productive_constructions WHERE predicate_class='dialogue_act' "
        "ORDER BY construction_id")]
    assert before and before == after
    rows = await database.fetchall("SELECT answer_event_id FROM language_observed_exchanges")
    assert [r[0] for r in rows] == ["discord:message:2"]
