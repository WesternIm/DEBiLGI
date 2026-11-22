import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository, behavior_key
from debilgi.language.communication import MessagePlan, ReaderGoal
from debilgi.language.generative_planner import DerivedPlan, DialogueObjective, plan_from_evidence
from debilgi.language.planning import ReplyContext
from debilgi.language.productive_nlg import ClauseThought, ConceptArgument, ConceptLexeme
from debilgi.language.productive_nlg import SemanticDependency as Dependency
from debilgi.storage.database import Database

NOW = datetime(2026, 9, 18, 12, tzinfo=UTC)
OBJECTIVE = DialogueObjective("respond", None, None, (), ())
GOAL = ReaderGoal("respond", None, (), None, False)


@pytest.fixture
async def database(tmp_path):
    db = Database(tmp_path / "intent.sqlite3", Path(__file__).resolve().parents[2] / "migrations")
    await db.initialize()
    yield db
    await db.close()


def event(name, seconds=0, **changes):
    value = Event(
        name,
        1,
        10,
        100,
        None,
        EventType.MESSAGE_CREATED,
        NOW + timedelta(seconds=seconds),
        {"content": "Маша читает."},
        "test",
    )
    return replace(value, **changes)


def proposal(predicate="читать", *, operation="continue_from_memory", question=False):
    claim = ClauseThought(
        ConceptLexeme(predicate, predicate, "verb", (("tense", "present"),)),
        (Dependency("subject", ConceptArgument(ConceptLexeme("маша", "Маша", "noun"))),),
        clause_kind="question" if question else "statement",
        confidence=0.9,
    )
    return DerivedPlan(MessagePlan(GOAL, claim), operation, ())


async def choose(db, repo, prompt, options, **kwargs):
    await EventRepository(db).append(prompt)
    return await repo.choose(prompt, options, OBJECTIVE, parent_intent_id="parent", **kwargs)


async def teach(db, repo, option, number, *, positive, state_features=()):
    prompt = event(f"prompt:{number}", number * 5)
    selected = await choose(db, repo, prompt, (option,), state_features=state_features)
    assert selected is not None
    response_id = 1000 + number
    assert await repo.record_sent(
        prompt,
        intention_id=selected.intention_id,
        response_message_id=response_id,
        created_at=prompt.utc_timestamp + timedelta(seconds=1),
    )
    feedback = event(
        f"feedback:{number}",
        number * 5 + 2,
        event_type=EventType.MESSAGE_REPLIED,
        payload={
            "reply_to_message_id": response_id,
            "content": "Отлично." if positive else "Неправильно.",
        },
    )
    await EventRepository(db).append(feedback)
    assert await repo.observe(feedback)
    return selected


async def test_prepared_meaning_is_not_an_executed_action_or_a_desire(database):
    repo = MeaningIntentRepository(database, exploration_rate=0)
    prompt = event("unsent")
    selected = await choose(database, repo, prompt, (proposal(),))
    assert selected is not None and selected.expected_reward == 0 and selected.exploring
    row = await database.fetchone("SELECT * FROM language_meaning_intentions")
    assert row["status"] == "prepared" and row["response_message_id"] is None
    assert row["action_json"] is None and row["reward"] is None
    assert "text" not in json.loads(row["plan_json"])
    assert json.loads(row["plan_json"])["claim"]["predicate"]["lemma"] == "читать"
    assert not await repo.learned_goals(event("next", 1))
    feedback = event(
        "unsent-feedback", 2, payload={"content": "Верно.", "reply_to_message_id": 1000}
    )
    await EventRepository(database).append(feedback)
    assert not await repo.observe(feedback)


async def test_same_input_and_database_choose_different_meanings_after_learning(database):
    repo = MeaningIntentRepository(database, exploration_rate=0)
    inform = proposal()
    ask = proposal("читать", operation="request_missing_argument", question=True)
    await teach(database, repo, inform, 1, positive=True)
    await teach(database, repo, ask, 2, positive=False)
    prompt = event("choice", 20)
    selected = await choose(database, repo, prompt, (ask, inform))
    assert selected is not None and selected.proposal == inform and not selected.exploring
    assert selected.expected_reward == 1 and selected.expected_outcome == "explicit_confirmation"
    # Change only action experience, keeping the input and available meanings.
    for number in range(5, 10):
        await teach(database, repo, inform, number, positive=False)
        await teach(database, repo, ask, number + 10, positive=True)
    selected = await choose(
        database, repo, replace(prompt, timestamp=NOW + timedelta(seconds=110)), (ask, inform)
    )
    assert selected is not None and selected.proposal == ask


async def test_values_transfer_operations_and_ignore_future_or_other_people(database):
    repo = MeaningIntentRepository(database, exploration_rate=0)
    await teach(database, repo, proposal(), 1, positive=True)
    unknown_claim = proposal("писать")
    selected = await choose(database, repo, event("transfer", 20), (unknown_claim,))
    assert selected.expected_reward == 1
    assert selected.alternatives[0]["value_origin"] == "operation_experience"
    for prompt in (
        event("past", 0),
        event("other-person", 20, actor_id=101),
        event("other-channel", 20, channel_id=11),
        event("other-guild", 20, guild_id=2),
    ):
        selected = await choose(database, repo, prompt, (unknown_claim,))
        assert selected.expected_reward == 0 and selected.sample_count == 0
        assert not await repo.learned_goals(prompt)


@pytest.mark.parametrize(
    "changes",
    [
        {"actor_id": 101},
        {"channel_id": 11},
        {"guild_id": 2},
        {"payload": {"content": "Верно."}},
        {"payload": {"reply_to_message_id": 1001, "content": "Расскажи ещё."}},
        {"payload": {"reply_to_message_id": 1001, "content": "Верно.", "author_is_bot": True}},
        {"timestamp": NOW},
        {"timestamp": NOW + timedelta(hours=1)},
        {"event_type": EventType.REACTION_REMOVED, "payload": {"message_id": 1001, "emoji": "👎"}},
    ],
)
async def test_unattributed_attention_and_silence_never_train_values(database, changes):
    repo = MeaningIntentRepository(database)
    prompt = event("actual-prompt", 5)
    selected = await choose(database, repo, prompt, (proposal(),))
    await repo.record_sent(
        prompt,
        intention_id=selected.intention_id,
        response_message_id=1001,
        created_at=NOW + timedelta(seconds=6),
    )
    feedback = replace(
        event("invalid-feedback", 7, payload={"reply_to_message_id": 1001, "content": "Верно."}),
        **changes,
    )
    await EventRepository(database).append(feedback)
    assert not await repo.observe(feedback)
    row = await database.fetchone("SELECT * FROM language_meaning_intentions")
    assert row["status"] == "sent" and row["reward"] is None


async def test_targeted_feedback_is_idempotent_and_goals_require_valued_execution(database):
    repo = MeaningIntentRepository(database)
    await teach(database, repo, proposal(), 1, positive=True)
    feedback = await EventRepository(database).get("feedback:1")
    assert not await repo.observe(feedback)
    goals = await repo.learned_goals(event("next", 20))
    # A single valued answer is useful for choice learning, not a preference.
    assert not goals
    await teach(database, repo, proposal(), 2, positive=False)
    assert not await repo.learned_goals(event("balanced", 20))


@pytest.mark.parametrize("answer", ["Да.", "Нет."])
async def test_answering_a_question_changes_knowledge_without_training_approval(database, answer):
    repo = MeaningIntentRepository(database)
    prompt = event("ask-prompt")
    selected = await choose(database, repo, prompt, (proposal(question=True),))
    await repo.record_sent(
        prompt, intention_id=selected.intention_id, response_message_id=1001,
        created_at=NOW + timedelta(seconds=1), text="Маша читает?",
    )
    feedback = event("answer", 2, payload={"content": answer, "reply_to_message_id": 1001})
    await EventRepository(database).append(feedback)
    assert not await repo.observe(feedback)
    consequence = await database.fetchone("SELECT * FROM language_meaning_consequences")
    assert consequence["outcome_type"] == "requested_information_received"
    assert consequence["external_value"] is None
    effect = json.loads(consequence["semantic_effect_json"])
    assert effect["predicate"]["lemma"] == "читать"
    assert effect["negated"] == (answer == "Нет.")
    assert not await repo.learned_goals(event("next", 3))
    selected = await choose(database, repo, event("predict", 4), (proposal(question=True),))
    assert selected.expected_reward == 0
    assert selected.alternatives[0]["predicted_consequences"] == {
        "requested_information_received": 1.0,
    }


async def test_negative_feedback_does_not_invent_an_opposite_desire(database):
    repo = MeaningIntentRepository(database)
    await teach(database, repo, proposal(), 1, positive=False)
    assert not await repo.learned_goals(event("negative-only", 10))


@pytest.mark.parametrize("content", ["Так да или нет?", "Да или нет.", "Верно ли это?"])
async def test_affirmative_words_inside_follow_up_questions_are_not_rewards(database, content):
    repo = MeaningIntentRepository(database)
    prompt = event("statement-prompt")
    selected = await choose(database, repo, prompt, (proposal(),))
    await repo.record_sent(
        prompt, intention_id=selected.intention_id, response_message_id=1001,
        created_at=NOW + timedelta(seconds=1), text="Маша читает.",
    )
    follow_up = event("follow-up", 2, payload={"content": content, "reply_to_message_id": 1001})
    await EventRepository(database).append(follow_up)
    assert not await repo.observe(follow_up)
    row = await database.fetchone("SELECT * FROM language_meaning_intentions")
    assert row["status"] == "sent" and row["reward"] is None
    consequence = await database.fetchone("SELECT * FROM language_meaning_consequences")
    assert consequence["outcome_type"] == "follow_up_question"
    assert consequence["external_value"] is None
    assert not await repo.learned_goals(event("next", 3))


async def test_learned_choice_uses_internal_state_without_fixed_interest_weights(database):
    repo = MeaningIntentRepository(database, exploration_rate=0)
    low = (("state_uncertainty", 0.0),)
    high = (("state_uncertainty", 1.0),)
    inform = proposal()
    ask = proposal(operation="request_missing_argument", question=True)
    await teach(database, repo, inform, 1, positive=True, state_features=low)
    await teach(database, repo, inform, 2, positive=False, state_features=high)
    await teach(database, repo, ask, 3, positive=False, state_features=low)
    await teach(database, repo, ask, 4, positive=True, state_features=high)
    prompt = event("state-choice", 50)
    selected = await choose(database, repo, prompt, (inform, ask), state_features=low)
    assert selected.proposal == inform
    selected = await choose(database, repo, prompt, (inform, ask), state_features=high)
    assert selected.proposal == ask


def test_choice_identity_uses_semantics_and_exploration_preserves_alternatives():
    original = proposal()
    variant = replace(
        original,
        plan=replace(
            original.plan,
            claim=replace(
                original.plan.claim,
                predicate=replace(original.plan.claim.predicate, surface="читает"),
                confidence=0.1,
                evidence=("different-source",),
            ),
        ),
    )
    assert behavior_key(original) == behavior_key(variant)
    result = plan_from_evidence(
        ReplyContext(prompt="Зарядка работает.", knowledge_facts=("Зарядка заряжает телефон.",)),
        explore_meanings=True,
    )
    assert {option.operation for option in result.proposals} >= {
        "continue_from_memory",
        "query_unobserved_relation",
    }


async def test_preferences_survive_database_restart(database):
    repo = MeaningIntentRepository(database, exploration_rate=0)
    await teach(database, repo, proposal(), 1, positive=True)
    path, migrations = database.path, database.migrations_path
    await database.close()
    await database.initialize()
    assert database.path == path and database.migrations_path == migrations
    restored = MeaningIntentRepository(database, exploration_rate=0)
    selected = await choose(database, restored, event("restored", 20), (proposal(),))
    assert selected.expected_reward == 1 and selected.sample_count == 1


async def test_night_hypothesis_said_by_day_is_remembered_as_the_delivered_turn(database):
    from dataclasses import asdict

    repo = MeaningIntentRepository(database)
    prompt = event("prompt:sleep", 1)
    selected = await choose(database, repo, prompt, (proposal(),))
    hypothesis = proposal("быть")
    plan = asdict(replace(hypothesis.plan, hypothesis=(("kind", "experience_opinion"),
                                                       ("operator", "наверное"))))
    adopted = await repo.adopt_hypothesis(selected, plan, "sleep:h1", "plan-key-h1")
    assert adopted.choice["sleep_hypothesis_id"] == "sleep:h1"
    assert await repo.record_sent(prompt, intention_id=selected.intention_id,
                                  response_message_id=777,
                                  created_at=prompt.utc_timestamp + timedelta(seconds=1),
                                  text="Наверное, я рад.")
    row = await database.fetchone(
        "SELECT operation,plan_key,response_text,plan_json FROM language_meaning_intentions "
        "WHERE intention_id=?", (selected.intention_id,))
    assert (row["operation"], row["plan_key"], row["response_text"]) == (
        "sleep_hypothesis", "plan-key-h1", "Наверное, я рад.")
    assert json.loads(row["plan_json"])["hypothesis"][1] == ["operator", "наверное"]
    # Said by day, it is neither a fact nor a pending "response:*" hypothesis awaiting a yes.
    for table in ("meaning_claims", "meaning_evidence"):
        assert (await database.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0
    assert (await database.fetchone(
        "SELECT COUNT(*) FROM continuous_cognition_hypotheses WHERE signature LIKE 'response:%'"
    ))[0] == 0
