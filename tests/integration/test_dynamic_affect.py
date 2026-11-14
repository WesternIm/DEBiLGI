from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.affect import AffectRepository, nearest_affect
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.social.models import PrimitiveVector
from debilgi.storage.database import Database

NOW = datetime(2026, 9, 20, 8, tzinfo=UTC)
POSITIVE = PrimitiveVector(1.0, 0.0, 1.0, 0.7, 0.0, 1.0, 0.8)
HOSTILE = PrimitiveVector(0.0, 1.0, 0.0, 0.2, 1.0, 0.0, 0.9)


def intent():
    return ActionIntent(
        "intent:test",
        1,
        ActionType.SHORT_REPLY,
        100,
        10,
        0.5,
        0.5,
        {},
        (),
        (),
        {},
        NOW,
        NOW + timedelta(minutes=1),
    )


@pytest.fixture
async def database(tmp_path):
    db = Database(tmp_path / "affect.sqlite3", Path(__file__).resolve().parents[2] / "migrations")
    await db.initialize()
    yield db
    await db.close()


def message(name, seconds=0, actor=100):
    return Event(
        name,
        1,
        10,
        actor,
        None,
        EventType.MESSAGE_CREATED,
        NOW + timedelta(seconds=seconds),
        {"content": name, "author_is_bot": False},
        "test",
    )


async def observed(db, repo, event, vector):
    await EventRepository(db).append(event)
    assert await repo.observe(event, vector)


async def test_external_mood_and_internal_emotion_have_different_time_scales(database):
    repo = AffectRepository(database)
    first = message("warm")
    await observed(database, repo, first, POSITIVE)
    state = await repo.state(1, 100, at=first.utc_timestamp)
    assert state.emotion.valence > state.mood.valence > 0
    assert state.labels[1] in {"радость", "интерес", "нейтральность"}

    hostile = message("hostile", 60)
    await observed(database, repo, hostile, HOSTILE)
    changed = await repo.state(1, 100, at=hostile.utc_timestamp)
    assert changed.emotion.valence < changed.mood.valence
    assert changed.emotion.arousal > 0

    later = await repo.state(1, 100, at=NOW + timedelta(hours=12))
    assert abs(later.emotion.valence) < abs(changed.emotion.valence) / 100
    assert abs(later.mood.valence) < abs(changed.mood.valence)


async def test_mood_is_shared_while_emotion_is_relationship_scoped(database):
    repo = AffectRepository(database)
    await observed(database, repo, message("warm"), POSITIVE)
    stranger = await repo.state(1, 200, at=NOW)
    familiar = await repo.state(1, 100, at=NOW)
    assert stranger.mood == familiar.mood
    assert stranger.emotion.valence == 0
    await observed(database, repo, message("attack", 10, actor=200), HOSTILE)
    assert (await repo.state(1, 100, at=NOW + timedelta(seconds=10))).emotion.valence > 0
    assert (await repo.state(1, 200, at=NOW + timedelta(seconds=10))).emotion.valence < 0


async def test_affect_observation_is_idempotent_and_contains_no_reply_text(database):
    repo = AffectRepository(database)
    current = message("same")
    await observed(database, repo, current, POSITIVE)
    assert not await repo.observe(current, HOSTILE)
    row = await database.fetchone("SELECT * FROM affect_observations")
    assert row["event_id"] == "same"
    assert "same" not in row["stimulus_json"]
    assert (await repo.state(1, 100, at=NOW)).emotion_observations == 1


async def test_affect_changes_expression_without_changing_genome(database):
    repo = AffectRepository(database)
    await observed(database, repo, message("warm"), POSITIVE)
    state = await repo.state(1, 100, at=NOW)
    base = intent()
    engine = PersonalityExpressionEngine(DEBILGI_GENOME)
    neutral = engine.express(base)
    affected = engine.express(
        replace(base, decision_context={**base.decision_context, **dict(state.features)})
    )
    assert affected.warmth > neutral.warmth
    assert affected.playfulness > neutral.playfulness
    assert DEBILGI_GENOME.signature == affected.genome_signature == neutral.genome_signature


def test_emotion_names_are_observational_points_only():
    from debilgi.cognition.affect import AffectVector

    assert nearest_affect(AffectVector()) == "нейтральность"
