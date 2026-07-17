from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository, ProcessingStatus
from debilgi.storage.database import Database
from debilgi.storage.migrations import MigrationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


def event(event_id: str, guild_id: int, event_type: EventType) -> Event:
    return Event(
        event_id=event_id,
        guild_id=guild_id,
        channel_id=20,
        actor_id=30,
        target_id=None,
        event_type=event_type,
        timestamp=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        payload={"text": "кириллица 😈"},
        source="test",
    )


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "events.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_migration_and_round_trip(database: Database) -> None:
    repository = EventRepository(database)
    original = event("one", 10, EventType.MESSAGE_CREATED)

    inserted = await repository.append(original)
    duplicate_inserted = await repository.append(original)
    restored = (await repository.recent(10))[0]

    assert await database.schema_version() == 15
    assert inserted is True
    assert duplicate_inserted is False
    assert restored == original


async def test_queries_are_isolated_by_guild(database: Database) -> None:
    repository = EventRepository(database)
    await repository.append(event("guild-1", 1, EventType.MESSAGE_CREATED))
    await repository.append(event("guild-2", 2, EventType.REACTION_ADDED))

    assert await repository.count(1) == 1
    assert await repository.count(2) == 1
    assert [item.event_id for item in await repository.recent(1)] == ["guild-1"]
    assert await repository.counts_by_type(1) == {"MESSAGE_CREATED": 1}


async def test_modified_applied_migration_is_rejected(tmp_path: Path) -> None:
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    migration = migration_dir / "001_initial.sql"
    migration.write_text("CREATE TABLE sample (id INTEGER PRIMARY KEY);", encoding="utf-8")
    db_path = tmp_path / "checksum.sqlite3"

    first = Database(db_path, migration_dir)
    await first.initialize()
    await first.close()

    migration.write_text("CREATE TABLE changed (id INTEGER PRIMARY KEY);", encoding="utf-8")
    second = Database(db_path, migration_dir)
    with pytest.raises(MigrationError, match="modified"):
        await second.initialize()
    await second.close()


async def test_existing_v1_database_is_backfilled_by_processing_migration(
    tmp_path: Path,
) -> None:
    v1_migrations = tmp_path / "v1_migrations"
    v1_migrations.mkdir()
    v1_source = MIGRATIONS / "001_initial_events.sql"
    (v1_migrations / v1_source.name).write_text(
        v1_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    db_path = tmp_path / "upgrade.sqlite3"

    old_database = Database(db_path, v1_migrations)
    await old_database.initialize()
    await EventRepository(old_database).append(
        event("before-upgrade", 1, EventType.MESSAGE_CREATED)
    )
    await old_database.close()

    upgraded_database = Database(db_path, MIGRATIONS)
    await upgraded_database.initialize()
    upgraded_repository = EventRepository(upgraded_database)
    try:
        assert await upgraded_database.schema_version() == 15
        assert (
            await upgraded_repository.processing_status("before-upgrade")
            is ProcessingStatus.PENDING
        )
    finally:
        await upgraded_database.close()


async def test_existing_v4_database_upgrades_to_cognitive_core_without_data_loss(
    tmp_path: Path,
) -> None:
    v4_migrations = tmp_path / "v4_migrations"
    v4_migrations.mkdir()
    for source in sorted(MIGRATIONS.glob("00[1-4]_*.sql")):
        (v4_migrations / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    db_path = tmp_path / "upgrade-v4.sqlite3"
    old_database = Database(db_path, v4_migrations)
    await old_database.initialize()
    await EventRepository(old_database).append(
        event("preserved-v4-event", 1, EventType.MESSAGE_CREATED)
    )
    await old_database.close()

    upgraded = Database(db_path, MIGRATIONS)
    await upgraded.initialize()
    try:
        assert await upgraded.schema_version() == 15
        assert await EventRepository(upgraded).get("preserved-v4-event") is not None
        tables = await upgraded.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        table_names = {str(row["name"]) for row in tables}
        assert {
            "cognition_episodes",
            "cognition_semantic_memories",
            "cognition_entity_models",
            "cognition_internal_states",
            "cognition_action_intents",
            "language_utterance_experiences",
            "language_experience_transitions",
            "cognition_self_models",
            "cognition_autonomy_goals",
            "cognition_initiative_states",
            "knowledge_concepts",
            "knowledge_aliases",
            "knowledge_queries",
            "knowledge_relations",
            "knowledge_acquisition_states",
            "knowledge_event_concepts",
            "knowledge_context_associations",
        } <= table_names
    finally:
        await upgraded.close()


async def test_existing_experience_is_backfilled_into_self_model(tmp_path: Path) -> None:
    v7_migrations = tmp_path / "v7_migrations"
    v7_migrations.mkdir()
    for source in sorted(MIGRATIONS.glob("00[1-7]_*.sql")):
        (v7_migrations / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    db_path = tmp_path / "upgrade-v7.sqlite3"
    old_database = Database(db_path, v7_migrations)
    await old_database.initialize()
    for number, reward in enumerate((0.9, -0.2), start=1):
        await old_database.execute(
            """
            INSERT INTO language_utterance_experiences (
                experience_id, guild_id, channel_id, user_id, prompt_event_id,
                intent_id, response_message_id, response_tokens_json,
                genome_signature, candidate_count, status, outcome_type,
                reward, created_at, resolved_at, initiative
            ) VALUES (?, 1, 10, 100, ?, 'intent:test', ?, '["word"]',
                'genome', 32, 'resolved', 'reply', ?, ?, ?, 0)
            """,
            (
                f"experience:{number}",
                f"message:{number}",
                1000 + number,
                reward,
                datetime(2026, 7, 17, 12, number, tzinfo=UTC).isoformat(),
                datetime(2026, 7, 17, 12, number, 1, tzinfo=UTC).isoformat(),
            ),
        )
    await old_database.close()

    upgraded = Database(db_path, MIGRATIONS)
    await upgraded.initialize()
    try:
        row = await upgraded.fetchone("SELECT * FROM cognition_self_models WHERE guild_id = 1")
        assert row is not None
        assert int(row["utterances_observed"]) == 2
        assert int(row["positive_outcomes"]) == 1
        assert int(row["negative_outcomes"]) == 1
        assert float(row["average_reward"]) == pytest.approx(0.35)
        assert float(row["efficacy"]) == pytest.approx(0.675)
    finally:
        await upgraded.close()
