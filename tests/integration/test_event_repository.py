import hashlib
import sqlite3
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

    assert await database.schema_version() == 40
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
        assert await upgraded_database.schema_version() == 40
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
        assert await upgraded.schema_version() == 40
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
            "learning_instruction_examples",
            "learning_dataset_feedback",
            "learning_graph_nodes",
            "learning_graph_edges",
            "learning_example_sources",
            "language_open_questions",
            "language_short_speech_acts",
            "background_cycle_runs",
            "background_graph_conflicts",
            "background_identity_checks",
            "background_initiative_queue",
        } <= table_names
    finally:
        await upgraded.close()


async def test_service_word_cleanup_removes_old_graph_noise(tmp_path: Path) -> None:
    db_path = tmp_path / "service-word-cleanup.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE learning_graph_nodes (
            node_id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, node_type TEXT NOT NULL,
            node_key TEXT NOT NULL, label TEXT NOT NULL
        );
        CREATE TABLE learning_graph_edges (
            edge_id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, relation TEXT NOT NULL,
            source_node_id TEXT NOT NULL REFERENCES learning_graph_nodes(node_id)
                ON DELETE CASCADE,
            target_node_id TEXT NOT NULL REFERENCES learning_graph_nodes(node_id)
                ON DELETE CASCADE
        );
        CREATE TABLE language_semantic_frames (
            source_id TEXT NOT NULL, frame_index INTEGER NOT NULL, actor TEXT NOT NULL,
            target TEXT, PRIMARY KEY (source_id, frame_index)
        );
        CREATE TABLE knowledge_relations (
            relation_id TEXT PRIMARY KEY, normalized_object_title TEXT NOT NULL
        );
            CREATE TABLE learning_instruction_examples (
                example_id TEXT PRIMARY KEY, source_event_id TEXT NOT NULL DEFAULT '',
                answer_text TEXT NOT NULL,
                status TEXT NOT NULL, quality_score REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
        CREATE TABLE memory_long_term_capsules (
            capsule_id TEXT PRIMARY KEY, summary TEXT NOT NULL
        );
            CREATE TABLE language_conversation_sessions (
                conversation_id TEXT PRIMARY KEY, current_summary TEXT NOT NULL
            );
            CREATE TABLE archive_messages (
                message_id INTEGER PRIMARY KEY
            );
            INSERT INTO learning_graph_nodes VALUES
            ('node:tipa', 1, 'concept', 'типо', 'типо'),
            ('node:person', 1, 'concept', 'человек', 'человек');
        INSERT INTO learning_graph_edges VALUES (
            'edge:noise', 1, 'associated_with', 'node:tipa', 'node:person'
        );
        INSERT INTO language_semantic_frames VALUES ('old:frame', 0, 'кто', 'человек');
        """
    )
    for source in sorted(MIGRATIONS.glob("*.sql")):
        version_text, name_with_suffix = source.name.split("_", 1)
        version = int(version_text)
        if version >= 28:
            continue
        sql = source.read_text(encoding="utf-8")
        connection.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            (
                version,
                name_with_suffix.removesuffix(".sql"),
                hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            ),
        )
    connection.commit()
    connection.close()

    upgraded = Database(db_path, MIGRATIONS)
    await upgraded.initialize()
    try:
        assert await upgraded.schema_version() == 40
        assert await upgraded.fetchone(
            "SELECT 1 FROM learning_graph_nodes WHERE node_id = 'node:tipa'"
        ) is None
        assert await upgraded.fetchone(
            "SELECT 1 FROM learning_graph_edges WHERE edge_id = 'edge:noise'"
        ) is None
        assert await upgraded.fetchone(
            "SELECT 1 FROM language_semantic_frames WHERE source_id = 'old:frame'"
        ) is None
        person = await upgraded.fetchone(
            "SELECT 1 FROM learning_graph_nodes WHERE node_id = 'node:person'"
        )
        audit = await upgraded.fetchone(
            "SELECT * FROM language_relation_filter_cleanups WHERE cleanup_id = 'migration:028'"
        )
        assert person is not None
        assert audit is not None
        assert int(audit["removed_graph_edges"]) == 1
        assert int(audit["removed_graph_nodes"]) == 1
        assert int(audit["removed_semantic_frames"]) == 1
    finally:
        await upgraded.close()


async def test_internal_surface_cleanup_preserves_raw_history_and_rejects_derivatives(
    tmp_path: Path,
) -> None:
    v29_migrations = tmp_path / "v29_migrations"
    v29_migrations.mkdir()
    for source in sorted(MIGRATIONS.glob("*.sql")):
        if int(source.name.split("_", 1)[0]) >= 30:
            continue
        (v29_migrations / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    db_path = tmp_path / "upgrade-v29.sqlite3"
    old_database = Database(db_path, v29_migrations)
    await old_database.initialize()
    await EventRepository(old_database).append(
        event("raw-event-survives-cleanup", 1, EventType.MESSAGE_CREATED)
    )
    timestamp = datetime(2026, 7, 19, 12, 0, tzinfo=UTC).isoformat()
    await old_database.execute(
        """
        INSERT INTO learning_instruction_examples (
            example_id, guild_id, channel_id, user_id, source_event_id,
            response_message_id, instruction_text, answer_text, prompt_hash,
            answer_hash, quality_score, status, created_at, updated_at
        ) VALUES (
            'polluted-example', 1, 20, 30, 'raw-event-survives-cleanup',
            1001, 'У тебя есть носки?', 'команды узнай названного объекта.',
            'prompt-hash', 'answer-hash', 0.9, 'accepted', ?, ?
        )
        """,
        (timestamp, timestamp),
    )
    await old_database.execute(
        """
        INSERT INTO learning_graph_nodes (
            node_id, guild_id, node_type, node_key, label, confidence,
            occurrence_count, created_at, updated_at
        ) VALUES (
            'polluted-node', 1, 'phrase', 'polluted-node',
            'значения и памяти. но не хватает фактов', 0.9, 2, ?, ?
        )
        """,
        (timestamp, timestamp),
    )
    await old_database.execute(
        """
        INSERT INTO memory_long_term_capsules (
            capsule_id, guild_id, channel_id, summary, importance, confidence,
            created_at, updated_at
        ) VALUES (
            'polluted-capsule', 1, 20,
            'в памяти пока не хватает фактов о носках', 0.8, 0.8, ?, ?
        )
        """,
        (timestamp, timestamp),
    )
    await old_database.execute(
        """
        INSERT INTO language_conversation_sessions (
            conversation_id, guild_id, channel_id, started_at, last_activity_at,
            current_summary
        ) VALUES (
            'polluted-session', 1, 20, ?, ?,
            'по реплике вижу тему носков'
        )
        """,
        (timestamp, timestamp),
    )
    await old_database.close()

    upgraded = Database(db_path, MIGRATIONS)
    await upgraded.initialize()
    try:
        assert await upgraded.schema_version() == 40
        assert await EventRepository(upgraded).get("raw-event-survives-cleanup") is not None
        example = await upgraded.fetchone(
            "SELECT status, quality_score FROM learning_instruction_examples "
            "WHERE example_id = 'polluted-example'"
        )
        assert example is not None
        assert example["status"] == "rejected"
        assert float(example["quality_score"]) == 0.0
        assert await upgraded.fetchone(
            "SELECT 1 FROM learning_graph_nodes WHERE node_id = 'polluted-node'"
        ) is None
        assert await upgraded.fetchone(
            "SELECT 1 FROM memory_long_term_capsules "
            "WHERE capsule_id = 'polluted-capsule'"
        ) is None
        session = await upgraded.fetchone(
            "SELECT current_summary FROM language_conversation_sessions "
            "WHERE conversation_id = 'polluted-session'"
        )
        assert session is not None
        assert session["current_summary"] == ""
    finally:
        await upgraded.close()


async def test_authored_warm_start_replies_are_removed_on_upgrade(tmp_path: Path) -> None:
    v38_migrations = tmp_path / "v38_migrations"
    v38_migrations.mkdir()
    for source in sorted(MIGRATIONS.glob("*.sql")):
        if int(source.name.split("_", 1)[0]) >= 39:
            continue
        (v38_migrations / source.name).write_text(
            source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    db_path = tmp_path / "upgrade-v38.sqlite3"
    old_database = Database(db_path, v38_migrations)
    await old_database.initialize()
    timestamp = datetime(2026, 7, 23, 12, 0, tzinfo=UTC).isoformat()
    prompt_message_id = 8_700_000_000_000_000_000
    response_message_id = prompt_message_id + 1
    for message_id, content, reply_to, author_is_bot in (
        (prompt_message_id, "Кто твоя мама?", None, 0),
        (
            response_message_id,
            "В человеческом смысле мамы у меня нет.",
            prompt_message_id,
            1,
        ),
    ):
        await old_database.execute(
            """
            INSERT INTO archive_messages (
                message_id, guild_id, channel_id, author_id, created_at,
                content, reply_to_message_id, author_is_bot
            ) VALUES (?, 1, 2, 3, ?, ?, ?, ?)
            """,
            (message_id, timestamp, content, reply_to, author_is_bot),
        )
    await old_database.execute(
        """
        INSERT INTO learning_instruction_examples (
            example_id, guild_id, channel_id, user_id, source_event_id,
            response_message_id, instruction_text, answer_text, prompt_hash,
            answer_hash, quality_score, status, created_at, updated_at
        ) VALUES (
            'warm-start-example', 1, 2, 3, 'curriculum:warm-start:0',
            ?, 'Кто твоя мама?', 'В человеческом смысле мамы у меня нет.',
            'warm-prompt', 'warm-answer', 0.95, 'accepted', ?, ?
        )
        """,
        (response_message_id, timestamp, timestamp),
    )
    await old_database.close()

    upgraded = Database(db_path, MIGRATIONS)
    await upgraded.initialize()
    try:
        assert await upgraded.schema_version() == 40
        archived = await upgraded.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id IN (?, ?)",
            (prompt_message_id, response_message_id),
        )
        assert archived is None
        example = await upgraded.fetchone(
            "SELECT status, quality_score FROM learning_instruction_examples "
            "WHERE example_id = 'warm-start-example'"
        )
        assert example is not None
        assert example["status"] == "rejected"
        assert float(example["quality_score"]) == 0.0
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
