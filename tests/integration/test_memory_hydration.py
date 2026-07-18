from datetime import UTC, datetime
from pathlib import Path

from debilgi.config import Settings
from debilgi.discord.bot import create_bot
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def test_bot_rebuilds_short_term_memory_from_recent_event_log(tmp_path: Path) -> None:
    database_path = tmp_path / "hydration.sqlite3"
    migrations_path = PROJECT_ROOT / "migrations"
    initial_database = Database(database_path, migrations_path)
    await initial_database.initialize()
    initial_repository = EventRepository(initial_database)
    await initial_repository.append(
        Event(
            event_id="persisted-recent-event",
            guild_id=1,
            channel_id=2,
            actor_id=3,
            target_id=None,
            event_type=EventType.MESSAGE_CREATED,
            timestamp=datetime.now(UTC),
            payload={"message_id": 4, "reply_to_message_id": None},
            source="test",
        )
    )
    await initial_database.close()

    settings = Settings(
        discord_token=None,
        database_path=database_path,
        migrations_path=migrations_path,
    )
    database = Database(database_path, migrations_path)
    repository = EventRepository(database)
    bot = create_bot(settings, database, repository)
    try:
        await bot.setup_hook()
        context = await bot.memory.channel_context(1, 2)
        assert [event.event_id for event in context] == ["persisted-recent-event"]
        await bot.pipeline.join()
        context_after_recovered_processing = await bot.memory.channel_context(1, 2)
        assert [event.event_id for event in context_after_recovered_processing] == [
            "persisted-recent-event"
        ]
    finally:
        await bot.close()
