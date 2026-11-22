"""The durable pipeline, observers and reply context share one source analysis."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from debilgi.events.normalizer import normalize_message_created
from debilgi.events.pipeline import EventPipeline
from debilgi.events.repository import EventRepository
from debilgi.language.planning import ReplyContext
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.structured_utterance import _cached_structure
from debilgi.storage.database import Database


async def test_durable_reload_and_observers_share_ir_without_reparsing(tmp_path):
    import debilgi.language.structured_utterance as parser

    database = Database(tmp_path / "ir.sqlite3", Path(__file__).parents[2] / "migrations")
    await database.initialize()
    source = normalize_message_created(
        message_id=100,
        guild_id=1,
        channel_id=2,
        author_id=11,
        created_at=datetime(2026, 11, 14, tzinfo=UTC),
        content="Я читаю газету.",
        author_is_bot=False,
        bot_user_id=22,
        bot_mentioned=True,
    )
    seen = []

    async def first(current):
        seen.append(current.meaning)
        await PropositionMemory(database).observe(current)

    async def second(current):
        seen.append(current.meaning)

    _cached_structure.cache_clear()
    pipeline = EventPipeline(EventRepository(database), (first, second))
    await pipeline.start()
    try:
        with patch.object(parser, "_analyze_structure", wraps=parser._analyze_structure) as build:
            await pipeline.publish(source)
            await pipeline.join()
            context = ReplyContext(prompt=source.meaning.text, input_meaning=source.meaning)
            assert seen[0] is seen[1] is context.structure
            assert build.call_count == 1
            rows = await database.fetchall("SELECT record_json FROM language_meaning_records")
            assert rows and "discord:user:11" in rows[0]["record_json"]
    finally:
        await pipeline.stop()
        await database.close()
