from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from debilgi.config import Settings
from debilgi.discord.bot import create_bot
from debilgi.events.repository import EventRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def test_shutdown_closes_database_when_background_stop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        discord_token=None,
        database_path=tmp_path / "shutdown.sqlite3",
        migrations_path=PROJECT_ROOT / "migrations",
        sync_commands=False,
    )
    database = Database(settings.database_path, settings.migrations_path)
    bot = create_bot(settings, database, EventRepository(database))
    await bot.setup_hook()
    original_stop = bot.background_learning.stop

    async def broken_stop() -> None:
        await original_stop()
        raise RuntimeError("background stop failed")

    close_database = AsyncMock(wraps=database.close)
    monkeypatch.setattr(bot.background_learning, "stop", broken_stop)
    monkeypatch.setattr(database, "close", close_database)
    with pytest.raises(RuntimeError, match="background stop failed"):
        await bot.close()
    close_database.assert_awaited_once()
    assert not bot.pipeline.running
    assert not bot.continuous_cognition.running
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = database.connection


@pytest.mark.filterwarnings("ignore:.*audioop.*:DeprecationWarning")
async def test_bot_registers_listener_and_diagnostic_group(tmp_path: Path) -> None:
    settings = Settings(
        discord_token=None,
        database_path=tmp_path / "bot.sqlite3",
        migrations_path=PROJECT_ROOT / "migrations",
        sync_commands=False,
    )
    database = Database(settings.database_path, settings.migrations_path)
    repository = EventRepository(database)
    bot = create_bot(settings, database, repository)

    try:
        await bot.setup_hook()
        assert bot.get_cog("EventListener") is not None
        assert bot.get_cog("Diagnostics") is not None
        assert not bot.reply_service.enabled
        assert bot.reply_service._semantic_consultant is bot.semantic_consultant
        assert bot.reply_service._assertion_experiences is bot.assertion_experiences
        assert bot.assertion_experiences.observe_feedback in bot.pipeline._handlers
        assert bot.russian_reference.worker_count == 4
        assert bot.background_learning.study_worker_count == 4
        assert bot.continuous_cognition.running
        assert (
            bot.reply_service._continuous_cognition
            is bot.continuous_cognition
        )
        group = bot.tree.get_command("debilgi")
        assert group is not None
        command_names = {command.name for command in group.commands}
        assert {
            "archive-start",
            "archive-status",
            "language-stats",
            "memory",
            "features",
            "expectations",
            "core",
            "dna",
            "experience",
            "reflection",
            "cognition-runtime",
            "literacy",
            "reactions",
            "autonomy",
            "knowledge",
        } <= command_names
        language = group.get_command("language")
        assert language is not None
        assert language.get_command("progress") is not None
        assert language.get_command("delivery") is not None
        assert language.get_command("progress-live") is not None
        assert language.get_command("quality-cases") is not None
    finally:
        await bot.close()
