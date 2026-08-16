from pathlib import Path

import pytest

from debilgi.config import Settings
from debilgi.discord.bot import create_bot
from debilgi.events.repository import EventRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        assert language.get_command("progress-live") is not None
        assert language.get_command("quality-cases") is not None
    finally:
        await bot.close()
