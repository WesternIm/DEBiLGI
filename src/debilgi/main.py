"""Application entry point."""

from __future__ import annotations

import asyncio
import logging

from debilgi.config import ConfigurationError, Settings
from debilgi.discord.bot import create_bot
from debilgi.events.repository import EventRepository
from debilgi.logging import configure_logging
from debilgi.storage.database import Database

logger = logging.getLogger(__name__)


async def run() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    token = settings.require_discord_token()

    database = Database(settings.database_path, settings.migrations_path)
    repository = EventRepository(database)
    bot = create_bot(settings, database, repository)

    try:
        await bot.start(token, reconnect=True)
    finally:
        await bot.close()


def main() -> None:
    try:
        asyncio.run(run())
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        logger.info("Shutdown requested")


if __name__ == "__main__":
    main()
