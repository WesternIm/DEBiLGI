"""Application entry point."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from debilgi.config import ConfigurationError, Settings
from debilgi.discord.bot import create_bot
from debilgi.events.repository import EventRepository
from debilgi.language.syntax_backend import configure_syntax_backend
from debilgi.logging import configure_logging
from debilgi.runtime_lock import InstanceAlreadyRunningError, SingleInstanceLock
from debilgi.secret_vault import SecretVaultError, prepare_runtime_secrets
from debilgi.storage.database import Database

logger = logging.getLogger(__name__)


async def run() -> None:
    prepare_runtime_secrets(Path(__file__).resolve().parents[2])
    settings = Settings.from_env()
    configure_logging(settings.log_level, settings.database_path.parent)
    token = settings.require_discord_token()
    instance_lock = SingleInstanceLock.for_database(settings.database_path)

    with instance_lock:
        await asyncio.to_thread(
            configure_syntax_backend,
            settings.slovnet_model_dir if settings.slovnet_enabled else None,
        )
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
    except (ConfigurationError, InstanceAlreadyRunningError, SecretVaultError) as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception:
        # Console windows may disappear with the launcher. Keep the traceback
        # in runtime.log as well, including failures during login/setup/close.
        logger.exception("Bot terminated unexpectedly")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
