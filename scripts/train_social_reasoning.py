"""Install abstract social policies without archiving authored bot replies."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from debilgi.config import Settings
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.language.productive_nlg import productive_pattern_curriculum
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.language.social_reasoning import SOCIAL_POLICY_CURRICULUM
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup_path = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        service = SemanticAssimilationService(database, SelfDatasetRepository(database))
        installed = (
            await service.install_social_curriculum(guild_id=guild_id)
            if not evaluate_only
            else 0
        )
        productive_installed = (
            await service.install_productive_curriculum(guild_id=guild_id)
            if not evaluate_only
            else 0
        )
        guidance = await service.guidance_for_prompt(guild_id, "социальная проверка")
        rows = await database.fetchone(
            """SELECT COUNT(*) AS amount
               FROM language_semantic_lessons
               WHERE guild_id = ? AND lesson_kind = 'dialogue'
                 AND status = 'active' AND payload_json LIKE '%social_policy%'""",
            (guild_id,),
        )
        print(f"backup={backup_path}")
        print(f"schema={await database.schema_version()}")
        print(f"curriculum_policies={len(SOCIAL_POLICY_CURRICULUM)}")
        print(f"installed_active={installed}")
        print(f"productive_curriculum={len(productive_pattern_curriculum())}")
        print(f"productive_installed={productive_installed}")
        print(f"stored_active={int(rows['amount']) if rows is not None else 0}")
        print(f"runtime_visible={len(guidance.social_policies)}")
        print("contains_authored_replies=false")
    finally:
        await database.close()


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(
        f"{database_path.stem}-before-social-reasoning-{stamp}.sqlite3"
    )
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(train(evaluate_only=arguments.evaluate_only))


if __name__ == "__main__":
    main()
