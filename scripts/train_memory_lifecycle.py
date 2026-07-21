# ruff: noqa: E501
"""Install and backfill the selective working-to-long-term memory lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.memory.long_term import LongTermMemoryRepository, memory_terms
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978

MEMORY_PRINCIPLES = (
    (
        "relevance",
        "Рабочая мысль выбирает прошлое по текущей теме, свежести и важности, а не тащит весь недавний текст.",
    ),
    (
        "forget_noise",
        "Слабая случайная реплика без решения, вопроса или значимой связи может быть забыта полностью.",
    ),
    (
        "protect_decisions",
        "Решения, выводы, причинные связи, обещания и незакрытые вопросы получают повышенную важность.",
    ),
    (
        "compress",
        "При выходе из рабочего кэша важные сообщения сжимаются в тематическую капсулу вместо хранения потока сырых слов.",
    ),
    (
        "recall_on_demand",
        "Долгосрочная память извлекается только при смысловом совпадении с текущим запросом, участником или каналом.",
    ),
    (
        "reinforce_recall",
        "Полезное воспоминание слегка укрепляется после извлечения, а невостребованное постепенно теряет силу.",
    ),
)

PROBES = (
    "как рабочая мысль выбирает прошлые сообщения",
    "что происходит со случайным шумом",
    "как хранить важные решения и выводы",
    "когда доставать долгосрочную память",
)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        working = ShortTermMemory(
            MemoryLimits(
                retention=timedelta(minutes=settings.memory_retention_minutes),
                max_context_events=settings.memory_max_context_events,
                relevance_half_life=timedelta(
                    minutes=settings.memory_relevance_half_life_minutes
                ),
                min_context_relevance=settings.memory_min_context_relevance,
            )
        )
        long_term = LongTermMemoryRepository(
            database,
            working,
            min_importance=settings.memory_consolidation_min_importance,
            fragment_batch_size=settings.memory_consolidation_batch_size,
            capsule_limit=settings.memory_long_term_capsule_limit,
            recall_limit=settings.memory_long_term_recall_limit,
        )
        if not evaluate_only:
            principles = await _install_principles(database, guild_id)
            episodes = await long_term.backfill_conversation_episodes(guild_id)
            decayed = await long_term.maintain(datetime.now(UTC))
            print(f"memory_principles_installed={principles}")
            print(f"conversation_capsules_added={episodes}")
            print(f"capsules_maintained={decayed}")
        await _report(database, long_term, guild_id, backup)
    finally:
        await database.close()


async def _install_principles(database: Database, guild_id: int) -> int:
    now = datetime.now(UTC).isoformat()
    changed = 0
    for key, summary in MEMORY_PRINCIPLES:
        cursor = await database.execute(
            """INSERT INTO memory_long_term_capsules (
                capsule_id, guild_id, channel_id, participant_ids_json,
                topic_terms_json, summary, source_event_ids_json, source_kind,
                importance, confidence, strength, decay_rate, created_at, updated_at
            ) VALUES (?, ?, NULL, '[]', ?, ?, '[]', 'memory_curriculum',
                      0.90, 0.96, 1.0, 0.002, ?, ?)
            ON CONFLICT (capsule_id) DO UPDATE SET
                topic_terms_json = excluded.topic_terms_json,
                summary = excluded.summary,
                importance = excluded.importance,
                confidence = excluded.confidence,
                strength = MAX(strength, excluded.strength),
                updated_at = excluded.updated_at""",
            (
                f"capsule:memory-principle:{key}",
                guild_id,
                json.dumps(memory_terms(summary), ensure_ascii=False),
                summary,
                now,
                now,
            ),
        )
        changed += max(0, cursor.rowcount)
    return changed


async def _report(
    database: Database,
    long_term: LongTermMemoryRepository,
    guild_id: int,
    backup: Path | None,
) -> None:
    stats = await long_term.stats(guild_id)
    source_rows = await database.fetchall(
        """SELECT source_kind, COUNT(*) AS count
           FROM memory_long_term_capsules WHERE guild_id = ? GROUP BY source_kind""",
        (guild_id,),
    )
    print(f"backup={backup}")
    print(f"schema={await database.schema_version()}")
    print(f"pending_fragments={stats.pending_fragments}")
    print(f"long_term_capsules={stats.capsules}")
    for row in source_rows:
        print(f"capsules[{row['source_kind']}]={row['count']}")
    print("recall_probes:")
    for probe in PROBES:
        memories = await long_term.recall(guild_id, probe, limit=2)
        print(f"- {probe} -> {memories[0].summary if memories else 'NO_MATCH'}")
    unrelated = await long_term.recall(
        guild_id, "археоптерикс калибрует тахионный спектрометр", limit=2
    )
    print(f"unrelated_recall_count={len(unrelated)}")


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(
        f"{database_path.stem}-before-memory-lifecycle-{stamp}.sqlite3"
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
