"""Offline JSONL replay for inspecting the full shadow cognition chain."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from debilgi.cognition.core import CognitiveCore, CognitiveCoreConfig
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.expectations import ExpectationPredictor
from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.cognition.learning import LearningConfig, LearningEngine
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    event_id: str
    pending_expectations: int
    resolved_expectations: int
    last_difference: float | None
    episode_id: str | None
    semantic_memories: int
    tension: float | None
    shadow_intent: str | None


async def replay_events(events: list[Event], database: Database) -> list[ReplayFrame]:
    await database.initialize()
    event_repository = EventRepository(database)
    memory = ShortTermMemory(MemoryLimits(retention=timedelta(days=7)))
    world_model = WorldModelRepository(database)
    learning = LearningEngine(
        world_model,
        memory,
        BehaviorFeatureExtractor(FeatureConfig(lookback=timedelta(days=7))),
        ExpectationPredictor(),
        LearningConfig(horizon=timedelta(minutes=2), poll_interval=timedelta(seconds=5)),
    )
    cognitive_repository = CognitiveRepository(database)
    core = CognitiveCore(
        cognitive_repository,
        world_model,
        config=CognitiveCoreConfig(
            episode_gap=timedelta(minutes=10),
            maintenance_interval=timedelta(minutes=1),
            raw_text_retention=timedelta(days=30),
        ),
    )
    frames: list[ReplayFrame] = []
    for event in sorted(events, key=lambda item: (item.utc_timestamp, item.event_id)):
        await event_repository.append(event)
        await memory.observe(event)
        await learning.observe(event)
        await core.observe(event)
        frames.append(await _frame(event, world_model, cognitive_repository))
    if events:
        final_time = max(event.utc_timestamp for event in events) + timedelta(minutes=11)
        await learning.expire_overdue(final_time)
        await core.maintenance_once(final_time)
        frames[-1] = await _frame(events[-1], world_model, cognitive_repository)
    return frames


async def _frame(
    event: Event,
    world_model: WorldModelRepository,
    repository: CognitiveRepository,
) -> ReplayFrame:
    world_stats = await world_model.stats(event.guild_id)
    differences = await repository.recent_differences(event.guild_id, 1)
    episodes = await repository.recent_episodes(event.guild_id, 1)
    semantic = await repository.semantic_memories(event.guild_id, limit=100)
    state = await repository.internal_state(event.guild_id, event.actor_id or 0)
    intents = await repository.recent_intents(event.guild_id, 1)
    return ReplayFrame(
        event_id=event.event_id,
        pending_expectations=world_stats.pending_expectations,
        resolved_expectations=world_stats.resolved_expectations,
        last_difference=float(differences[0]["scalar_d"]) if differences else None,
        episode_id=episodes[0].episode_id if episodes else None,
        semantic_memories=len(semantic),
        tension=state.tension if state else None,
        shadow_intent=intents[0].action_type.value if intents else None,
    )


def _event_from_json(data: dict[str, Any]) -> Event:
    timestamp = datetime.fromisoformat(str(data["timestamp"]))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return Event(
        event_id=str(data["event_id"]),
        guild_id=int(data["guild_id"]),
        channel_id=int(data["channel_id"]) if data.get("channel_id") is not None else None,
        actor_id=int(data["actor_id"]) if data.get("actor_id") is not None else None,
        target_id=int(data["target_id"]) if data.get("target_id") is not None else None,
        event_type=EventType(str(data["event_type"])),
        timestamp=timestamp,
        payload=dict(data.get("payload", {})),
        source=str(data.get("source", "replay")),
    )


async def _run_cli(input_path: Path, database_path: Path, migrations_path: Path) -> None:
    events = [
        _event_from_json(json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    database = Database(database_path, migrations_path)
    try:
        for frame in await replay_events(events, database):
            print(json.dumps(asdict(frame), ensure_ascii=False, sort_keys=True))
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file with normalized events")
    parser.add_argument("--database", type=Path, default=Path("data/replay.sqlite3"))
    parser.add_argument("--migrations", type=Path, default=Path("migrations"))
    args = parser.parse_args()
    asyncio.run(_run_cli(args.input, args.database, args.migrations))


if __name__ == "__main__":
    main()
