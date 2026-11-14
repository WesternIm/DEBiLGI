"""Persistent continuous affect shaped by observed interaction consequences."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime

from debilgi.events.models import Event, EventType
from debilgi.social.models import PrimitiveVector
from debilgi.social.repository import SocialReactionRepository
from debilgi.storage.database import Database

_MESSAGES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


@dataclass(frozen=True, slots=True)
class AffectVector:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    novelty: float = 0.0
    affiliation: float = 0.0

    def features(self, prefix: str) -> tuple[tuple[str, float], ...]:
        return tuple((f"{prefix}_{key}", value) for key, value in asdict(self).items())


@dataclass(frozen=True, slots=True)
class AffectState:
    guild_id: int
    actor_id: int
    mood: AffectVector
    emotion: AffectVector
    mood_observations: int
    emotion_observations: int
    mood_updated_at: datetime | None
    emotion_updated_at: datetime | None

    @property
    def features(self) -> tuple[tuple[str, float], ...]:
        return (*self.mood.features("mood"), *self.emotion.features("emotion"))

    @property
    def labels(self) -> tuple[str, str]:
        return nearest_affect(self.mood), nearest_affect(self.emotion)


# Labels make the wheel inspectable. They never select an action or phrase.
_POINTS = {
    "радость": (0.85, 0.45),
    "спокойствие": (0.65, -0.65),
    "скука": (-0.35, -0.85),
    "печаль": (-0.80, -0.40),
    "злость": (-0.72, 0.76),
    "тревога": (-0.45, 0.88),
    "удивление": (0.15, 0.95),
    "интерес": (0.48, 0.68),
    "нейтральность": (0.0, 0.0),
}


def nearest_affect(vector: AffectVector) -> str:
    return min(
        _POINTS,
        key=lambda name: (
            (vector.valence - _POINTS[name][0]) ** 2 + (vector.arousal - _POINTS[name][1]) ** 2
        ),
    )


def stimulus(vector: PrimitiveVector) -> AffectVector:
    positive = (vector.approval + vector.agreement + vector.trust) / 3.0
    negative = (vector.resistance + vector.aversion) / 2.0
    return AffectVector(
        valence=_signed(positive - negative),
        arousal=_signed(vector.arousal * 2.0 - 1.0),
        dominance=_signed(vector.agreement + vector.trust - vector.resistance - 0.5),
        novelty=_signed(vector.curiosity * 2.0 - 1.0),
        affiliation=_signed(vector.trust + vector.approval - vector.aversion - 0.5),
    )


class AffectRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def observe(self, event: Event, social: PrimitiveVector) -> bool:
        if (
            event.event_type not in _MESSAGES
            or event.payload.get("author_is_bot") is True
            or event.actor_id is None
        ):
            return False
        if await self._database.fetchone(
            "SELECT 1 FROM affect_observations WHERE event_id = ?", (event.event_id,)
        ):
            return False
        current = await self.state(event.guild_id, event.actor_id, at=event.utc_timestamp)
        impulse = stimulus(social)
        mood = _advance(
            current.mood,
            impulse,
            event.utc_timestamp,
            current.mood_updated_at,
            half_life_seconds=21600.0,
            learning_rate=0.08,
        )
        emotion = _advance(
            current.emotion,
            impulse,
            event.utc_timestamp,
            current.emotion_updated_at,
            half_life_seconds=720.0,
            learning_rate=0.38,
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO affect_moods VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id) DO UPDATE SET
                    valence=excluded.valence, arousal=excluded.arousal,
                    dominance=excluded.dominance, novelty=excluded.novelty,
                    affiliation=excluded.affiliation,
                    observations=affect_moods.observations+1, updated_at=excluded.updated_at""",
                (
                    event.guild_id,
                    *asdict(mood).values(),
                    current.mood_observations + 1,
                    event.utc_timestamp.isoformat(),
                ),
            )
            await connection.execute(
                """INSERT OR IGNORE INTO affect_emotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id,actor_id) DO UPDATE SET
                    valence=excluded.valence, arousal=excluded.arousal,
                    dominance=excluded.dominance, novelty=excluded.novelty,
                    affiliation=excluded.affiliation,
                    observations=affect_emotions.observations+1, updated_at=excluded.updated_at""",
                (
                    event.guild_id,
                    event.actor_id,
                    *asdict(emotion).values(),
                    current.emotion_observations + 1,
                    event.utc_timestamp.isoformat(),
                ),
            )
            await connection.execute(
                """INSERT INTO affect_observations VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.guild_id,
                    event.actor_id,
                    _json(impulse),
                    _json(mood),
                    _json(emotion),
                    event.utc_timestamp.isoformat(),
                ),
            )
        return True

    async def state(
        self, guild_id: int, actor_id: int, *, at: datetime | None = None
    ) -> AffectState:
        mood = await self._database.fetchone(
            "SELECT * FROM affect_moods WHERE guild_id=?", (guild_id,)
        )
        emotion = await self._database.fetchone(
            "SELECT * FROM affect_emotions WHERE guild_id=? AND actor_id=?", (guild_id, actor_id)
        )
        mood_updated = datetime.fromisoformat(mood["updated_at"]) if mood else None
        emotion_updated = datetime.fromisoformat(emotion["updated_at"]) if emotion else None
        mood_vector = _decay(_vector(mood), at, mood_updated, 21600.0)
        emotion_vector = _decay(_vector(emotion), at, emotion_updated, 720.0)
        return AffectState(
            guild_id,
            actor_id,
            mood_vector,
            emotion_vector,
            int(mood["observations"]) if mood else 0,
            int(emotion["observations"]) if emotion else 0,
            mood_updated,
            emotion_updated,
        )


class AffectEngine:
    def __init__(self, repository: AffectRepository, reactions: SocialReactionRepository) -> None:
        self.repository = repository
        self._reactions = reactions

    async def observe(self, event: Event) -> None:
        reaction = await self._reactions.reaction_for_event(event.event_id)
        if reaction is not None:
            await self.repository.observe(event, reaction.vector)


def _advance(previous, impulse, now, updated_at, *, half_life_seconds, learning_rate):
    elapsed = max(0.0, (now - updated_at).total_seconds()) if updated_at else half_life_seconds * 8
    retention = 0.5 ** (elapsed / half_life_seconds)
    return AffectVector(
        **{
            key: _signed(
                getattr(previous, key) * retention * (1.0 - learning_rate)
                + getattr(impulse, key) * learning_rate
            )
            for key in asdict(previous)
        }
    )


def _decay(vector, at, updated_at, half_life_seconds):
    if at is None or updated_at is None:
        return vector
    elapsed = max(0.0, (at - updated_at).total_seconds())
    retention = 0.5 ** (elapsed / half_life_seconds)
    return AffectVector(**{key: getattr(vector, key) * retention for key in asdict(vector)})


def _vector(row) -> AffectVector:
    return (
        AffectVector(**{key: float(row[key]) for key in asdict(AffectVector())})
        if row
        else AffectVector()
    )


def _json(value) -> str:
    return json.dumps(asdict(value), ensure_ascii=False, sort_keys=True)


def _signed(value: float) -> float:
    return max(-1.0, min(1.0, value if math.isfinite(value) else 0.0))
