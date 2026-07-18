"""Persistence for per-event reactions and slowly learned relationship baselines."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from debilgi.social.models import (
    PrimitiveVector,
    ReactionStats,
    RelationState,
    SocialReaction,
)
from debilgi.storage.database import Database


class SocialReactionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(self, reaction: SocialReaction) -> bool:
        vector = reaction.vector
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO social_reactions (
                    reaction_id, event_id, guild_id, actor_id, agreement, resistance,
                    approval, curiosity, aversion, trust, arousal, primary_component,
                    secondary_component, blend_id, confidence, signals_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), ?)
                """,
                (
                    reaction.reaction_id,
                    reaction.event_id,
                    reaction.guild_id,
                    reaction.actor_id,
                    vector.agreement,
                    vector.resistance,
                    vector.approval,
                    vector.curiosity,
                    vector.aversion,
                    vector.trust,
                    vector.arousal,
                    reaction.primary_component,
                    reaction.secondary_component,
                    reaction.blend_id,
                    reaction.confidence,
                    _json_array(reaction.signals),
                    reaction.created_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1 or reaction.actor_id is None:
                return cursor.rowcount == 1
            prior = await (
                await connection.execute(
                    """
                    SELECT * FROM social_relation_states
                    WHERE guild_id = ? AND actor_id = ?
                    """,
                    (reaction.guild_id, reaction.actor_id),
                )
            ).fetchone()
            observations = int(prior["observations"]) if prior is not None else 0
            baseline = self._vector_from_row(prior) if prior is not None else vector
            learning_rate = max(0.08, min(0.30, 1.0 / (observations + 1)))
            updated = baseline.blended(vector, learning_rate)
            await connection.execute(
                """
                INSERT INTO social_relation_states (
                    guild_id, actor_id, agreement, resistance, approval, curiosity,
                    aversion, trust, arousal, observations, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT (guild_id, actor_id) DO UPDATE SET
                    agreement = excluded.agreement,
                    resistance = excluded.resistance,
                    approval = excluded.approval,
                    curiosity = excluded.curiosity,
                    aversion = excluded.aversion,
                    trust = excluded.trust,
                    arousal = excluded.arousal,
                    observations = social_relation_states.observations + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    reaction.guild_id,
                    reaction.actor_id,
                    updated.agreement,
                    updated.resistance,
                    updated.approval,
                    updated.curiosity,
                    updated.aversion,
                    updated.trust,
                    updated.arousal,
                    reaction.created_at.isoformat(),
                ),
            )
        return True

    async def reaction_for_event(self, event_id: str) -> SocialReaction | None:
        row = await self._database.fetchone(
            "SELECT * FROM social_reactions WHERE event_id = ?", (event_id,)
        )
        return self._reaction_from_row(row) if row else None

    async def relation_state(self, guild_id: int, actor_id: int) -> RelationState | None:
        row = await self._database.fetchone(
            "SELECT * FROM social_relation_states WHERE guild_id = ? AND actor_id = ?",
            (guild_id, actor_id),
        )
        return (
            RelationState(
                guild_id=guild_id,
                actor_id=actor_id,
                vector=self._vector_from_row(row),
                observations=int(row["observations"]),
                updated_at=datetime.fromisoformat(str(row["updated_at"])),
            )
            if row
            else None
        )

    async def recent(self, guild_id: int, limit: int = 10) -> tuple[SocialReaction, ...]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM social_reactions WHERE guild_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 50))),
        )
        return tuple(self._reaction_from_row(row) for row in rows)

    async def stats(self, guild_id: int) -> ReactionStats:
        row = await self._database.fetchone(
            """
            SELECT COUNT(*) AS reactions,
                   SUM(secondary_component IS NOT NULL) AS composites
            FROM social_reactions WHERE guild_id = ?
            """,
            (guild_id,),
        )
        relation = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM social_relation_states WHERE guild_id = ?",
            (guild_id,),
        )
        return ReactionStats(
            reactions=int(row["reactions"] or 0) if row else 0,
            composite_reactions=int(row["composites"] or 0) if row else 0,
            relation_models=int(relation["count"] or 0) if relation else 0,
        )

    @classmethod
    def _reaction_from_row(cls, row: Any) -> SocialReaction:
        import json

        return SocialReaction(
            reaction_id=str(row["reaction_id"]),
            event_id=str(row["event_id"]),
            guild_id=int(row["guild_id"]),
            actor_id=int(row["actor_id"]) if row["actor_id"] is not None else None,
            vector=cls._vector_from_row(row),
            primary_component=str(row["primary_component"]),
            secondary_component=(
                str(row["secondary_component"]) if row["secondary_component"] is not None else None
            ),
            blend_id=str(row["blend_id"]),
            confidence=float(row["confidence"]),
            signals=tuple(str(value) for value in json.loads(str(row["signals_json"]))),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _vector_from_row(row: Any) -> PrimitiveVector:
        return PrimitiveVector(
            agreement=float(row["agreement"]),
            resistance=float(row["resistance"]),
            approval=float(row["approval"]),
            curiosity=float(row["curiosity"]),
            aversion=float(row["aversion"]),
            trust=float(row["trust"]),
            arousal=float(row["arousal"]),
        )


def _json_array(values: tuple[str, ...]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))
