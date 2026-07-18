"""Persistence for character observations without retaining additional raw text."""

from __future__ import annotations

import json

from debilgi.literacy.models import LiteracyStats, SymbolObservation
from debilgi.storage.database import Database


class LiteracyRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(self, observation: SymbolObservation) -> bool:
        analysis = observation.analysis
        query_kinds = set(observation.query_kinds)
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO cognition_symbol_observations (
                    event_id, guild_id, actor_id, letters, digits, punctuation,
                    symbols, whitespace, numbers_json, scripts_json,
                    query_kinds_json, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?), json(?), ?)
                """,
                (
                    observation.event_id,
                    observation.guild_id,
                    observation.actor_id,
                    analysis.letters,
                    analysis.digits,
                    analysis.punctuation,
                    analysis.symbols,
                    analysis.whitespace,
                    _json(analysis.numbers),
                    _json(analysis.scripts),
                    _json(observation.query_kinds),
                    observation.observed_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return False
            await connection.execute(
                """
                INSERT INTO cognition_literacy_states (
                    guild_id, messages_observed, letters_observed, digits_observed,
                    punctuation_observed, symbols_observed, arithmetic_queries,
                    alphabet_queries, counting_queries, comparison_queries,
                    symbol_queries, updated_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (guild_id) DO UPDATE SET
                    messages_observed = cognition_literacy_states.messages_observed + 1,
                    letters_observed = cognition_literacy_states.letters_observed
                        + excluded.letters_observed,
                    digits_observed = cognition_literacy_states.digits_observed
                        + excluded.digits_observed,
                    punctuation_observed = cognition_literacy_states.punctuation_observed
                        + excluded.punctuation_observed,
                    symbols_observed = cognition_literacy_states.symbols_observed
                        + excluded.symbols_observed,
                    arithmetic_queries = cognition_literacy_states.arithmetic_queries
                        + excluded.arithmetic_queries,
                    alphabet_queries = cognition_literacy_states.alphabet_queries
                        + excluded.alphabet_queries,
                    counting_queries = cognition_literacy_states.counting_queries
                        + excluded.counting_queries,
                    comparison_queries = cognition_literacy_states.comparison_queries
                        + excluded.comparison_queries,
                    symbol_queries = cognition_literacy_states.symbol_queries
                        + excluded.symbol_queries,
                    updated_at = excluded.updated_at
                """,
                (
                    observation.guild_id,
                    analysis.letters,
                    analysis.digits,
                    analysis.punctuation,
                    analysis.symbols,
                    int("arithmetic" in query_kinds),
                    int("alphabet" in query_kinds),
                    int("counting" in query_kinds),
                    int("comparison" in query_kinds),
                    int("symbol" in query_kinds),
                    observation.observed_at.isoformat(),
                ),
            )
        return True

    async def stats(self, guild_id: int) -> LiteracyStats:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_literacy_states WHERE guild_id = ?", (guild_id,)
        )
        if row is None:
            return LiteracyStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return LiteracyStats(
            messages_observed=int(row["messages_observed"]),
            letters_observed=int(row["letters_observed"]),
            digits_observed=int(row["digits_observed"]),
            punctuation_observed=int(row["punctuation_observed"]),
            symbols_observed=int(row["symbols_observed"]),
            arithmetic_queries=int(row["arithmetic_queries"]),
            alphabet_queries=int(row["alphabet_queries"]),
            counting_queries=int(row["counting_queries"]),
            comparison_queries=int(row["comparison_queries"]),
            symbol_queries=int(row["symbol_queries"]),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
