"""Backfill and export DEBiLGI's self-curated instruction and graph datasets."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from debilgi.config import Settings
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.storage.database import Database


async def export(arguments: argparse.Namespace) -> None:
    settings = Settings.from_env()
    database = Database(settings.database_path, settings.migrations_path)
    await database.initialize()
    try:
        guild_id = arguments.guild_id or settings.dev_guild_id
        if guild_id is None:
            row = await database.fetchone(
                """SELECT guild_id, COUNT(*) AS count FROM events
                   GROUP BY guild_id ORDER BY count DESC LIMIT 1"""
            )
            if row is None:
                raise RuntimeError("No guild data is available for export")
            guild_id = int(row["guild_id"])
        repository = SelfDatasetRepository(database)
        backfilled = (
            0
            if arguments.no_backfill
            else await repository.backfill_experiences(guild_id, limit=arguments.backfill_limit)
        )
        reassessed = await repository.reassess_examples(guild_id)
        graph_nodes, graph_edges = await repository.rebuild_graph(guild_id)
        counts = await repository.export(
            guild_id,
            arguments.output_dir.resolve(),
            include_candidates=arguments.include_candidates,
            minimum_quality=arguments.minimum_quality,
        )
        stats = await repository.stats(guild_id)
        print(f"guild_id={guild_id}")
        print(f"backfilled={backfilled}")
        print(f"rejected_by_reassessment={reassessed}")
        print(f"graph_nodes_rebuilt={graph_nodes}")
        print(f"graph_edges_rebuilt={graph_edges}")
        print(f"output_dir={arguments.output_dir.resolve()}")
        print(f"instruction_qa_exported={counts['instruction_qa']}")
        print(f"graph_nodes_exported={counts['graph_nodes']}")
        print(f"graph_edges_exported={counts['graph_edges']}")
        print(f"candidates={stats.candidates}")
        print(f"accepted={stats.accepted}")
        print(f"rejected={stats.rejected}")
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guild-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("data/self-dataset"))
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--minimum-quality", type=float, default=0.55)
    parser.add_argument("--no-backfill", action="store_true")
    parser.add_argument("--backfill-limit", type=int, default=10_000)
    asyncio.run(export(parser.parse_args()))


if __name__ == "__main__":
    main()
