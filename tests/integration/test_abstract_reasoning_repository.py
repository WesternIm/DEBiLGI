from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.language.abstract_reasoning import AbstractReasoningService
from debilgi.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "abstract.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_mental_models_persist_separately_and_elaborate_in_background(
    database: Database,
) -> None:
    service = AbstractReasoningService(database)
    model = await service.observe_turn(
        guild_id=1,
        channel_id=10,
        source_event_id="discord:abstract:1",
        prompt=(
            "Представь, что дом слева от башни, "
            "а башня слева от реки. Где дом?"
        ),
        observed_at=NOW,
    )

    assert model is not None
    assert model.mode == "hypothetical"
    stored = await database.fetchone(
        """SELECT mode, assumptions_json FROM language_mental_models
           WHERE source_event_id = 'discord:abstract:1'"""
    )
    assert stored is not None
    assert str(stored["mode"]) == "hypothetical"
    assert await database.fetchone(
        """SELECT 1 FROM knowledge_concepts
           WHERE normalized_title IN ('дом', 'башня', 'река')"""
    ) is None

    # Remove derived closure to prove the idle pass can reconstruct it.
    await database.execute(
        """DELETE FROM language_mental_relations
           WHERE model_id = ? AND inferred = 1""",
        (model.model_id,),
    )
    elaborated = await service.background_elaborate(observed_at=NOW)

    assert elaborated >= 3
    inferred = await database.fetchone(
        """SELECT 1 FROM language_mental_relations
           WHERE model_id = ? AND subject_key = 'дом'
             AND relation = 'left_of' AND object_key = 'река'
             AND inferred = 1""",
        (model.model_id,),
    )
    assert inferred is not None
