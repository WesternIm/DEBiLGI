from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.language.usage_learning import (
    LexicalStudyMaterial,
    LexicalUsageLearningService,
    generalize_usage_pattern,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 18, 45, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "usage-learning.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_reference_examples_become_pattern_but_are_not_stored(
    database: Database,
) -> None:
    dataset = SelfDatasetRepository(database)
    service = LexicalUsageLearningService(database, dataset)
    examples = (
        "Клоун выступал на арене.",
        "Этот клоун выступает каждый вечер.",
        "Весёлый клоун выступил перед публикой.",
    )
    lesson = await service.learn(
        1,
        LexicalStudyMaterial(
            word="клоун",
            lemma="клоун",
            definition="комический артист цирка, который развлекает публику",
            examples=examples,
            source="wiktionary",
        ),
        observed_at=NOW,
    )

    pattern = await database.fetchone("SELECT * FROM language_usage_patterns")
    assert pattern is not None
    assert str(pattern["pattern_status"]) == "stable"
    assert str(pattern["preferred_case"]) == "nominative"
    assert str(pattern["preferred_number"]) == "singular"
    assert "выступать" in str(pattern["neighbor_lemmas_json"])
    assert lesson.pattern.example_count == 3
    assert lesson.generated_example not in examples

    concepts = await database.fetchall("SELECT * FROM language_reference_concepts")
    assert concepts
    assert {str(row["concept_key"]) for row in concepts} & {"артист", "цирк", "публика"}
    graph_nodes = await database.fetchall(
        "SELECT label, properties_json FROM learning_graph_nodes WHERE guild_id = 1"
    )
    serialized = " ".join(
        f'{row["label"]} {row["properties_json"]}' for row in graph_nodes
    )
    assert all(example not in serialized for example in examples)
    assert "комический артист цирка" not in serialized

    guidance = await service.guidance_for_text(1, "Этот клоун снова здесь")
    assert guidance and guidance[0].stable is True
    assert guidance[0].preferred_case == "nominative"

    await dataset.rebuild_graph(1)
    rebuilt = await database.fetchone(
        """SELECT 1 FROM learning_graph_nodes
           WHERE guild_id = 1 AND node_type = 'usage_pattern'"""
    )
    association = await database.fetchone(
        """SELECT 1 FROM learning_graph_edges
           WHERE guild_id = 1 AND relation = 'reference_association'"""
    )
    assert rebuilt is not None
    assert association is not None


def test_missing_or_divergent_examples_leave_an_uncertain_soft_pattern() -> None:
    pattern = generalize_usage_pattern(
        word="кванг",
        lemma="кванг",
        observations=(),
        total_examples=0,
    )

    assert pattern.status == "uncertain"
    assert pattern.preferred_case is None
    assert pattern.neighbors == ()
    assert pattern.confidence <= 0.38
