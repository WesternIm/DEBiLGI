import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.sentence_understanding import (
    UtteranceUnderstandingRepository,
    understand_utterance,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 19, 40, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "sentence-understanding.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def word(word: str, *, known: bool) -> WordAnalysis:
    return WordAnalysis(
        surface_form=word,
        normalized_form=word,
        lemma=word,
        part_of_speech="VERB" if word == "закинуть" else "NOUN",
        grammatical_features={},
        dictionary_meanings=("known",) if known else (),
        style_labels=(),
        source="test" if known else "unknown",
        confidence=ConfidenceProfile(0.8, 0.8 if known else 0.0, 0.7, 0.5, 0.5),
        unknown_kind=None if known else UnknownKind.UNKNOWN_FORM,
    )


async def test_persists_terms_and_relations_without_reducing_event_to_one_word(
    database: Database,
) -> None:
    understanding = understand_utterance(
        "Я закинул трошку в сумку",
        (
            word("закинуть", known=True),
            word("трошка", known=False),
            word("сумка", known=True),
        ),
    )
    repository = UtteranceUnderstandingRepository(database)
    await repository.save(
        event_id="event:sentence",
        guild_id=1,
        channel_id=10,
        understanding=understanding,
        observed_at=NOW,
    )

    row = await database.fetchone(
        "SELECT * FROM language_utterance_understandings WHERE event_id = 'event:sentence'"
    )
    assert row is not None
    assert json.loads(str(row["unknown_terms_json"])) == ["трошка"]
    assert {"закинуть", "сумка"} <= set(json.loads(str(row["known_terms_json"])))
    assert json.loads(str(row["relations_json"]))
    assert int(row["sufficient"]) == 1
