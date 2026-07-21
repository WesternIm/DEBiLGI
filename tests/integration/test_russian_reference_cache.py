from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from debilgi.language.lexicon import LexiconService
from debilgi.language.russian_reference import RemoteLexeme, RussianReferenceService
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "russian-reference.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


class FakeClient:
    def __init__(self) -> None:
        self.lookups = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def lookup(self, word: str) -> RemoteLexeme | None:
        self.lookups += 1
        return RemoteLexeme(
            word,
            "INTJ",
            ("разговорное междометие удивления",),
            f"https://ru.wiktionary.org/wiki/{word}",
        )


async def test_remote_result_is_cached_in_local_dictionary(database: Database) -> None:
    lexicon = LexiconService(database)
    client = FakeClient()
    service = RussianReferenceService(
        database,
        lexicon,
        client,  # type: ignore[arg-type]
        enabled=True,
    )

    result = await service.refresh_word("шушпанчик")
    cached = await service.resolve_for_context("шушпанчик")
    analysis = await lexicon.analyze_word("шушпанчик")
    lookup = await database.fetchone(
        "SELECT status, source FROM language_reference_lookups WHERE normalized_form = ?",
        ("шушпанчик",),
    )

    assert result is not None
    assert cached is not None
    assert cached.meanings == result.meanings
    assert client.lookups == 1
    assert analysis.dictionary_meanings == ("разговорное междометие удивления",)
    assert analysis.style_labels == ("wiktionary_reference",)
    assert lookup is not None
    assert (lookup["status"], lookup["source"]) == ("found", "ru_wiktionary")
