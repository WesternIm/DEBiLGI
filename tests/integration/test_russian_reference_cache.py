import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.language.lexicon import LexiconService
from debilgi.language.russian_reference import (
    RemoteLexeme,
    RussianReferenceService,
    WiktionaryUnavailable,
)
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


async def test_foreground_timeout_leaves_shared_lookup_running(database):
    gate = asyncio.Event()

    class DelayedClient(FakeClient):
        async def lookup(self, word):
            await gate.wait()
            return await super().lookup(word)

    client = DelayedClient()
    lexicon = LexiconService(database)
    reference = RussianReferenceService(database, lexicon, client, enabled=True)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reference.resolve_for_context("кваргония"), timeout=0.02)
        assert await reference.retry_delay_for(("кваргония",)) == 5
        gate.set()
        result = await reference.resolve_for_context("кваргония")
        assert result is not None
        assert client.lookups == 1
        assert await reference.retry_delay_for(("кваргония",)) is None
        assert (await lexicon.analyze_word("кваргония")).has_usable_meaning
    finally:
        await reference.stop()


async def test_missing_entry_is_terminal_but_transport_error_is_retryable(database):
    class MissingClient(FakeClient):
        async def lookup(self, word):
            return None

    reference = RussianReferenceService(
        database, LexiconService(database), MissingClient(), enabled=True,
    )
    assert await reference.resolve_for_context("кваргония") is None
    assert await reference.retry_delay_for(("кваргония",)) is None

    class OfflineClient(FakeClient):
        async def lookup(self, word):
            raise OSError("offline")

    offline = RussianReferenceService(
        database, LexiconService(database), OfflineClient(), enabled=True,
    )
    assert await offline.resolve_for_context("стромба") is None
    assert 1 <= await offline.retry_delay_for(("стромба",)) <= 300


async def test_access_denial_is_recorded_as_error_with_actionable_diagnostics(database, caplog):
    class BlockedClient(FakeClient):
        async def lookup(self, word):
            raise WiktionaryUnavailable("Check Wikimedia User-Agent/contact", status=403)

    service = RussianReferenceService(database, LexiconService(database), BlockedClient())
    assert await service.refresh_word("ручка") is None
    row = await database.fetchone(
        "SELECT status FROM language_reference_lookups WHERE normalized_form = ?", ("ручка",)
    )
    assert row["status"] == "error"
    assert "User-Agent/contact" in caplog.text
    assert "403" in caplog.text


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


@pytest.mark.parametrize("status,age,refresh", [
    ("error", timedelta(minutes=1), False), ("error", timedelta(minutes=6), True),
    ("missing", timedelta(hours=2), False), ("missing", timedelta(days=2), True),
    ("found", timedelta(days=2), False),
])
async def test_temporary_failure_does_not_poison_dictionary_for_a_month(
    database, status, age, refresh,
):
    lexicon = LexiconService(database)
    client = FakeClient()
    service = RussianReferenceService(database, lexicon, client, enabled=True)
    await service._record("ручка", status, None)
    await database.execute(
        "UPDATE language_reference_lookups SET checked_at = ? WHERE normalized_form = ?",
        ((datetime.now(UTC) - age).isoformat(), "ручка"),
    )
    assert await service._needs_refresh("ручка") is refresh
