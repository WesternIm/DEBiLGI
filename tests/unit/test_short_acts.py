from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.short_acts import ShortSpeechActService, detect_short_speech_act
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("да", "confirmation"),
        ("ага", "confirmation"),
        ("именно", "confirmation"),
        ("ты понял", "confirmation"),
        ("нет же", "rejection"),
        ("не так", "rejection"),
        ("и?", "continue"),
        ("ну и?", "continue"),
        ("дальше", "continue"),
    ],
)
def test_explicit_short_speech_acts_preempt_full_parser(text: str, kind: str) -> None:
    detection = detect_short_speech_act(text)

    assert detection is not None
    assert detection.kind == kind
    assert detection.recognized is True
    assert detection.suppress_full_parse is True


@pytest.mark.parametrize(("text", "kind"), (("дааа", "confirmation"), ("нееет", "rejection")))
def test_stretched_short_acts_use_their_canonical_meaning(text: str, kind: str) -> None:
    detection = detect_short_speech_act(text)

    assert detection is not None
    assert detection.kind == kind
    assert detection.recognized


def test_complete_short_statements_and_unknown_terms_keep_full_parser() -> None:
    assert detect_short_speech_act("кот спит") is None
    assert detect_short_speech_act("monkey") is None
    assert detect_short_speech_act("monkey это обезьяна") is None
    assert detect_short_speech_act("ты ведь умный?") is None
    assert detect_short_speech_act("мордобой ты") is None


def test_unresolved_short_pronoun_uses_safe_coreference_fallback() -> None:
    detection = detect_short_speech_act("ты это")

    assert detection is not None
    assert detection.kind == "ambiguous"
    assert detection.recognized is False
    assert detection.coreference_to_bot is True
    assert detection.suppress_full_parse is True


class _ExperienceSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    async def apply_short_act(
        self, event: Any, *, response_message_id: int, accepted: bool
    ) -> bool:
        self.calls.append((response_message_id, accepted))
        return True


class _DatasetSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    async def record_short_act_feedback(
        self, event: Any, *, response_message_id: int, accepted: bool
    ) -> bool:
        self.calls.append((response_message_id, accepted))
        return True


async def test_short_act_binds_to_last_bot_turn_and_records_feedback(tmp_path: Path) -> None:
    database = Database(tmp_path / "short-acts.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        events = EventRepository(database)
        discourse = DiscourseMemoryRepository(database)
        experience = _ExperienceSpy()
        dataset = _DatasetSpy()
        service = ShortSpeechActService(database, experience, dataset)  # type: ignore[arg-type]
        created_at = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
        bot_turn = normalize_message_created(
            message_id=700,
            guild_id=1,
            channel_id=2,
            author_id=99,
            created_at=created_at,
            content="Я объяснил значение слова.",
            author_is_bot=True,
        )
        reaction = normalize_message_created(
            message_id=701,
            guild_id=1,
            channel_id=2,
            author_id=7,
            created_at=created_at + timedelta(seconds=5),
            content="нет же",
            author_is_bot=False,
        )
        await events.append(bot_turn)
        await discourse.observe(bot_turn)
        await events.append(reaction)

        await service.observe(reaction)
        resolution = await service.resolution_for_event(reaction.event_id)

        assert resolution is not None
        assert resolution.kind == "rejection"
        assert resolution.target_message_id == 700
        assert experience.calls == [(700, False)]
        assert dataset.calls == [(700, False)]
    finally:
        await database.close()


async def test_ambiguous_short_phrase_is_logged_without_false_feedback(tmp_path: Path) -> None:
    database = Database(tmp_path / "ambiguous-short-act.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        events = EventRepository(database)
        experience = _ExperienceSpy()
        dataset = _DatasetSpy()
        service = ShortSpeechActService(database, experience, dataset)  # type: ignore[arg-type]
        event = normalize_message_created(
            message_id=800,
            guild_id=1,
            channel_id=2,
            author_id=7,
            created_at=datetime(2026, 7, 18, 12, 0, tzinfo=UTC),
            content="ты это",
            author_is_bot=False,
        )
        await events.append(event)

        await service.observe(event)
        resolution = await service.resolution_for_event(event.event_id)

        assert resolution is not None
        assert resolution.kind == "ambiguous"
        assert resolution.reply_text() == "Это продолжение предыдущей мысли?"
        assert experience.calls == []
        assert dataset.calls == []
    finally:
        await database.close()


async def test_short_act_prefers_last_bot_turn_from_shared_working_context(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "working-short-act.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        events = EventRepository(database)
        experience = _ExperienceSpy()
        dataset = _DatasetSpy()
        working = WorkingMemoryContextGate(dataset)  # type: ignore[arg-type]
        service = ShortSpeechActService(
            database,
            experience,  # type: ignore[arg-type]
            dataset,  # type: ignore[arg-type]
            working_context=working,
        )
        created_at = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
        await working.commit_outgoing(
            guild_id=1,
            channel_id=2,
            message_id=900,
            text="Ключ означает предмет для отпирания замка.",
            bot_user_id=99,
            target_user_id=7,
            observed_at=created_at,
        )
        reaction = normalize_message_created(
            message_id=901,
            guild_id=1,
            channel_id=2,
            author_id=7,
            created_at=created_at + timedelta(seconds=1),
            content="именно",
            author_is_bot=False,
        )
        await events.append(reaction)

        await service.observe(reaction)
        resolution = await service.resolution_for_event(reaction.event_id)

        assert resolution is not None
        assert resolution.target_message_id == 900
        assert resolution.target_content == "Ключ означает предмет для отпирания замка."
        assert experience.calls == [(900, True)]
    finally:
        await database.close()
