from pathlib import Path

import pytest

from debilgi.language.semantic_consultant import (
    ConsultationRequest,
    ConsultationResult,
    SemanticConsultationService,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


class _Consultant:
    def __init__(self, confidence: float = 0.8) -> None:
        self.calls = 0
        self.confidence = confidence

    async def consult(self, request: ConsultationRequest) -> ConsultationResult:
        self.calls += 1
        return ConsultationResult.from_payload(
            {
                "interpretation": "Вопрос о значении сленгового слова.",
                "intent": "definition_question",
                "entities": [{"surface": "база", "lemma": "база", "type": "abstract"}],
                "relations": [],
                "ambiguities": [],
                "facts": ["В сленге слово выражает одобрение."],
                "response_text": "Здесь «база» означает согласие или одобрение.",
                "confidence": self.confidence,
            },
            provider="fake-puter",
            model="fake-model",
        )

    async def close(self) -> None:
        return None


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "consultations.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def _request() -> ConsultationRequest:
    return ConsultationRequest(
        prompt="Это база?",
        context_messages=("Обсуждаем странное мнение.",),
        known_terms=("это",),
        unknown_terms=("база",),
        local_relations=(),
        local_confidence=0.35,
        reason_codes=("unknown_terms",),
    )


async def test_consultation_is_cached_and_stays_out_of_learning_graph(
    database: Database,
) -> None:
    provider = _Consultant()
    service = SemanticConsultationService(database, provider, enabled=True)

    first = await service.consult(
        event_id="discord:one",
        guild_id=1,
        channel_id=10,
        request=_request(),
    )
    second = await service.consult(
        event_id="discord:two",
        guild_id=1,
        channel_id=10,
        request=_request(),
    )

    assert first is not None and not first.cached
    assert second is not None and second.cached
    assert provider.calls == 1
    row = await database.fetchone(
        "SELECT status, occurrence_count FROM language_semantic_consultations"
    )
    assert row is not None
    assert row["status"] == "provisional"
    assert int(row["occurrence_count"]) == 1
    assert await database.fetchone(
        "SELECT 1 FROM learning_graph_nodes WHERE label LIKE '%база%'"
    ) is None


async def test_low_confidence_consultation_is_not_persisted(database: Database) -> None:
    provider = _Consultant(confidence=0.3)
    service = SemanticConsultationService(
        database,
        provider,
        enabled=True,
        minimum_result_confidence=0.6,
    )

    result = await service.consult(
        event_id="discord:low",
        guild_id=1,
        channel_id=10,
        request=_request(),
    )

    assert result is None
    assert await database.fetchone(
        "SELECT 1 FROM language_semantic_consultations"
    ) is None


async def test_background_lexical_consultation_can_answer_later_without_second_call(
    database: Database,
) -> None:
    provider = _Consultant()
    service = SemanticConsultationService(database, provider, enabled=True)
    lexical = ConsultationRequest(
        prompt="Что означает слово «база» и как оно употребляется?",
        context_messages=(),
        known_terms=(),
        unknown_terms=("база",),
        local_relations=(),
        local_confidence=0.0,
        reason_codes=("background_lexical_sources_exhausted",),
        mode="lexical",
    )
    await service.consult(
        event_id=None,
        guild_id=1,
        channel_id=10,
        request=lexical,
    )

    reused = await service.consult(
        event_id="discord:later",
        guild_id=1,
        channel_id=10,
        request=ConsultationRequest(
            prompt="Что значит слово база?",
            context_messages=(),
            known_terms=("слово",),
            unknown_terms=("база",),
            local_relations=(),
            local_confidence=0.35,
            reason_codes=("unknown_terms",),
        ),
    )

    assert reused is not None and reused.cached
    assert reused.response_text is None
    assert reused.facts == ("В сленге слово выражает одобрение.",)
    assert provider.calls == 1


async def test_lexical_cache_does_not_replace_full_utterance_intent(
    database: Database,
) -> None:
    provider = _Consultant()
    service = SemanticConsultationService(database, provider, enabled=True)
    await service.consult(
        event_id=None,
        guild_id=1,
        channel_id=10,
        request=ConsultationRequest(
            prompt="Что означает слово «ключ»?",
            context_messages=(),
            known_terms=(),
            unknown_terms=("ключ",),
            local_relations=(),
            local_confidence=0.0,
            reason_codes=("background_lexical_sources_exhausted",),
            mode="lexical",
        ),
    )

    result = await service.consult(
        event_id="discord:experience",
        guild_id=1,
        channel_id=10,
        request=ConsultationRequest(
            prompt="Ты терял когда-то ключи?",
            context_messages=("user: расскажи о своём опыте",),
            known_terms=("терять",),
            unknown_terms=("ключ",),
            local_relations=(("ты", "терять", "ключ"),),
            local_confidence=0.7,
            reason_codes=("no_reply_bound_local_candidate",),
        ),
    )

    assert result is not None
    assert provider.calls == 2
