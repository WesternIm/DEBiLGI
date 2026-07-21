from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.language.semantic_consultant import (
    ConsultationRequest,
    ConsultationResult,
    SemanticConsultationRepository,
)
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "semantic-lessons.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def _request(prompt: str) -> ConsultationRequest:
    return ConsultationRequest(
        prompt=prompt,
        context_messages=(),
        known_terms=(),
        unknown_terms=("тарелка",),
        local_relations=(),
        local_confidence=0.2,
        reason_codes=("unknown_terms",),
        mode="lexical",
    )


def _result() -> ConsultationResult:
    return ConsultationResult.from_payload(
        {
            "interpretation": "Запрос о значении слова «тарелка».",
            "intent": "definition_question",
            "entities": [
                {"surface": "тарелка", "lemma": "тарелка", "type": "object"}
            ],
            "relations": [
                {"source": "тарелка", "relation": "is_a", "target": "посуда"}
            ],
            "ambiguities": [],
            "tokens": [
                {
                    "surface": "тарелка",
                    "lemma": "тарелка",
                    "part_of_speech": "noun",
                    "role": "object",
                    "features": {"case": "nominative", "number": "singular"},
                }
            ],
            "sentence_pattern": ["object"],
            "grammar_rules": ["Существительное называет предмет запроса."],
            "constructions": [
                {
                    "key": "object_first_possession",
                    "response_act": "relation_answer",
                    "relation": "possession",
                    "truth_state": "known",
                    "sequence": [
                        "polar_marker",
                        "comma",
                        "object:truth_case",
                        "existence_predicate",
                        "possession_preposition",
                        "subject:genitive",
                    ],
                }
            ],
            "expected_response_acts": ["brief_statement"],
            "register": "neutral",
            "facts": ["Тарелка — предмет посуды, из которого едят."],
            "response_text": "Тарелка — предмет посуды, из которого обычно едят.",
            "confidence": 0.94,
        },
        provider="fake-puter",
        model="fake-gemini",
    )


async def test_assimilation_keeps_candidates_and_applies_locally_supported_patterns(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))

    active = await service.assimilate(
        guild_id=1,
        request=_request("Что означает слово тарелка?"),
        result=_result(),
    )

    assert active >= 3
    lexical = await database.fetchone(
        """SELECT status, evidence_count FROM language_semantic_lessons
           WHERE lesson_kind = 'lexical'"""
    )
    assert lexical is not None
    assert lexical["status"] == "candidate"
    assert int(lexical["evidence_count"]) == 1
    assert await database.fetchone(
        """SELECT 1 FROM language_morphology_observations
           WHERE lemma = 'тарелка' AND grammatical_features_json LIKE '%semantic_consultant%'"""
    ) is not None
    assert await database.fetchone("SELECT 1 FROM language_grammar_patterns") is not None
    assert await database.fetchone("SELECT 1 FROM language_dialogue_act_patterns") is not None

    guidance = await service.guidance_for_prompt(1, "Что такое тарелка?")
    assert ("noun",) in guidance.syntax_patterns
    assert "brief_statement" in guidance.expected_response_acts


async def test_repeated_independent_lexical_evidence_enters_learning_graph(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))
    await service.assimilate(
        guild_id=1,
        request=_request("Что означает слово тарелка?"),
        result=_result(),
    )
    await service.assimilate(
        guild_id=1,
        request=_request("Объясни значение слова тарелка."),
        result=_result(),
    )

    lexical = await database.fetchone(
        """SELECT status, evidence_count FROM language_semantic_lessons
           WHERE lesson_kind = 'lexical'"""
    )
    assert lexical is not None
    assert lexical["status"] == "active"
    assert int(lexical["evidence_count"]) == 2
    assert await database.fetchone(
        """SELECT 1 FROM learning_graph_nodes
           WHERE guild_id = 1 AND node_type = 'concept' AND node_key = 'тарелка'"""
    ) is not None
    guidance = await service.guidance_for_prompt(1, "Расскажи про тарелку")
    assert "Тарелка — предмет посуды, из которого едят." in guidance.facts
    assert any(item.key == "object_first_possession" for item in guidance.constructions)


async def test_teacher_role_order_becomes_a_word_free_dependency_pattern(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))
    request = ConsultationRequest(
        prompt="Ты понял идею?",
        context_messages=(),
        known_terms=("ты", "понять", "идея"),
        unknown_terms=(),
        local_relations=(("ты", "понять", "идея"),),
        local_confidence=0.92,
        reason_codes=("background_dependency_induction",),
    )
    result = ConsultationResult.from_payload(
        {
            "interpretation": "Вопрос о понимании идеи.",
            "intent": "self_state_question",
            "tokens": [
                {
                    "surface": "ты",
                    "lemma": "ты",
                    "part_of_speech": "pronoun",
                    "role": "subject",
                    "features": {"case": "nominative"},
                },
                {
                    "surface": "понял",
                    "lemma": "понять",
                    "part_of_speech": "verb",
                    "role": "predicate",
                    "features": {"tense": "past"},
                },
                {
                    "surface": "идею",
                    "lemma": "идея",
                    "part_of_speech": "noun",
                    "role": "direct_object",
                    "features": {"case": "accusative"},
                },
            ],
            "sentence_pattern": ["subject", "predicate", "direct_object"],
            "grammar_rules": [],
            "facts": [],
            "confidence": 0.94,
        },
        provider="fake-puter",
        model="fake-gemini",
    )

    await service.assimilate(guild_id=1, request=request, result=result)
    guidance = await service.guidance_for_prompt(1, "Ты понял идею?")

    assert any(
        pattern.roles == ("subject", "predicate", "direct_object")
        for pattern in guidance.dependency_patterns
    )
    assert all(
        "понял" not in pattern.roles and "идея" not in pattern.roles
        for pattern in guidance.dependency_patterns
    )


async def test_local_parser_teaches_the_same_dependency_memory_without_ai(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))

    learned = await service.observe_local_structure(
        guild_id=1,
        structure=analyze_structured_utterance("Я помню наш разговор."),
    )
    guidance = await service.guidance_for_prompt(1, "Ты знаешь правило?")

    assert learned == 1
    assert any(
        pattern.roles == ("subject", "predicate", "direct_object")
        and pattern.source == "semantic_lesson"
        for pattern in guidance.dependency_patterns
    )


async def test_local_structure_persists_a_reusable_scored_construction(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))
    structure = analyze_structured_utterance("Я помню наш разговор.")

    await service.observe_local_structure(guild_id=1, structure=structure)
    await service.observe_local_structure(
        guild_id=1,
        structure=analyze_structured_utterance("Ты понимаешь новую идею."),
    )

    row = await database.fetchone(
        "SELECT * FROM language_productive_constructions WHERE guild_id = 1"
    )
    guidance = await service.guidance_for_prompt(1, "Я понимаю новую идею.")
    assert row is not None
    assert int(row["observation_count"]) == 2
    assert "помню" not in str(row["roles_json"])
    assert any(
        pattern.construction_id == row["construction_id"]
        and pattern.predicate_class == "lexical"
        and pattern.evidence_count == 2
        for pattern in guidance.dependency_patterns
    )


async def test_existing_syntax_lessons_backfill_productive_memory_once(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))
    await service.observe_local_structure(
        guild_id=1,
        structure=analyze_structured_utterance("Я помню наш разговор."),
    )
    await database.execute("DELETE FROM language_productive_constructions")

    assert await service.backfill_productive_constructions() == 1
    assert await service.backfill_productive_constructions() == 0
    row = await database.fetchone(
        "SELECT * FROM language_productive_constructions WHERE guild_id = 1"
    )
    assert row is not None
    assert row["source"] == "backfilled_semantic_lesson"


async def test_social_pretraining_stores_moves_without_authored_answers(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))

    installed = await service.install_social_curriculum(guild_id=1)
    guidance = await service.guidance_for_prompt(1, "У тебя есть носки?")

    assert installed >= 10
    assert any(
        policy.primary_move == "answer" and policy.answer_first
        for policy in guidance.social_policies
    )
    rows = await database.fetchall(
        """SELECT payload_json FROM language_semantic_lessons
           WHERE guild_id = 1 AND lesson_kind = 'dialogue'
             AND payload_json LIKE '%social_policy%'"""
    )
    assert rows
    assert all('"text"' not in str(row["payload_json"]) for row in rows)


async def test_productive_pretraining_stores_only_safe_abstract_roles(
    database: Database,
) -> None:
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))

    installed = await service.install_productive_curriculum(guild_id=1)
    repeated = await service.install_productive_curriculum(guild_id=1)
    rows = await database.fetchall(
        """SELECT roles_json, negation_mode, source
           FROM language_productive_constructions
           WHERE guild_id = 1 AND source = 'productive_curriculum'"""
    )

    assert installed >= 8
    assert repeated == 0
    assert rows
    assert all('"subject"' in str(row["roles_json"]) for row in rows)
    assert all('"predicate"' in str(row["roles_json"]) for row in rows)
    assert all(
        row["negation_mode"] != "required"
        or '"negation"' in str(row["roles_json"])
        for row in rows
    )


async def test_backfill_preserves_lessons_from_older_lexical_consultations(
    database: Database,
) -> None:
    request = _request("Что означает слово тарелка?")
    now = datetime.now(UTC)
    await SemanticConsultationRepository(database).save(
        event_id=None,
        guild_id=1,
        channel_id=10,
        request=request,
        result=_result(),
        now=now,
        ttl=timedelta(days=14),
    )
    service = SemanticAssimilationService(database, SelfDatasetRepository(database))

    assert await service.backfill() == 1
    assert await service.backfill() == 0
    assert await database.fetchone(
        """SELECT 1 FROM language_semantic_lessons
           WHERE guild_id = 1 AND lesson_kind = 'lexical'"""
    ) is not None
