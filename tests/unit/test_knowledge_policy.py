from types import SimpleNamespace

from debilgi.events.models import EventType
from debilgi.knowledge.policy import (
    KnowledgeQueryPolicy,
    _clean_lookup_query,
    canonical_lexical_query,
)
from debilgi.knowledge.service import (
    _encyclopedia_query_variants,
    _title_matches_query,
    concise_reference,
)
from debilgi.knowledge.text import normalize_key


def test_lookup_command_separates_address_from_query_subject() -> None:
    followup = KnowledgeQueryPolicy.followup_query

    assert followup("узнай, умник") == (True, None)
    assert followup("узнай умник") == (True, None)
    assert followup("узнай, Маша") == (True, None)
    assert followup("узнай значение слова умник") == (True, "умник")
    assert followup("узнай что значит умник") == (True, "умник")
    assert followup("узнай что значит слово умник") == (True, "умник")
    assert followup("найди умника") == (True, "умника")
    assert followup("умник узнай") == (True, None)
    assert followup("умник узнай что значит мороженое") == (True, "мороженое")
    assert followup("изучи GTA 5 и ответь") == (True, "gta 5")
    assert followup("ну так узнай что это") == (True, None)


def test_deictic_lookup_waits_for_dialogue_resolution_instead_of_searching_pronouns() -> None:
    policy = KnowledgeQueryPolicy(enabled=True)

    decision = policy.decide(
        SimpleNamespace(
            channel_id=10,
            actor_id=20,
            event_type=EventType.MESSAGE_CREATED,
            payload={
                "content": "Ну так узнай что это",
                "author_is_bot": False,
            },
        ),
        None,
    )

    assert not decision.should_query
    assert decision.query is None
    assert decision.reason == "contextual_followup"


def test_entity_lookup_preserves_numeric_discriminator_and_strips_next_action() -> None:
    assert normalize_key("GTA 5") == "gta 5"
    assert _clean_lookup_query("гта 5 и ответь") == "гта 5"


def test_acronym_lookup_can_match_expanded_media_title() -> None:
    assert _encyclopedia_query_variants("гта 5") == ("гта 5", "gta 5")
    assert _title_matches_query("GTA 5", "Grand Theft Auto V")
    assert normalize_key("ИИ") == "ии"
    assert _title_matches_query("ИИ", "Искусственный интеллект")


def test_reference_dump_is_limited_to_two_sentences_with_a_more_note() -> None:
    result = concise_reference(
        "Первое определение. Второе полезное уточнение. "
        "Третья длинная часть статьи уже не должна попадать в ответ целиком."
    )

    assert result.startswith("Первое определение. Второе полезное уточнение.")
    assert "Третья" not in result
    assert "Если нужно подробнее" not in result
    assert len(result) <= 420
    assert len(result) <= 420


def test_lexical_query_uses_contextual_exact_lemma_and_safe_typo_correction() -> None:
    assert canonical_lexical_query("варенье", "что такое варенье?") == "варенье"
    assert canonical_lexical_query("варение", "что такое варение?") == "варение"
    assert canonical_lexical_query("таоелка", "что такое таоелка?") == "тарелка"
