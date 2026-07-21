from debilgi.knowledge.policy import KnowledgeQueryPolicy, canonical_lexical_query
from debilgi.knowledge.service import concise_reference


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


def test_reference_dump_is_limited_to_two_sentences_with_a_more_note() -> None:
    result = concise_reference(
        "Первое определение. Второе полезное уточнение. "
        "Третья длинная часть статьи уже не должна попадать в ответ целиком."
    )

    assert result.startswith("Первое определение. Второе полезное уточнение.")
    assert "Третья" not in result
    assert result.endswith("Если нужно подробнее — уточни.")
    assert len(result) <= 420


def test_lexical_query_uses_contextual_exact_lemma_and_safe_typo_correction() -> None:
    assert canonical_lexical_query("варенье", "что такое варенье?") == "варенье"
    assert canonical_lexical_query("варение", "что такое варение?") == "варение"
    assert canonical_lexical_query("таоелка", "что такое таоелка?") == "тарелка"
