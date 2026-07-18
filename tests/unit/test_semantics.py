from debilgi.language.semantics import extract_semantic_frames, lexeme_key


def test_extracts_actor_action_object_and_location() -> None:
    frames = extract_semantic_frames("Лиса взяла красный ключ со стола.")

    assert len(frames) == 1
    frame = frames[0]
    assert frame.actor == "лиса"
    assert frame.action == "взяла"
    assert frame.target == "красный ключ"
    assert ("со", "стола") in frame.modifiers


def test_inherits_actor_across_coordinated_actions_and_keeps_negation() -> None:
    frames = extract_semantic_frames("Лиса взяла ключ и не положила его в ящик.")

    assert len(frames) == 2
    assert frames[1].actor == "лиса"
    assert frames[1].action == "положила"
    assert frames[1].target == "его"
    assert frames[1].negated
    assert ("в", "ящик") in frames[1].modifiers


def test_lexeme_key_reduces_simple_russian_inflection() -> None:
    assert lexeme_key("ключ") == lexeme_key("ключа")
    assert lexeme_key("коробка") == lexeme_key("коробке")


def test_question_is_not_stored_as_observed_fact() -> None:
    assert extract_semantic_frames("Кто положил ключ в ящик?") == ()
