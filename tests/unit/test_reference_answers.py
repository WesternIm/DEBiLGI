import pytest

from debilgi.language.reference_answers import (
    definition_action_thought,
    definition_head,
    definition_requested,
    same_reference_text,
    unrequested_definition,
)


@pytest.mark.parametrize(("prompt", "title", "expected"), [
    ("Как это понимать?", "понимать", False),
    ("Я это не спрашивал", "понимать", False),
    ("Почему вода кипит?", "вода", False),
    ("Найди, почему вода кипит", "вода", False),
    ("Что значит понимать?", "понимать", True),
    ("Объясни значение слова понимать", "понимать", True),
    ("Расскажи о драконе", "дракон", True),
    ("Изучи дракона", "дракон", True),
    ("Найди значение слова понимать", "понимать", True),
    ("Изучи что такое небо и скажи своё мнение", "небо", True),
    ("Что такое небо?", "мнение", False),
    ("Что это значит?", "понимать", False),
])
def test_definition_requires_requested_topic(prompt, title, expected):
    assert definition_requested(prompt, title) is expected


def test_reference_provenance_does_not_depend_on_dictionary_labels():
    definitions = (("понимать", "Постигать смысл высказывания."),)
    assert unrequested_definition("постигать смысл высказывания", "Как это понимать?", definitions)
    assert not unrequested_definition(
        "Постигать смысл высказывания.", "Что значит понимать?", definitions
    )


def test_old_accented_headword_is_recognized():
    assert definition_head("Отве́т — реплика на вопрос.") == "Ответ"
    assert same_reference_text("Отве́т — реплика.", "Ответ — реплика")


@pytest.mark.parametrize(("title", "definition", "action", "obj"), [
    ("понимать", "постигать смысл высказывания", "постигать", "смысл"),
    ("хранить", "сохранять старые письма", "сохранять", "письмо"),
])
def test_action_definition_is_compiled_to_changeable_roles(title, definition, action, obj):
    from dataclasses import replace

    from debilgi.language.productive_nlg import realize_thought

    thought = definition_action_thought(f"Что значит {title}?", title, definition)
    assert thought is not None
    roles = {dep.role: dep.argument for dep in thought.dependencies}
    assert roles["action_complement"].head.lemma == action
    assert roles["direct_object"].head.lemma == obj
    surface = realize_thought(thought, limit=1)[0]
    assert surface.text != definition
    assert "означает" in surface.text
    # The realizer consumes the roles. It does not fetch the evidence string.
    changed = replace(thought, dependencies=tuple(
        replace(dep, argument=replace(dep.argument, head=replace(
            dep.argument.head, lemma="изучать", surface="изучать",
        ))) if dep.role == "action_complement" else dep
        for dep in thought.dependencies
    ))
    assert "изучать" in realize_thought(changed, limit=1)[0].text
    assert action not in realize_thought(changed, limit=1)[0].text


@pytest.mark.parametrize("definition", [
    "не постигать смысл", "постигать смысл без объяснения",
    "постигать смысл, который скрыт", "постигать смысл или угадывать намерения",
])
def test_definition_composer_does_not_drop_semantic_restrictions(definition):
    assert definition_action_thought("Что значит понимать?", "понимать", definition) is None
