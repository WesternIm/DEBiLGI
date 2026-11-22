"""Stage 4.37 (Э1): evaluation / definition readers of parse_proposition, flag only."""

import pytest

from debilgi.language.evaluative_reading import evaluative_qualities
from debilgi.language.semantic_graph import parse_proposition


@pytest.fixture
def flag(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def read(text):
    return parse_proposition(text, allow_question=True, evaluation=True, definition=True)


def roles(thought):
    return {(d.role, d.argument.head.lemma) for d in thought.dependencies}


def test_short_adjective_is_a_quality_not_the_noun_right(flag):
    thought = read("Гитлер был прав?")
    assert ("predicate_complement", "правый") in roles(thought)
    assert [(s, q.lemma) for s, q, _ in evaluative_qualities(thought)] == [("гитлер", "правый")]


def test_evaluative_adjective_of_predicate_noun(flag):
    thought = read("Гитлер был плохим человеком")
    assert [(s, q.lemma) for s, q, _ in evaluative_qualities(thought)] == [("гитлер", "плохой")]
    assert ("predicate_complement", "человек") in roles(thought)


def test_definition_keeps_dependents(flag):
    thought = read("шампунь это средство для мытья волос")
    topic = next(d for d in thought.dependencies if d.role == "topic")
    assert topic.head == "predicate_complement" and topic.argument.preposition == "для"
    assert [m.lemma for m in topic.argument.postmodifiers] == ["волос"]
    relative = read("Гитлер это диктатор, который начал войну")
    assert ("relative_action", "начать") in roles(relative)
    assert ("direct_object", "война") in roles(relative)
    assert read("Гитлер — диктатор") is not None


def test_comparison(flag):
    thought = read("кошки лучше собак?")
    assert [(s, q.lemma, o.lemma) for s, q, o in evaluative_qualities(thought)] == [
        ("кошка", "хороший", "собака")
    ]


@pytest.mark.parametrize("text", [
    "Я Александр", "Спасибо, это правда", "— Привет, как дела", "нравится — мое качество)))",
    "Вася — иди сюда", "лучше пойти спать", "что это?", "кто лучше, кошки или собаки?",
])
def test_not_a_definition_or_evaluation(flag, text):
    assert read(text) is None


def test_readers_are_inert_without_flag_or_without_modes(monkeypatch):
    parse_proposition.cache_clear()  # the cache does not see the process flag
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "0")
    assert read("шампунь это средство для мытья волос") is None
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")
    assert parse_proposition("шампунь это средство для мытья волос") is None
    old = parse_proposition("Гитлер был прав?", allow_question=True)
    assert old.dependencies[1].argument.head.lemma == "право"
