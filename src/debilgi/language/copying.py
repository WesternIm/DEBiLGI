"""A repetition of input isn't a distributionally generated response."""

from collections import Counter

from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.tokenizer import tokenize


def is_input_copy(text: str, prompt: str) -> bool:
    def words(value: str) -> tuple[str, ...]:
        return tuple(
            DEFAULT_MORPHOLOGY.lemma(t)
            for t in tokenize(value)
            if any(char.isalnum() for char in t) and not t.startswith("<")
        )

    actual, source = words(text), words(prompt)
    if not actual or not source:
        return False
    perspective = {
        "я": "@person",
        "ты": "@person",
        "мы": "@persons",
        "вы": "@persons",
        "мой": "@possessive",
        "твой": "@possessive",
        "наш": "@possessives",
        "ваш": "@possessives",
    }
    actual = tuple(perspective.get(t, t) for t in actual)
    source = tuple(perspective.get(t, t) for t in source)
    if Counter(actual) == Counter(source):
        return True
    questions = {"почему", "зачем", "где", "когда", "как", "кто", "что", "какой"}
    if questions.intersection(actual) - questions.intersection(source):
        return False
    # Negating or affirming the proposition changes its truth conditions.  A
    # role-preserving polarity decision may reuse the topic and predicate, but
    # it is an actual stance rather than a cosmetic reordering of the input.
    if ("не" in actual) != ("не" in source):
        return False
    actual_content = {perspective.get(term, term) for term in content_lemmas(text)}
    source_content = {perspective.get(term, term) for term in content_lemmas(prompt)}
    if source_content and actual_content == source_content:
        return True
    shared = sum((Counter(actual) & Counter(source)).values())
    return len(source) >= 3 and shared / len(source) >= 0.9 and shared / len(actual) >= 0.8
