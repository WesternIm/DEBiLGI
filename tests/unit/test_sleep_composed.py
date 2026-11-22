from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.language.dialogue_learning import reaction_slot_tokens
from debilgi.language.structured_utterance import (
    analytic_future_events,
    analyze_structured_utterance,
)


def test_analytic_future_names_the_infinitive_not_the_auxiliary():
    structure = analyze_structured_utterance("так и будешь молчать?")
    events = analytic_future_events(structure)
    assert [t.lemma for t in events.values()] == ["молчать"]
    assert not analytic_future_events(analyze_structured_utterance("ты будешь дома"))


def test_predicate_slot_binds_the_event_not_the_auxiliary():
    structure = analyze_structured_utterance("так и будешь молчать?")
    slot = {"source_role": "predicate", "part": "verb", "features": {}, "lemma": "заниматься"}
    lemmas = [t.lemma for t in reaction_slot_tokens(structure, slot)]
    assert "быть" not in lemmas
    assert lemmas == ["молчать"]


def test_only_an_admission_of_not_knowing_counts_as_unknown():
    assert ContinuousCognitionService.admits_not_knowing("Я не знаю, почему я рад.")
    assert not ContinuousCognitionService.admits_not_knowing("чем будешь?!")
