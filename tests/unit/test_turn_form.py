from datetime import UTC, datetime

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.dialogue_acts import analyze_turn_form


def test_factual_followup_with_predicate_does_not_become_own_turn_repair():
    form = analyze_turn_form("что я сегодня читал?", addressed=True, follows_bot=True,
                             previous_bot_text="Читал ты книгу вчера.")
    assert form.kind == "question"


def test_repeated_own_turn_explanation_keeps_the_original_quote():
    from dataclasses import asdict
    from types import SimpleNamespace

    from debilgi.language.dialogue_learning import own_turn_plan
    from debilgi.language.structured_utterance import analyze_structured_utterance

    prior = {"plan": {}, "decision": {}, "operations": {},
             "prompt_event_id": "sent:0", "response_text": "ты отдыхал вчера?"}
    context = SimpleNamespace(prior_bot_turn=prior,
                              structure=analyze_structured_utterance("почему?"))
    quotes = []
    for i in range(5):
        proposal = own_turn_plan(context, None)
        quotes.append(next(d.argument.head.surface for d in proposal.plan.claim.dependencies
                           if d.role == "direct_object"))
        prior.update(plan=asdict(proposal.plan), prompt_event_id=f"sent:{i + 1}",
                     response_text="Я сказал " + quotes[-1])
    assert len(set(quotes)) == 1


@pytest.mark.parametrize("text,kind", [
    ("эй", "call"), ("алоу бро", "call"), ("ты зумбрастый", "evaluation"),
    ("я зумбрал штуку", "self_report"), ("зумбраст?", "question"),
    ("зумбруй", "request"), ("аузын жап котакбас", "fragment"),
])
def test_turn_form_does_not_need_morphology(monkeypatch, text, kind):
    from debilgi.language.morphology import RussianMorphology

    def forbidden(*args, **kwargs):
        raise AssertionError("form recognition requested dictionary analysis")

    monkeypatch.setattr(RussianMorphology, "parse_word", forbidden)
    assert analyze_turn_form(text).kind == kind


def test_address_and_reply_are_available_before_meaning():
    event = normalize_message_created(
        message_id=1, guild_id=1, channel_id=2, author_id=3,
        created_at=datetime(2026, 11, 15, tzinfo=UTC), content="ЗУМБРАСТ!!!)",
        author_is_bot=False, bot_user_id=99, reply_to_author_id=99, reply_to_message_id=7,
    )
    form = event.turn_form
    assert event._meaning is None
    assert form.addressed and form.follows_bot and "reply_to_bot" in form.address_evidence
    assert {"caps", "emphatic", "smile"} <= set(form.tone)


def test_habitual_coarseness_and_tone_change_are_distinct():
    ordinary = analyze_turn_form("ну ты бля крутой", habitual_tone=("coarse",))
    assert "coarse" in ordinary.tone and "coarse" not in ordinary.tone_change
    assert "coarse" in analyze_turn_form("ну ты бля крутой").tone_change
