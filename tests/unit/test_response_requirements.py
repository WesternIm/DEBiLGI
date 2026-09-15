import pytest

from debilgi.discord.replies import _previous_answer_grounding
from debilgi.language.arbitration import ArbitrationProfile, ResponseArbitrator, ResponseCandidate
from debilgi.language.grounded_repair import prior_claim_review_thought
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.response_requirements import requirement_rejection, response_requirement


@pytest.mark.parametrize("prompt", [
    "Почему дерево упало?", "А почему так считаешь?",
    "Ну типо почему ты так себя представляешь",
])
def test_reason_goal_survives_discourse_prefixes(prompt):
    assert response_requirement(prompt) == "explain_reason"


@pytest.mark.parametrize("source", ["semantic_reasoning", "recovery", "dialogue_move"])
def test_high_confidence_cannot_replace_a_reason_with_a_polarity(source):
    candidate = ResponseCandidate.create(
        event_id="one", source=source, renderer="test", text="Нет, я не представляю себя.",
        speech_act="question", source_confidence=1.0,
        metadata={"truth_grounding": 1.0, "semantic_frame_fit": 1.0},
    )
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id="one", prompt="Ну типо почему ты так себя представляешь",
        focus_concepts=(), candidates=(candidate,), recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is None
    assert "reason_required_not_polarity_or_word_meaning" in decision.scores[0].rejection_reasons


def test_explanation_requires_evidence_or_an_explicit_knowledge_gap():
    prompt = "Почему дерево упало?"
    assert requirement_rejection(prompt, {"answer_slot": "почему"}) is not None
    assert requirement_rejection(prompt, {
        "causal_direction": "cause", "causal_path": ("storm->tree_fall",),
    }) is None
    assert requirement_rejection(prompt, {"unresolved_requirement": "explain_reason"}) is None


@pytest.mark.parametrize("text", ["Я считаю себя хищником.", "Я не считаю себя хищником."])
def test_own_utterance_is_a_claim_not_verified_truth(text):
    facts = dict(_previous_answer_grounding((f"agent: {text}",)))
    assert "truth=" not in facts["previous_answer.statement"]
    assert "epistemic_status=unverified_claim" in facts["previous_answer.statement"]


def test_own_question_cannot_become_an_asserted_fact():
    assert _previous_answer_grounding(("agent: Ты считаешь себя хищником?",)) == ()


@pytest.mark.parametrize("history", [
    ("user: Я считаю себя хищником.",),
    ("agent: Ты считаешь себя хищником?",),
    ("agent: Я люблю музыку.",),
    (),
])
def test_review_cannot_invent_a_matching_own_claim(history):
    assert prior_claim_review_thought("Но ты сказал, что ты хищник", history) is None


def test_review_preserves_the_actual_claim_including_negation():
    claim = "Я не считаю себя хищником."
    thought, matched = prior_claim_review_thought(
        "Но ты сказал, что ты хищник", (f"agent: {claim}",),
    )
    assert matched == claim
    assert "«Я не считаю себя хищником»" in realize_thought(thought, limit=1)[0].text
