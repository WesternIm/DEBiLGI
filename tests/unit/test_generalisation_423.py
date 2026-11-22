"""Stage 4.23: the mechanisms behind fantasy and the variety of not-knowing, on unseen material."""

import pytest

from debilgi.language.arbitration import ArbitrationProfile, ResponseArbitrator, ResponseCandidate
from debilgi.language.communication import (
    MessagePlan,
    ResponseComposer,
    construction_repetition,
    is_amusement_marker,
    plan_issues,
    reader_goal,
    surface_construction,
)
from debilgi.language.lexical_matching import verb_family
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.request_focus import is_invention_request
from debilgi.language.structured_utterance import analyze_structured_utterance


def thought(verb="съесть", *, mood=True, obj="торт", elided=False):
    features = (("tense", "past"), ("gender", "masculine"), ("number", "singular"))
    if mood:
        features += (("mood", "conditional"),)
    subject = ConceptLexeme("я", "я", "pronoun", (("elided", "true"),) if elided else ())
    deps = [SemanticDependency("subject", ConceptArgument(subject))]
    if obj:
        deps.append(SemanticDependency("direct_object", ConceptArgument(
            ConceptLexeme(obj, obj, "noun", (("number", "singular"),)),
            grammatical_case="accusative")))
    return ClauseThought(ConceptLexeme(verb, verb, "verb", features), tuple(deps),
                         confidence=0.7, evidence=("m",))


def fantasy(claim, **hypothesis):
    pairs = (("kind", "fantasy"), *hypothesis.items())
    goal = reader_goal(analyze_structured_utterance("а ты бы?"))
    return MessagePlan(goal, claim, speech_act="opinion", hypothesis=pairs,
                       mixture=(("fantasy", 1.0),))


# ------------------------------------------------------------- the clause skeleton

@pytest.mark.parametrize("left,right", [
    ("Я не знаю, летал ли я на самолёте.", "Я не знаю, что я хочу."),
    ("Я не знаю, ел ли я пиццу.", "Я не знаю, хотел ли бы я полететь."),
    ("Я проснулся.", "Я заснул."),
    ("Я не знаю ответа.", "Я не знаю ответа на этот вопрос."),
])
def test_one_construction_whatever_the_verbs_and_nouns(left, right):
    assert surface_construction(left) == surface_construction(right)


@pytest.mark.parametrize("left,right", [
    ("Я не знаю, летал ли я.", "Я не знаю ответа на этот вопрос."),
    ("Я не знаю, летал ли я.", "Не знаю, летал ли я."),
    ("Я проснулся.", "Я сказал «Я проснулся.»."),
    ("Я не знаю ответа.", "Я знаю ответ."),
    ("Я купил хлеб.", "Что я купил?"),
])
def test_different_skeletons_are_different_constructions(left, right):
    assert surface_construction(left) != surface_construction(right)


def test_repetition_weighs_the_nearest_reply_most():
    recent = ("Я не знаю, ел ли я.", "Я проснулся.")
    assert construction_repetition("Я не знаю, спал ли я.", recent) == 1.0
    assert construction_repetition("Я не знаю, спал ли я.", recent[::-1]) == 0.5
    assert construction_repetition("Не знаю, спал ли я.", recent) == 0.0
    assert construction_repetition("Я не знаю, спал ли я.", ()) == 0.0


# ------------------------------------------------------------- invention requests

@pytest.mark.parametrize("text", [
    "пошути", "придумай что-нибудь", "расскажи что-нибудь смешное", "сочини стих",
    "придумывай", "шути", "сочиняй", "выдумай что-то", "пофантазируй", "давай придумай что-нибудь",
    "расскажи анекдот", "пошути, пожалуйста",
])
def test_requests_to_make_something_up_in_any_aspect(text):
    assert is_invention_request(text) is True


@pytest.mark.parametrize("text", [
    "расскажи про кошку", "запомни", "ответь", "придумай имя коту", "расскажи, как дела?",
    "ты умеешь шутить?", "он пошутил", "сделай что-нибудь", "придумай мне сказку про дракона",
])
def test_requests_for_information_or_other_acts_are_not_invitations_to_play(text):
    assert is_invention_request(text) is False


def test_a_bare_tell_may_only_ask_to_go_on_with_the_answer_before_it():
    assert is_invention_request("расскажи") is True
    assert is_invention_request("расскажи", bare=False) is False
    assert is_invention_request("расскажи что-нибудь", bare=False) is True


# ------------------------------------------------------------- marks of play

def test_laughter_cues_are_read_by_the_pragmatic_reader_not_listed():
    assert all(is_amusement_marker(w) for w in ("😂", "🤣", "😹", "ахахах", "лол"))
    assert not any(is_amusement_marker(w) for w in ("привет", "наверное", "😭", "дом"))


def test_a_fantasy_is_honest_only_when_marked_as_play():
    assert plan_issues(fantasy(thought())) == ()  # «бы»
    plain = thought(mood=False)
    assert plan_issues(fantasy(plain)) == ("unmarked_fantasy",)
    assert plan_issues(fantasy(plain, operator="😂", operator_source="m")) == ()
    assert plan_issues(fantasy(plain, operator="наверное", operator_source="m")) == (
        "unmarked_fantasy",)  # a hedge is not a laughter marker
    assert plan_issues(fantasy(plain, operator="😂", operator_source="other")) == (
        "unmarked_fantasy",)  # the marker must be one the agent actually observed
    about_someone_else = ClauseThought(
        plain.predicate, (SemanticDependency("subject", ConceptArgument(
            ConceptLexeme("ты", "ты", "pronoun"))),), confidence=0.7)
    assert plan_issues(fantasy(about_someone_else, operator="😂", operator_source="m")) == (
        "unmarked_fantasy",)


@pytest.mark.parametrize("verb,obj,expected", [
    ("съесть", "торт", "Я съел бы торт"),
    ("купить", "машина", "Я купил бы машину"),
    ("построить", "дом", "Я построил бы дом"),
])
def test_the_composer_says_the_unreal_with_the_grammar_of_the_unreal(verb, obj, expected):
    composer = ResponseComposer(analyze_structured_utterance("а ты бы?"))
    texts = [s.text.rstrip(".") for s in composer.compose(fantasy(thought(verb, obj=obj)))]
    assert expected in texts


def test_the_learned_laughter_marker_closes_the_statement():
    composer = ResponseComposer(analyze_structured_utterance("а ты бы?"))
    plan = fantasy(thought("купить", obj="машина"), operator="🤣", operator_source="m")
    assert "Я купил бы машину 🤣" in [s.text for s in composer.compose(plan)]


def test_the_person_may_be_carried_by_the_verb_alone():
    plain = [s.text for s in realize_thought(thought("знать", mood=False, obj=None))]
    carried = [s.text for s in realize_thought(thought("знать", mood=False, obj=None,
                                                       elided=True))]
    assert "Я знал." in plain and "Знал." in carried
    assert all(not t.startswith("Я ") for t in carried)


# ------------------------------------------------------------- verbs of one family

@pytest.mark.parametrize("left,right", [
    ("летать", "полететь"), ("лететь", "летать"), ("улететь", "лететь"),
    ("делать", "сделать"), ("строить", "построить"), ("читать", "прочитать"),
])
def test_one_verb_in_any_aspect_prefix_or_determinacy(left, right):
    assert verb_family(left, right) is True


@pytest.mark.parametrize("left,right", [
    ("любить", "ловить"), ("писать", "плакать"), ("хотеть", "мочь"), ("мыть", "мыться"),
])
def test_other_verbs_are_not_one_family(left, right):
    assert verb_family(left, right) is False


# ------------------------------------------------------------- the arbiter weighs constructions

def limit_candidate(text):
    return ResponseCandidate.create(
        event_id="event:one", source="semantic_reasoning", renderer="evidence_planner:limit",
        text=text, speech_act="brief_statement", source_confidence=0.8,
        metadata={"communicative_status": "express_limit", "locally_composed": True})


def test_the_construction_used_a_moment_ago_loses_to_another_way_of_not_knowing():
    first = limit_candidate("Я не знаю, ел ли я пиццу.")
    other = limit_candidate("Я не знаю ответа на этот вопрос.")
    decision = ResponseArbitrator(minimum_score=-10.0).decide(
        event_id="event:one", prompt="а ты ел пиццу?", focus_concepts=("пицца",),
        candidates=(first, other), recent_replies=("Я не знаю, спал ли я.",),
        profile=ArbitrationProfile.baseline(1))
    assert decision.winner.text == other.text
    scores = {s.candidate.text: s for s in decision.scores}
    assert scores[first.text].construction_repeat == 1.0
    assert scores[other.text].construction_repeat == 0.0
    fresh = ResponseArbitrator(minimum_score=-10.0).decide(
        event_id="event:one", prompt="а ты ел пиццу?", focus_concepts=("пицца",),
        candidates=(first, other), recent_replies=(), profile=ArbitrationProfile.baseline(1))
    assert all(s.construction_repeat == 0.0 for s in fresh.scores)


def test_a_fact_is_not_weighed_against_a_familiar_clause_shape():
    fact = ResponseCandidate.create(
        event_id="event:one", source="semantic_reasoning", renderer="x",
        text="Я съел торт.", speech_act="brief_statement", source_confidence=0.8,
        metadata={"communicative_status": "inform"})
    decision = ResponseArbitrator(minimum_score=-10.0).decide(
        event_id="event:one", prompt="что ты ел?", focus_concepts=("торт",),
        candidates=(fact,), recent_replies=("Я проснулся.",),
        profile=ArbitrationProfile.baseline(1))
    assert decision.scores[0].construction_repeat == 0.0


def test_a_personal_subject_behind_its_verb_never_beats_the_neutral_wording_of_the_same_plan():
    def worded(text, order):
        return ResponseCandidate.create(
            event_id="event:one", source="semantic_reasoning", renderer="x", text=text,
            speech_act="brief_statement", source_confidence=0.8,
            metadata={"role_order": order, "message_plan_key": "same-plan",
                      "communicative_status": "inform", "locally_composed": True})

    neutral = worded("Я построил бы дом.", ("subject", "predicate", "direct_object"))
    marked = worded("Построил бы я дом.", ("predicate", "subject", "direct_object"))
    for recent in ((), ("Я построил бы дом.",)):  # even when the neutral one was just said
        decision = ResponseArbitrator(minimum_score=-10.0).decide(
            event_id="event:one", prompt="что бы ты построил?", focus_concepts=("дом",),
            candidates=(marked, neutral), recent_replies=recent,
            profile=ArbitrationProfile.baseline(1))
        assert decision.winner.text == neutral.text


def test_a_reader_check_accepts_the_person_carried_by_the_verb_only_when_it_is_marked_so():
    from debilgi.language.communication import surface_issues

    elided = thought("знать", mood=False, obj=None, elided=True)
    assert surface_issues(elided, "Знал.") == ()
    plain = thought("знать", mood=False, obj=None)
    assert surface_issues(plain, "Знал.") == ("missing_subject",)
