from debilgi.cognition.difference import categorical_difference
from debilgi.cognition.expectations import (
    ExpectationPredictor,
    OutcomeType,
    PatternEvidence,
    Situation,
)


def situation(**changes: object) -> Situation:
    values = {
        "trigger_type": "MESSAGE_CREATED",
        "author_is_bot": False,
        "is_reply": False,
        "has_mentions": False,
        "channel_activity": "low",
        "actor_charge": "low",
        "message_length": "medium",
        "actor_message_rate": "low",
        "actor_uppercase_ratio": "none",
    }
    values.update(changes)
    return Situation(**values)


def test_untrained_prediction_uses_uniform_prior() -> None:
    prediction = ExpectationPredictor().predict(situation(), [])

    assert prediction.outcome is OutcomeType.NO_RESPONSE
    assert prediction.probability == 0.2
    assert set(prediction.distribution.values()) == {0.2}
    assert prediction.evidence_support == 0.0


def test_exact_situation_has_full_similarity() -> None:
    predictor = ExpectationPredictor()

    assert predictor.similarity(situation(), situation()) == 1.0


def test_different_trigger_type_is_not_considered_similar() -> None:
    predictor = ExpectationPredictor()

    assert predictor.similarity(situation(), situation(trigger_type="REACTION_ADDED")) == 0.0


def test_similar_situations_contribute_weighted_evidence() -> None:
    predictor = ExpectationPredictor()
    evidence = [
        PatternEvidence(
            situation=situation(channel_activity="medium"),
            outcome=OutcomeType.REPLY,
            observations=9,
        )
    ]

    prediction = predictor.predict(situation(), evidence)

    assert prediction.outcome is OutcomeType.REPLY
    assert prediction.probability > 0.6
    assert prediction.evidence_support > 0


def test_difference_is_one_minus_probability_of_actual_outcome() -> None:
    distribution = {
        OutcomeType.NO_RESPONSE: 0.1,
        OutcomeType.OTHER_MESSAGE: 0.1,
        OutcomeType.SAME_ACTOR_MESSAGE: 0.1,
        OutcomeType.REPLY: 0.6,
        OutcomeType.REACTION: 0.1,
    }

    assert categorical_difference(distribution, OutcomeType.REPLY) == 0.4
    assert categorical_difference(distribution, OutcomeType.REACTION) == 0.9


def test_situation_json_round_trip_is_stable() -> None:
    original = situation(has_mentions=True)

    assert Situation.from_json(original.to_json()) == original
