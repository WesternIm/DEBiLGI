from decimal import Decimal

from debilgi.language.productive_nlg import realize_thought
from debilgi.language.quantitative_reasoning import resolve_travel_time_question
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_travel_time_is_derived_from_rate_and_distance_with_units() -> None:
    structure = analyze_structured_utterance(
        "А если машина едет со скоростью 200 км/ч, когда она доедет 654 километра"
    )

    resolution = resolve_travel_time_question(structure)

    assert resolution is not None
    assert resolution.value_seconds == Decimal("11772")
    assert realize_thought(resolution.thought)[0].text == (
        "Машина доедет за 3 часа 16 минут 12 секунд."
    )
    assert "rate_equation:time=distance/speed" in resolution.evidence


def test_travel_time_converts_meters_per_second() -> None:
    structure = analyze_structured_utterance(
        "Если робот движется со скоростью 10 м/с, когда он пройдёт 150 метров"
    )

    resolution = resolve_travel_time_question(structure)

    assert resolution is not None
    assert resolution.value_seconds == Decimal("15")
    assert realize_thought(resolution.thought)[0].text == ("Робот пройдёт за 15 секунд.")


def test_travel_time_rejects_zero_speed() -> None:
    structure = analyze_structured_utterance(
        "Если машина едет со скоростью 0 км/ч, когда она доедет 10 километров"
    )

    assert resolve_travel_time_question(structure) is None
