import pytest

from debilgi.language.productive_nlg import realize_thought
from debilgi.language.spatial_dynamics import (
    realize_boundary_entry_explanation,
    resolve_boundary_entry_explanation,
    resolve_boundary_entry_question,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


@pytest.mark.parametrize(
    ("direction", "edge"),
    (
        ("налево", "справа"),
        ("вправо", "слева"),
        ("вверх", "снизу"),
        ("вниз", "сверху"),
    ),
)
def test_entry_edge_is_inferred_from_opposite_motion_direction(
    direction: str,
    edge: str,
) -> None:
    structure = analyze_structured_utterance(
        f"Вагонетка мчится {direction}, откуда она появится на экране сперва"
    )

    resolution = resolve_boundary_entry_question(structure)

    assert resolution is not None
    assert edge in resolution.focus_terms
    assert realize_thought(resolution.thought)[0].text == (f"Вагонетка появится {edge}.")
    assert "entry_edge_opposes_motion_direction" in resolution.evidence


def test_motion_rule_requires_a_bounded_view_question() -> None:
    structure = analyze_structured_utterance("Вагонетка мчится налево, откуда она появилась вчера")

    assert resolve_boundary_entry_question(structure) is None


def test_motion_rule_does_not_bind_an_incompatible_pronoun() -> None:
    structure = analyze_structured_utterance(
        "Вагонетка мчится налево, откуда он появится на экране"
    )

    assert resolve_boundary_entry_question(structure) is None


def test_reason_follow_up_explains_the_previous_motion_deduction() -> None:
    structure = analyze_structured_utterance("Почему же оттуда поедет")

    explanation = resolve_boundary_entry_explanation(
        structure,
        (
            "Вагонетка мчится налево, откуда она появится на экране сперва",
            "Вагонетка появится справа.",
        ),
    )

    assert explanation is not None
    surface = realize_boundary_entry_explanation(explanation)[0].text
    assert surface == ("Вагонетка появится справа, потому что она двигается налево.")
    assert "effect_explained_by_motion_direction" in explanation.evidence
