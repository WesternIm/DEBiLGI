from decimal import Decimal

import pytest

from debilgi.literacy.reasoner import analyze_symbols, calculate, facts_for


def fact_text(prompt: str) -> str:
    facts = facts_for(prompt)
    assert facts
    return facts[0].text


def test_unicode_character_classes_and_scripts_are_distinguished() -> None:
    analysis = analyze_symbols("Абc 12+!")

    assert analysis.letters == 3
    assert analysis.digits == 2
    assert analysis.whitespace == 1
    assert analysis.punctuation == 1
    assert analysis.symbols == 1
    assert analysis.numbers == ("12",)
    assert analysis.scripts == ("cyrillic", "latin")


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        ("2 + 3 * 4", Decimal("14")),
        ("(2 + 3) * 4", Decimal("20")),
        ("10 / 4", Decimal("2.5")),
        ("2^3", Decimal("8")),
        ("7 % 4", Decimal("3")),
        ("-3 + 5", Decimal("2")),
    ),
)
def test_bounded_arithmetic_obeys_operator_rules(expression: str, expected: Decimal) -> None:
    assert calculate(expression) == expected


def test_arithmetic_rejects_code_and_unsafe_operations() -> None:
    with pytest.raises(ValueError):
        calculate("__import__('os')")
    with pytest.raises(ArithmeticError):
        calculate("1 / 0")
    with pytest.raises(ValueError):
        calculate("2 ^ 999")


def test_arithmetic_query_returns_an_exact_fact() -> None:
    assert fact_text("сколько будет 2 + 3 * 4?") == "2 + 3 * 4 = 14"
    assert fact_text("12,5 × 2") == "12.5 * 2 = 25"
    assert fact_text("сколько будет двадцать один плюс девять?") == "21 + 9 = 30"
    assert facts_for("2026-07-18") == ()


def test_russian_and_english_alphabet_queries() -> None:
    assert "идёт «ж»" in fact_text("какая буква после ё?")
    assert "номер 33" in fact_text("номер буквы я")
    assert "— «я»" in fact_text("33-я буква русского алфавита")
    english = fact_text("покажи весь английский алфавит")
    assert english.startswith("Английский алфавит: a, b, c")
    assert english.endswith("x, y, z.")


def test_count_comparison_parity_and_symbol_meaning() -> None:
    assert fact_text("сколько букв в слове ёлка?") == "Букв в «ёлка»: 4."
    assert fact_text("что больше: 12 или 7?") == "12 > 7; больше 12."
    assert fact_text("число 8 чётное или нечётное?") == "Число 8 — чётное."
    assert "сложение" in fact_text("что означает символ +?")
