"""Pure symbolic analysis, alphabets and bounded arithmetic."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from debilgi.literacy.models import LiteracyFact, SymbolAnalysis

RUSSIAN_ALPHABET = tuple("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
ENGLISH_ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz")

_NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?(?![\w])")
_DATE_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")
_ARITHMETIC_RE = re.compile(r"(?<!\w)[\d\s.,()+\-*/%^×÷·:]{3,}(?!\w)")
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|\*\*|//|[()+\-*/%]")
_COUNT_RE = re.compile(
    r"сколько\s+(букв|цифр|символов|знаков|пробелов)\s+(?:в|во)\s+"
    r"(?:(?:слове|фразе|строке|тексте)\s+)?(.+)",
    re.IGNORECASE,
)
_AFTER_RE = re.compile(r"(?:буква\s+)?после\s+[«\"']?([a-zа-яё])[»\"']?", re.IGNORECASE)
_BEFORE_RE = re.compile(r"(?:буква\s+)?(?:перед|до)\s+[«\"']?([a-zа-яё])[»\"']?", re.IGNORECASE)
_LETTER_NUMBER_RE = re.compile(
    r"(?:номер|позици\w*|по\s+сч[её]ту)\s+(?:буквы\s+)?[«\"']?([a-zа-яё])[»\"']?",
    re.IGNORECASE,
)
_ORDINAL_LETTER_RE = re.compile(r"\b(\d{1,2})(?:-?я|-?ая)\s+буква\b", re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"(?:что|какое\s+число)\s+(больше|меньше)\D+([-+]?\d+(?:[.,]\d+)?)\D+"
    r"([-+]?\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_PARITY_RE = re.compile(r"(?:число\s+)?([-+]?\d+)\s+(ч[её]тное|неч[её]тное)", re.IGNORECASE)
_SYMBOL_QUERY_RE = re.compile(
    r"(?:что\s+(?:значит|означает)|значение)\s+"
    r"(?:символ(?:а)?\s+|знака\s+)?[«\"']?([^\s«»\"']+)",
    re.IGNORECASE,
)

_SYMBOL_MEANINGS = {
    "+": "сложение или положительное число",
    "-": "вычитание, отрицательное число или дефис",
    "*": "умножение",
    "×": "умножение",
    "/": "деление или разделитель",
    "÷": "деление",
    "%": "процент или остаток от деления",
    "=": "равенство",
    "==": "проверка равенства в языках программирования",
    "!=": "неравенство",
    "<": "меньше",
    ">": "больше",
    "<=": "меньше или равно",
    ">=": "больше или равно",
    "?": "вопрос",
    "!": "восклицание или логическое отрицание",
    "@": "обозначение адресата или имени пользователя",
    "#": "номер, решётка или обозначение темы",
    "&": "союз «и» или побитовое И",
    "|": "разделитель или побитовое ИЛИ",
    "^": "степень в обычной записи или исключающее ИЛИ в программировании",
    ".": "конец предложения или десятичный разделитель",
    ",": "разделение частей предложения или десятичный разделитель",
    ":": "пояснение, отношение или деление",
    ";": "разделение связанных частей",
    "_": "нижнее подчёркивание",
    "(": "начало круглой группы",
    ")": "конец круглой группы",
    "[": "начало квадратной группы",
    "]": "конец квадратной группы",
    "{": "начало фигурной группы",
    "}": "конец фигурной группы",
}

_NUMBER_WORDS = {
    "ноль": 0,
    "один": 1,
    "одна": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}
_WORD_OPERATORS = {
    "плюс": "+",
    "минус": "-",
    "умножить": "*",
    "помножить": "*",
    "разделить": "/",
    "делить": "/",
}
_ALPHABET_NAMES = {
    "russian": ("русский", "русском", "русского"),
    "english": ("английский", "английском", "английского"),
}


def analyze_symbols(text: str) -> SymbolAnalysis:
    scripts: set[str] = set()
    letters = digits = punctuation = symbols = whitespace = 0
    for character in text:
        category = unicodedata.category(character)
        if character.isalpha():
            letters += 1
            script = _script(character)
            if script:
                scripts.add(script)
        elif character.isdigit():
            digits += 1
        elif character.isspace():
            whitespace += 1
        elif category.startswith("P"):
            punctuation += 1
        else:
            symbols += 1
    return SymbolAnalysis(
        letters=letters,
        digits=digits,
        punctuation=punctuation,
        symbols=symbols,
        whitespace=whitespace,
        numbers=tuple(match.group(0).replace(",", ".") for match in _NUMBER_RE.finditer(text)),
        scripts=tuple(sorted(scripts)),
    )


def facts_for(text: str) -> tuple[LiteracyFact, ...]:
    normalized = text.casefold().strip()
    facts: list[LiteracyFact] = []
    arithmetic = _arithmetic_fact(normalized)
    if arithmetic is not None:
        facts.append(arithmetic)
    for detector in (
        _alphabet_fact,
        _count_fact,
        _comparison_fact,
        _parity_fact,
        _symbol_fact,
    ):
        fact = detector(normalized)
        if fact is not None and fact not in facts:
            facts.append(fact)
    return tuple(facts)


def query_kinds(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(fact.kind for fact in facts_for(text)))


def calculate(expression: str) -> Decimal:
    parser = _ArithmeticParser(expression)
    return parser.parse()


def _arithmetic_fact(text: str) -> LiteracyFact | None:
    requested = any(word in text for word in ("посчитай", "вычисли", "сколько будет", "реши"))
    word_expression = _word_arithmetic_expression(text) if requested else None
    if word_expression is not None:
        try:
            result = calculate(word_expression)
        except (ValueError, ArithmeticError):
            pass
        else:
            return LiteracyFact(
                "arithmetic",
                f"{_display_expression(word_expression)} = {_format_decimal(result)}",
            )
    matches = tuple(_ARITHMETIC_RE.finditer(text))
    for match in matches:
        candidate = match.group(0).strip(" .,:;=?")
        if _DATE_RE.fullmatch(candidate):
            continue
        if not candidate or not any(operator in candidate for operator in "+-*/%^×÷·:"):
            continue
        if not requested and text.strip(" ?=") != match.group(0).strip(" ?="):
            continue
        try:
            result = calculate(candidate)
        except (ValueError, ArithmeticError):
            continue
        rendered = _display_expression(candidate)
        return LiteracyFact("arithmetic", f"{rendered} = {_format_decimal(result)}")
    return None


def _alphabet_fact(text: str) -> LiteracyFact | None:
    alphabet, language = _selected_alphabet(text)
    nominative, locative, _ = _ALPHABET_NAMES[language]
    if "сколько букв" in text and ("алфавит" in text or "азбук" in text):
        return LiteracyFact("alphabet", f"В {locative} алфавите {len(alphabet)} букв.")
    if ("алфавит" in text or "азбук" in text) and any(
        word in text for word in ("покажи", "напиши", "назови", "целиком", "весь")
    ):
        return LiteracyFact(
            "alphabet", f"{nominative.capitalize()} алфавит: {', '.join(alphabet)}."
        )
    match = _AFTER_RE.search(text)
    direction = 1
    if match is None:
        match = _BEFORE_RE.search(text)
        direction = -1
    if match is not None:
        letter = match.group(1).casefold()
        chosen, chosen_language = _alphabet_for_letter(letter)
        _, chosen_locative, chosen_genitive = _ALPHABET_NAMES[chosen_language]
        index = chosen.index(letter)
        target = index + direction
        if 0 <= target < len(chosen):
            relation = "после" if direction == 1 else "перед"
            return LiteracyFact(
                "alphabet",
                f"В {chosen_locative} алфавите {relation} «{letter}» идёт «{chosen[target]}».",
            )
        edge = "последняя" if direction == 1 else "первая"
        return LiteracyFact("alphabet", f"«{letter}» — {edge} буква {chosen_genitive} алфавита.")
    match = _LETTER_NUMBER_RE.search(text)
    if match is not None:
        letter = match.group(1).casefold()
        chosen, chosen_language = _alphabet_for_letter(letter)
        _, chosen_locative, _ = _ALPHABET_NAMES[chosen_language]
        return LiteracyFact(
            "alphabet",
            f"«{letter}» — буква номер {chosen.index(letter) + 1} в {chosen_locative} алфавите.",
        )
    match = _ORDINAL_LETTER_RE.search(text)
    if match is not None:
        position = int(match.group(1))
        if 1 <= position <= len(alphabet):
            return LiteracyFact(
                "alphabet",
                f"Буква номер {position} в {locative} алфавите — «{alphabet[position - 1]}».",
            )
    return None


def _count_fact(text: str) -> LiteracyFact | None:
    match = _COUNT_RE.search(text)
    if match is None:
        return None
    requested, value = match.groups()
    value = value.strip().strip("«»\"'.?!")
    analysis = analyze_symbols(value)
    if requested == "букв":
        count = analysis.letters
        label = "Букв"
    elif requested == "цифр":
        count = analysis.digits
        label = "Цифр"
    elif requested == "пробелов":
        count = analysis.whitespace
        label = "Пробелов"
    elif requested == "знаков":
        count = analysis.punctuation + analysis.symbols
        label = "Знаков"
    else:
        count = len(value)
        label = "Символов"
    return LiteracyFact("counting", f"{label} в «{value}»: {count}.")


def _comparison_fact(text: str) -> LiteracyFact | None:
    match = _COMPARE_RE.search(text)
    if match is None:
        return None
    direction, left_raw, right_raw = match.groups()
    left = Decimal(left_raw.replace(",", "."))
    right = Decimal(right_raw.replace(",", "."))
    if left == right:
        return LiteracyFact("comparison", f"{_format_decimal(left)} = {_format_decimal(right)}.")
    selected = max(left, right) if direction == "больше" else min(left, right)
    sign = ">" if left > right else "<"
    return LiteracyFact(
        "comparison",
        f"{_format_decimal(left)} {sign} {_format_decimal(right)}; "
        f"{direction} {_format_decimal(selected)}.",
    )


def _parity_fact(text: str) -> LiteracyFact | None:
    match = _PARITY_RE.search(text)
    if match is None:
        return None
    value = int(match.group(1))
    parity = "чётное" if value % 2 == 0 else "нечётное"
    return LiteracyFact("comparison", f"Число {value} — {parity}.")


def _symbol_fact(text: str) -> LiteracyFact | None:
    match = _SYMBOL_QUERY_RE.search(text)
    if match is None:
        return None
    symbol = match.group(1).rstrip("?.!,")
    meaning = _SYMBOL_MEANINGS.get(symbol)
    return LiteracyFact("symbol", f"Символ «{symbol}» означает: {meaning}.") if meaning else None


def _selected_alphabet(text: str) -> tuple[tuple[str, ...], str]:
    if "англ" in text or re.search(r"[a-z]", text):
        return ENGLISH_ALPHABET, "english"
    return RUSSIAN_ALPHABET, "russian"


def _alphabet_for_letter(letter: str) -> tuple[tuple[str, ...], str]:
    return (
        (ENGLISH_ALPHABET, "english")
        if letter in ENGLISH_ALPHABET
        else (
            RUSSIAN_ALPHABET,
            "russian",
        )
    )


def _word_arithmetic_expression(text: str) -> str | None:
    lowered = text.replace("?", " ").replace("=", " ")
    for marker in ("сколько будет", "посчитай", "вычисли", "реши"):
        if marker in lowered:
            lowered = lowered.split(marker, 1)[1]
            break
    lowered = lowered.replace("умножить на", "умножить").replace("разделить на", "разделить")
    tokens = re.findall(r"[а-яё]+|[+\-*/]", lowered)
    if not tokens:
        return None
    output: list[str] = []
    current_number: list[str] = []
    for token in tokens:
        operator = _WORD_OPERATORS.get(token, token if token in "+-*/" else None)
        if operator is None:
            current_number.append(token)
            continue
        number = _parse_number_words(current_number)
        if number is None:
            return None
        output.extend((str(number), operator))
        current_number = []
    number = _parse_number_words(current_number)
    if number is None or not output:
        return None
    output.append(str(number))
    return " ".join(output)


def _parse_number_words(words: list[str]) -> int | None:
    if not words or any(word not in _NUMBER_WORDS for word in words):
        return None
    values = [_NUMBER_WORDS[word] for word in words]
    if len(values) > 3 or sum(value >= 100 for value in values) > 1:
        return None
    if any(left <= right for left, right in zip(values, values[1:], strict=False)):
        return None
    total = sum(values)
    return total if 0 <= total <= 999 else None


def _script(character: str) -> str | None:
    code = ord(character.casefold()[0])
    if 0x0400 <= code <= 0x052F:
        return "cyrillic"
    if 0x0041 <= code <= 0x024F:
        return "latin"
    return "other"


def _display_expression(value: str) -> str:
    normalized = _normalize_expression(value).replace(",", ".")
    return " ".join(_TOKEN_RE.findall(normalized)).replace("**", "^")


def _normalize_expression(value: str) -> str:
    return (
        value.replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace(":", "/")
        .replace("^", "**")
    )


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f").rstrip("0").rstrip(".")


@dataclass(slots=True)
class _ArithmeticParser:
    expression: str
    _tokens: tuple[str, ...] = field(init=False)
    _position: int = field(init=False)

    def __post_init__(self) -> None:
        normalized = _normalize_expression(self.expression).replace(",", ".")
        self._tokens = tuple(_TOKEN_RE.findall(normalized))
        compact = re.sub(r"\s+", "", normalized)
        if not self._tokens or "".join(self._tokens) != compact or len(self._tokens) > 100:
            raise ValueError("unsupported arithmetic expression")
        self._position = 0

    def parse(self) -> Decimal:
        with localcontext() as context:
            context.prec = 40
            value = self._expression()
        if (
            self._position != len(self._tokens)
            or not value.is_finite()
            or abs(value) > Decimal("1e100")
        ):
            raise ValueError("arithmetic result is outside supported bounds")
        return value

    def _expression(self) -> Decimal:
        value = self._term()
        while (operator := self._peek()) in {"+", "-"}:
            self._position += 1
            right = self._term()
            value = value + right if operator == "+" else value - right
        return value

    def _term(self) -> Decimal:
        value = self._power()
        while (operator := self._peek()) in {"*", "/", "//", "%"}:
            self._position += 1
            right = self._power()
            try:
                if operator == "*":
                    value *= right
                elif operator == "/":
                    value /= right
                elif operator == "//":
                    value = value // right
                else:
                    value %= right
            except (DivisionByZero, InvalidOperation, ZeroDivisionError) as error:
                raise ArithmeticError("invalid arithmetic operation") from error
        return value

    def _power(self) -> Decimal:
        value = self._unary()
        if self._peek() == "**":
            self._position += 1
            exponent = self._unary()
            if exponent != exponent.to_integral_value() or abs(exponent) > 12:
                raise ValueError("exponent must be a small integer")
            value = value ** int(exponent)
        return value

    def _unary(self) -> Decimal:
        operator = self._peek()
        if operator in {"+", "-"}:
            self._position += 1
            value = self._unary()
            return value if operator == "+" else -value
        return self._primary()

    def _primary(self) -> Decimal:
        token = self._peek()
        if token == "(":
            self._position += 1
            value = self._expression()
            if self._peek() != ")":
                raise ValueError("unclosed parenthesis")
            self._position += 1
            return value
        if token is None or not token[0].isdigit() or len(token) > 40:
            raise ValueError("number expected")
        self._position += 1
        return Decimal(token.replace(",", "."))

    def _peek(self) -> str | None:
        return self._tokens[self._position] if self._position < len(self._tokens) else None
