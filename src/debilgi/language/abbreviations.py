"""Recognition, expansion, and alias generation for abbreviations."""

from __future__ import annotations

import re

_COMMON_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "ai": ("artificial intelligence",),
    "api": ("application programming interface",),
    "cpu": ("central processing unit",),
    "css": ("cascading style sheets",),
    "gpu": ("graphics processing unit",),
    "gta": ("Grand Theft Auto",),
    "html": ("hypertext markup language",),
    "http": ("hypertext transfer protocol",),
    "https": ("hypertext transfer protocol secure",),
    "json": ("JavaScript Object Notation",),
    "js": ("JavaScript",),
    "llm": ("large language model",),
    "npc": ("non-player character",),
    "ram": ("random access memory",),
    "rpg": ("role-playing game",),
    "sql": ("structured query language",),
    "ui": ("user interface",),
    "url": ("uniform resource locator",),
    "ux": ("user experience",),
    "бд": ("база данных",),
    "вуз": ("высшее учебное заведение",),
    "гта": ("Grand Theft Auto",),
    "ии": ("искусственный интеллект",),
    "жкх": ("жилищно-коммунальное хозяйство",),
    "ос": ("операционная система",),
    "пк": ("персональный компьютер",),
    "рф": ("Российская Федерация",),
    "сми": ("средства массовой информации",),
    "снг": ("Содружество Независимых Государств",),
    "тз": ("техническое задание",),
}
_DOTTED_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "тд": ("так далее",),
    "те": ("то есть",),
    "тк": ("так как",),
    "тп": ("тому подобное",),
}
_IGNORED_INITIAL_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "а",
        "без",
        "в",
        "для",
        "и",
        "из",
        "к",
        "на",
        "о",
        "об",
        "от",
        "по",
        "с",
    }
)
_ROMAN_NUMBERS = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}
_NORMALIZE_RE = re.compile(r"[\s._-]+")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*|[0-9]+")
_DOTTED_RE = re.compile(r"^(?:[A-Za-zА-Яа-яЁё]\.){2,}[A-Za-zА-Яа-яЁё]?$")
_MENTION_RE = re.compile(
    r"(?<!\w)(?:[A-Za-zА-Яа-яЁё]\.){2,}[A-Za-zА-Яа-яЁё]?\.?"
    r"|(?<!\w)[A-ZА-ЯЁ][A-ZА-ЯЁ0-9]{1,11}(?!\w)"
)
_SHORT_FIRST_RE = re.compile(
    r"(?<!\w)(?P<short>(?:[A-Za-zА-Яа-яЁё]\.){2,}|"
    r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]{1,11})"
    r"\s*(?:[—–=]\s*(?:это\s+)?|"
    r"(?:означает|расшифровывается\s+как)\s+)"
    r"(?P<expansion>[^.!?;\n]{3,180})",
    re.IGNORECASE,
)
_PARENTHETICAL_SHORT_RE = re.compile(
    r"\((?P<short>(?:[A-Za-zА-Яа-яЁё]\.){2,}|"
    r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9]{1,11})\)"
)


def normalize_abbreviation(value: str) -> str:
    """Return a comparison form that ignores case, dots, spaces, and separators."""

    return _NORMALIZE_RE.sub("", value.casefold().replace("ё", "е"))


def abbreviation_meanings(value: str) -> tuple[str, ...]:
    """Return built-in expansions for common conversational and technical forms."""

    normalized = normalize_abbreviation(value)
    if "." in value:
        return _DOTTED_ABBREVIATIONS.get(normalized, ())
    return _COMMON_ABBREVIATIONS.get(normalized, ())


def common_abbreviations() -> tuple[str, ...]:
    return tuple(_COMMON_ABBREVIATIONS)


def is_abbreviation_candidate(value: str) -> bool:
    """Recognize conventional uppercase, dotted, and known lowercase abbreviations."""

    raw = value.strip("«»\"'()[]{} ,;:!?")
    normalized = normalize_abbreviation(raw)
    if normalized in _COMMON_ABBREVIATIONS:
        return True
    if not 2 <= len(normalized) <= 12:
        return False
    if _DOTTED_RE.fullmatch(raw):
        return True
    letters = "".join(character for character in raw if character.isalpha())
    return bool(
        len(letters) >= 2
        and all(character.isalnum() or character in "._-" for character in raw)
        and letters.upper() == letters
    )


def abbreviation_mentions(text: str) -> tuple[str, ...]:
    """Extract explicit abbreviation forms while retaining known lowercase forms."""

    found = [normalize_abbreviation(match.group(0)) for match in _MENTION_RE.finditer(text)]
    for match in _WORD_RE.finditer(text):
        token = match.group(0)
        if is_abbreviation_candidate(token):
            found.append(normalize_abbreviation(token))
    return tuple(dict.fromkeys(item for item in found if item))


def expansion_initials(value: str) -> str:
    """Build an initialism from the significant words of a full name."""

    words = tuple(
        word.casefold().replace("ё", "е")
        for word in _WORD_RE.findall(value)
        if word[:1].isalpha()
    )
    significant = tuple(word for word in words if word not in _IGNORED_INITIAL_WORDS)
    return "".join(word[0] for word in significant)


def extract_explicit_abbreviation_definition(text: str) -> tuple[str, str] | None:
    """Learn both ``API — ...`` and ``full name (API)`` constructions."""

    for match in _SHORT_FIRST_RE.finditer(text):
        short = match.group("short")
        expansion = " ".join(match.group("expansion").split()).strip(" ,:—–-")
        if is_abbreviation_candidate(short) and _initials_match(short, expansion):
            return normalize_abbreviation(short), expansion

    for match in _PARENTHETICAL_SHORT_RE.finditer(text):
        short = match.group("short")
        if not is_abbreviation_candidate(short):
            continue
        prefix = text[: match.start()]
        sentence = re.split(r"[.!?;\n]", prefix)[-1]
        words = tuple(_WORD_RE.findall(sentence))[-10:]
        for size in range(2, len(words) + 1):
            expansion = " ".join(words[-size:])
            if _initials_match(short, expansion):
                return normalize_abbreviation(short), expansion
    return None


def generated_abbreviation_aliases(title: str) -> tuple[str, ...]:
    """Generate conservative lookup aliases from a resolved concept title."""

    words = tuple(_WORD_RE.findall(title))
    alpha_words = tuple(word for word in words if word[:1].isalpha())
    significant = tuple(
        word
        for word in alpha_words
        if word.casefold().replace("ё", "е") not in _IGNORED_INITIAL_WORDS
    )
    aliases: list[str] = []
    if len(significant) >= 2:
        acronym = normalize_abbreviation("".join(word[0] for word in significant))
        if 2 <= len(acronym) <= 12:
            aliases.append(acronym)
    if significant:
        last = significant[-1].casefold()
        if last in _ROMAN_NUMBERS and len(significant) >= 3:
            base = normalize_abbreviation("".join(word[0] for word in significant[:-1]))
            aliases.extend((base, f"{base} {_ROMAN_NUMBERS[last]}"))
    normalized_titles = {
        " ".join(word.casefold().replace("ё", "е") for word in words)
    }
    if words and (
        words[-1].isdigit() or words[-1].casefold() in _ROMAN_NUMBERS
    ):
        normalized_titles.add(
            " ".join(word.casefold().replace("ё", "е") for word in words[:-1])
        )
    for short, expansions in _COMMON_ABBREVIATIONS.items():
        if any(
            " ".join(
                word.casefold().replace("ё", "е")
                for word in _WORD_RE.findall(expansion)
            )
            in normalized_titles
            for expansion in expansions
        ):
            aliases.append(short)
    return tuple(dict.fromkeys(alias for alias in aliases if len(alias) >= 2))


def _initials_match(short: str, expansion: str) -> bool:
    normalized = normalize_abbreviation(short)
    initials = normalize_abbreviation(expansion_initials(expansion))
    return bool(initials and (normalized == initials or initials.startswith(normalized)))
