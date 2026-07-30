"""Russian morphology, case government and agreement checks backed by pymorphy3."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from debilgi.events.models import Event, EventType
from debilgi.language.character_markov import is_word_form, normalize_word
from debilgi.language.lexical_matching import false_friend_group, fuzzy_match_allowed
from debilgi.language.tokenizer import normalize_chat_token, tokenize
from debilgi.storage.database import Database

try:
    import pymorphy3  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - deployment fallback
    pymorphy3 = None

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
morphology_logger = logging.getLogger("debilgi.morphology_disambiguation")
spelling_logger = logging.getLogger("debilgi.spelling_correction")
_RUSSIAN_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_NUMERIC_LITERAL_RE = re.compile(
    r"^[+-]?\d+(?:(?:\.\d+)+|,\d+)?(?:[%‰])?$"
)
_INFINITIVE_GOVERNOR_LEMMAS = frozenset(
    {
        "быть",
        "мочь",
        "хотеть",
        "должный",
        "намереваться",
        "пытаться",
        "начать",
        "стать",
        "продолжить",
        "перестать",
        "решить",
        "любить",
    }
)
CASE_NAMES = {
    "nomn": "nominative",
    "gent": "genitive",
    "gen1": "genitive",
    "gen2": "genitive",
    "datv": "dative",
    "accs": "accusative",
    "acc2": "accusative",
    "ablt": "instrumental",
    "loct": "prepositional",
    "loc1": "prepositional",
    "loc2": "prepositional",
    "voct": "vocative",
}
CASE_GRAMMEMES = {
    "nominative": "nomn",
    "genitive": "gent",
    "dative": "datv",
    "accusative": "accs",
    "instrumental": "ablt",
    "prepositional": "loct",
}
FEATURE_GRAMMEMES = {
    "singular": "sing",
    "plural": "plur",
    "masculine": "masc",
    "feminine": "femn",
    "neuter": "neut",
    "past": "past",
    "present": "pres",
    "future": "futr",
    "first": "1per",
    "second": "2per",
    "third": "3per",
    "animate": "anim",
    "inanimate": "inan",
}
POS_NAMES = {
    "NOUN": "noun",
    "ADJF": "adjective",
    "ADJS": "adjective",
    "COMP": "adjective",
    "VERB": "verb",
    "INFN": "verb",
    "PRTF": "participle",
    "PRTS": "participle",
    "GRND": "gerund",
    "NUMR": "number",
    "ADVB": "adverb",
    "NPRO": "pronoun",
    "PRED": "predicate",
    "PREP": "preposition",
    "CONJ": "conjunction",
    "PRCL": "particle",
    "INTJ": "interjection",
}
PREPOSITION_CASES: dict[str, frozenset[str]] = {
    "без": frozenset({"genitive"}),
    "для": frozenset({"genitive"}),
    "до": frozenset({"genitive"}),
    "из": frozenset({"genitive"}),
    "от": frozenset({"genitive"}),
    "у": frozenset({"genitive"}),
    "около": frozenset({"genitive"}),
    "вокруг": frozenset({"genitive"}),
    "к": frozenset({"dative"}),
    "ко": frozenset({"dative"}),
    "по": frozenset({"dative", "prepositional", "accusative"}),
    "о": frozenset({"prepositional"}),
    "об": frozenset({"prepositional"}),
    "обо": frozenset({"prepositional"}),
    "при": frozenset({"prepositional"}),
    "над": frozenset({"instrumental"}),
    "перед": frozenset({"instrumental"}),
    "между": frozenset({"instrumental", "genitive"}),
    "под": frozenset({"instrumental", "accusative"}),
    "с": frozenset({"instrumental", "genitive"}),
    "со": frozenset({"instrumental", "genitive"}),
    "в": frozenset({"accusative", "prepositional"}),
    "во": frozenset({"accusative", "prepositional"}),
    "на": frozenset({"accusative", "prepositional"}),
    "за": frozenset({"accusative", "instrumental"}),
    "через": frozenset({"accusative"}),
    "про": frozenset({"accusative"}),
}


@dataclass(frozen=True, slots=True)
class MorphologicalReading:
    surface_form: str
    lemma: str
    part_of_speech: str | None
    case: str | None
    number: str | None
    gender: str | None
    tense: str | None
    person: str | None
    animacy: str | None
    aspect: str | None
    mood: str | None
    voice: str | None
    transitivity: str | None
    score: float
    grammemes: frozenset[str]
    corrected_form: str | None = None
    is_known: bool = False

    def features(self) -> dict[str, str]:
        values = {
            "case": self.case,
            "number": self.number,
            "gender": self.gender,
            "tense": self.tense,
            "person": self.person,
            "animacy": self.animacy,
            "aspect": self.aspect,
            "mood": self.mood,
            "voice": self.voice,
            "transitivity": self.transitivity,
        }
        return {name: value for name, value in values.items() if value is not None}

    @property
    def analysis_kind(self) -> str:
        if "LITR" in self.grammemes:
            return "numeric_literal"
        if self.corrected_form is not None:
            return "corrected"
        return "dictionary" if self.is_known else "guessed"


@dataclass(frozen=True, slots=True)
class GrammarIssue:
    kind: str
    token_index: int
    message: str
    expected: tuple[str, ...] = ()
    actual: str | None = None


@dataclass(frozen=True, slots=True)
class SentenceAnalysis:
    text: str
    tokens: tuple[str, ...]
    readings: tuple[MorphologicalReading | None, ...]
    alternatives: tuple[tuple[MorphologicalReading, ...], ...]
    issues: tuple[GrammarIssue, ...]
    agreement_score: float
    confidence: float

    @property
    def lemmas(self) -> tuple[str, ...]:
        return tuple(
            reading.lemma if reading is not None else token.casefold()
            for token, reading in zip(self.tokens, self.readings, strict=True)
            if token[:1].isalnum()
        )

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(reading.case for reading in self.readings if reading and reading.case)


def _existential_holder_index(
    tokens: tuple[str, ...],
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> int | None:
    """Find the holder in Russian ``у X есть Y`` without choosing a word sense."""

    for index, token in enumerate(tokens[:-1]):
        if normalize_word(token) != "у":
            continue
        for following in range(index + 1, min(len(tokens), index + 3)):
            if any(
                item.part_of_speech in {"noun", "pronoun"}
                and item.case == "genitive"
                for item in alternatives[following]
            ):
                return following
    return None


def _existential_predicate_index(
    tokens: tuple[str, ...],
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> int | None:
    if _existential_holder_index(tokens, alternatives) is None:
        return None
    return next(
        (
            index
            for index, token in enumerate(tokens)
            if normalize_word(token) in {"есть", "нет", "имеется", "имеются"}
        ),
        None,
    )


class RussianMorphology:
    def __init__(self) -> None:
        self._analyzer = pymorphy3.MorphAnalyzer() if pymorphy3 is not None else None

    @lru_cache(maxsize=50_000)  # noqa: B019 - process-lifetime singleton/service cache
    def parse_word(self, word: str) -> tuple[MorphologicalReading, ...]:
        normalized = normalize_word(word)
        if not normalized or self._analyzer is None:
            return ()
        if _NUMERIC_LITERAL_RE.fullmatch(normalized):
            return (_numeric_reading(normalized),)
        if not is_word_form(normalized):
            return ()
        chat_form = normalize_chat_token(
            normalized,
            is_known=self._analyzer.word_is_known,
        )
        analysis_form = chat_form.canonical
        parsed = self._analyzer.parse(analysis_form)
        corrected_form: str | None = (
            analysis_form if analysis_form != normalized else None
        )
        if corrected_form is not None:
            spelling_logger.info(
                "word=%r corrected=%r reason=expressive_%s intensity=%.3f",
                normalized,
                corrected_form,
                chat_form.kind,
                chat_form.intensity,
            )
        if (
            parsed
            and not any(item.is_known for item in parsed)
            and not self._productive_ka_form_is_plausible(
                analysis_form,
                parsed,
            )
        ):
            spelling_correction = self._spelling_correction(analysis_form)
            if spelling_correction is not None:
                corrected_form = spelling_correction
                parsed = self._analyzer.parse(spelling_correction)
                spelling_logger.info(
                    "word=%r corrected=%r reason=unique_known_single_edit",
                    normalized,
                    spelling_correction,
                )
        readings: list[MorphologicalReading] = []
        seen: set[tuple[str, frozenset[str]]] = set()
        for item in parsed:
            key = (
                str(item.normal_form),
                frozenset(str(grammeme) for grammeme in item.tag.grammemes),
            )
            if key in seen:
                continue
            seen.add(key)
            readings.append(_reading(normalized, item, corrected_form=corrected_form))
        return tuple(readings)

    def best(self, word: str) -> MorphologicalReading | None:
        readings = self.parse_word(word)
        if not readings:
            return None
        normalized = normalize_word(word)
        if normalized.endswith("ку"):
            productive_lemma = f"{normalized[:-1]}а"
            productive = tuple(
                reading
                for reading in readings
                if reading.lemma == productive_lemma
                and reading.part_of_speech == "noun"
                and reading.case == "accusative"
                and reading.gender == "feminine"
            )
            if productive:
                return max(productive, key=lambda reading: reading.score)
        if normalized.endswith("ка"):
            feminine_lemma = tuple(
                reading
                for reading in readings
                if reading.lemma == normalized
                and reading.part_of_speech == "noun"
                and reading.gender == "feminine"
                and "Fixd" not in reading.grammemes
            )
            if feminine_lemma:
                return max(feminine_lemma, key=lambda reading: reading.score)
        exact_lemma = tuple(reading for reading in readings if reading.lemma == normalized)
        if exact_lemma and false_friend_group(normalized) is not None:
            return max(exact_lemma, key=lambda reading: reading.score)
        return readings[0]

    def reading_in_context(
        self,
        word: str,
        context: str | None,
    ) -> MorphologicalReading | None:
        normalized = normalize_word(word)
        if context:
            sentence = self.analyze_sentence(context)
            for token, reading in zip(sentence.tokens, sentence.readings, strict=True):
                if normalize_word(token) == normalized and reading is not None:
                    return reading
        return self.best(normalized)

    def analyze_sentence(self, text: str) -> SentenceAnalysis:
        tokens = tokenize(text)
        alternatives = tuple(self.parse_word(token) for token in tokens)
        readings = self._contextual_readings(text, tokens, alternatives)
        issues = (
            *self._preposition_issues(tokens, readings),
            *self._agreement_issues(tokens, readings),
        )
        meaningful = tuple(reading for reading in readings if reading is not None)
        morphology_confidence = (
            sum(reading.score for reading in meaningful) / len(meaningful) if meaningful else 0.0
        )
        agreement = max(0.0, 1.0 - len(issues) / max(1, len(meaningful)))
        return SentenceAnalysis(
            text=text,
            tokens=tokens,
            readings=readings,
            alternatives=alternatives,
            issues=tuple(issues),
            agreement_score=agreement,
            confidence=min(1.0, morphology_confidence * 0.65 + agreement * 0.35),
        )

    def _contextual_readings(
        self,
        text: str,
        tokens: tuple[str, ...],
        alternatives: tuple[tuple[MorphologicalReading, ...], ...],
    ) -> tuple[MorphologicalReading | None, ...]:
        selected: list[MorphologicalReading | None] = []
        for index, options in enumerate(alternatives):
            if not options:
                selected.append(None)
                continue
            scored = tuple(
                (
                    self._contextual_score(index, candidate, tokens, alternatives),
                    candidate,
                )
                for candidate in options
            )
            score, chosen = max(scored, key=lambda item: item[0])
            selected.append(chosen)
            if len(options) > 1:
                morphology_logger.info(
                    "word=%r alternatives=%s selected=%s reason=%s context=%r",
                    tokens[index],
                    tuple(_reading_log_value(item) for item in options),
                    _reading_log_value(chosen),
                    _selection_reason(index, chosen, tokens, alternatives, score),
                    " ".join(text.split())[:400],
                )
        return tuple(selected)

    def _contextual_score(
        self,
        index: int,
        candidate: MorphologicalReading,
        tokens: tuple[str, ...],
        alternatives: tuple[tuple[MorphologicalReading, ...], ...],
    ) -> float:
        score = candidate.score
        surface = normalize_word(tokens[index])
        likely_finite_predicate = _has_likely_finite_predicate(alternatives)
        infinitive_governor = _infinitive_governor(index, alternatives)
        has_infinitive_alternative = any(
            reading.part_of_speech == "verb" and "INFN" in reading.grammemes
            for reading in alternatives[index]
        )
        if "INFN" in candidate.grammemes and infinitive_governor is not None:
            score += 1.45
        elif (
            candidate.part_of_speech == "noun"
            and has_infinitive_alternative
            and infinitive_governor is not None
        ):
            score -= 0.72
        if surface == "ли":
            if candidate.part_of_speech == "particle":
                score += 2.20
            elif candidate.part_of_speech == "conjunction":
                score += 0.55
            else:
                score -= 1.80
        elif surface in {"и", "а", "но", "зато", "либо"}:
            score += 1.10 if candidate.part_of_speech == "conjunction" else -0.55
        elif surface in {"не", "ни", "же", "бы", "ведь"}:
            score += 1.10 if candidate.part_of_speech == "particle" else -0.55
        if surface == "это":
            next_lexical_index = next(
                (
                    following
                    for following in range(index + 1, len(tokens))
                    if tokens[following][:1].isalnum()
                ),
                None,
            )
            next_has_noun = bool(
                next_lexical_index is not None
                and any(
                    reading.part_of_speech == "noun"
                    for reading in alternatives[next_lexical_index]
                )
            )
            if next_has_noun:
                # Demonstrative determiner: ``это имя``, ``это мнение``.
                if candidate.part_of_speech == "adjective" and candidate.lemma == "этот":
                    score += 1.65
                elif candidate.part_of_speech == "particle":
                    score -= 0.55
            else:
                has_later_finite_predicate = any(
                    reading.part_of_speech in {"verb", "predicate"}
                    and "INFN" not in reading.grammemes
                    for options in alternatives[index + 1 :]
                    for reading in options
                )
                if candidate.part_of_speech == "pronoun" and candidate.case == "nominative":
                    score += 1.40 if has_later_finite_predicate else 0.72
                elif candidate.part_of_speech == "particle" and has_later_finite_predicate:
                    score -= 0.48
        existential_predicate = _existential_predicate_index(tokens, alternatives)
        existential_holder = _existential_holder_index(tokens, alternatives)
        if surface == "есть" and existential_holder is not None:
            if candidate.lemma == "быть" and "VERB" in candidate.grammemes:
                score += 2.40
            elif "INFN" in candidate.grammemes:
                score -= 0.90
        if (
            existential_predicate is not None
            and index > existential_predicate
            and candidate.part_of_speech == "noun"
            and surface not in {"ли", "же", "бы", "ведь"}
        ):
            score += 0.82 if candidate.case == "nominative" else -0.28
        if candidate.lemma == surface:
            score += 0.30
        expected = _governing_preposition_cases(self, index, tokens, alternatives)
        if expected:
            if candidate.part_of_speech in {"noun", "pronoun", "adjective", "participle"}:
                score += 1.20 if candidate.case in expected else -0.40
            else:
                score -= 0.45
        previous_options = alternatives[index - 1] if index > 0 else ()
        next_options = alternatives[index + 1] if index + 1 < len(alternatives) else ()
        follows_transitive_verb = any(
            item.part_of_speech == "verb" and "tran" in item.grammemes for item in previous_options
        )
        next_has_noun = any(item.part_of_speech == "noun" for item in next_options)
        previous_transitive = _nearest_verb(
            alternatives,
            index,
            direction=-1,
            maximum_distance=3,
            require_transitive=True,
        )
        following_transitive = _nearest_verb(
            alternatives,
            index,
            direction=1,
            maximum_distance=3,
            require_transitive=True,
        )
        if candidate.part_of_speech in {"noun", "pronoun"}:
            definitional_subject = (
                index >= 2
                and normalize_word(tokens[index - 2]) == "что"
                and normalize_word(tokens[index - 1])
                in {"такой", "такая", "такое", "такие"}
            )
            if definitional_subject:
                score += 1.10 if candidate.case == "nominative" else -0.32
            if previous_transitive is not None:
                verb_index, verb = previous_transitive
                imperative = "impr" in verb.grammemes
                if candidate.case == "accusative":
                    score += 0.78 if imperative else 0.54
                elif candidate.case == "nominative" and imperative:
                    score -= 0.34
            if following_transitive is not None:
                verb_index, verb = following_transitive
                has_other_subject = _has_nominative_argument(
                    alternatives,
                    excluded_index=index,
                    # Only an overt nominative between a preposed nominal and
                    # its verb licenses the object reading (``небо ты
                    # любишь``). Looking past the verb made an unrelated
                    # object candidate corrupt the subject itself: ``лиса
                    # нашла ключ`` was selected as masculine ``лис`` in the
                    # accusative merely because ``ключ`` also has a
                    # nominative dictionary reading.
                    start=index + 1,
                    end=verb_index,
                )
                implicit_person = (
                    verb.person in {"first", "second"}
                    or "impr" in verb.grammemes
                )
                if has_other_subject or implicit_person:
                    if candidate.case == "accusative":
                        score += 0.82
                    elif candidate.case == "nominative":
                        score -= 0.26
            if index > 0 and normalize_word(tokens[index - 1]) == "это":
                preceding_finite = _nearest_verb(
                    alternatives,
                    index,
                    direction=-1,
                    maximum_distance=3,
                    require_transitive=False,
                )
                if preceding_finite is None or "intr" in preceding_finite[1].grammemes:
                    score += 0.64 if candidate.case == "nominative" else -0.18
            if not likely_finite_predicate and not expected:
                score += 0.32 if candidate.case == "nominative" else -0.08
        if (
            candidate.part_of_speech == "pronoun"
            and candidate.case == "accusative"
            and follows_transitive_verb
        ):
            score += 0.58
        if (
            candidate.part_of_speech == "adjective"
            and "Apro" in candidate.grammemes
            and not next_has_noun
        ):
            score -= 0.52
        if candidate.part_of_speech == "verb":
            agreement = _subject_agreement_features(index, tokens, alternatives)
            if "INFN" not in candidate.grammemes and agreement:
                score += _predicate_agreement_score(candidate, agreement)
            if any(
                item.part_of_speech == "pronoun" and item.case == "nominative"
                for item in previous_options
            ):
                score += 0.35
            if any(
                item.part_of_speech == "noun" and item.case == "accusative" for item in next_options
            ):
                score += 0.25
        if candidate.part_of_speech == "noun":
            for adjective in previous_options:
                if adjective.part_of_speech != "adjective":
                    continue
                if not _feature_mismatches(adjective, candidate, ("case", "number", "gender")):
                    score += 0.35
                    break
        if candidate.part_of_speech == "adjective":
            for noun in next_options:
                if noun.part_of_speech != "noun":
                    continue
                if not _feature_mismatches(candidate, noun, ("case", "number", "gender")):
                    score += 0.45
                    break
            if not likely_finite_predicate and not expected and next_has_noun:
                score += 0.34 if candidate.case == "nominative" else -0.06
        return score

    def _spelling_correction(self, word: str) -> str | None:
        if self._analyzer is None or not 5 <= len(word) <= 32:
            return None
        candidates = {
            candidate
            for candidate in _single_edit_candidates(word)
            if candidate[:1] == word[:1]
            and fuzzy_match_allowed(word, candidate)
            and self._analyzer.word_is_known(candidate)
        }
        if len(candidates) == 1:
            return next(iter(candidates))
        if candidates:
            spelling_logger.info(
                "word=%r correction=none reason=ambiguous_single_edit candidates=%s",
                word,
                tuple(sorted(candidates)[:20]),
            )
        return None

    def _productive_ka_form_is_plausible(
        self,
        word: str,
        parsed: list[Any],
    ) -> bool:
        """Preserve colloquial ``base + ка`` nouns such as ``мусорку``."""

        if (
            self._analyzer is None
            or not word.endswith("ку")
            or len(word) < 5
        ):
            return False
        base = word[:-2]
        expected_lemma = f"{word[:-1]}а"
        if not self._analyzer.word_is_known(base):
            return False
        return any(
            str(item.normal_form) == expected_lemma
            and str(item.tag.POS) == "NOUN"
            and "femn" in item.tag
            and "accs" in item.tag
            for item in parsed
        )

    def inflect(self, word: str, **features: str) -> str | None:
        if self._analyzer is None:
            return None
        normalized_word = normalize_word(word)
        readings = self._analyzer.parse(normalized_word)
        # The input of the realization layer is a lemma.  Pymorphy can also
        # parse that spelling as an already-inflected form of another lemma
        # (``носок`` is both the lemma for a sock and genitive plural of
        # ``носка``).  Prefer analyses whose normal form is the requested
        # lemma, otherwise generation silently changes the concept.
        lemma_readings = tuple(
            reading for reading in readings if reading.normal_form == normalized_word
        )
        if lemma_readings:
            readings = lemma_readings
        if normalized_word.endswith("ка"):
            productive_feminine = tuple(
                reading
                for reading in readings
                if str(reading.tag.POS) == "NOUN"
                and "femn" in reading.tag
                and "Fixd" not in reading.tag
            )
            if productive_feminine:
                readings = (
                    *productive_feminine,
                    *(
                        reading
                        for reading in readings
                        if reading not in productive_feminine
                    ),
                )
        grammemes: set[str] = set()
        for name, value in features.items():
            normalized = value.casefold()
            if name == "case":
                grammemes.add(CASE_GRAMMEMES.get(normalized, normalized))
            else:
                grammemes.add(FEATURE_GRAMMEMES.get(normalized, normalized))
        for reading in readings:
            inflected = reading.inflect(grammemes)
            if inflected is not None:
                result = str(inflected.word)
                # Pymorphy normalizes every output to lowercase.  That is
                # correct for lexical comparison but not for realization:
                # geographical names, people and organizations must retain
                # the proper-name signal supplied by the semantic layer.
                if word[:1].isupper():
                    result = result[:1].upper() + result[1:]
                return result
        return None

    def lemma(self, word: str) -> str:
        reading = self.best(word)
        return reading.lemma if reading else normalize_word(word)

    def role(self, word: str) -> str | None:
        reading = self.best(word)
        return reading.part_of_speech if reading else None

    @staticmethod
    def expected_cases(preposition: str) -> frozenset[str]:
        return PREPOSITION_CASES.get(normalize_word(preposition), frozenset())

    def _preposition_issues(
        self,
        tokens: tuple[str, ...],
        readings: tuple[MorphologicalReading | None, ...],
    ) -> tuple[GrammarIssue, ...]:
        issues: list[GrammarIssue] = []
        for index, token in enumerate(tokens[:-1]):
            expected = self.expected_cases(token)
            if not expected:
                continue
            target_index = next(
                (
                    following
                    for following in range(index + 1, min(len(tokens), index + 5))
                    if readings[following]
                    and readings[following].part_of_speech in {"noun", "pronoun"}
                ),
                None,
            )
            if target_index is None:
                continue
            actual = readings[target_index].case
            if actual is not None and actual not in expected:
                issues.append(
                    GrammarIssue(
                        kind="preposition_case",
                        token_index=target_index,
                        message=f"{token} requires {', '.join(sorted(expected))}",
                        expected=tuple(sorted(expected)),
                        actual=actual,
                    )
                )
        return tuple(issues)

    def _agreement_issues(
        self,
        tokens: tuple[str, ...],
        readings: tuple[MorphologicalReading | None, ...],
    ) -> tuple[GrammarIssue, ...]:
        issues: list[GrammarIssue] = []
        for index in range(len(tokens) - 1):
            left, right = readings[index], readings[index + 1]
            if left is None or right is None:
                continue
            if left.part_of_speech == "adjective" and right.part_of_speech == "noun":
                mismatches = _feature_mismatches(left, right, ("case", "number", "gender"))
                if mismatches:
                    issues.append(
                        GrammarIssue(
                            kind="adjective_noun_agreement",
                            token_index=index,
                            message=f"{tokens[index]} does not agree with {tokens[index + 1]}",
                            expected=tuple(mismatches),
                        )
                    )
        subject = next(
            (
                (index, reading)
                for index, reading in enumerate(readings)
                if reading
                and reading.part_of_speech in {"noun", "pronoun"}
                and reading.case == "nominative"
            ),
            None,
        )
        verb = next(
            (
                (index, reading)
                for index, reading in enumerate(readings)
                if reading and reading.part_of_speech == "verb"
            ),
            None,
        )
        if subject and verb and subject[0] < verb[0]:
            mismatches = _feature_mismatches(subject[1], verb[1], ("number",))
            if verb[1].tense == "past":
                mismatches.extend(_feature_mismatches(subject[1], verb[1], ("gender",)))
            if mismatches:
                issues.append(
                    GrammarIssue(
                        kind="subject_verb_agreement",
                        token_index=verb[0],
                        message=f"{tokens[subject[0]]} does not agree with {tokens[verb[0]]}",
                        expected=tuple(dict.fromkeys(mismatches)),
                    )
                )
        return tuple(issues)


def _nearest_verb(
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
    index: int,
    *,
    direction: int,
    maximum_distance: int,
    require_transitive: bool,
) -> tuple[int, MorphologicalReading] | None:
    for distance in range(1, maximum_distance + 1):
        candidate_index = index + direction * distance
        if not 0 <= candidate_index < len(alternatives):
            break
        verbs = tuple(
            reading
            for reading in alternatives[candidate_index]
            if reading.part_of_speech == "verb"
            and (
                ("tran" in reading.grammemes)
                if require_transitive
                else ("INFN" not in reading.grammemes)
            )
        )
        if verbs:
            return candidate_index, max(verbs, key=lambda reading: reading.score)
    return None


def _has_likely_finite_predicate(
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> bool:
    """Ignore low-probability verb guesses when identifying a nominal clause."""

    return any(
        reading.part_of_speech in {"verb", "predicate"}
        and "INFN" not in reading.grammemes
        and (reading.score >= 0.45 or len(options) == 1)
        for options in alternatives
        for reading in options
    )


def _infinitive_governor(
    index: int,
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> tuple[int, MorphologicalReading] | None:
    for previous in range(index - 1, max(-1, index - 4), -1):
        candidates = tuple(
            reading
            for reading in alternatives[previous]
            if reading.lemma in _INFINITIVE_GOVERNOR_LEMMAS
            and (
                reading.part_of_speech == "predicate"
                or (
                    reading.part_of_speech == "verb"
                    and "INFN" not in reading.grammemes
                )
                or reading.lemma == "должный"
            )
        )
        if candidates:
            return previous, max(candidates, key=lambda reading: reading.score)
        if any(
            reading.part_of_speech in {"noun", "pronoun"}
            for reading in alternatives[previous]
        ):
            break
    return None


def _subject_agreement_features(
    predicate_index: int,
    tokens: tuple[str, ...],
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> dict[str, str]:
    """Infer only robust overt-subject features before a predicate candidate."""

    start = max(0, predicate_index - 5)
    for index in range(predicate_index - 1, start - 1, -1):
        if RussianMorphology.expected_cases(tokens[index]):
            continue
        nominatives = tuple(
            reading
            for reading in alternatives[index]
            if reading.part_of_speech in {"noun", "pronoun"}
            and reading.case == "nominative"
            and reading.score >= 0.08
        )
        if nominatives:
            subject = max(nominatives, key=lambda reading: reading.score)
            return {
                name: value
                for name in ("number", "gender", "person")
                if (value := getattr(subject, name)) is not None
            }

        # Cardinal numerals govern a genitive nominal but the whole phrase is
        # normally plural for predicate agreement: ``два кота пришли``.
        genitives = tuple(
            reading
            for reading in alternatives[index]
            if reading.part_of_speech == "noun" and reading.case == "genitive"
        )
        previous_is_cardinal = bool(
            index > start
            and any(
                reading.part_of_speech == "number"
                for reading in alternatives[index - 1]
            )
        )
        if genitives and previous_is_cardinal:
            return {"number": "plural"}
    return {}


def _predicate_agreement_score(
    candidate: MorphologicalReading,
    subject_features: dict[str, str],
) -> float:
    score = 0.0
    for feature, reward in (("number", 0.64), ("person", 0.42)):
        subject_value = subject_features.get(feature)
        predicate_value = getattr(candidate, feature)
        if subject_value is not None and predicate_value is not None:
            score += reward if subject_value == predicate_value else -reward
    if candidate.tense == "past":
        subject_gender = subject_features.get("gender")
        if subject_gender is not None and candidate.gender is not None:
            score += 0.34 if subject_gender == candidate.gender else -0.34
    if "impr" in candidate.grammemes and subject_features.get("person") not in {None, "second"}:
        score -= 0.52
    return score


def _governing_preposition_cases(
    morphology: RussianMorphology,
    index: int,
    tokens: tuple[str, ...],
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
) -> frozenset[str]:
    """Return cases only when the token belongs to that prepositional phrase.

    The old four-token lookback made ``у тебя носки`` treat both ``тебя`` and
    ``носки`` as governed by ``у``.  A preceding noun or pronoun closes the
    phrase; adjectives may remain between a preposition and its noun.
    """

    for previous in range(index - 1, max(-1, index - 5), -1):
        expected = morphology.expected_cases(tokens[previous])
        if expected:
            blocked = any(
                any(
                    reading.part_of_speech in {"noun", "pronoun"}
                    for reading in alternatives[between]
                )
                for between in range(previous + 1, index)
            )
            return frozenset() if blocked else expected
        if any(
            reading.part_of_speech == "verb" and "INFN" not in reading.grammemes
            for reading in alternatives[previous]
        ):
            break
    return frozenset()


def _has_nominative_argument(
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
    *,
    excluded_index: int,
    start: int,
    end: int,
) -> bool:
    return any(
        index != excluded_index
        and any(
            reading.part_of_speech in {"noun", "pronoun"}
            and reading.case == "nominative"
            for reading in alternatives[index]
        )
        for index in range(start, end)
    )


class MorphologyRepository:
    def __init__(self, database: Database, morphology: RussianMorphology | None = None) -> None:
        from debilgi.language.addressing import DEFAULT_ADDRESSING

        self._database = database
        self.morphology = morphology or RussianMorphology()
        self.addressing = DEFAULT_ADDRESSING

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        if event.payload.get("suppress_full_parse") is True:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        analysis = self.morphology.analyze_sentence(content)
        addressing = self.addressing.analyze(content)
        address_indices = {
            index
            for span in addressing.addresses
            for index in range(span.token_start, span.token_end)
        }
        observed_at = event.utc_timestamp.isoformat()
        morphology_rows = []
        roles: list[str] = []
        cases: list[str] = []
        for index, (token, reading) in enumerate(
            zip(analysis.tokens, analysis.readings, strict=True)
        ):
            if reading is None:
                continue
            is_address = index in address_indices
            roles.append("address" if is_address else reading.part_of_speech or "content")
            if is_address:
                cases.append("vocative")
            elif reading.case:
                cases.append(reading.case)
            features = reading.features()
            if is_address:
                features["syntactic_role"] = "address"
            morphology_rows.append(
                (
                    event.guild_id,
                    token,
                    reading.lemma,
                    reading.part_of_speech,
                    json.dumps(features, ensure_ascii=False, sort_keys=True),
                    reading.score,
                    observed_at,
                    observed_at,
                )
            )
        pattern_key = hashlib.sha256("|".join(roles).encode("utf-8")).hexdigest()[:24]
        async with self._database.transaction() as connection:
            if morphology_rows:
                await connection.executemany(
                    """INSERT INTO language_morphology_observations (
                        guild_id, surface_form, lemma, part_of_speech,
                        grammatical_features_json, confidence, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (
                        guild_id, surface_form, lemma, grammatical_features_json
                    ) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        confidence = MAX(confidence, excluded.confidence),
                        last_seen_at = excluded.last_seen_at""",
                    morphology_rows,
                )
            if roles:
                await connection.execute(
                    """INSERT INTO language_grammar_patterns (
                        guild_id, pattern_key, roles_json, cases_json,
                        confidence, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (guild_id, pattern_key) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        confidence = MIN(0.98, confidence + 0.01),
                        updated_at = excluded.updated_at""",
                    (
                        event.guild_id,
                        pattern_key,
                        json.dumps(roles, ensure_ascii=False),
                        json.dumps(cases, ensure_ascii=False),
                        analysis.confidence,
                        observed_at,
                    ),
                )


def _reading(
    surface: str,
    parsed: Any,
    *,
    corrected_form: str | None = None,
) -> MorphologicalReading:
    raw_case = str(parsed.tag.case) if parsed.tag.case else None
    raw_score = float(parsed.score)
    return MorphologicalReading(
        surface_form=surface,
        lemma=str(parsed.normal_form),
        part_of_speech=(
            POS_NAMES.get(str(parsed.tag.POS))
            if parsed.tag.POS
            else "number"
            if "NUMB" in parsed.tag.grammemes
            else None
        ),
        case=CASE_NAMES.get(raw_case, raw_case) if raw_case else None,
        number=_feature_name(getattr(parsed.tag, "number", None)),
        gender=_feature_name(getattr(parsed.tag, "gender", None)),
        tense=_feature_name(getattr(parsed.tag, "tense", None)),
        person=_feature_name(getattr(parsed.tag, "person", None)),
        animacy=_feature_name(getattr(parsed.tag, "animacy", None)),
        aspect=_feature_name(getattr(parsed.tag, "aspect", None)),
        mood=_feature_name(getattr(parsed.tag, "mood", None)),
        voice=_feature_name(getattr(parsed.tag, "voice", None)),
        transitivity=_feature_name(getattr(parsed.tag, "transitivity", None)),
        # Pymorphy assigns probabilities to productive guesses too.  A guessed
        # neologism can therefore receive 1.0 even though only its suffix was
        # recognized.  Keep the hypothesis, but never expose it as dictionary
        # certainty to downstream semantic reasoning.
        score=raw_score if parsed.is_known else min(raw_score, 0.38),
        grammemes=frozenset(str(item) for item in parsed.tag.grammemes),
        corrected_form=corrected_form,
        is_known=bool(parsed.is_known),
    )


def _numeric_reading(surface: str) -> MorphologicalReading:
    numeric_kind = "percentage" if surface.endswith(("%", "‰")) else (
        "real" if "." in surface or "," in surface else "integer"
    )
    return MorphologicalReading(
        surface_form=surface,
        lemma=surface,
        part_of_speech="number",
        case=None,
        number=None,
        gender=None,
        tense=None,
        person=None,
        animacy=None,
        aspect=None,
        mood=None,
        voice=None,
        transitivity=None,
        score=0.99,
        grammemes=frozenset({"NUMB", "LITR", numeric_kind}),
        is_known=True,
    )


def _single_edit_candidates(word: str) -> frozenset[str]:
    candidates: set[str] = set()
    for index in range(len(word)):
        candidates.add(word[:index] + word[index + 1 :])
    for index in range(len(word) - 1):
        candidates.add(word[:index] + word[index + 1] + word[index] + word[index + 2 :])
    for index, original in enumerate(word):
        candidates.update(
            word[:index] + character + word[index + 1 :]
            for character in _RUSSIAN_ALPHABET
            if character != original
        )
    for index in range(len(word) + 1):
        candidates.update(
            word[:index] + character + word[index:] for character in _RUSSIAN_ALPHABET
        )
    candidates.discard(word)
    return frozenset(candidates)


def _reading_log_value(reading: MorphologicalReading) -> tuple[object, ...]:
    return (
        reading.lemma,
        reading.part_of_speech,
        reading.case,
        reading.number,
        round(reading.score, 4),
    )


def _selection_reason(
    index: int,
    chosen: MorphologicalReading,
    tokens: tuple[str, ...],
    alternatives: tuple[tuple[MorphologicalReading, ...], ...],
    score: float,
) -> str:
    for previous in range(max(0, index - 4), index):
        expected = RussianMorphology.expected_cases(tokens[previous])
        if expected and chosen.case in expected:
            return "grammatical_context:preposition_case"
    previous_options = alternatives[index - 1] if index > 0 else ()
    next_options = alternatives[index + 1] if index + 1 < len(alternatives) else ()
    if chosen.part_of_speech == "verb" and (
        any(
            item.part_of_speech == "pronoun" and item.case == "nominative"
            for item in previous_options
        )
        or any(item.part_of_speech == "noun" and item.case == "accusative" for item in next_options)
    ):
        return "grammatical_context:verb_frame"
    if chosen.part_of_speech == "noun" and any(
        item.part_of_speech == "adjective" for item in previous_options
    ):
        return "grammatical_context:adjective_agreement"
    if chosen.lemma == normalize_word(tokens[index]):
        return "exact_surface_lemma"
    return f"probability_fallback:{score:.4f}"


def _feature_name(value: Any) -> str | None:
    if value is None:
        return None
    aliases = {
        "sing": "singular",
        "plur": "plural",
        "masc": "masculine",
        "femn": "feminine",
        "neut": "neuter",
        "pres": "present",
        "past": "past",
        "futr": "future",
        "1per": "first",
        "2per": "second",
        "3per": "third",
        "anim": "animate",
        "inan": "inanimate",
        "perf": "perfective",
        "impf": "imperfective",
        "indc": "indicative",
        "impr": "imperative",
        "actv": "active",
        "pssv": "passive",
        "tran": "transitive",
        "intr": "intransitive",
    }
    raw = str(value)
    return aliases.get(raw, raw)


def _feature_mismatches(
    left: MorphologicalReading,
    right: MorphologicalReading,
    features: tuple[str, ...],
) -> list[str]:
    mismatches: list[str] = []
    for feature in features:
        left_value = getattr(left, feature)
        right_value = getattr(right, feature)
        if left_value is not None and right_value is not None and left_value != right_value:
            mismatches.append(feature)
    return mismatches


DEFAULT_MORPHOLOGY = RussianMorphology()
