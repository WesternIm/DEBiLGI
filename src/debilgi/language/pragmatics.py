"""Contextual roles for Russian particles, interjections and discourse markers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from debilgi.language.tokenizer import ChatTokenNormalization, normalize_chat_token, tokenize


@dataclass(frozen=True, slots=True)
class PragmaticCue:
    surface: str
    category: str
    function: str
    confidence: float
    intensity: float = 0.0


@dataclass(frozen=True, slots=True)
class PragmaticAnalysis:
    cues: tuple[PragmaticCue, ...]

    @property
    def functions(self) -> frozenset[str]:
        return frozenset(cue.function for cue in self.cues)

    @property
    def question_force(self) -> float:
        return max(
            (cue.intensity for cue in self.cues if cue.function == "question_or_doubt"),
            default=0.0,
        )

    @property
    def uncertainty(self) -> float:
        return max(
            (
                cue.intensity
                for cue in self.cues
                if cue.function in {"uncertainty", "hesitation"}
            ),
            default=0.0,
        )

    @property
    def emotional_intensity(self) -> float:
        return max(
            (cue.intensity for cue in self.cues if cue.category == "interjection"),
            default=0.0,
        )


_PARTICLES: dict[str, tuple[str, float]] = {
    "не": ("negation", 0.75),
    "ни": ("reinforced_negation", 0.80),
    "бы": ("hypothetical_or_softening", 0.55),
    "же": ("emphasis_or_shared_context", 0.50),
    "ли": ("question_or_doubt", 0.70),
    "разве": ("question_or_doubt", 0.78),
    "неужели": ("question_or_doubt", 0.88),
    "только": ("restriction", 0.65),
    "лишь": ("restriction", 0.65),
    "именно": ("focus", 0.72),
    "даже": ("unexpected_inclusion", 0.62),
    "ведь": ("shared_reason", 0.58),
    "вот": ("demonstration_or_focus", 0.55),
    "вон": ("demonstration_or_focus", 0.55),
    "уж": ("intensification", 0.52),
    "якобы": ("reported_doubt", 0.74),
    "мол": ("reported_speech", 0.68),
    "типа": ("approximation_or_quote", 0.52),
    "типо": ("approximation_or_quote", 0.48),
}

_PARTICLE_PHRASES: dict[tuple[str, ...], tuple[str, float]] = {
    ("вряд", "ли"): ("uncertainty", 0.88),
    ("едва", "ли"): ("uncertainty", 0.82),
    ("как", "бы"): ("hypothetical_or_softening", 0.64),
    ("вовсе", "не"): ("reinforced_negation", 0.84),
    ("далеко", "не"): ("reinforced_negation", 0.76),
    ("что", "ли"): ("question_or_doubt", 0.60),
}

_INTERJECTIONS: dict[str, tuple[str, float]] = {
    "ой": ("surprise_pain_or_alarm", 0.72),
    "ах": ("surprise_delight_or_regret", 0.70),
    "увы": ("regret", 0.74),
    "фу": ("aversion", 0.86),
    "ух": ("strong_impression", 0.72),
    "ого": ("surprise_or_admiration", 0.82),
    "эй": ("attention_call", 0.70),
    "ау": ("attention_call", 0.68),
    "хм": ("hesitation", 0.66),
    "мда": ("skepticism_or_disappointment", 0.74),
    "блин": ("annoyance", 0.58),
    "ура": ("joy", 0.86),
}

_DISCOURSE_MARKERS: dict[str, tuple[str, float]] = {
    "ну": ("turn_management", 0.52),
    "короче": ("summary_or_topic_shift", 0.68),
    "значит": ("inference_or_transition", 0.66),
    "кстати": ("side_note", 0.72),
    "вообще": ("generalization_or_emphasis", 0.50),
    "наверное": ("uncertainty", 0.70),
    "кажется": ("uncertainty", 0.66),
    "хуле": ("coarse_emphasis", 0.78),
    "хули": ("coarse_emphasis", 0.78),
}

_GREETINGS = frozenset(
    {"привет", "здравствуй", "здравствуйте", "дарова", "здорово", "хай", "ку"}
)
_REDUPLICATION_FUNCTIONS: dict[str, tuple[str, float]] = {
    "бла": ("dismissive_placeholder", 0.68),
    "да": ("emphatic_affirmation", 0.78),
    "нет": ("emphatic_rejection", 0.82),
    "не": ("emphatic_rejection", 0.78),
    "ну": ("skepticism_or_warning", 0.68),
    "ха": ("amusement", 0.82),
    "хе": ("amusement", 0.76),
}
_ELONGATED_VOCALIZATIONS: dict[str, tuple[str, float]] = {
    "а": ("realization_or_alarm", 0.72),
    "м": ("hesitation_or_appraisal", 0.66),
    "о": ("surprise_or_realization", 0.76),
    "у": ("strong_impression", 0.72),
    "э": ("hesitation", 0.70),
}


def _construction_cues(words: tuple[str, ...]) -> tuple[PragmaticCue, ...]:
    """Recognize reusable conversational constructions above individual words."""

    cues: list[PragmaticCue] = []
    if len(words) >= 3 and words[0] in {"ты", "вы"} and words[1] == "что":
        cues.append(
            PragmaticCue(
                f"{words[0]} что …",
                "construction",
                "rhetorical_challenge",
                0.86,
                0.82,
            )
        )
    if any(words[index : index + 2] == ("как", "думаешь") for index in range(len(words) - 1)):
        cues.append(
            PragmaticCue(
                "как думаешь",
                "construction",
                "opinion_request",
                0.94,
                0.84,
            )
        )
    if any(words[index : index + 3] == ("да", "или", "нет") for index in range(len(words) - 2)):
        cues.append(
            PragmaticCue(
                "да или нет",
                "construction",
                "binary_answer_demand",
                0.96,
                0.90,
            )
        )
    if len(words) <= 4 and words[:2] == ("а", "что") and "тогда" in words:
        cues.append(
            PragmaticCue(
                "а что тогда",
                "construction",
                "ellipsis_alternative_request",
                0.94,
                0.82,
            )
        )
    if len(words) >= 2 and words[-1] in {
        "прикинь",
        "представь",
        "понимаешь",
        "знаешь",
    }:
        cues.append(
            PragmaticCue(
                words[-1],
                "construction",
                "parenthetical_attention_marker",
                0.88,
                0.58,
            )
        )
    return tuple(cues)


def _expressive_cue(
    word: str,
    normalized: ChatTokenNormalization,
) -> PragmaticCue | None:
    canonical = normalized.canonical
    if canonical in _GREETINGS:
        return PragmaticCue(
            word,
            "social_routine",
            "greeting",
            0.96,
            max(0.42, normalized.intensity),
        )
    if normalized.kind == "reduplication" and canonical in _REDUPLICATION_FUNCTIONS:
        function, base_intensity = _REDUPLICATION_FUNCTIONS[canonical]
        return PragmaticCue(
            word,
            "interjection",
            function,
            0.92,
            min(1.0, max(base_intensity, normalized.intensity)),
        )
    if normalized.kind == "elongation":
        if canonical in _ELONGATED_VOCALIZATIONS:
            function, base_intensity = _ELONGATED_VOCALIZATIONS[canonical]
            return PragmaticCue(
                word,
                "interjection",
                function,
                0.90,
                min(1.0, max(base_intensity, normalized.intensity)),
            )
        if canonical in _INTERJECTIONS:
            function, base_intensity = _INTERJECTIONS[canonical]
            return PragmaticCue(
                word,
                "interjection",
                function,
                0.92,
                min(1.0, max(base_intensity, normalized.intensity)),
            )
        if canonical == "да":
            return PragmaticCue(word, "particle", "emphatic_affirmation", 0.92, 0.78)
        if canonical in {"нет", "не"}:
            return PragmaticCue(word, "particle", "emphatic_rejection", 0.92, 0.82)
    if re.fullmatch(r"хм+", word):
        return PragmaticCue(word, "interjection", "hesitation", 0.86, 0.68)
    if re.fullmatch(r"х[ахе]{3,}", word):
        return PragmaticCue(word, "interjection", "amusement", 0.88, 0.76)
    if re.fullmatch(r"мда+", word):
        return PragmaticCue(word, "interjection", "skepticism_or_disappointment", 0.86, 0.74)
    if re.fullmatch(r"о+го+", word):
        return PragmaticCue(word, "interjection", "surprise_or_admiration", 0.84, 0.82)
    return None


def analyze_pragmatics(text: str) -> PragmaticAnalysis:
    raw_words = tuple(
        token.casefold().replace("ё", "е")
        for token in tokenize(text)
        if any(character.isalnum() for character in token)
    )
    normalized_words = tuple(normalize_chat_token(word) for word in raw_words)
    words = tuple(item.canonical for item in normalized_words)
    cues: list[PragmaticCue] = list(_construction_cues(words))
    occupied: set[int] = set()
    for phrase, (function, intensity) in sorted(
        _PARTICLE_PHRASES.items(), key=lambda item: -len(item[0])
    ):
        width = len(phrase)
        for index in range(len(words) - width + 1):
            if any(position in occupied for position in range(index, index + width)):
                continue
            if words[index : index + width] == phrase:
                cues.append(
                    PragmaticCue(" ".join(phrase), "particle", function, 0.92, intensity)
                )
                occupied.update(range(index, index + width))
    for index, (word, raw_word, normalized) in enumerate(
        zip(words, raw_words, normalized_words, strict=True)
    ):
        if index in occupied:
            continue
        expressive = _expressive_cue(raw_word, normalized)
        if expressive is not None:
            cues.append(expressive)
        elif word in _INTERJECTIONS:
            function, intensity = _INTERJECTIONS[word]
            cues.append(PragmaticCue(word, "interjection", function, 0.90, intensity))
        elif word in _PARTICLES:
            function, intensity = _PARTICLES[word]
            cues.append(PragmaticCue(word, "particle", function, 0.88, intensity))
        elif word in _DISCOURSE_MARKERS:
            function, intensity = _DISCOURSE_MARKERS[word]
            category = "interjection_particle" if word == "ну" else "discourse_marker"
            confidence = 0.66 if word == "ну" else 0.82
            cues.append(PragmaticCue(word, category, function, confidence, intensity))
    return PragmaticAnalysis(tuple(cues))
