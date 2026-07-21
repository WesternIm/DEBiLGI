"""Turn reference definitions and temporary examples into reusable graph knowledge."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.language.character_markov import normalize_word
from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.semantics import extract_semantic_frames
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class LexicalStudyMaterial:
    word: str
    lemma: str
    definition: str
    examples: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class ExampleObservation:
    case: str | None
    number: str | None
    role: str | None
    neighbors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UsagePattern:
    word: str
    lemma: str
    preferred_case: str | None
    preferred_number: str | None
    preferred_role: str | None
    neighbors: tuple[str, ...]
    status: str
    confidence: float
    example_count: int


@dataclass(frozen=True, slots=True)
class UsageGuidance:
    word: str
    lemma: str
    preferred_case: str | None
    preferred_number: str | None
    preferred_role: str | None
    neighbors: tuple[str, ...]
    stable: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class LexicalLesson:
    concepts: tuple[str, ...]
    pattern: UsagePattern
    generated_example: str


class LexicalUsageLearningService:
    """Learn meaning links and soft collocation guidance without storing examples."""

    def __init__(self, database: Database, dataset: SelfDatasetRepository) -> None:
        self._database = database
        self._dataset = dataset

    async def learn(
        self,
        guild_id: int,
        material: LexicalStudyMaterial,
        *,
        observed_at: datetime,
    ) -> LexicalLesson:
        word = normalize_word(material.word)
        lemma = normalize_word(material.lemma) or DEFAULT_MORPHOLOGY.lemma(word)
        concepts = tuple(
            term
            for term in dict.fromkeys(content_lemmas(material.definition))
            if term not in {word, lemma}
        )[:16]
        relations = _definition_relations(material.definition)
        confidence = 0.76 if material.source == "wiktionary" else 0.66
        await self._dataset.link_reference_concepts(
            guild_id=guild_id,
            word=word,
            concepts=concepts,
            relations=relations,
            source=material.source,
            observed_at=observed_at,
            confidence=confidence,
        )
        observations = tuple(
            item
            for example in material.examples[:5]
            if (item := analyze_usage_example(example, word=word, lemma=lemma)) is not None
        )
        pattern = generalize_usage_pattern(
            word=word,
            lemma=lemma,
            observations=observations,
            total_examples=len(material.examples[:5]),
        )
        await self._dataset.link_usage_pattern(
            guild_id=guild_id,
            word=word,
            lemma=lemma,
            preferred_case=pattern.preferred_case,
            preferred_number=pattern.preferred_number,
            preferred_role=pattern.preferred_role,
            neighbor_lemmas=pattern.neighbors,
            pattern_status=pattern.status,
            confidence=pattern.confidence,
            example_count=pattern.example_count,
            source=material.source,
            observed_at=observed_at,
        )
        return LexicalLesson(
            concepts=concepts,
            pattern=pattern,
            generated_example=generate_usage_example(pattern),
        )

    async def guidance_for_text(
        self, guild_id: int, text: str, *, limit: int = 8
    ) -> tuple[UsageGuidance, ...]:
        lemmas = tuple(
            dict.fromkeys(
                DEFAULT_MORPHOLOGY.lemma(token)
                for token in tokenize(text)
                if token[:1].isalnum()
            )
        )
        if not lemmas:
            return ()
        placeholders = ",".join("?" for _ in lemmas)
        rows = await self._database.fetchall(
            f"""SELECT * FROM language_usage_patterns
                  WHERE guild_id = ?
                    AND (lemma IN ({placeholders}) OR normalized_form IN ({placeholders}))
                  ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
            (guild_id, *lemmas, *lemmas, max(1, min(limit, 32))),
        )
        return tuple(
            UsageGuidance(
                word=str(row["normalized_form"]),
                lemma=str(row["lemma"]),
                preferred_case=(str(row["preferred_case"]) if row["preferred_case"] else None),
                preferred_number=(
                    str(row["preferred_number"]) if row["preferred_number"] else None
                ),
                preferred_role=(str(row["preferred_role"]) if row["preferred_role"] else None),
                neighbors=tuple(json.loads(str(row["neighbor_lemmas_json"]))),
                stable=str(row["pattern_status"]) == "stable",
                confidence=float(row["confidence"]),
            )
            for row in rows
        )


def analyze_usage_example(
    example: str, *, word: str, lemma: str
) -> ExampleObservation | None:
    analysis = DEFAULT_MORPHOLOGY.analyze_sentence(example)
    target_index = next(
        (
            index
            for index, (token, reading) in enumerate(
                zip(analysis.tokens, analysis.readings, strict=True)
            )
            if normalize_word(token) == word or (reading is not None and reading.lemma == lemma)
        ),
        None,
    )
    if target_index is None:
        return None
    target = analysis.readings[target_index]
    if target is None:
        return None
    neighbors: list[str] = []
    start = max(0, target_index - 3)
    end = min(len(analysis.tokens), target_index + 4)
    for index in range(start, end):
        if index == target_index:
            continue
        reading = analysis.readings[index]
        if reading is None or reading.lemma in {word, lemma}:
            continue
        if reading.part_of_speech in {"verb", "preposition"}:
            neighbors.append(reading.lemma)
    case = target.case
    role = (
        "subject_candidate"
        if case == "nominative"
        else "prepositional_object"
        if case == "prepositional"
        else "object_candidate"
        if case in {"genitive", "dative", "accusative", "instrumental"}
        else None
    )
    return ExampleObservation(case, target.number, role, tuple(dict.fromkeys(neighbors)))


def generalize_usage_pattern(
    *,
    word: str,
    lemma: str,
    observations: tuple[ExampleObservation, ...],
    total_examples: int,
) -> UsagePattern:
    located = len(observations)
    preferred_case = _repeated_value(item.case for item in observations)
    preferred_number = _repeated_value(item.number for item in observations)
    preferred_role = _repeated_value(item.role for item in observations)
    neighbor_counts = Counter(
        neighbor for observation in observations for neighbor in set(observation.neighbors)
    )
    neighbors = tuple(
        neighbor
        for neighbor, count in neighbor_counts.most_common(4)
        if count >= 2 and count / max(1, located) >= 0.5
    )
    stable = located >= 2 and any(
        (preferred_case, preferred_number, preferred_role, neighbors)
    )
    coverage = located / max(1, total_examples)
    repeated_dimensions = sum(
        value is not None for value in (preferred_case, preferred_number, preferred_role)
    ) + int(bool(neighbors))
    confidence = min(0.86, 0.25 + coverage * 0.35 + repeated_dimensions * 0.08)
    if not stable:
        confidence = min(confidence, 0.38)
    return UsagePattern(
        word=word,
        lemma=lemma,
        preferred_case=preferred_case,
        preferred_number=preferred_number,
        preferred_role=preferred_role,
        neighbors=neighbors,
        status="stable" if stable else "uncertain",
        confidence=confidence,
        example_count=total_examples,
    )


def generate_usage_example(pattern: UsagePattern) -> str:
    features = {
        name: value
        for name, value in (
            ("case", pattern.preferred_case),
            ("number", pattern.preferred_number),
        )
        if value
    }
    surface = DEFAULT_MORPHOLOGY.inflect(pattern.lemma, **features) or pattern.word
    neighbor = pattern.neighbors[0] if pattern.neighbors else None
    if neighbor and DEFAULT_MORPHOLOGY.role(neighbor) == "preposition":
        return f"Можно сказать: «{neighbor} {surface}»."
    if neighbor and DEFAULT_MORPHOLOGY.role(neighbor) == "verb":
        return f"Кто-то может {neighbor} {surface}."
    return f"В разговоре можно употребить слово «{surface}»."


def _repeated_value(values: Iterable[str | None]) -> str | None:
    counts = Counter(value for value in values if isinstance(value, str) and value)
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    total = sum(counts.values())
    return value if count >= 2 and count / total >= 0.5 else None


def _definition_relations(text: str) -> tuple[tuple[str, str, str, float], ...]:
    relations: list[tuple[str, str, str, float]] = []
    for frame in extract_semantic_frames(text)[:8]:
        if not frame.target:
            continue
        actor = next(iter(content_lemmas(frame.actor)), "")
        relation = next(iter(content_lemmas(frame.action)), normalize_word(frame.action))
        target = next(iter(content_lemmas(frame.target)), "")
        if actor and relation and target:
            relations.append((actor, relation, target, frame.confidence))
    return tuple(relations)
