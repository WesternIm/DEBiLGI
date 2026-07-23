"""Sentence-first comprehension over lexical gaps, relations and recent context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from debilgi.language.character_markov import normalize_word
from debilgi.language.content import content_lemmas
from debilgi.language.entities import EntityAnalysis, EntityMention
from debilgi.language.function_words import service_word_reason
from debilgi.language.lexicon import WordAnalysis
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.valency import predicate_valency
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.relation_filter")


@dataclass(frozen=True, slots=True)
class WordRelation:
    source: str
    relation: str
    target: str
    kind: str

    def terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.source, self.relation, self.target)))


@dataclass(frozen=True, slots=True)
class UtteranceUnderstanding:
    content_terms: tuple[str, ...]
    known_terms: tuple[str, ...]
    unknown_terms: tuple[str, ...]
    relations: tuple[WordRelation, ...]
    context_links: tuple[str, ...]
    comprehension_score: float
    sufficient: bool
    entities: tuple[EntityMention, ...] = ()

    @property
    def subject_phrase(self) -> str:
        selected = self.content_terms[:6] or self.unknown_terms[:6]
        return " ".join(selected)


class UtteranceUnderstandingRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(
        self,
        *,
        event_id: str,
        guild_id: int,
        channel_id: int | None,
        understanding: UtteranceUnderstanding,
        observed_at: datetime,
    ) -> None:
        await self._database.execute(
            """INSERT INTO language_utterance_understandings (
                   event_id, guild_id, channel_id, content_terms_json, known_terms_json,
                   unknown_terms_json, relations_json, context_links_json,
                   comprehension_score, sufficient, observed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (event_id) DO UPDATE SET
                   content_terms_json = excluded.content_terms_json,
                   known_terms_json = excluded.known_terms_json,
                   unknown_terms_json = excluded.unknown_terms_json,
                   relations_json = excluded.relations_json,
                   context_links_json = excluded.context_links_json,
                   comprehension_score = excluded.comprehension_score,
                   sufficient = excluded.sufficient,
                   observed_at = excluded.observed_at""",
            (
                event_id,
                guild_id,
                channel_id,
                _json(understanding.content_terms),
                _json(understanding.known_terms),
                _json(understanding.unknown_terms),
                _json(
                    tuple(
                        {
                            "source": item.source,
                            "relation": item.relation,
                            "target": item.target,
                            "kind": item.kind,
                        }
                        for item in understanding.relations
                    )
                ),
                _json(understanding.context_links),
                understanding.comprehension_score,
                int(understanding.sufficient),
                observed_at.isoformat(),
            ),
        )


def understand_utterance(
    prompt: str,
    analyses: tuple[WordAnalysis, ...],
    context_messages: tuple[str, ...] = (),
    entity_analysis: EntityAnalysis | None = None,
    resolved_bindings: tuple[tuple[str, str], ...] = (),
) -> UtteranceUnderstanding:
    content_terms = tuple(dict.fromkeys(content_lemmas(prompt)))
    entity_terms = entity_analysis.identity_terms if entity_analysis is not None else frozenset()
    unknown_terms = tuple(
        dict.fromkeys(
            item.lemma or item.normalized_form
            for item in analyses
            if item.requires_lexical_study
            if (item.lemma or item.normalized_form) not in entity_terms
        )
    )
    known_terms = tuple(
        dict.fromkeys(
            item.lemma or item.normalized_form
            for item in analyses
            if not item.requires_lexical_study
            or item.has_usable_meaning
        )
    )
    known_terms = tuple(
        dict.fromkeys(
            (
                *known_terms,
                *entity_terms,
                *(lemma for _, lemma in resolved_bindings),
                *(entity_analysis.action_terms if entity_analysis is not None else ()),
                *(entity_analysis.object_terms if entity_analysis is not None else ()),
            )
        )
    )
    relations = _extract_relations(
        prompt,
        set(unknown_terms),
        resolved_bindings=resolved_bindings,
    )
    recent_terms = {
        term for message in context_messages[-5:] for term in content_lemmas(message)
    }
    context_links = tuple(
        dict.fromkeys(
            (
                *(term for term in content_terms if term in recent_terms),
                *(lemma for _, lemma in resolved_bindings),
            )
        )
    )
    total = max(1, len(analyses))
    known_ratio = len(known_terms) / total
    unknown_ratio = len(unknown_terms) / total
    relation_score = min(1.0, len(relations) / 2.0)
    context_score = min(1.0, len(context_links) / 2.0)
    comprehension_score = min(
        1.0,
        known_ratio * 0.58 + relation_score * 0.27 + context_score * 0.15,
    )
    sufficient = (
        not unknown_terms
        or (
            bool(known_terms)
            and unknown_ratio <= 0.5
            and bool(relations or context_links)
        )
        or (len(known_terms) >= 2 and unknown_ratio <= 0.4)
    )
    return UtteranceUnderstanding(
        content_terms=content_terms,
        known_terms=known_terms,
        unknown_terms=unknown_terms,
        relations=relations,
        context_links=context_links,
        comprehension_score=comprehension_score,
        sufficient=sufficient,
        entities=entity_analysis.mentions if entity_analysis is not None else (),
    )


def _extract_relations(
    prompt: str,
    unknown_terms: set[str],
    *,
    resolved_bindings: tuple[tuple[str, str], ...] = (),
) -> tuple[WordRelation, ...]:
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(prompt)
    lexical: list[tuple[int, str, str | None]] = []
    for index, (token, reading) in enumerate(
        zip(sentence.tokens, sentence.readings, strict=True)
    ):
        if not token[:1].isalnum():
            continue
        lemma = reading.lemma if reading is not None else normalize_word(token)
        part = reading.part_of_speech if reading is not None else None
        lexical.append((index, lemma, part))
    relations: list[WordRelation] = []
    structure = analyze_structured_utterance(prompt)
    for frame in structure.clauses:
        if frame.predicate_index is None:
            continue
        predicate = structure.tokens[frame.predicate_index].lemma
        explicit_subject = (
            structure.tokens[frame.subject_indices[0]].lemma
            if frame.subject_indices
            else ""
        )
        inherited_subject = {
            "addressee": "ты",
            "speaker": "я",
        }.get(frame.implicit_subject or "", explicit_subject)
        direct_targets = tuple(
            structure.tokens[index].lemma for index in frame.direct_object_indices
        )
        indirect_targets = tuple(
            structure.tokens[index].lemma for index in frame.indirect_object_indices
        )
        if predicate == "нравиться" and explicit_subject and indirect_targets:
            relations.append(
                WordRelation(
                    indirect_targets[0],
                    predicate,
                    explicit_subject,
                    "experiencer_state",
                )
            )
            continue
        if frame.action_complement_indices:
            action_index = frame.action_complement_indices[-1]
            nested_action = structure.tokens[action_index].lemma
            nested_targets = tuple(
                structure.tokens[index].lemma
                for index in frame.direct_object_indices
                if index > action_index
            )
            if inherited_subject and nested_targets:
                relations.extend(
                    WordRelation(
                        inherited_subject,
                        nested_action,
                        target,
                        "controlled_action",
                    )
                    for target in nested_targets
                )
            if inherited_subject:
                relations.append(
                    WordRelation(
                        inherited_subject,
                        predicate,
                        nested_action,
                        "action_complement",
                    )
                )
            continue
        if inherited_subject and direct_targets:
            relations.extend(
                WordRelation(inherited_subject, predicate, target, "action")
                for target in direct_targets
            )
        elif direct_targets:
            relations.extend(
                WordRelation(predicate, "acts_on", target, "action_target")
                for target in direct_targets
            )
    if not relations:
        for verb_index, action, part in lexical:
            if part != "verb":
                continue
            source = next(
                (
                    lemma
                    for index, lemma, candidate_part in reversed(lexical)
                    if index < verb_index and candidate_part in {"noun", "pronoun"}
                ),
                "",
            )
            target = next(
                (
                    lemma
                    for index, lemma, candidate_part in lexical
                    if index > verb_index
                    and candidate_part in {"noun", "pronoun", "adjective"}
                ),
                "",
            )
            if not target:
                target = next(
                    (
                        lemma
                        for index, lemma, _ in lexical
                        if index > verb_index and lemma in unknown_terms
                    ),
                    "",
                )
            if source and target:
                relations.append(WordRelation(source, action, target, "action"))
            elif target:
                relations.append(WordRelation(action, "acts_on", target, "action_target"))
    resolved_objects = tuple(
        lemma for semantic_role, lemma in resolved_bindings if semantic_role == "direct_object"
    )
    if resolved_objects:
        for verb_index, action, part in lexical:
            if part != "verb":
                continue
            profile = predicate_valency(action)
            requires_direct_object = bool(
                profile
                and any(
                    slot.role == "direct_object" and slot.required_in_isolated_fact
                    for slot in profile.slots
                )
            )
            if not requires_direct_object:
                continue
            already_bound = any(
                item.relation == action and item.target in resolved_objects
                for item in relations
            )
            if already_bound:
                continue
            reading = sentence.readings[verb_index]
            source = (
                "ты"
                if reading is not None and reading.person == "second"
                else "я"
                if reading is not None and reading.person == "first"
                else action
            )
            relations.append(
                WordRelation(
                    source,
                    action,
                    resolved_objects[0],
                    "resolved_action_target",
                )
            )
    if not relations:
        terms = tuple(dict.fromkeys(lemma for _, lemma, _ in lexical if len(lemma) > 1))
        if len(terms) >= 2:
            relations.extend(
                WordRelation(terms[0], "associated_with", target, "cooccurrence")
                for target in terms[1:4]
            )
            question_context = prompt.rstrip().endswith("?")
            parts = {lemma: part for _, lemma, part in lexical}
            semantic_terms = tuple(
                term
                for term in terms
                if service_word_reason(
                    term,
                    part_of_speech=parts.get(term),
                    question_context=question_context,
                )
                is None
            )
            if len(semantic_terms) >= 2 and semantic_terms[0] != terms[0]:
                relations.extend(
                    WordRelation(
                        semantic_terms[0],
                        "associated_with",
                        target,
                        "cooccurrence",
                    )
                    for target in semantic_terms[1:4]
                )
    parts = {lemma: part for _, lemma, part in lexical}
    question_context = prompt.rstrip().endswith("?")
    lexical_lemmas = tuple(lemma for _, lemma, _ in lexical)
    binary_scaffold = any(
        lexical_lemmas[index : index + 3] == ("да", "или", "нет")
        for index in range(max(0, len(lexical_lemmas) - 2))
    )
    filtered: list[WordRelation] = []
    for relation in dict.fromkeys(relations):
        source_reason = service_word_reason(
            relation.source,
            part_of_speech=parts.get(relation.source),
            question_context=question_context,
        )
        target_reason = service_word_reason(
            relation.target,
            part_of_speech=parts.get(relation.target),
            question_context=question_context,
        )
        if binary_scaffold and relation.source in {"да", "нет"}:
            source_reason = "binary_answer_scaffold"
        if binary_scaffold and relation.target in {"да", "нет"}:
            target_reason = "binary_answer_scaffold"
        if source_reason is not None or target_reason is not None:
            logger.info(
                "filtered relation source=%r relation=%r target=%r kind=%s "
                "source_reason=%s target_reason=%s prompt=%r",
                relation.source,
                relation.relation,
                relation.target,
                relation.kind,
                source_reason,
                target_reason,
                " ".join(prompt.split())[:400],
            )
            continue
        filtered.append(relation)
    return tuple(filtered)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
