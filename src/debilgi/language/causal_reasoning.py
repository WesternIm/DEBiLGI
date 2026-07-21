"""Durable cause, condition, prevention and consequence reasoning."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime

from debilgi.events.models import Event, EventType
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.structured_utterance import (
    ClauseFrame,
    StructuredUtterance,
    analyze_structured_utterance,
)
from debilgi.storage.database import Database

_MESSAGE_TYPES = frozenset({EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED})
logger = logging.getLogger("debilgi.causal_reasoning")
_TRIM = " \t\r\n,.;:!?—–-"
_QUERY_WORDS = frozenset(
    {
        "что",
        "какой",
        "почему",
        "зачем",
        "если",
        "причина",
        "следствие",
        "последствие",
        "случиться",
        "произойти",
        "происходить",
        "быть",
        "стать",
        "вызвать",
        "вызывать",
        "привести",
        "приводить",
        "получиться",
        "из-за",
    }
)

# These are semantic action classes, not reply templates.  They let the
# reasoner compose a consequence from grammatical roles when no memorised
# cause/effect sentence exists.  The learned graph remains the preferred
# source; schemas are a conservative physical-world fallback.
_FORCE_DISPLACEMENT_ACTIONS = frozenset(
    {
        "толкнуть",
        "столкнуть",
        "сбросить",
        "уронить",
        "опрокинуть",
    }
)
_SUPPORT_SURFACE_LEMMAS = frozenset(
    {
        "стол",
        "полка",
        "стул",
        "табурет",
        "тумба",
        "подоконник",
        "кровать",
        "диван",
        "шкаф",
        "край",
        "поверхность",
    }
)


@dataclass(frozen=True, slots=True)
class CausalCue:
    cue: str
    cue_kind: str
    relation: str
    orientation: str
    confidence: float


@dataclass(frozen=True, slots=True)
class CausalProposition:
    proposition_id: str
    kind: str
    surface: str
    subject_lemma: str | None
    predicate_lemma: str | None
    object_lemma: str | None
    terms: tuple[str, ...]
    negated: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class CausalObservation:
    cause: CausalProposition
    effect: CausalProposition
    relation: str
    confidence: float
    supports_link: bool
    evidence_surface: str


@dataclass(frozen=True, slots=True)
class CausalQuery:
    direction: str
    focus_terms: tuple[str, ...]
    confidence: float
    predicate_lemma: str | None = None
    object_lemma: str | None = None
    source_lemma: str | None = None


@dataclass(frozen=True, slots=True)
class CausalLink:
    link_id: str
    relation: str
    cause: CausalProposition
    effect: CausalProposition
    confidence: float
    evidence_surface: str


@dataclass(frozen=True, slots=True)
class CausalInference:
    text: str
    confidence: float
    direction: str
    path: tuple[CausalLink, ...]
    evidence: tuple[str, ...]


def extract_causal_observations(
    text: str,
    cues: tuple[CausalCue, ...],
) -> tuple[CausalObservation, ...]:
    """Extract oriented links; temporal order remains distinct from causation."""

    sentence_parts = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    )
    if len(sentence_parts) > 1:
        return tuple(
            observation
            for part in sentence_parts
            for observation in extract_causal_observations(part, cues)
        )
    cleaned = DEFAULT_ADDRESSING.analyze(text).content_text.strip()
    if not cleaned or cleaned.rstrip().endswith("?"):
        return ()
    lowered = cleaned.casefold().replace("ё", "е")
    observations: list[CausalObservation] = []
    marker_cues = sorted(
        (cue for cue in cues if cue.cue_kind == "marker"),
        key=lambda item: len(item.cue),
        reverse=True,
    )
    for cue in marker_cues:
        marker = cue.cue.casefold().replace("ё", "е")
        leading_cause = (
            cue.orientation == "effect_before_cause"
            and lowered.lstrip().startswith(marker)
        )
        if leading_cause:
            split = _split_leading_cause(cleaned, lowered, marker)
        elif cue.orientation == "condition_then_effect":
            split = _split_condition(cleaned, lowered, marker)
        else:
            split = _split_marker(cleaned, lowered, marker)
        if split is None:
            continue
        left, right = split
        if cue.orientation == "effect_before_cause" and not leading_cause:
            cause_text, effect_text = right, left
        else:
            cause_text, effect_text = left, right
        cause = _proposition_from_text(cause_text)
        effect = _proposition_from_text(effect_text)
        if cause is None or effect is None:
            continue
        observations.append(
            CausalObservation(
                cause,
                effect,
                cue.relation,
                min(cue.confidence, cause.confidence, effect.confidence),
                True,
                cleaned,
            )
        )
        break

    if observations:
        return tuple(observations)
    structure = analyze_structured_utterance(cleaned)
    predicate_cues = {
        cue.cue: cue
        for cue in cues
        if cue.cue_kind == "predicate" and cue.orientation == "subject_then_object"
    }
    for clause in structure.clauses:
        if clause.predicate_index is None:
            continue
        predicate = structure.tokens[clause.predicate_index]
        cue = predicate_cues.get(predicate.lemma)
        if cue is None:
            continue
        cause = _argument_proposition(structure, clause, role_name="subject")
        effect = _argument_proposition(
            structure,
            clause,
            role_name="object",
            complement_only=predicate.lemma in {"привести", "приводить"},
        )
        if cause is None or effect is None:
            continue
        observations.append(
            CausalObservation(
                cause,
                effect,
                cue.relation,
                min(cue.confidence, clause.confidence),
                not clause.negated,
                cleaned,
            )
        )
    return tuple(observations)


def analyze_causal_query(
    text: str,
    context_messages: tuple[str, ...] = (),
) -> CausalQuery | None:
    cleaned = DEFAULT_ADDRESSING.analyze(text).content_text.strip()
    structure = analyze_structured_utterance(cleaned)
    question_like = cleaned.rstrip().endswith("?") or any(
        clause.clause_kind == "question" for clause in structure.clauses
    )
    if not question_like:
        return None
    lemmas = tuple(content_lemmas(cleaned))
    lemma_set = set(lemmas)
    normalized = cleaned.casefold().replace("ё", "е")
    normalized_syntax = " ".join(re.sub(r"[,;:]+", " ", normalized).split())
    asks_cause = bool(
        lemma_set & {"почему", "причина"}
        or normalized.startswith("почему ")
        or normalized.rstrip(" ?!") == "почему"
        or "из-за чего" in normalized
        or "что вызвало" in normalized
        or "что стало причиной" in normalized
    )
    asks_effect = bool(
        "что будет если" in normalized_syntax
        or "что произойдет если" in normalized_syntax
        or "что случится если" in normalized_syntax
        or normalized.rstrip(" ?!") in {"что будет", "и что будет"}
        or "что из этого следует" in normalized
        or "к чему привед" in normalized
        or "к чему привод" in normalized
        or lemma_set & {"последствие", "следствие"}
    )
    if not asks_cause and not asks_effect:
        return None
    focus = tuple(dict.fromkeys(lemma for lemma in lemmas if lemma not in _QUERY_WORDS))
    if not focus:
        focus = next(
            (
                tuple(
                    dict.fromkeys(
                        lemma
                        for lemma in content_lemmas(message)
                        if lemma not in _QUERY_WORDS
                    )
                )
                for message in reversed(context_messages[-8:])
                if content_lemmas(message)
            ),
            (),
        )
    if not focus:
        return None
    predicate_lemma, object_lemma, source_lemma = _query_roles(structure)
    return CausalQuery(
        "cause" if asks_cause else "effect",
        focus,
        min(0.96, structure.confidence * 0.72 + 0.24),
        predicate_lemma,
        object_lemma,
        source_lemma,
    )


def infer_compositional_consequence(query: CausalQuery) -> CausalInference | None:
    """Compose a physical consequence from roles instead of copying a sentence.

    This deliberately covers only consequences supported by a small semantic
    action class.  It does not pretend that mere topical overlap is causal
    evidence, which is what previously let an unrelated fact about Boris win.
    """

    if (
        query.direction != "effect"
        or query.predicate_lemma not in _FORCE_DISPLACEMENT_ACTIONS
        or query.object_lemma is None
    ):
        return None
    removes_support = query.source_lemma in _SUPPORT_SURFACE_LEMMAS
    directly_causes_fall = query.predicate_lemma in {
        "столкнуть",
        "сбросить",
        "уронить",
        "опрокинуть",
    }
    if not removes_support and not directly_causes_fall:
        return None

    subject = (
        DEFAULT_MORPHOLOGY.inflect(query.object_lemma, case="nominative")
        or query.object_lemma
    )
    predicate = (
        DEFAULT_MORPHOLOGY.inflect(
            "упасть",
            tense="future",
            number="singular",
            person="third",
        )
        or "упадёт"
    )
    text = f"{subject[:1].upper()}{subject[1:]}, скорее всего, {predicate}."
    confidence = min(0.82, query.confidence * (0.86 if removes_support else 0.91))
    return CausalInference(
        text=text,
        confidence=confidence,
        direction="effect",
        path=(),
        evidence=("schema:force_displacement_removes_support",),
    )


def _query_roles(
    structure: StructuredUtterance,
) -> tuple[str | None, str | None, str | None]:
    predicate = next(
        (
            token.lemma
            for token in structure.tokens
            if token.part_of_speech == "verb" and token.lemma not in _QUERY_WORDS
        ),
        None,
    )
    object_lemma = next(
        (
            structure.tokens[index].lemma
            for clause in structure.clauses
            for index in clause.direct_object_indices
            if structure.tokens[index].part_of_speech in {"noun", "pronoun"}
        ),
        None,
    )
    source_lemma = next(
        (
            structure.tokens[index].lemma
            for clause in structure.clauses
            for complement in clause.complements
            if complement.role in {"source", "source_or_companion"}
            for index in complement.object_indices
            if structure.tokens[index].part_of_speech in {"noun", "pronoun"}
        ),
        None,
    )
    return predicate, object_lemma, source_lemma


class CausalReasoningRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def cues(self) -> tuple[CausalCue, ...]:
        rows = await self._database.fetchall(
            """SELECT cue, cue_kind, relation, orientation, confidence
               FROM language_causal_cues
               WHERE confidence >= 0.55
               ORDER BY confidence DESC, evidence_count DESC"""
        )
        return tuple(
            CausalCue(
                str(row["cue"]),
                str(row["cue_kind"]),
                str(row["relation"]),
                str(row["orientation"]),
                float(row["confidence"]),
            )
            for row in rows
        )

    async def observe(self, event: Event) -> None:
        if (
            event.event_type not in _MESSAGE_TYPES
            or event.payload.get("author_is_bot") is True
            or event.payload.get("suppress_full_parse") is True
        ):
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        observations = extract_causal_observations(content, await self.cues())
        for observation in observations:
            await self._store_observation(
                guild_id=event.guild_id,
                source_event_id=event.event_id,
                source="event",
                observation=observation,
                observed_at=event.utc_timestamp,
            )
            logger.info(
                "event=%s relation=%s cause=%r effect=%r confidence=%.3f supports=%s",
                event.event_id,
                observation.relation,
                observation.cause.surface,
                observation.effect.surface,
                observation.confidence,
                observation.supports_link,
            )

    async def infer(
        self,
        guild_id: int,
        query: CausalQuery,
        *,
        maximum_depth: int = 3,
    ) -> CausalInference | None:
        links = await self._links(guild_id)
        schema_inference = infer_compositional_consequence(query)
        if not links:
            return schema_inference
        start_side = "effect" if query.direction == "cause" else "cause"
        ranked = sorted(
            (
                (_term_fit(query.focus_terms, getattr(link, start_side).terms), link)
                for link in links
                if link.relation != "precedes"
            ),
            key=lambda item: (item[0] * item[1].confidence, item[1].confidence),
            reverse=True,
        )
        ranked = [item for item in ranked if item[0] > 0.0]
        if not ranked:
            return schema_inference
        first_fit, first = ranked[0]
        path = [first]
        visited = {first.cause.proposition_id, first.effect.proposition_id}
        current = first.cause if query.direction == "cause" else first.effect
        for _ in range(max(0, min(maximum_depth, 4) - 1)):
            next_links = [
                link
                for link in links
                if link.relation != "precedes"
                and (
                    link.effect.proposition_id == current.proposition_id
                    if query.direction == "cause"
                    else link.cause.proposition_id == current.proposition_id
                )
            ]
            next_links.sort(key=lambda item: item.confidence, reverse=True)
            if not next_links:
                break
            selected = next_links[0]
            following = selected.cause if query.direction == "cause" else selected.effect
            if following.proposition_id in visited:
                break
            path.append(selected)
            visited.add(following.proposition_id)
            current = following
        confidence = min(
            0.96,
            query.confidence
            * first_fit
            * min(link.confidence * (0.92**index) for index, link in enumerate(path)),
        )
        if confidence < 0.28:
            return schema_inference
        inference = CausalInference(
            _render_path(tuple(path), query.direction),
            confidence,
            query.direction,
            tuple(path),
            tuple(link.link_id for link in path),
        )
        logger.info(
            "guild=%s query=%s focus=%s path=%s confidence=%.3f",
            guild_id,
            query.direction,
            query.focus_terms,
            inference.evidence,
            inference.confidence,
        )
        return inference

    async def rebuild_events(self, guild_id: int, *, limit: int = 5000) -> int:
        rows = await self._database.fetchall(
            """SELECT event_id, payload_json, occurred_at
               FROM events
               WHERE guild_id = ? AND event_type IN ('MESSAGE_CREATED', 'MESSAGE_REPLIED')
               ORDER BY occurred_at ASC LIMIT ?""",
            (guild_id, max(1, min(limit, 50_000))),
        )
        cues = await self.cues()
        stored = 0
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"]))
                content = payload.get("content")
                observed_at = datetime.fromisoformat(str(row["occurred_at"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if payload.get("author_is_bot") is True or not isinstance(content, str):
                continue
            for observation in extract_causal_observations(content, cues):
                await self._store_observation(
                    guild_id=guild_id,
                    source_event_id=str(row["event_id"]),
                    source="event_backfill",
                    observation=observation,
                    observed_at=observed_at,
                )
                stored += 1
        return stored

    async def rebuild_archive(self, guild_id: int, *, limit: int = 50_000) -> int:
        rows = await self._database.fetchall(
            """SELECT message_id, content, created_at
               FROM archive_messages
               WHERE guild_id = ? AND content IS NOT NULL AND author_is_bot = 0
               ORDER BY created_at ASC LIMIT ?""",
            (guild_id, max(1, min(limit, 250_000))),
        )
        cues = await self.cues()
        stored = 0
        for row in rows:
            content = str(row["content"])
            try:
                observed_at = datetime.fromisoformat(str(row["created_at"]))
            except ValueError:
                continue
            for observation in extract_causal_observations(content, cues):
                await self._store_observation(
                    guild_id=guild_id,
                    source_event_id=f"archive:message:{int(row['message_id'])}",
                    source="archive_backfill",
                    observation=observation,
                    observed_at=observed_at,
                )
                stored += 1
        return stored

    async def rebuild_knowledge(self, *, limit: int = 20_000) -> int:
        rows = await self._database.fetchall(
            """SELECT concept_id, definition, confidence, fetched_at
               FROM knowledge_concepts
               WHERE definition IS NOT NULL AND definition != ''
               ORDER BY confidence DESC, fetched_at DESC LIMIT ?""",
            (max(1, min(limit, 100_000)),),
        )
        guild_rows = await self._database.fetchall(
            "SELECT DISTINCT guild_id FROM events"
        )
        cues = await self.cues()
        stored = 0
        for guild in guild_rows:
            guild_id = int(guild["guild_id"])
            for row in rows:
                try:
                    observed_at = datetime.fromisoformat(str(row["fetched_at"]))
                except ValueError:
                    continue
                for observation in extract_causal_observations(str(row["definition"]), cues):
                    await self._store_observation(
                        guild_id=guild_id,
                        source_event_id=f"knowledge:concept:{row['concept_id']}",
                        source="reference_knowledge",
                        observation=observation,
                        observed_at=observed_at,
                    )
                    stored += 1
        return stored

    async def _store_observation(
        self,
        *,
        guild_id: int,
        source_event_id: str,
        source: str,
        observation: CausalObservation,
        observed_at: datetime,
    ) -> None:
        now = observed_at.isoformat()
        cause_id = _scoped_proposition_id(guild_id, observation.cause)
        effect_id = _scoped_proposition_id(guild_id, observation.effect)
        cause = _with_id(observation.cause, cause_id)
        effect = _with_id(observation.effect, effect_id)
        link_material = f"{guild_id}|{cause_id}|{effect_id}|{observation.relation}"
        link_id = f"causal-link:{hashlib.sha256(link_material.encode()).hexdigest()[:24]}"
        already_stored = await self._database.fetchone(
            """SELECT 1 FROM language_causal_evidence
               WHERE link_id = ? AND source_event_id = ?""",
            (link_id, source_event_id),
        )
        if already_stored is not None:
            return
        async with self._database.transaction() as connection:
            for proposition in (cause, effect):
                await connection.execute(
                    """INSERT INTO language_causal_propositions (
                           proposition_id, guild_id, proposition_kind, surface,
                           subject_lemma, predicate_lemma, object_lemma, terms_json,
                           negated, confidence, evidence_count, first_seen_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                       ON CONFLICT (proposition_id) DO UPDATE SET
                           confidence = MIN(
                               0.98,
                               (language_causal_propositions.confidence
                                * language_causal_propositions.evidence_count
                                + excluded.confidence)
                               / (language_causal_propositions.evidence_count + 1)
                           ),
                           evidence_count = evidence_count + 1,
                           updated_at = excluded.updated_at""",
                    (
                        proposition.proposition_id,
                        guild_id,
                        proposition.kind,
                        proposition.surface,
                        proposition.subject_lemma,
                        proposition.predicate_lemma,
                        proposition.object_lemma,
                        json.dumps(proposition.terms, ensure_ascii=False),
                        int(proposition.negated),
                        proposition.confidence,
                        now,
                        now,
                    ),
                )
            existing = await connection.execute(
                """SELECT positive_evidence, negative_evidence
                   FROM language_causal_links WHERE link_id = ?""",
                (link_id,),
            )
            row = await existing.fetchone()
            positive = (int(row[0]) if row else 0) + int(observation.supports_link)
            negative = (int(row[1]) if row else 0) + int(not observation.supports_link)
            evidence_confidence = (positive + 1.0) / (positive + negative + 2.0)
            confidence = min(
                0.98,
                max(0.05, observation.confidence * 0.72 + evidence_confidence * 0.28),
            )
            await connection.execute(
                """INSERT INTO language_causal_links (
                       link_id, guild_id, cause_proposition_id, effect_proposition_id,
                       relation, confidence, positive_evidence, negative_evidence,
                       evidence_surface, source, first_seen_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (link_id) DO UPDATE SET
                       confidence = excluded.confidence,
                       positive_evidence = excluded.positive_evidence,
                       negative_evidence = excluded.negative_evidence,
                       evidence_surface = CASE
                           WHEN excluded.positive_evidence > language_causal_links.positive_evidence
                           THEN excluded.evidence_surface
                           ELSE language_causal_links.evidence_surface
                       END,
                       updated_at = excluded.updated_at""",
                (
                    link_id,
                    guild_id,
                    cause_id,
                    effect_id,
                    observation.relation,
                    confidence,
                    positive,
                    negative,
                    observation.evidence_surface,
                    source,
                    now,
                    now,
                ),
            )
            evidence_id = _digest(f"{link_id}|{source_event_id}", "causal-evidence")
            await connection.execute(
                """INSERT OR IGNORE INTO language_causal_evidence (
                       evidence_id, link_id, source_event_id, supports_link,
                       confidence, surface, observed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence_id,
                    link_id,
                    source_event_id,
                    int(observation.supports_link),
                    observation.confidence,
                    observation.evidence_surface,
                    now,
                ),
            )

    async def _links(self, guild_id: int) -> tuple[CausalLink, ...]:
        rows = await self._database.fetchall(
            """SELECT link.link_id, link.relation, link.confidence,
                      link.evidence_surface,
                      cause.proposition_id AS cause_id,
                      cause.proposition_kind AS cause_kind,
                      cause.surface AS cause_surface,
                      cause.subject_lemma AS cause_subject,
                      cause.predicate_lemma AS cause_predicate,
                      cause.object_lemma AS cause_object,
                      cause.terms_json AS cause_terms,
                      cause.negated AS cause_negated,
                      cause.confidence AS cause_confidence,
                      effect.proposition_id AS effect_id,
                      effect.proposition_kind AS effect_kind,
                      effect.surface AS effect_surface,
                      effect.subject_lemma AS effect_subject,
                      effect.predicate_lemma AS effect_predicate,
                      effect.object_lemma AS effect_object,
                      effect.terms_json AS effect_terms,
                      effect.negated AS effect_negated,
                      effect.confidence AS effect_confidence
               FROM language_causal_links AS link
               JOIN language_causal_propositions AS cause
                 ON cause.proposition_id = link.cause_proposition_id
               JOIN language_causal_propositions AS effect
                 ON effect.proposition_id = link.effect_proposition_id
               WHERE link.guild_id = ? AND link.confidence >= 0.42
                 AND link.positive_evidence > link.negative_evidence
               ORDER BY link.confidence DESC LIMIT 2000""",
            (guild_id,),
        )
        return tuple(
            CausalLink(
                str(row["link_id"]),
                str(row["relation"]),
                _row_proposition(row, "cause"),
                _row_proposition(row, "effect"),
                float(row["confidence"]),
                str(row["evidence_surface"]),
            )
            for row in rows
        )


def _split_marker(text: str, lowered: str, marker: str) -> tuple[str, str] | None:
    index = lowered.find(marker)
    if index < 0:
        return None
    left = text[:index].strip(_TRIM)
    right = text[index + len(marker) :].strip(_TRIM)
    return (left, right) if left and right else None


def _split_condition(text: str, lowered: str, marker: str) -> tuple[str, str] | None:
    index = lowered.find(marker)
    if index < 0:
        return None
    comma = text.find(",", index + len(marker))
    if comma < 0:
        return None
    condition = text[index + len(marker) : comma].strip(_TRIM)
    effect = re.sub(r"^\s*то\s+", "", text[comma + 1 :], flags=re.IGNORECASE).strip(_TRIM)
    return (condition, effect) if condition and effect else None


def _split_leading_cause(text: str, lowered: str, marker: str) -> tuple[str, str] | None:
    index = lowered.find(marker)
    comma = text.find(",", index + len(marker))
    if index < 0 or comma < 0:
        return None
    cause = text[index + len(marker) : comma].strip(_TRIM)
    effect = text[comma + 1 :].strip(_TRIM)
    return (cause, effect) if cause and effect else None


def _proposition_from_text(text: str) -> CausalProposition | None:
    surface = text.strip(_TRIM)
    if not surface or len(surface) > 240:
        return None
    structure = analyze_structured_utterance(surface)
    frame = next((item for item in structure.clauses if item.predicate_index is not None), None)
    if frame is None or frame.predicate_index is None:
        return None
    subject = _first_token_lemma(structure, frame.subject_indices)
    direct_object = _first_token_lemma(structure, frame.direct_object_indices)
    predicate = structure.tokens[frame.predicate_index].lemma
    terms = tuple(content_lemmas(surface))
    if not terms:
        return None
    key = "|".join((subject or "", predicate, direct_object or "", str(frame.negated)))
    return CausalProposition(
        _digest(key, "causal-proposition"),
        "clause",
        surface,
        subject,
        predicate,
        direct_object,
        terms,
        frame.negated,
        min(0.96, frame.confidence * 0.84 + 0.10),
    )


def _argument_proposition(
    structure: StructuredUtterance,
    clause: ClauseFrame,
    *,
    role_name: str,
    complement_only: bool = False,
) -> CausalProposition | None:
    if role_name == "subject":
        indices = clause.subject_indices
    else:
        indices = () if complement_only else clause.direct_object_indices
        if not indices:
            indices = next(
                (item.object_indices for item in clause.complements if item.object_indices),
                (),
            )
    if not indices:
        return None
    tokens = tuple(structure.tokens[index] for index in indices)
    surface = " ".join(token.surface for token in tokens)
    terms = tuple(dict.fromkeys(token.lemma for token in tokens if token.lemma))
    if not terms:
        return None
    return CausalProposition(
        _digest("|".join(terms), "causal-proposition"),
        "nominal",
        surface,
        terms[0],
        None,
        None,
        terms,
        False,
        min(0.92, clause.confidence * 0.82 + 0.08),
    )


def _first_token_lemma(structure: StructuredUtterance, indices: tuple[int, ...]) -> str | None:
    return structure.tokens[indices[0]].lemma if indices else None


def _scoped_proposition_id(guild_id: int, proposition: CausalProposition) -> str:
    return _digest(f"{guild_id}|{proposition.proposition_id}", "causal-proposition")


def _with_id(proposition: CausalProposition, proposition_id: str) -> CausalProposition:
    return CausalProposition(
        proposition_id,
        proposition.kind,
        proposition.surface,
        proposition.subject_lemma,
        proposition.predicate_lemma,
        proposition.object_lemma,
        proposition.terms,
        proposition.negated,
        proposition.confidence,
    )


def _row_proposition(row, prefix: str) -> CausalProposition:
    return CausalProposition(
        str(row[f"{prefix}_id"]),
        str(row[f"{prefix}_kind"]),
        str(row[f"{prefix}_surface"]),
        str(row[f"{prefix}_subject"]) if row[f"{prefix}_subject"] is not None else None,
        str(row[f"{prefix}_predicate"])
        if row[f"{prefix}_predicate"] is not None
        else None,
        str(row[f"{prefix}_object"]) if row[f"{prefix}_object"] is not None else None,
        tuple(str(item) for item in json.loads(str(row[f"{prefix}_terms"]))),
        bool(row[f"{prefix}_negated"]),
        float(row[f"{prefix}_confidence"]),
    )


def _term_fit(query_terms: tuple[str, ...], proposition_terms: tuple[str, ...]) -> float:
    query = set(query_terms)
    proposition = set(proposition_terms)
    overlap = len(query & proposition)
    return overlap / max(1, len(query))


def _render_path(path: tuple[CausalLink, ...], direction: str) -> str:
    ordered = tuple(reversed(path)) if direction == "cause" else path
    surfaces = tuple(dict.fromkeys(_render_link(link) for link in ordered))
    return " ".join(surfaces[:3])


def _render_link(link: CausalLink) -> str:
    cause = link.cause.surface.strip(_TRIM)
    effect = link.effect.surface.strip(_TRIM)
    if link.cause.kind == "clause" and link.effect.kind == "clause":
        if link.relation == "enables":
            return f"Если {cause}, то {_lower_initial(effect)}."
        if link.relation == "causes":
            return f"{effect[:1].upper() + effect[1:]}, потому что {_lower_initial(cause)}."
        if link.relation == "precedes":
            return f"{cause[:1].upper() + cause[1:]}, затем {_lower_initial(effect)}."
    evidence = link.evidence_surface.strip()
    evidence = evidence[:1].upper() + evidence[1:] if evidence else evidence
    return evidence if evidence.endswith((".", "!", "?")) else f"{evidence}."


def _lower_initial(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _digest(material: str, prefix: str) -> str:
    return f"{prefix}:{hashlib.sha256(material.encode()).hexdigest()[:24]}"
