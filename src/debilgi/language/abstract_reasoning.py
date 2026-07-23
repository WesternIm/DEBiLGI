"""Separated working-world models for abstraction, space and imagination.

Mental models are deliberately not factual memory.  They represent the world
described by the current turn, including hypothetical and fictional premises,
and may support local deductions without promoting those deductions to truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from debilgi.language.content import content_lemmas
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.storage.database import Database

_WORD = r"[а-яёa-z][а-яёa-z0-9_-]{0,47}"
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:представь|представим|вообрази|допустим|предположим|пусть|если)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(?:если\s+бы|как\s+будто\s+бы|могло\s+бы|мог\s+бы|могла\s+бы)\b",
    re.IGNORECASE,
)
_FICTIONAL_RE = re.compile(
    r"\b(?:вымышленн\w*|фантастическ\w*|сказочн\w*|воображаем\w*)\b",
    re.IGNORECASE,
)
_METAPHOR_RE = re.compile(
    rf"\b(?P<subject>{_WORD})\s+(?:—|-)?\s*(?:это\s+)?"
    rf"(?:как|словно|будто)\s+"
    rf"(?P<object>{_WORD})\b",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(
    rf"\bгде\s+(?:же\s+)?(?:находится\s+|лежит\s+|стоит\s+)?(?P<focus>{_WORD})\b",
    re.IGNORECASE,
)
_SPATIAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"внутри\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "inside",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"слева\s+от\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "left_of",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"справа\s+от\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "right_of",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|висит\s+|стоит\s+)?"
            rf"над\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "above",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"под\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "below",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"перед\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "in_front_of",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"за\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "behind",
    ),
    (
        re.compile(
            rf"\b(?P<subject>{_WORD})\s+(?:находится\s+|лежит\s+|стоит\s+)?"
            rf"рядом\s+с\s+(?P<object>{_WORD})\b",
            re.IGNORECASE,
        ),
        "near",
    ),
)
_INVERSE_RELATIONS = {
    "inside": "contains",
    "contains": "inside",
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "near": "near",
}
_TRANSITIVE_RELATIONS = frozenset(
    {"inside", "left_of", "right_of", "above", "below", "in_front_of", "behind"}
)
_RELATION_SURFACES = {
    "inside": "внутри",
    "contains": "вокруг",
    "left_of": "слева от",
    "right_of": "справа от",
    "above": "над",
    "below": "под",
    "in_front_of": "перед",
    "behind": "за",
    "near": "рядом с",
    "maps_to": "подобно",
}
_RELATION_OBJECT_CASES = {
    "inside": "genitive",
    "contains": "genitive",
    "left_of": "genitive",
    "right_of": "genitive",
    "above": "instrumental",
    "below": "instrumental",
    "in_front_of": "instrumental",
    "behind": "instrumental",
    "near": "instrumental",
}


@dataclass(frozen=True, slots=True)
class AbstractRelation:
    subject: str
    relation: str
    object: str
    confidence: float
    inferred: bool = False
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MentalModel:
    model_id: str
    guild_id: int
    channel_id: int | None
    source_event_id: str | None
    mode: str
    concepts: tuple[str, ...]
    assumptions: tuple[str, ...]
    relations: tuple[AbstractRelation, ...]
    question_focus: str | None
    confidence: float
    status: str = "working"
    parent_model_id: str | None = None

    @property
    def imaginative(self) -> bool:
        return self.mode in {
            "hypothetical",
            "counterfactual",
            "metaphorical",
            "fictional",
        }


@dataclass(frozen=True, slots=True)
class AbstractResolution:
    text: str
    confidence: float
    focus_terms: tuple[str, ...]
    evidence: tuple[str, ...]
    mode: str


class AbstractReasoningService:
    """Build, retain and elaborate non-factual working-world models."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def observe_turn(
        self,
        *,
        guild_id: int,
        channel_id: int | None,
        source_event_id: str,
        prompt: str,
        context_messages: tuple[str, ...] = (),
        observed_at: datetime | None = None,
    ) -> MentalModel | None:
        model = build_mental_model(
            prompt,
            guild_id=guild_id,
            channel_id=channel_id,
            source_event_id=source_event_id,
            context_messages=context_messages,
        )
        if model is None:
            return None
        await self.remember(model, observed_at=observed_at)
        return model

    async def remember(
        self,
        model: MentalModel,
        *,
        observed_at: datetime | None = None,
    ) -> None:
        now = (observed_at or datetime.now(UTC)).isoformat()
        await self._database.execute(
            """INSERT INTO language_mental_models (
                   model_id, guild_id, channel_id, source_event_id, parent_model_id,
                   mode, concepts_json, assumptions_json, question_focus,
                   confidence, status, created_at, last_used_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (guild_id, source_event_id) DO UPDATE SET
                   mode = excluded.mode,
                   concepts_json = excluded.concepts_json,
                   assumptions_json = excluded.assumptions_json,
                   question_focus = excluded.question_focus,
                   confidence = MAX(confidence, excluded.confidence),
                   last_used_at = excluded.last_used_at""",
            (
                model.model_id,
                model.guild_id,
                model.channel_id,
                model.source_event_id,
                model.parent_model_id,
                model.mode,
                json.dumps(model.concepts, ensure_ascii=False),
                json.dumps(model.assumptions, ensure_ascii=False),
                model.question_focus,
                model.confidence,
                model.status,
                now,
                now,
            ),
        )
        stored = await self._database.fetchone(
            """SELECT model_id FROM language_mental_models
               WHERE guild_id = ? AND source_event_id = ?""",
            (model.guild_id, model.source_event_id),
        )
        stored_id = str(stored["model_id"]) if stored is not None else model.model_id
        if model.relations:
            async with self._database.transaction() as connection:
                await connection.executemany(
                    """INSERT INTO language_mental_relations (
                           model_id, subject_key, relation, object_key,
                           confidence, inferred, evidence_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (model_id, subject_key, relation, object_key)
                       DO UPDATE SET
                           confidence = MAX(confidence, excluded.confidence),
                           inferred = MIN(inferred, excluded.inferred),
                           evidence_json = excluded.evidence_json""",
                    (
                        (
                            stored_id,
                            relation.subject,
                            relation.relation,
                            relation.object,
                            relation.confidence,
                            int(relation.inferred),
                            json.dumps(relation.evidence, ensure_ascii=False),
                        )
                        for relation in model.relations
                    ),
                )

    async def background_elaborate(
        self,
        *,
        limit: int = 12,
        observed_at: datetime | None = None,
    ) -> int:
        """Derive safe spatial closure while idle; never promote it to facts."""

        rows = await self._database.fetchall(
            """SELECT model_id, guild_id, channel_id, source_event_id, parent_model_id,
                      mode, concepts_json, assumptions_json, question_focus,
                      confidence, status
               FROM language_mental_models
               WHERE status IN ('working', 'retained')
               ORDER BY last_used_at DESC LIMIT ?""",
            (max(1, min(limit, 100)),),
        )
        inserted = 0
        now = observed_at or datetime.now(UTC)
        for row in rows:
            relation_rows = await self._database.fetchall(
                """SELECT subject_key, relation, object_key, confidence,
                          inferred, evidence_json
                   FROM language_mental_relations WHERE model_id = ?""",
                (str(row["model_id"]),),
            )
            relations = tuple(
                AbstractRelation(
                    str(item["subject_key"]),
                    str(item["relation"]),
                    str(item["object_key"]),
                    float(item["confidence"]),
                    bool(item["inferred"]),
                    tuple(json.loads(str(item["evidence_json"]))),
                )
                for item in relation_rows
            )
            expanded = infer_relations(relations)
            known = {
                (item.subject, item.relation, item.object)
                for item in relations
            }
            additions = tuple(
                item
                for item in expanded
                if (item.subject, item.relation, item.object) not in known
            )
            if not additions:
                continue
            model = MentalModel(
                model_id=str(row["model_id"]),
                guild_id=int(row["guild_id"]),
                channel_id=(
                    int(row["channel_id"]) if row["channel_id"] is not None else None
                ),
                source_event_id=(
                    str(row["source_event_id"])
                    if row["source_event_id"] is not None
                    else None
                ),
                parent_model_id=(
                    str(row["parent_model_id"])
                    if row["parent_model_id"] is not None
                    else None
                ),
                mode=str(row["mode"]),
                concepts=tuple(json.loads(str(row["concepts_json"]))),
                assumptions=tuple(json.loads(str(row["assumptions_json"]))),
                relations=additions,
                question_focus=(
                    str(row["question_focus"])
                    if row["question_focus"] is not None
                    else None
                ),
                confidence=float(row["confidence"]),
                status=str(row["status"]),
            )
            await self.remember(model, observed_at=now)
            inserted += len(additions)
        return inserted


def build_mental_model(
    prompt: str,
    *,
    guild_id: int = 0,
    channel_id: int | None = None,
    source_event_id: str | None = None,
    context_messages: tuple[str, ...] = (),
) -> MentalModel | None:
    """Extract an abstract working model from explicit premises."""

    normalized = " ".join(prompt.casefold().replace("ё", "е").split())
    mode = _detect_mode(normalized)
    relations = list(_spatial_relations(normalized))
    metaphor = _METAPHOR_RE.search(normalized)
    if metaphor is not None:
        relations.append(
            AbstractRelation(
                _lemma(metaphor.group("subject"), expected_case="nominative"),
                "maps_to",
                _lemma(metaphor.group("object"), expected_case="nominative"),
                0.72,
                evidence=("explicit_analogy",),
            )
        )
        mode = "metaphorical"
    focus_match = _WHERE_RE.search(normalized)
    focus = (
        _lemma(focus_match.group("focus"), expected_case="nominative")
        if focus_match is not None
        else None
    )

    contextual_assumptions: list[str] = []
    if focus is not None:
        contextual_relations, contextual_modes, contextual_assumptions = (
            _connected_context_relations(focus, context_messages)
        )
        known_relation_keys = {
            (item.subject, item.relation, item.object)
            for item in relations
        }
        relations.extend(
            item
            for item in contextual_relations
            if (item.subject, item.relation, item.object) not in known_relation_keys
        )
        if mode == "grounded" and contextual_modes:
            mode = contextual_modes[0]

    if not relations and mode == "grounded":
        return None
    expanded = infer_relations(tuple(relations))
    concepts = tuple(
        dict.fromkeys(
            (
                *content_lemmas(prompt),
                *(item.subject for item in expanded),
                *(item.object for item in expanded),
            )
        )
    )[:32]
    assumptions = (
        tuple(
            dict.fromkeys(
                (
                    normalized[:500],
                    *(item[:500] for item in contextual_assumptions),
                )
            )
        )
        if mode in {"hypothetical", "counterfactual", "fictional", "metaphorical"}
        else ()
    )
    fingerprint = (
        source_event_id
        or hashlib.sha256(
            f"{guild_id}|{channel_id}|{normalized}".encode()
        ).hexdigest()[:24]
    )
    return MentalModel(
        model_id=f"mental-model:{fingerprint}:{uuid4().hex[:8]}",
        guild_id=guild_id,
        channel_id=channel_id,
        source_event_id=source_event_id,
        mode=mode,
        concepts=concepts,
        assumptions=assumptions,
        relations=expanded,
        question_focus=focus,
        confidence=(
            0.88
            if relations and focus
            else 0.78
            if relations
            else 0.62
        ),
    )


def resolve_abstract_question(model: MentalModel | None) -> AbstractResolution | None:
    if model is None or model.question_focus is None:
        return None
    relation = next(
        (
            item
            for item in model.relations
            if item.subject == model.question_focus
            and item.relation in _RELATION_SURFACES
            and item.relation != "maps_to"
        ),
        None,
    )
    if relation is None:
        return None
    prefix = (
        "В этой воображаемой ситуации "
        if model.imaginative
        else "По описанной схеме "
    )
    object_case = _RELATION_OBJECT_CASES.get(relation.relation)
    text = (
        f"{prefix}{_surface(model.question_focus, grammatical_case='nominative')} "
        f"находится {_RELATION_SURFACES[relation.relation]} "
        f"{_surface(relation.object, grammatical_case=object_case)}."
    )
    return AbstractResolution(
        text=text,
        confidence=min(model.confidence, relation.confidence),
        focus_terms=(model.question_focus, relation.object),
        evidence=(
            f"mental_mode:{model.mode}",
            *relation.evidence,
        ),
        mode=model.mode,
    )


def infer_relations(
    relations: tuple[AbstractRelation, ...],
) -> tuple[AbstractRelation, ...]:
    """Add inverses and bounded transitive spatial consequences."""

    result: dict[tuple[str, str, str], AbstractRelation] = {
        (item.subject, item.relation, item.object): item
        for item in relations
        if item.subject != item.object
    }
    for item in tuple(result.values()):
        inverse = _INVERSE_RELATIONS.get(item.relation)
        if inverse is None:
            continue
        key = (item.object, inverse, item.subject)
        result.setdefault(
            key,
            AbstractRelation(
                item.object,
                inverse,
                item.subject,
                item.confidence * 0.96,
                True,
                (*item.evidence, f"inverse:{item.relation}"),
            ),
        )
    for _ in range(2):
        additions: list[AbstractRelation] = []
        current = tuple(result.values())
        for left in current:
            if left.relation not in _TRANSITIVE_RELATIONS:
                continue
            for right in current:
                if (
                    right.relation != left.relation
                    or left.object != right.subject
                    or left.subject == right.object
                ):
                    continue
                key = (left.subject, left.relation, right.object)
                if key in result:
                    continue
                additions.append(
                    AbstractRelation(
                        left.subject,
                        left.relation,
                        right.object,
                        min(left.confidence, right.confidence) * 0.82,
                        True,
                        (
                            *left.evidence,
                            *right.evidence,
                            f"transitive:{left.relation}",
                        ),
                    )
                )
        if not additions:
            break
        for item in additions:
            result.setdefault((item.subject, item.relation, item.object), item)
    return tuple(result.values())


def _spatial_relations(text: str) -> tuple[AbstractRelation, ...]:
    relations: list[AbstractRelation] = []
    for pattern, relation in _SPATIAL_PATTERNS:
        for match in pattern.finditer(text):
            subject = _lemma(match.group("subject"), expected_case="nominative")
            object_ = _lemma(match.group("object"), expected_case="genitive")
            if subject == object_:
                continue
            relations.append(
                AbstractRelation(
                    subject,
                    relation,
                    object_,
                    0.90,
                    evidence=(f"explicit_spatial:{relation}",),
                )
            )
    return tuple(
        {
            (item.subject, item.relation, item.object): item
            for item in relations
        }.values()
    )


def _detect_mode(text: str) -> str:
    if _COUNTERFACTUAL_RE.search(text):
        return "counterfactual"
    if _FICTIONAL_RE.search(text):
        return "fictional"
    if _HYPOTHETICAL_RE.search(text):
        return "hypothetical"
    if _METAPHOR_RE.search(text):
        return "metaphorical"
    return "grounded"


def _connected_context_relations(
    focus: str,
    context_messages: tuple[str, ...],
) -> tuple[tuple[AbstractRelation, ...], tuple[str, ...], list[str]]:
    """Return only the recent context component connected to the question focus."""

    entries: list[tuple[str, str, tuple[AbstractRelation, ...]]] = []
    for message in reversed(context_messages[-6:]):
        normalized = " ".join(message.casefold().replace("ё", "е").split())
        candidates = _spatial_relations(normalized)
        if candidates:
            entries.append((normalized, _detect_mode(normalized), candidates))

    connected = {focus}
    selected_keys: set[tuple[str, str, str]] = set()
    changed = True
    while changed:
        changed = False
        for _, _, candidates in entries:
            for item in candidates:
                key = (item.subject, item.relation, item.object)
                if key in selected_keys:
                    continue
                if item.subject not in connected and item.object not in connected:
                    continue
                selected_keys.add(key)
                connected.update((item.subject, item.object))
                changed = True

    selected_relations: list[AbstractRelation] = []
    selected_modes: list[str] = []
    selected_assumptions: list[str] = []
    for normalized, entry_mode, candidates in entries:
        matching = tuple(
            item
            for item in candidates
            if (item.subject, item.relation, item.object) in selected_keys
        )
        if not matching:
            continue
        selected_relations.extend(matching)
        selected_assumptions.append(normalized)
        if entry_mode != "grounded" and entry_mode not in selected_modes:
            selected_modes.append(entry_mode)

    return (
        tuple(selected_relations),
        tuple(selected_modes),
        selected_assumptions,
    )


def _lemma(value: str, *, expected_case: str | None = None) -> str:
    normalized = value.casefold().replace("ё", "е")
    if expected_case is not None:
        reading = next(
            (
                item
                for item in DEFAULT_MORPHOLOGY.parse_word(normalized)
                if item.case == expected_case
            ),
            None,
        )
        if reading is not None:
            return reading.lemma
    return DEFAULT_MORPHOLOGY.lemma(normalized)


def _surface(value: str, *, grammatical_case: str | None = None) -> str:
    normalized = value.replace("_", " ")
    if " " in normalized or grammatical_case is None:
        return normalized
    return (
        DEFAULT_MORPHOLOGY.inflect(
            normalized,
            case=grammatical_case,
            number="singular",
        )
        or normalized
    )
