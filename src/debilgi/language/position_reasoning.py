"""Evidence-aware personal positions without stored reply wording.

The repository remembers propositions and abstract expression modes.  Surface
sentences are rebuilt from semantic roles every time, so feedback can develop
the agent's stance and conversational choices without creating canned replies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from debilgi.events.models import Event, EventType
from debilgi.language.content import content_lemmas, content_tokens
from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    ProductiveSurface,
    SemanticDependency,
    realize_thought,
)
from debilgi.language.tokenizer import tokenize
from debilgi.social.signals import AFFIRMATION, APPROVAL, AVERSION, DISAPPROVAL, REFUSAL
from debilgi.storage.database import Database

_POSITIVE_REACTIONS = frozenset({"❤️", "❤", "👍", "✅", "🔥", "💯", "🥰", "😍", "🤩"})
_NEGATIVE_REACTIONS = frozenset({"👎", "🤮", "💩", "❌", "🤢"})
_POSITION_PREDICATES = frozenset({"быть", "считать", "думать", "полагать"})


@dataclass(frozen=True, slots=True)
class StoredPosition:
    position_key: str
    target: str
    quality: str
    stance: str
    score: float
    confidence: float
    evidence: tuple[str, ...]
    observations: int


@dataclass(frozen=True, slots=True)
class PositionDecision:
    position_key: str
    target: str
    quality: str
    stance: str
    score: float
    confidence: float
    evidence: tuple[str, ...]
    act_kind: str
    response_obligation: str
    move: str
    expression_mode: str


class PositionRepository:
    """Durable propositions plus feedback on abstract expression choices."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def recall(self, guild_id: int, position_key: str) -> StoredPosition | None:
        row = await self._database.fetchone(
            """SELECT * FROM language_positions
               WHERE guild_id = ? AND position_key = ?""",
            (guild_id, position_key),
        )
        if row is None:
            return None
        return StoredPosition(
            position_key=str(row["position_key"]),
            target=str(row["target"]),
            quality=str(row["quality"]),
            stance=str(row["stance"]),
            score=float(row["score"]),
            confidence=float(row["confidence"]),
            evidence=tuple(json.loads(str(row["evidence_json"]))),
            observations=int(row["observation_count"]),
        )

    async def preferred_expression_mode(
        self,
        guild_id: int,
        *,
        act_kind: str,
        response_obligation: str,
        move: str,
    ) -> str | None:
        rows = await self._database.fetchall(
            """SELECT expression_mode, utility
               FROM language_dialogue_intent_preferences
               WHERE guild_id = ? AND act_kind = ? AND response_obligation = ?
                 AND move = ?""",
            (guild_id, act_kind, response_obligation, move),
        )
        if not rows:
            return None
        # Untried modes are real alternatives with neutral utility.  This is
        # what lets negative feedback move the system away from a repeatedly
        # unsuccessful realization instead of selecting that same mode merely
        # because it is the only row observed so far.
        utilities = {"direct": 0.0, "qualified": 0.0}
        for row in rows:
            mode = str(row["expression_mode"])
            if mode in utilities:
                utilities[mode] = float(row["utility"])
        best_mode, best_utility = max(
            utilities.items(),
            key=lambda item: (item[1], item[0] == "direct"),
        )
        return best_mode if best_utility != 0.0 or any(
            utility < 0.0 for utility in utilities.values()
        ) else None

    async def record_expressed(
        self,
        decision: PositionDecision,
        *,
        guild_id: int,
        source_event_id: str,
        candidate_id: str | None,
        response_message_id: int,
        observed_at: datetime,
    ) -> None:
        now = observed_at.isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_positions (
                       guild_id, position_key, target, quality, stance, score,
                       confidence, evidence_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (guild_id, position_key) DO UPDATE SET
                       target = excluded.target,
                       quality = excluded.quality,
                       stance = CASE
                           WHEN excluded.confidence >= language_positions.confidence
                           THEN excluded.stance ELSE language_positions.stance END,
                       score = language_positions.score * 0.65 + excluded.score * 0.35,
                       confidence = MIN(
                           0.99,
                           language_positions.confidence * 0.72
                           + excluded.confidence * 0.28 + 0.01
                       ),
                       evidence_json = excluded.evidence_json,
                       observation_count = observation_count + 1,
                       updated_at = excluded.updated_at""",
                (
                    guild_id,
                    decision.position_key,
                    decision.target,
                    decision.quality,
                    decision.stance,
                    decision.score,
                    decision.confidence,
                    _json(decision.evidence),
                    now,
                    now,
                ),
            )
            await connection.execute(
                """INSERT INTO language_dialogue_intent_preferences (
                       guild_id, act_kind, response_obligation, move,
                       expression_mode, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (
                       guild_id, act_kind, response_obligation, move, expression_mode
                   ) DO UPDATE SET observation_count = observation_count + 1,
                                   updated_at = excluded.updated_at""",
                (
                    guild_id,
                    decision.act_kind,
                    decision.response_obligation,
                    decision.move,
                    decision.expression_mode,
                    now,
                ),
            )
            await connection.execute(
                """INSERT OR REPLACE INTO language_expressed_positions (
                       response_message_id, guild_id, source_event_id, candidate_id,
                       position_key, stance, score, confidence, act_kind,
                       response_obligation, move, expression_mode, expressed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    response_message_id,
                    guild_id,
                    source_event_id,
                    candidate_id,
                    decision.position_key,
                    decision.stance,
                    decision.score,
                    decision.confidence,
                    decision.act_kind,
                    decision.response_obligation,
                    decision.move,
                    decision.expression_mode,
                    now,
                ),
            )

    async def observe_feedback(self, event: Event) -> None:
        response_message_id = _feedback_target(event)
        if response_message_id is None:
            return
        row = await self._database.fetchone(
            """SELECT * FROM language_expressed_positions
               WHERE guild_id = ? AND response_message_id = ?""",
            (event.guild_id, response_message_id),
        )
        if row is None:
            return
        reward, kind, attribution = _feedback_signal(event)
        if reward == 0.0 or attribution < 0.60:
            return
        inserted = await self._database.execute(
            """INSERT OR IGNORE INTO language_position_feedback (
                   feedback_event_id, response_message_id, guild_id, position_key,
                   feedback_kind, reward, attribution_confidence, observed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                response_message_id,
                event.guild_id,
                str(row["position_key"]),
                kind,
                reward,
                attribution,
                event.utc_timestamp.isoformat(),
            ),
        )
        if inserted.rowcount != 1:
            return
        delta = max(-0.12, min(0.12, reward * attribution * 0.10))
        positive = int(reward > 0.0)
        negative = int(reward < 0.0)
        await self._database.execute(
            """UPDATE language_positions
               SET confidence = MAX(0.15, MIN(0.99, confidence + ?)),
                   score = score * CASE WHEN ? < 0 THEN 0.92 ELSE 1.0 END,
                   positive_feedback = positive_feedback + ?,
                   negative_feedback = negative_feedback + ?,
                   updated_at = ?
               WHERE guild_id = ? AND position_key = ?""",
            (
                delta,
                reward,
                positive,
                negative,
                event.utc_timestamp.isoformat(),
                event.guild_id,
                str(row["position_key"]),
            ),
        )
        await self._database.execute(
            """UPDATE language_dialogue_intent_preferences
               SET utility = MAX(-1.0, MIN(1.0, utility + ?)),
                   positive_feedback = positive_feedback + ?,
                   negative_feedback = negative_feedback + ?,
                   updated_at = ?
               WHERE guild_id = ? AND act_kind = ? AND response_obligation = ?
                 AND move = ? AND expression_mode = ?""",
            (
                delta,
                positive,
                negative,
                event.utc_timestamp.isoformat(),
                event.guild_id,
                str(row["act_kind"]),
                str(row["response_obligation"]),
                str(row["move"]),
                str(row["expression_mode"]),
            ),
        )
        await self._database.execute(
            """UPDATE language_expressed_positions SET feedback_reward = ?
               WHERE response_message_id = ?""",
            (reward, response_message_id),
        )


class PositionReasoningService:
    def __init__(self, repository: PositionRepository | None = None) -> None:
        self.repository = repository

    async def resolve(
        self,
        guild_id: int,
        frame: DialogueActFrame | None,
        *,
        facts: tuple[str, ...] = (),
        context_messages: tuple[str, ...] = (),
    ) -> PositionDecision | None:
        request = _position_request(frame)
        if request is None or frame is None:
            return None
        target, quality = request
        key = _position_key(target, quality)
        stored = await self.repository.recall(guild_id, key) if self.repository else None
        score, confidence, evidence = _derive_score(
            target,
            quality,
            facts=facts,
            context_messages=context_messages,
            stored=stored,
        )
        stance = "support" if score >= 0.22 else "oppose" if score <= -0.22 else "withhold"
        default_mode = "direct" if stance != "withhold" and confidence >= 0.64 else "qualified"
        preferred = (
            await self.repository.preferred_expression_mode(
                guild_id,
                act_kind=frame.kind,
                response_obligation=frame.response_obligation,
                move="state_position",
            )
            if self.repository
            else None
        )
        mode = preferred if preferred in {"direct", "qualified"} else default_mode
        if stance == "withhold":
            mode = "qualified"
        return PositionDecision(
            key,
            target,
            quality,
            stance,
            score,
            confidence,
            evidence,
            frame.kind,
            frame.response_obligation,
            "state_position",
            mode,
        )


def realize_position(
    decision: PositionDecision,
    patterns: tuple[DependencyPattern, ...] = (),
) -> tuple[ProductiveSurface, ...]:
    """Express a decision through dependencies rather than an answer string."""

    negated = decision.stance != "support"
    dependencies: list[SemanticDependency] = [
        SemanticDependency(
            "subject",
            ConceptArgument(
                ConceptLexeme(
                    "я",
                    "я",
                    "pronoun",
                    (("number", "singular"), ("person", "first")),
                )
            ),
        ),
        SemanticDependency("direct_object", _target_argument(decision.target)),
        SemanticDependency(
            "predicate_complement",
            ConceptArgument(ConceptLexeme(decision.quality, decision.quality, "noun")),
        ),
    ]
    learned_patterns = patterns
    if decision.expression_mode == "qualified":
        dependencies.append(
            SemanticDependency(
                "manner",
                ConceptArgument(ConceptLexeme("пока", "пока", "adverb")),
            )
        )
        learned_patterns = (
            DependencyPattern(
                (
                    "subject",
                    "manner",
                    "negation",
                    "predicate",
                    "direct_object",
                    "predicate_complement",
                ),
                confidence=0.98,
                evidence_count=8,
                source="qualified_position_information_structure",
                focus_role="predicate_complement",
                negation_mode="required",
            ),
            *patterns,
        )
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "считать",
            "считаю",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "first")),
        ),
        dependencies=tuple(dependencies),
        negated=negated,
        response_polarity=not negated,
        focus_role="predicate_complement",
        confidence=decision.confidence,
        evidence=decision.evidence,
    )
    surfaces = realize_thought(thought, learned_patterns, limit=4)
    if decision.expression_mode != "qualified":
        return surfaces
    return tuple(
        surface
        for surface in surfaces
        if "manner" in surface.role_order
        and surface.role_order.index("manner")
        < surface.role_order.index("predicate")
    )


def _position_request(frame: DialogueActFrame | None) -> tuple[str, str] | None:
    if (
        frame is None
        or not frame.complete
        or frame.response_obligation != "express_position"
        or len(frame.objects) < 2
        or (frame.predicate is not None and frame.predicate not in _POSITION_PREDICATES)
    ):
        return None
    target, quality = frame.objects[:2]
    if target.casefold() == quality.casefold():
        return None
    return target.casefold(), quality.casefold()


def _derive_score(
    target: str,
    quality: str,
    *,
    facts: tuple[str, ...],
    context_messages: tuple[str, ...],
    stored: StoredPosition | None,
) -> tuple[float, float, tuple[str, ...]]:
    weighted: list[tuple[float, float, str]] = []
    for fact in facts:
        if _mentions_proposition(fact, target, quality):
            negative = _is_negated(fact)
            weighted.append((-0.88 if negative else 0.88, 0.92, "grounded_fact"))
    for message in context_messages[-10:]:
        if _mentions_proposition(message, target, quality):
            negative = _is_negated(message)
            weighted.append((-0.42 if negative else 0.42, 0.48, "dialogue_claim"))
    if stored is not None:
        weighted.append((stored.score, stored.confidence, "stored_position"))
    if not weighted:
        # Lack of support is not evidence for a negative fact.  It creates a
        # provisional refusal to endorse the label, which remains revisable.
        return -0.16, 0.46, ("no_supporting_evidence", "provisional_non_endorsement")
    numerator = sum(score * weight for score, weight, _ in weighted)
    denominator = sum(weight for _, weight, _ in weighted)
    score = max(-1.0, min(1.0, numerator / max(0.01, denominator)))
    confidence = max(
        0.38,
        min(0.96, sum(weight for _, weight, _ in weighted) / len(weighted)),
    )
    return score, confidence, tuple(dict.fromkeys(reason for _, _, reason in weighted))


def _mentions_proposition(text: str, target: str, quality: str) -> bool:
    surfaces = frozenset(content_tokens(text))
    lemmas = frozenset(content_lemmas(text))
    target_forms = {target, *content_tokens(target), *content_lemmas(target)}
    quality_forms = {quality, *content_tokens(quality), *content_lemmas(quality)}
    return bool((surfaces | lemmas) & target_forms) and bool(
        (surfaces | lemmas) & quality_forms
    )


def _is_negated(text: str) -> bool:
    words = {
        item.casefold().replace("ё", "е")
        for item in tokenize(text)
        if item[:1].isalnum()
    }
    return bool(words & {"не", "нет", "никогда", "ни"})


def _target_argument(target: str) -> ConceptArgument:
    reading = DEFAULT_MORPHOLOGY.best(target)
    if reading is not None and reading.is_known:
        return ConceptArgument(
            ConceptLexeme(
                reading.lemma,
                target,
                reading.part_of_speech,
                tuple(
                    (name, value)
                    for name, value in (
                        ("number", reading.number),
                        ("gender", reading.gender),
                        ("animacy", reading.animacy),
                    )
                    if value is not None
                ),
            )
        )
    return ConceptArgument(
        ConceptLexeme(
            "он",
            "его",
            "pronoun",
            (
                ("case", "accusative"),
                ("number", "singular"),
                ("gender", "masculine"),
                ("person", "third"),
            ),
        ),
        grammatical_case="accusative",
    )


def _position_key(target: str, quality: str) -> str:
    target_reading = DEFAULT_MORPHOLOGY.best(target)
    quality_reading = DEFAULT_MORPHOLOGY.best(quality)
    target_key = (
        target_reading.lemma
        if target_reading is not None and target_reading.is_known
        else target.casefold()
    )
    quality_key = (
        quality_reading.lemma
        if quality_reading is not None and quality_reading.is_known
        else quality.casefold()
    )
    digest = hashlib.sha256(f"{target_key}|{quality_key}".encode()).hexdigest()[:24]
    return f"position:{digest}"


def _feedback_target(event: Event) -> int | None:
    if event.event_type in {EventType.REACTION_ADDED, EventType.REACTION_REMOVED}:
        value = event.payload.get("message_id")
    elif event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}:
        value = event.payload.get("reply_to_message_id")
    else:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _feedback_signal(event: Event) -> tuple[float, str, float]:
    if event.event_type in {EventType.REACTION_ADDED, EventType.REACTION_REMOVED}:
        emoji = str(event.payload.get("emoji", ""))
        direction = 1.0 if event.event_type is EventType.REACTION_ADDED else -1.0
        if emoji in _POSITIVE_REACTIONS:
            return 0.8 * direction, "positive_reaction", 0.90
        if emoji in _NEGATIVE_REACTIONS:
            return -0.8 * direction, "negative_reaction", 0.90
        return 0.0, "neutral_reaction", 0.0
    content = event.payload.get("content")
    if not isinstance(content, str):
        return 0.0, "empty_reply", 0.0
    words = set(tokenize(content.casefold().replace("ё", "е")))
    if words & (DISAPPROVAL | AVERSION):
        return -1.0, "explicit_rejection", 0.95
    if words & (APPROVAL | AFFIRMATION):
        return 1.0, "explicit_confirmation", 0.95
    if words & REFUSAL:
        return -0.65, "refusal", 0.75
    return 0.0, "conversation_continued", 0.10


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
