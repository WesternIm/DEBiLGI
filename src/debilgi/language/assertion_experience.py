"""Learn applicability of local hypotheses from explicit, attributed feedback.

No reply wording is stored. Votes concern the usefulness of a conversational
interpretation, not the truth of a proposition about the world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.assertion_memory import ContinuationPlan
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class ExperienceGuidance:
    confirmations: int = 0
    rejections: int = 0

    @property
    def confidence_adjustment(self) -> float:
        return (
            0.2
            * (self.confirmations - self.rejections)
            / (4 + self.confirmations + self.rejections)
        )

    @property
    def suppressed(self) -> bool:
        return self.rejections >= 2 and self.rejections > self.confirmations


def experience_proposal(prompt: str, plan: ContinuationPlan) -> dict[str, object]:
    # Generalize concrete objects (door/window), retain the action, roles and cue.
    # Cue tokens are input recognition data, never a pool of reply strings.
    cue = sorted(
        {word for word in tokenize(prompt) if word[:1].isalnum()}
        - {"ну", "всё", "все", "вот", "а", "и"}
    )
    key = json.dumps(
        {
            "version": 1,
            "kind": plan.kind,
            "cue": cue,
            "predicate": plan.thought.predicate.lemma,
            "features": [
                (k, v)
                for k, v in plan.thought.predicate.features
                if k in {"tense", "aspect", "mood"}
            ],
            "roles": [
                (
                    dep.role,
                    dep.argument.head.part_of_speech,
                    (
                        "interlocutor"
                        if (dep.argument.head.entity_kind or "").startswith("user:")
                        else dep.argument.head.entity_kind
                    ),
                    (
                        dep.argument.head.lemma
                        if dep.role in {"predicate_complement", "action_complement"}
                        or dep.argument.head.part_of_speech == "pronoun"
                        else None
                    ),
                    dict(dep.argument.head.features).get("number"),
                )
                for dep in plan.thought.dependencies
            ],
            "negated": plan.thought.negated,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "learning_key": key,
        "anchor_event_id": plan.source_event_ids[0],
        "negated": plan.thought.negated,
    }


def explicit_feedback(text: str, *, negated_question: bool) -> bool | None:
    if any(not char.isalnum() and not char.isspace() and char not in ".,!" for char in text):
        return None
    tokens = tuple(tokenize(text))
    if any(not word[:1].isalnum() and word not in {".", ",", "!"} for word in tokens):
        return None
    words = tuple(word for word in tokens if word[:1].isalnum())
    # Punctuation that changes the speech act is material; sarcasm/qualification
    # and open-ended comments are not forced into a yes/no training label.
    if any(mark in text for mark in ("?", "«", "»", '"', "`")):
        return None
    if words in {
        ("не", "это"),
        ("нет", "не", "это"),
        ("ты", "неправильно", "понял"),
        ("ты", "неправильно", "поняла"),
    }:
        return False
    if words in {("именно", "это"), ("да", "именно", "это")}:
        return True
    # Russian 'да/нет' to a negative question does not identify its polarity.
    if negated_question:
        return None
    if words in {("да",), ("ага",), ("верно",), ("именно",), ("да", "верно")}:
        return True
    if words in {("нет",), ("неверно",)}:
        return False
    return None


class AssertionExperienceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def guidance(self, current: Event, proposal: dict[str, object]) -> ExperienceGuidance:
        row = await self.database.fetchone(
            """SELECT COALESCE(SUM(f.accepted), 0) AS confirmations,
                      COALESCE(SUM(1 - f.accepted), 0) AS rejections
               FROM assertion_experiences e JOIN assertion_feedback f USING(prompt_event_id)
               WHERE e.guild_id = ? AND e.channel_id = ? AND e.user_id = ?
                 AND e.learning_key = ? AND julianday(f.observed_at) < julianday(?)
                 AND julianday(f.observed_at) >= julianday(?)""",
            (
                current.guild_id,
                current.channel_id,
                current.actor_id,
                proposal["learning_key"],
                current.utc_timestamp.isoformat(),
                (current.utc_timestamp - timedelta(days=30)).isoformat(),
            ),
        )
        return ExperienceGuidance(row["confirmations"], row["rejections"])

    async def adapt(
        self,
        current: Event,
        prompt: str,
        plan: ContinuationPlan,
    ) -> tuple[ContinuationPlan | None, ExperienceGuidance]:
        guidance = await self.guidance(current, experience_proposal(prompt, plan))
        if guidance.suppressed:
            return None, guidance
        confidence = max(0.0, min(0.95, plan.confidence + guidance.confidence_adjustment))
        return replace(
            plan, confidence=confidence, thought=replace(plan.thought, confidence=confidence)
        ), guidance

    async def record_sent(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        proposal: dict[str, object],
        created_at: datetime,
    ) -> None:
        if prompt.channel_id is None or prompt.actor_id is None:
            return
        if not isinstance(proposal.get("learning_key"), str):
            raise ValueError("assertion experience requires a semantic learning key")
        if not isinstance(proposal.get("negated"), bool):
            raise ValueError("assertion experience requires question polarity")
        await self.database.execute(
            """INSERT OR IGNORE INTO assertion_experiences
               (prompt_event_id, anchor_event_id, guild_id, channel_id, user_id,
                response_message_id, learning_key, negated, sent_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prompt.event_id,
                proposal["anchor_event_id"],
                prompt.guild_id,
                prompt.channel_id,
                prompt.actor_id,
                response_message_id,
                proposal["learning_key"],
                int(proposal["negated"]),
                created_at.astimezone(UTC).isoformat(),
            ),
        )
        # Feedback can reach the event pipeline before send bookkeeping finishes,
        # or while a sent outbox job awaits recovery. Reconcile that narrow window.
        feedback_rows = await self.database.fetchall(
            """SELECT event_id FROM events
               WHERE guild_id = ? AND channel_id = ? AND actor_id = ?
                 AND json_extract(payload_json, '$.reply_to_message_id') = ?
                 AND julianday(occurred_at) > julianday(?)
                 AND julianday(occurred_at) <= julianday(?)
               ORDER BY julianday(occurred_at), event_id""",
            (
                prompt.guild_id,
                prompt.channel_id,
                prompt.actor_id,
                response_message_id,
                created_at.astimezone(UTC).isoformat(),
                (created_at + timedelta(minutes=5)).astimezone(UTC).isoformat(),
            ),
        )
        events = EventRepository(self.database)
        for row in feedback_rows:
            feedback = await events.get(row["event_id"])
            if feedback is not None:
                await self.observe_feedback(feedback)

    async def observe_feedback(self, event: Event) -> None:
        if (
            event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
            or event.payload.get("author_is_bot")
            or event.actor_id is None
            or event.channel_id is None
            or event.source != "discord"
        ):
            return
        reference = event.payload.get("reply_to_message_id")
        # Require a reply reference. Bare channel 'да' has no reliable addressee.
        if not isinstance(reference, int) or isinstance(reference, bool):
            return
        text = event.payload.get("content")
        if not isinstance(text, str):
            return
        async with self.database.transaction() as connection:
            async with connection.execute(
                """SELECT * FROM assertion_experiences
                   WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                     AND response_message_id = ?
                     AND julianday(sent_at) < julianday(?)
                     AND julianday(sent_at) >= julianday(?)""",
                (
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    reference,
                    event.utc_timestamp.isoformat(),
                    (event.utc_timestamp - timedelta(minutes=5)).isoformat(),
                ),
            ) as cursor:
                episode = await cursor.fetchone()
            if episode is None:
                return
            verdict = explicit_feedback(text, negated_question=bool(episode["negated"]))
            if verdict is None:
                return
            # One vote per sent hypothesis. Retries/replays and repeated replies
            # cannot inflate its weight or turn one experience into a consensus.
            await connection.execute(
                """INSERT OR IGNORE INTO assertion_feedback
                   (prompt_event_id, feedback_event_id, accepted, observed_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    episode["prompt_event_id"],
                    event.event_id,
                    int(verdict),
                    event.utc_timestamp.isoformat(),
                ),
            )
