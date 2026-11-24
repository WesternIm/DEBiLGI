"""Persistent local cognition that keeps working between Discord events."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from debilgi.events.models import Event, EventType
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.continuous_cognition")

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


@dataclass(frozen=True, slots=True)
class ContinuousCognitionConfig:
    active_interval: timedelta = timedelta(seconds=2)
    idle_interval: timedelta = timedelta(seconds=15)
    work_per_pulse: int = 8
    trace_retry_limit: int = 20
    sleep_after: timedelta = timedelta(minutes=30)
    sleep_cycle: timedelta = timedelta(minutes=90)
    sleep_experiments: int = 2

    def __post_init__(self) -> None:
        if self.active_interval <= timedelta(0) or self.idle_interval <= timedelta(0):
            raise ValueError("continuous cognition intervals must be positive")
        if self.sleep_after <= timedelta(0) or not (
            timedelta(hours=1) <= self.sleep_cycle <= timedelta(hours=2)
        ) or self.sleep_experiments < 0:
            raise ValueError("invalid sleep rhythm or experiment budget")
        if self.work_per_pulse <= 0 or self.trace_retry_limit <= 0:
            raise ValueError("continuous cognition limits must be positive")


@dataclass(frozen=True, slots=True)
class CognitivePulseStats:
    pulse_id: str
    work_claimed: int = 0
    hypotheses_reviewed: int = 0
    gaps_recorded: int = 0
    signals_recorded: int = 0


class ContinuousCognitionService:
    """Keep a bounded cognitive actor alive for the lifetime of the bot.

    The actor never authors or sends Discord messages. It turns completed
    semantic inquiry traces into revisable hypotheses, preserves unresolved
    questions as durable gaps and exposes relevant retained knowledge to later
    reply planning.
    """

    def __init__(
        self,
        database: Database,
        *,
        enabled: bool = True,
        config: ContinuousCognitionConfig | None = None,
        clock: Any | None = None,
        activity_filter: Any | None = None,
        presence: Any | None = None,
        replay: Any | None = None,
        maintenance: Any | None = None,
    ) -> None:
        self._database = database
        self.enabled = enabled
        self.config = config or ContinuousCognitionConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._task: asyncio.Task[None] | None = None
        self._pulse_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._sleep_lock = asyncio.Lock()
        self._rhythm_lock = asyncio.Lock()
        self._awake = asyncio.Event()
        self._awake.set()
        self.sleeping = False
        self._sleep_loaded = False
        self._wake_requested = False
        self._wake_event: Event | None = None  # the tag that asked to be answered while asleep
        self._activity_filter = activity_filter
        self._presence = presence
        self._replay = replay
        self._maintenance = maintenance
        self._sleep_guilds = {}
        self._trial_candidates = set()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.enabled or self.running:
            return
        await self.restore_sleep()
        await self.resume_sleep_after_restart()
        now = self._now()
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'queued', next_attempt_at = ?, updated_at = ?,
                   last_error = 'recovered_after_restart'
               WHERE status = 'processing'""",
            (now.isoformat(), now.isoformat()),
        )
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
               SET status = 'running', started_at = ?, last_heartbeat_at = ?,
                   stopped_at = NULL, last_error = NULL
               WHERE runtime_id = 1""",
            (now.isoformat(), now.isoformat()),
        )
        self._task = asyncio.create_task(
            self._run_loop(),
            name="debilgi-continuous-cognition",
        )
        logger.info(
            "runtime=started active_interval=%.3f idle_interval=%.3f work_per_pulse=%d",
            self.config.active_interval.total_seconds(),
            self.config.idle_interval.total_seconds(),
            self.config.work_per_pulse,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        now = self._now()
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
               SET status = 'stopped', stopped_at = ?, last_heartbeat_at = ?
               WHERE runtime_id = 1""",
            (now.isoformat(), now.isoformat()),
        )
        logger.info("runtime=stopped")

    async def observe(self, event: Event) -> bool:
        """Queue human language for continued processing after the live turn."""

        if (
            not self.enabled
            or event.event_type not in _MESSAGE_TYPES
            or event.payload.get("author_is_bot") is True
        ):
            return False
        if "corpus" not in event.source and not event.source.startswith("seed:"):
            await self.note_activity(event)
            await self.observe_sleep_feedback(event)
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        now = self._now()
        addressed = event.payload.get("bot_mentioned") is True
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_work_items (
                   work_id, source_event_id, guild_id, channel_id, priority,
                   next_attempt_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"continuous-work:{event.event_id}",
                event.event_id,
                event.guild_id,
                event.channel_id,
                0.90 if addressed else 0.55,
                now.isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        self._wake.set()
        return cursor.rowcount == 1

    def wake(self) -> None:
        """Notify the actor that a response trace or another evidence source changed."""

        self._wake.set()

    async def restore_sleep(self) -> None:
        if self._sleep_loaded:
            return
        row = await self._database.fetchone(
            "SELECT * FROM continuous_cognition_runtime WHERE runtime_id=1"
        )
        self.sleeping = row["sleep_state"] == "sleeping"
        self._wake_requested = bool(row["wake_requested"])
        if self.sleeping:
            self._awake.clear()
        if row["last_activity_at"] is None:
            await self._database.execute(
                "UPDATE continuous_cognition_runtime SET last_activity_at=? WHERE runtime_id=1",
                (self._now().isoformat(),),
            )
        self._sleep_loaded = True
        try:
            await self.merge_sleep_gaps()
        except Exception:
            # Re-keying is maintenance; it must never block a reply or a wake.
            logger.exception("sleep=gaps_rekey_failed")
        await self.sync_presence()

    async def resume_sleep_after_restart(self) -> str:
        """Recompute a sleep restored from disk instead of continuing it on paper.

        The previous process' cycle budgets were spent by that process. An
        expired cycle follows the ordinary boundary rule (pending messages wake,
        silence begins the next cycle); an unexpired one restarts its cycle from
        this real start, so revisits and experiments have a budget again.
        """
        await self.restore_sleep()
        if not self.sleeping:
            return "awake"
        async with self._rhythm_lock:
            moment = self._now()
            row = await self._database.fetchone(
                "SELECT * FROM continuous_cognition_runtime WHERE runtime_id=1"
            )
            boundary = row["cycle_ends_at"] is None or moment >= datetime.fromisoformat(
                row["cycle_ends_at"])
            if (self._wake_requested or row["wake_requested"]
                    or boundary and row["pending_activity"]):
                state = await self._advance_rhythm(moment)
            else:
                await self._begin_cycle(moment)
                state = "sleeping"
            await self._database.execute(
                "UPDATE continuous_cognition_runtime SET sleep_restarts=sleep_restarts+1 "
                "WHERE runtime_id=1")
        logger.info("sleep=restored state=%s cycle_restarted=%s", state, state == "sleeping")
        self._wake.set()
        return state

    async def _begin_cycle(self, moment: datetime) -> None:
        await self._database.execute(
            "UPDATE continuous_cognition_runtime SET sleep_state='sleeping', "
            "cycle_started_at=?,cycle_ends_at=?,sleep_cycles=sleep_cycles+1, "
            "experiment_units=0,maintenance_at=NULL WHERE runtime_id=1",
            (moment.isoformat(), (moment + self.config.sleep_cycle).isoformat()),
        )
        self.sleeping = True
        self._awake.clear()

    async def sync_presence(self) -> None:
        if self._presence is not None:
            try:
                await self._presence("idle" if self.sleeping else "online")
            except Exception:
                logger.exception("Sleep presence update failed; delivery remains available")

    async def note_activity(self, event: Event) -> None:
        if (not self.enabled or event.event_type not in _MESSAGE_TYPES
                or event.payload.get("author_is_bot") or "corpus" in event.source
                or event.source.startswith("seed:") or event.channel_id is None):
            return
        addressed = event.turn_form.addressed
        relevant = addressed
        if self._activity_filter is not None:
            relevant = await self._activity_filter(event)
        if not relevant:
            return
        await self.restore_sleep()
        async with self._rhythm_lock:
            if addressed and self.sleeping and self._wake_event is None:
                self._wake_event = event
            self._wake_requested |= addressed and self.sleeping
            await self._database.execute(
                "UPDATE continuous_cognition_runtime SET last_activity_at=?, "
                "pending_activity=CASE WHEN sleep_state='sleeping' THEN 1 ELSE 0 END, "
                "wake_requested=MAX(wake_requested,?) WHERE runtime_id=1",
                (self._now().isoformat(), int(self._wake_requested)),
            )
        self._wake.set()

    async def wait_until_awake(self, event: Event) -> None:
        if not self.enabled:
            return
        # The event and its delivery job already exist. Observation never waits
        # on this barrier; only reply preparation waits for the current unit.
        if self.sleeping and event.turn_form.addressed:
            await self.note_activity(event)
            async with self._sleep_lock:
                await self.rhythm_once()
        await self._awake.wait()

    async def rhythm_once(self, now: datetime | None = None) -> str:
        async with self._rhythm_lock:
            return await self._advance_rhythm(now)

    async def _advance_rhythm(self, now: datetime | None = None) -> str:
        await self.restore_sleep()
        moment = self._aware(now or self._now())
        row = await self._database.fetchone(
            "SELECT * FROM continuous_cognition_runtime WHERE runtime_id=1"
        )
        if self.sleeping:
            boundary = moment >= datetime.fromisoformat(row["cycle_ends_at"])
            if (self._wake_requested or row["wake_requested"]
                    or boundary and row["pending_activity"]):
                await self._database.execute(
                    "UPDATE continuous_cognition_runtime SET sleep_state='awake', "
                    "last_activity_at=?, pending_activity=0,wake_requested=0 WHERE runtime_id=1",
                    (moment.isoformat(),),
                )
                woken_by, self._wake_event = (
                    (self._wake_event, None) if self._wake_requested else (None, None)
                )
                self.sleeping = self._wake_requested = False
                self._awake.set()
                await self.sync_presence()
                if woken_by is not None:
                    await self._record_wake(woken_by, row["cycle_started_at"], moment)
                return "awake"
            if not boundary:
                return "sleeping"
        elif moment - datetime.fromisoformat(row["last_activity_at"]) < self.config.sleep_after:
            return "awake"
        await self._begin_cycle(moment)
        await self.sync_presence()
        return "sleeping"

    async def _record_wake(
        self, event: Event, cycle_started_at: str | None, moment: datetime
    ) -> None:
        """The real transition asleep→awake, caused by this addressed event, is an own action.

        Only an observed transition with its waking event is recorded (like a delivery with its
        receipt); a boundary wake, a restart or any guess writes nothing. The claim belongs to the
        guild and channel of that event, never to others. It never blocks the reply.
        """
        try:
            from debilgi.language.canonical_claims import activity_thought
            from debilgi.language.proposition_memory import PropositionMemory

            await PropositionMemory.for_meaning(self._database).record_agent_activity(
                event,
                source_key=f"sleep_wake:{cycle_started_at}",
                operation="wake",
                thought=activity_thought("проснуться"),
                occurred_at=moment,
            )
        except Exception:  # noqa: BLE001 - own-state bookkeeping must not stop the day answer
            logging.getLogger(__name__).warning("sleep wake not recorded", exc_info=True)

    async def _is_play_turn(self, event: Event) -> bool:
        """Was the turn the agent delivered to this event a make-believe one?"""
        row = await self._database.fetchone(
            "SELECT selection_json FROM language_meaning_intentions WHERE prompt_event_id=? "
            "AND response_message_id IS NOT NULL ORDER BY sent_at DESC LIMIT 1", (event.event_id,))
        return bool(row and (json.loads(row[0]).get("basis") or {}).get("fantasy"))

    async def _rejects_play(self, event: Event, feedback) -> bool:
        """«Не смешно» about a fantasy weighs the choice of play; it is no hole to repair."""
        row = await self._database.fetchone(
            "SELECT selection_json FROM language_meaning_intentions WHERE guild_id=? AND "
            "channel_id=? AND user_id=? AND response_message_id IS NOT NULL AND sent_at<? "
            "AND (response_message_id=? OR (? IS NULL AND sent_at>=?)) "
            "ORDER BY sent_at DESC LIMIT 1",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             feedback.target_message_id, feedback.target_message_id,
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        return bool(row and (json.loads(row[0]).get("basis") or {}).get("fantasy"))

    async def capture_sleep_gap(self, event_id: str, *, now: datetime | None = None) -> int:
        """Classify completed day episodes with the existing form and feedback readers."""
        from debilgi.events.repository import EventRepository
        from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR

        event = await EventRepository(self._database).get(event_id)
        if (event is None or event.payload.get("author_is_bot") or "corpus" in event.source
                or event.source.startswith("seed:") or not event.turn_form.addressed):
            return 0
        if await self._database.fetchone(
            "SELECT 1 FROM continuous_cognition_signals WHERE source_event_id=?", (event_id,)
        ):
            return 0
        trace = await self._database.fetchone(
            "SELECT * FROM language_response_traces WHERE source_event_id=?", (event_id,)
        )
        if trace is None or trace["status"] == "started" or trace["sleep_captured"]:
            return 0
        steps = await self._database.fetchall(
            "SELECT s.component,s.details_json FROM language_response_trace_steps s "
            "JOIN language_response_traces t ON t.trace_id=s.trace_id "
            "WHERE t.source_event_id=? ORDER BY s.sequence", (event_id,)
        )
        details = {r["component"]: _json_mapping(r["details_json"]) for r in steps}
        decision = details.get("response_mode", {}).get("decision", {})
        text = str(trace["final_text"] or "")
        if await self._is_play_turn(event):
            return 0  # make-believe leaves no hole for the night to repair
        kinds = []
        if trace["status"] in {"withheld", "failed"}:
            kinds.append("withhold")
        if decision.get("act") == "express_limit" or text.casefold().startswith("я не знаю"):
            kinds.append("unknown")
        if event.turn_form.kind == "repair":
            kinds.append("own_turn_repair")
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        if (feedback.reward < 0 and not (feedback.kind == "refusal"
                                        and event.turn_form.kind == "question")
                and not await self._rejects_play(event, feedback)):
            kinds.append("human_rejection")
        if text and await self._database.fetchone(
            "SELECT 1 FROM language_response_traces t JOIN events e "
            "ON e.event_id=t.source_event_id WHERE t.guild_id=? AND e.channel_id=? "
            "AND e.actor_id=? AND t.final_text=? AND t.source_event_id!=? "
            "AND e.occurred_at<? AND t.status='sent' LIMIT 1",
            (event.guild_id, event.channel_id, event.actor_id, text, event_id,
             event.utc_timestamp.isoformat()),
        ):
            kinds.append("repetition")
        # The existing reader identifies unresolved anaphora, including learned
        # parts carrying 'их' with no bound referent in the delivered episode.
        generation = details.get("evidence_planner", {})
        if any(c.get("metadata", {}).get("unresolved_requirement")
               for c in generation.get("candidates", ())):
            kinds.append("unbound_slot")
        if text:
            from debilgi.language.dialogue_state import analyze_dialogue_state
            neighbours = await self._database.fetchall(
                "SELECT json_extract(payload_json,'$.content') content FROM events "
                "WHERE guild_id=? AND channel_id=? AND occurred_at<=? AND "
                "(actor_id=? OR target_id=?) ORDER BY occurred_at DESC LIMIT 6",
                (event.guild_id, event.channel_id, event.utc_timestamp.isoformat(),
                 event.actor_id, event.actor_id))
            if analyze_dialogue_state(text, tuple(
                str(r[0]) for r in reversed(neighbours) if r[0]
            )).unresolved_references:
                kinds.append("unbound_slot")
        selection = details.get("response_arbitrator", {})
        scores = selection.get("scores", ())
        if not kinds and scores and any(
            r.get("eligible") and r.get("admission_score", 1) < 0.65 for r in scores
        ):
            kinds.append("weak_answer")
        if not kinds:
            return 0
        normalized = self._sleep_prompt(event)
        scope, signature = await self.sleep_gap_identity(event)
        stamp = self._aware(now or self._now()).isoformat()
        details = {"kinds": kinds, "actor_id": event.actor_id,
                   "channel_id": event.channel_id, "prompt": normalized,
                   "old_text": text, "meaning_scope": scope}
        existing = await self._database.fetchone(
            "SELECT gap_id,details_json FROM continuous_cognition_gaps "
            "WHERE guild_id=? AND signature=?", (event.guild_id, signature))
        if existing is None:
            await self._database.execute(
                "INSERT INTO continuous_cognition_gaps "
                "(gap_id,guild_id,signature,question_kind,answer_role,source_event_id,"
                "first_seen_at,last_seen_at,details_json,meaning_scope) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"sleep-gap:{event.guild_id}:{signature[6:]}", event.guild_id, signature,
                 kinds[0], "dialogue_response", event_id, stamp, stamp,
                 json.dumps({**details, "occurrences": [event_id]}, ensure_ascii=False),
                 scope),
            )
        else:
            # The same meaning asked again is one hole seen more often, whatever
            # its wording. Earlier occurrences and tried plans stay with it, and
            # the new material lifts an exhaustion mark.
            merged = _merge_gap_details(_json_mapping(existing["details_json"]), details,
                                        (event_id,))
            await self._database.execute(
                "UPDATE continuous_cognition_gaps SET occurrence_count=occurrence_count+1,"
                "source_event_id=?,last_seen_at=MAX(last_seen_at,?),details_json=?,"
                "question_kind=CASE WHEN question_kind='weak_answer' THEN ? "
                "ELSE question_kind END,"
                "status='open',exhausted_at=NULL,meaning_scope=? WHERE gap_id=?",
                (event_id, stamp, json.dumps(merged, ensure_ascii=False), kinds[0], scope,
                 existing["gap_id"]),
            )
        await self._database.execute(
            "UPDATE language_response_traces SET sleep_captured=1 WHERE source_event_id=?",
            (event_id,),
        )
        return 1

    @staticmethod
    def sleep_scope(event: Event, canonical=None, text: str | None = None) -> str:
        """Meaning identity of a hole: the 4.13 key used for evidence, plus the slot.

        Wording, discourse particles and the analytic future do not split a
        hole. The requested slot (question word) and the act (question or
        assertion) do. When the bot is addressed, a personal subject that
        includes it (ты, inclusive мы/вы) is the same participant direction.
        A prompt without a readable proposition keeps its normalized surface.
        """

        from debilgi.language.communication import reader_goal
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import parse_proposition
        from debilgi.language.structured_utterance import analyze_structured_utterance

        prompt = (ContinuousCognitionService._sleep_prompt(event) if text is None
                  else ContinuousCognitionService._normalized_prompt(text))
        structure = analyze_structured_utterance(prompt)
        thought = parse_proposition(structure, allow_question=True, future_auxiliary=True)
        if thought is None:
            return json.dumps(["surface", prompt], ensure_ascii=False, separators=(",", ":"))
        if event.turn_form.addressed:
            thought = replace(thought, dependencies=tuple(
                replace(d, argument=replace(d.argument, head=replace(
                    d.argument.head, lemma="я", surface="я")))
                if d.role == "subject" and d.argument.head.lemma in {"я", "мы", "вы"} else d
                for d in thought.dependencies))
        question = any(c.clause_kind == "question" for c in structure.clauses)
        return json.dumps(["meaning", meaning_key(thought, canonical),
                           reader_goal(structure).question_word, question],
                          ensure_ascii=False, separators=(",", ":"))

    async def sleep_gap_identity(
        self, event: Event, text: str | None = None
    ) -> tuple[str, str]:
        from debilgi.language.lexicon import LexiconService

        # The confirmed canon only: working synonym hypotheses never merge holes.
        canonical = await LexiconService.for_meaning(self._database).canonical_lemmas(
            event.guild_id)
        scope = self.sleep_scope(event, canonical or None, text)
        return scope, "sleep:" + hashlib.sha256(
            f"{event.channel_id}:{event.actor_id}:{scope}".encode()
        ).hexdigest()[:32]

    async def merge_sleep_gaps(self) -> int:
        """Re-key holes recorded by surface (schema 066) and merge their duplicates."""
        from debilgi.events.repository import EventRepository

        rows = await self._database.fetchall(
            "SELECT * FROM continuous_cognition_gaps WHERE signature LIKE 'sleep:%' "
            "AND meaning_scope IS NULL ORDER BY first_seen_at,gap_id")
        merged = 0
        events = EventRepository(self._database)
        for row in rows:
            details = dict(_json_mapping(row["details_json"]))
            event = await events.get(row["source_event_id"])
            if event is None:
                scope = json.dumps(["surface", details.get("prompt", "")],
                                   ensure_ascii=False, separators=(",", ":"))
                signature = "sleep:" + hashlib.sha256(
                    f"{details.get('channel_id')}:{details.get('actor_id')}:{scope}".encode()
                ).hexdigest()[:32]
            else:
                scope, signature = await self.sleep_gap_identity(event)
            details.setdefault("occurrences", [row["source_event_id"]])
            details["meaning_scope"] = scope
            target = await self._database.fetchone(
                "SELECT * FROM continuous_cognition_gaps WHERE guild_id=? AND signature=? "
                "AND gap_id!=?", (row["guild_id"], signature, row["gap_id"]))
            async with self._database.transaction() as connection:
                if target is None:
                    await connection.execute(
                        "UPDATE continuous_cognition_gaps SET signature=?,meaning_scope=?,"
                        "details_json=? WHERE gap_id=?",
                        (signature, scope, json.dumps(details, ensure_ascii=False),
                         row["gap_id"]))
                    continue
                newer = row["last_seen_at"] > target["last_seen_at"]
                current = _json_mapping(target["details_json"])
                combined = (_merge_gap_details(current, details, later=details) if newer
                            else _merge_gap_details(current, {}, earlier=details))
                await connection.execute(
                    "UPDATE continuous_cognition_hypotheses SET sleep_gap_id=? "
                    "WHERE sleep_gap_id=?", (target["gap_id"], row["gap_id"]))
                await connection.execute(
                    "UPDATE continuous_cognition_gaps SET "
                    "occurrence_count=occurrence_count+?,replayed_count=replayed_count+?,"
                    "visited_occurrences=visited_occurrences+?,"
                    "first_seen_at=MIN(first_seen_at,?),last_seen_at=MAX(last_seen_at,?),"
                    "source_event_id=CASE WHEN ? THEN ? ELSE source_event_id END,"
                    "details_json=?,status='open',meaning_scope=? WHERE gap_id=?",
                    (row["occurrence_count"], row["replayed_count"],
                     row["visited_occurrences"], row["first_seen_at"], row["last_seen_at"],
                     newer, row["source_event_id"],
                     json.dumps(combined, ensure_ascii=False), scope, target["gap_id"]))
                await connection.execute(
                    "DELETE FROM continuous_cognition_gaps WHERE gap_id=?", (row["gap_id"],))
            merged += 1
        if rows:
            logger.info("sleep=gaps_rekeyed gaps=%d merged=%d", len(rows), merged)
        return merged

    async def sleep_once(self) -> int:
        """One bounded local replay, or a smaller experiment/maintenance unit.

        New occurrences come first. Then each open hole without a surviving
        hypothesis is revisited once per cycle with a wider set of existing
        pieces (other occurrences, more human donors, more compatible acts),
        until a hypothesis is found or no untried plan remains. An exhausted
        hole waits for new human material. Every executed unit is counted.
        """
        from debilgi.events.repository import EventRepository

        async with self._sleep_lock:
            await self.rhythm_once()
            if not self.sleeping:
                return 0
            await self.review_person_hypotheses()
            await self.consolidate_chat_norms()
            produced = 0
            unit = None
            try:
                async with asyncio.timeout(120):
                    runtime = await self._database.fetchone(
                        "SELECT experiment_units,maintenance_at,sleep_cycles FROM "
                        "continuous_cognition_runtime WHERE runtime_id=1"
                    )
                    await self._release_exhausted_gaps()
                    gap, revisit, experiment = await self._next_sleep_gap(runtime)
                    if gap is not None:
                        unit = "revisit" if revisit else "replay"
                        details = dict(_json_mapping(gap["details_json"]))
                        tried = list(details.get("tried", ()))
                        occurrences = list(details.get("occurrences", ())) or [
                            gap["source_event_id"]]
                        visit = int(gap["replayed_count"]) if revisit else 0
                        # The first replay reads the latest occurrence; revisits
                        # rotate back through earlier wordings of the same meaning.
                        source = (occurrences[-1 - visit % len(occurrences)] if revisit
                                  else gap["source_event_id"])
                        events = EventRepository(self._database)
                        event = await events.get(source) or await events.get(
                            gap["source_event_id"])
                        variation = {"visit": visit, "tried": tuple(tried), "examined": []}
                        candidates = ()
                        if self._replay and event:
                            candidates = await (self._replay(event, variation=variation)
                                                if self._replay_varies()
                                                else self._replay(event))
                        examined = list(variation["examined"])
                        for candidate in candidates[:1 if experiment else 3]:
                            if not self.sleep_surface_allowed(event, candidate):
                                continue
                            data = {**details, "candidate": candidate}
                            data.pop("tried", None)
                            # A rejected derivation stays rejected. Merely sleeping
                            # or asking again cannot supply a new confirming donor.
                            signature = "sleep:" + hashlib.sha256(json.dumps(
                                [gap["gap_id"], candidate["renderer"],
                                 candidate.get("metadata", {}).get("message_plan_key")
                                 or candidate["text"]],
                                sort_keys=True,
                                ensure_ascii=False).encode()).hexdigest()[:32]
                            cursor = await self._database.execute(
                                "INSERT OR IGNORE INTO continuous_cognition_hypotheses "
                                "(hypothesis_id,guild_id,signature,predicate_key,roles_json,"
                                "epistemic_status,confidence,first_seen_at,last_reviewed_at,"
                                "sleep_gap_id,sleep_revision) VALUES (?,?,?,'sleep_response',?,"
                                "'hypothesis',0.5,?,?,?,?)",
                                (signature, gap["guild_id"], signature,
                                 json.dumps(data, ensure_ascii=False), self._now().isoformat(),
                                 self._now().isoformat(), gap["gap_id"], gap["occurrence_count"]),
                            )
                            produced += int(cursor.rowcount == 1)
                            self._sleep_guilds[gap["guild_id"]] = True
                            key = candidate.get("metadata", {}).get("intention_plan_key")
                            if key and key not in examined:
                                examined.append(key)
                        fresh = [k for k in examined if k not in tried]
                        # Nothing new was even examined: the window is exhausted
                        # until new material arrives. A replay that cannot report
                        # its plans is exhausted by a revisit that found nothing.
                        exhausted = revisit and not produced and (
                            not fresh and visit + 1 >= len(occurrences)
                            if self._replay_varies() else True)
                        if experiment:
                            await self._database.execute(
                                "UPDATE continuous_cognition_runtime SET "
                                "experiment_units=experiment_units+1 WHERE runtime_id=1")
                        details["tried"] = list(dict.fromkeys([*tried, *fresh]))[-64:]
                        await self._database.execute(
                            "UPDATE continuous_cognition_gaps SET replayed_count=replayed_count+1,"
                            "visited_occurrences=?,visit_cycle=?,details_json=?,"
                            "exhausted_at=CASE WHEN ? THEN ? ELSE exhausted_at END "
                            "WHERE gap_id=?",
                            (gap["occurrence_count"], runtime["sleep_cycles"],
                             json.dumps(details, ensure_ascii=False), exhausted,
                             self._now().isoformat(), gap["gap_id"]),
                        )
                        logger.info(
                            "sleep_unit=%s gap=%s visit=%d source=%s candidates=%d "
                            "new_plans=%d hypotheses=%d exhausted=%s",
                            unit, gap["gap_id"], visit + 1, source, len(candidates),
                            len(fresh), produced, exhausted)
                    elif runtime["experiment_units"] < self.config.sleep_experiments:
                        unit = "synonyms"
                        from debilgi.language.lexicon import LexiconService
                        lexicon = LexiconService.for_meaning(self._database)
                        await lexicon.discover_synonym_candidates(limit=8)
                        await self._database.execute(
                            "UPDATE continuous_cognition_runtime SET "
                            "experiment_units=experiment_units+1 WHERE runtime_id=1"
                        )
                    elif runtime["maintenance_at"] is None and self._maintenance:
                        unit = "maintenance"
                        completed = await self._maintenance()
                        if completed is not False:
                            await self._database.execute(
                                "UPDATE continuous_cognition_runtime SET maintenance_at=? "
                                "WHERE runtime_id=1", (self._now().isoformat(),)
                            )
                    if unit is not None:
                        await self._database.execute(
                            "UPDATE continuous_cognition_runtime SET sleep_units=sleep_units+1,"
                            "total_work_processed=total_work_processed+1 WHERE runtime_id=1")
            finally:
                # A tag interrupts at this boundary, after durable writes.
                await self.rhythm_once()
            return produced

    def _replay_varies(self) -> bool:
        import inspect

        try:
            parameters = inspect.signature(self._replay).parameters
        except (TypeError, ValueError):
            return False
        return "variation" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    async def _release_exhausted_gaps(self) -> None:
        """New human messages or exchanges are new material for exhausted holes."""
        await self._database.execute(
            "UPDATE continuous_cognition_gaps SET exhausted_at=NULL "
            "WHERE exhausted_at IS NOT NULL AND signature LIKE 'sleep:%' AND ("
            "EXISTS (SELECT 1 FROM events e WHERE e.guild_id=continuous_cognition_gaps.guild_id "
            "AND e.occurred_at>continuous_cognition_gaps.exhausted_at "
            "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0) "
            "OR EXISTS (SELECT 1 FROM language_observed_exchanges x "
            "WHERE x.guild_id=continuous_cognition_gaps.guild_id "
            "AND x.observed_at>continuous_cognition_gaps.exhausted_at))")

    async def _next_sleep_gap(self, runtime) -> tuple[Any, bool, bool]:
        """(gap, revisit, experiment): frequency first, weak answers on a budget."""
        fresh = ("SELECT * FROM continuous_cognition_gaps WHERE status='open' "
                 "AND signature LIKE 'sleep:%' AND occurrence_count>visited_occurrences "
                 "AND question_kind{} 'weak_answer' "
                 "ORDER BY occurrence_count DESC,last_seen_at DESC,gap_id LIMIT 1")
        revisit = ("SELECT * FROM continuous_cognition_gaps g WHERE g.status='open' "
                   "AND g.signature LIKE 'sleep:%' AND g.occurrence_count<=g.visited_occurrences "
                   "AND g.visit_cycle<? AND g.exhausted_at IS NULL AND g.question_kind{} "
                   "'weak_answer' AND NOT EXISTS (SELECT 1 FROM continuous_cognition_hypotheses h "
                   "WHERE h.sleep_gap_id=g.gap_id AND h.status!='rejected') "
                   "ORDER BY g.occurrence_count DESC,g.replayed_count,g.last_seen_at DESC,"
                   "g.gap_id LIMIT 1")
        cycle = runtime["sleep_cycles"]
        steps = [(fresh.format("!="), (), False, False),
                 (revisit.format("!="), (cycle,), True, False)]
        if runtime["experiment_units"] < self.config.sleep_experiments:
            steps += [(fresh.format("="), (), False, True),
                      (revisit.format("="), (cycle,), True, True)]
        for sql, parameters, again, experiment in steps:
            gap = await self._database.fetchone(sql, parameters)
            if gap is not None:
                return gap, again, experiment
        return None, False, False

    @staticmethod
    def sleep_surface_allowed(event: Event, candidate: dict) -> bool:
        return ContinuousCognitionService.sleep_rejection(event, candidate) is None

    @staticmethod
    def admits_not_knowing(text: str) -> bool:
        from debilgi.language.structured_utterance import analyze_structured_utterance

        earlier = analyze_structured_utterance(text)
        return any(c.negated and c.predicate_index is not None
                   and earlier.tokens[c.predicate_index].lemma in {"знать", "помнить"}
                   for c in earlier.clauses)

    @staticmethod
    def sleep_rejection(event: Event, candidate: dict) -> str | None:
        """Check the requested reader update, including the previous failure."""
        from debilgi.language.communication import reader_goal
        from debilgi.language.dialogue_state import analyze_dialogue_state
        from debilgi.language.structured_utterance import analyze_structured_utterance

        metadata = candidate.get("metadata", {})
        text = candidate["text"]
        previous = metadata.get("sleep_previous_text", "")
        if previous and text.casefold().strip() == previous.casefold().strip():
            return "same_failure"
        if metadata.get("derivation") == "retrieve_delivered_act":
            return "quote_instead_of_explanation"
        query_goal = metadata.get("sleep_query_goal") or {}
        if (metadata.get("derivation") == "human_answer_crossover"
                and metadata.get("sleep_answer_slot")
                and query_goal.get("addressed") and query_goal.get("question_word")):
            # A donor's answer to an open slot about the agent itself is another
            # person's experience. Hypotheses made before that rule stay out too.
            return "donor_answer_about_self"
        state = analyze_dialogue_state(text, ())
        if any(r.kind == "explicit_pronoun" for r in state.unresolved_references):
            return "unbound_reference"
        structure = analyze_structured_utterance(text)
        from debilgi.language.request_focus import polarity_of

        for clause in structure.clauses:
            if clause.predicate_index is not None and any(
                polarity_of(t.lemma) is not None
                for t in structure.tokens[:clause.predicate_index]
            ):
                return "asserted_question_preface"
        prompt = str(event.payload.get("content", ""))
        goal = reader_goal(analyze_structured_utterance(prompt))
        query = metadata.get("sleep_query_goal")
        if query and (goal.predicate in {"предположить", "прикинуть"}
                      or metadata.get("sleep_composed")):
            from debilgi.language.communication import ReaderGoal
            goal = ReaderGoal(**query)
        if metadata.get("communicative_status") == "express_limit":
            if metadata.get("repair_unbound_reference"):
                return None if metadata.get("reasoning_evidence") else "missing_episode"
            if metadata.get("sleep_opinion_request"):
                return "unknown_instead_of_unknown"
            if previous and ContinuousCognitionService.admits_not_knowing(previous):
                # Only an earlier admission of not knowing is replaced by one.
                return "unknown_instead_of_unknown"
            return None
        answer_goal = reader_goal(structure)
        from debilgi.language.structured_utterance import analytic_future_events

        future = analytic_future_events(structure)
        if answer_goal.predicate == "быть" and len(future) == 1:
            # An analytic future names the infinitive's event, not its auxiliary.
            answer_goal = replace(answer_goal, predicate=next(iter(future.values())).lemma)
        canonical = metadata.get("sleep_canonical", {})
        same_predicate = (goal.predicate is not None and
                          canonical.get(goal.predicate, goal.predicate) ==
                          canonical.get(answer_goal.predicate, answer_goal.predicate))
        question = any(c.clause_kind == "question" for c in structure.clauses)
        if question:
            if same_predicate and answer_goal.question_word:
                return "predicate_echo_question"
            if not (metadata.get("genuine_clarification") and same_predicate
                    and metadata.get("bound_query_slot")):
                return "unrelated_counterquestion"
        subject = set(answer_goal.subject)
        if any(c.implicit_subject == "speaker" for c in structure.clauses):
            subject.add("я")
        if goal.addressed and (answer_goal.addressed or subject & {"ты", "вы"}):
            return "wrong_addressee"
        if set(goal.subject) & {"я", "мы"} and subject & {"я", "мы"}:
            return "wrong_addressee"
        action_slot = (metadata.get("sleep_answer_slot") and goal.predicate == "делать"
                       and goal.question_word == "что")
        bound_repair = (metadata.get("repair_reference_binding") and
                        metadata.get("reasoning_evidence"))
        if not same_predicate and not action_slot and not bound_repair:
            return "different_predicate"
        if metadata.get("sleep_opinion") and not any(
                t.lemma in {"наверное", "возможно", "вероятно"} for t in structure.tokens):
            return "unmarked_hypothesis"
        if not question and not metadata.get("sleep_opinion") and not bound_repair:
            return "unsupported_assertion"
        return None

    @staticmethod
    def _sleep_prompt(event: Event) -> str:
        from debilgi.language.entities import render_message_content

        rendered = render_message_content(str(event.payload.get("content", "")), event.payload)
        return ContinuousCognitionService._normalized_prompt(rendered)

    @staticmethod
    def _normalized_prompt(text: str) -> str:
        from debilgi.language.addressing import DEFAULT_ADDRESSING
        from debilgi.language.tokenizer import normalize_chat_token, tokenize

        clean = DEFAULT_ADDRESSING.analyze(" ".join(text.split())).content_text
        return " ".join(normalize_chat_token(t).canonical for t in tokenize(clean))

    async def sleep_candidates(self, event: Event, prompt: str, *, context=None) -> tuple:
        """Recall competitors only for the same server, channel, person and request."""
        from debilgi.language.arbitration import ResponseCandidate

        if event.guild_id not in self._sleep_guilds:
            self._sleep_guilds[event.guild_id] = bool(await self._database.fetchone(
                "SELECT 1 FROM continuous_cognition_hypotheses WHERE guild_id=? "
                "AND predicate_key='sleep_response' LIMIT 1", (event.guild_id,)))
        if not self._sleep_guilds[event.guild_id]:
            return ()
        _, signature = await self.sleep_gap_identity(event)
        signatures = [signature]
        if context is not None and prompt and self._normalized_prompt(prompt) != (
                self._sleep_prompt(event)):
            # An elliptical turn continues the request it was resolved to; the
            # hypotheses made for that meaning are its competitors too.
            signatures.append((await self.sleep_gap_identity(event, prompt))[1])
        rows = await self._database.fetchall(
            "SELECT h.hypothesis_id,h.confidence,h.roles_json FROM continuous_cognition_gaps g "
            "JOIN continuous_cognition_hypotheses h ON h.sleep_gap_id=g.gap_id "
            f"WHERE g.guild_id=? AND g.signature IN ({','.join('?' * len(signatures))}) "
            "AND h.predicate_key='sleep_response' "
            "AND h.status!='rejected' ORDER BY h.confidence DESC,h.last_reviewed_at DESC LIMIT 3",
            (event.guild_id, *signatures),
        )
        result = []
        for row in rows:
            data = json.loads(row["roles_json"])
            # The gap signature already fixes the meaning, person and channel;
            # a wording of the same request may differ. Re-check the scope.
            if data["actor_id"] != event.actor_id or data["channel_id"] != event.channel_id:
                continue
            candidate = data["candidate"]
            metadata = candidate.get("metadata", {})
            if context is not None and metadata.get("sleep_opinion_request"):
                from debilgi.language.dialogue_obligations import thought_from_data
                from debilgi.language.proposition_memory import meaning_key

                prior = context.prior_bot_turn
                question = ((prior["plan"].get("knowledge_limit") or {}).get("question")
                            if prior else None)
                if not question or meaning_key(thought_from_data(question),
                        metadata.get("sleep_canonical")) != metadata.get("sleep_query_key"):
                    continue
            if not self.sleep_surface_allowed(event, candidate):
                continue
            candidate = {**candidate, "metadata": {**metadata, **self._own_evidence_stance(
                event, candidate, context, await self._own_recent_positions(event),
                await self._own_affect_events(event),
                await self._own_position_history(event, candidate))}}
            result.append(ResponseCandidate.create(
                event_id=event.event_id, source="semantic_reasoning",
                **{k: v for k, v in candidate.items()
                   if k not in {"metadata", "source_confidence"}},
                source_confidence=min(float(candidate["source_confidence"]), float(row[1])),
                metadata={**candidate.get("metadata", {}), "sleep_hypothesis_id": row[0],
                          "epistemic_status": "hypothesis", "sleep_weight": row[1],
                          "focused_request": prompt},
            ))
        self._trial_candidates.update(c.candidate_id for c in result[:3])
        if len(self._trial_candidates) > 512:
            self._trial_candidates = {c.candidate_id for c in result[:3]}
        return tuple(result[:3])

    async def _own_recent_positions(self, event: Event) -> tuple[tuple[dict, ...], tuple[str, ...]]:
        """Propositions the agent itself put forward to this person lately, newest first.

        A statement is a position; a "why / how" question presupposes its proposition.
        Only delivered turns count (the intention's response message exists).
        """
        rows = await self._database.fetchall(
            "SELECT plan_json,response_text FROM language_meaning_intentions WHERE guild_id=? "
            "AND channel_id=? AND user_id=? AND response_message_id IS NOT NULL "
            "AND sent_at<? AND sent_at>=? "
            "ORDER BY sent_at DESC LIMIT 6",
            (event.guild_id, event.channel_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(minutes=10)).isoformat()))
        positions = []
        for row in rows:
            plan = json.loads(row[0])
            claim = plan.get("claim") or {}
            if dict(plan.get("hypothesis") or ()).get("kind") == "fantasy":
                continue  # make-believe is no position of the agent
            if claim.get("clause_kind") == "statement" and not plan.get("knowledge_limit"):
                positions.append(claim)
            question = (plan.get("knowledge_limit") or {}).get("question") or {}
            if question.get("focus_role") in {"cause", "manner"}:
                positions.append({**question, "dependencies": [
                    d for d in question.get("dependencies", ())
                    if d.get("role") != question["focus_role"]]})
        return tuple(positions), tuple(row[1] or "" for row in rows)

    async def _own_position_history(self, event: Event, candidate: dict) -> dict:
        """The agent's earlier positions on the candidate's meaning, however long ago.

        Delivered turns to this person on this server, filtered by meaning key; nothing of
        other people or servers. A guess that flips the sign of the latest position needs
        a new ground after it: a statement of this same person that says what the guess
        says (a question or a retelling presupposes, it does not state).
        """
        metadata = candidate.get("metadata", {})
        claim = (metadata.get("message_plan") or {}).get("claim")
        if not metadata.get("sleep_opinion") or not claim or event.actor_id is None:
            return {}
        from debilgi.language.dialogue_learning import own_turn_history
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import parse_proposition

        canonical = metadata.get("sleep_canonical") or None
        try:
            key = meaning_key(thought_from_data(claim), canonical)
            positions = []
            for item in await own_turn_history(
                    self._database, event.guild_id, event.actor_id, event.utc_timestamp):
                if item["claim"] and meaning_key(thought_from_data(item["claim"]),
                                                 canonical) == key:
                    positions.append(item)
        except (KeyError, TypeError, ValueError):
            return {}
        grounds = []
        if positions and bool(positions[0]["claim"].get("negated")) != bool(claim.get("negated")):
            rows = await self._database.fetchall(
                "SELECT source_id,content FROM language_discourse_turns WHERE guild_id=? "
                "AND actor_id=? AND observed_at>? AND observed_at<? AND source_id!=? "
                "ORDER BY observed_at DESC LIMIT 64",
                (event.guild_id, event.actor_id, positions[0]["sent_at"],
                 event.utc_timestamp.isoformat(), event.event_id))
            for row in rows:
                said = parse_proposition(row["content"], perspective="user")
                if said is not None and meaning_key(said, canonical) == key and (
                        bool(said.negated) == bool(claim.get("negated"))):
                    grounds.append(row["source_id"])
        return {"positions": positions, "grounds": grounds}

    async def _own_affect_events(self, event: Event) -> tuple[tuple[str, float], ...]:
        """Events of this person that moved the agent's emotion toward them, newest first.

        Same server and same person only (feelings about one person are not another's);
        the window is two half-lives of the emotion (affect.py), the stimulus is read as
        recorded.
        """
        rows = await self._database.fetchall(
            "SELECT event_id,stimulus_json FROM affect_observations WHERE guild_id=? "
            "AND actor_id=? AND observed_at<? AND observed_at>=? AND event_id!=? "
            "ORDER BY observed_at DESC LIMIT 32",
            (event.guild_id, event.actor_id, event.utc_timestamp.isoformat(),
             (event.utc_timestamp - timedelta(seconds=1440)).isoformat(), event.event_id))
        return tuple((row[0], float(json.loads(row[1]).get("valence", 0.0))) for row in rows)

    @staticmethod
    def _own_evidence_stance(event: Event, candidate: dict, context, recent=((), ()),
                             affect_events=(), history=None) -> dict:
        """Tie a guess's polarity to what the agent itself said and felt.

        A guess stays a hypothesis, but it must not contradict the agent's own earlier turn on
        the same meaning, nor repeat that turn word for word, and its valence is compared with
        the agent's own affect. Evidence is agent-side only (delivered turns, own affect); with
        none, the arbiter's stance feature stays unset. Changing a position takes a person's
        new statement: a repeated question is not new evidence.
        """
        if context is None:
            return {}
        metadata = candidate.get("metadata", {})
        plan = metadata.get("message_plan") or {}
        if not metadata.get("sleep_opinion") or not plan.get("claim"):
            return {}
        from math import copysign

        from debilgi.cognition.affect import lexical_valence
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.proposition_memory import meaning_key

        positions, said, *_ = recent
        history = history or {}
        evidence, own = [], 0.0
        canonical = metadata.get("sleep_canonical") or None
        for position in positions:
            try:
                same = meaning_key(thought_from_data(position), canonical) == meaning_key(
                    thought_from_data(plan["claim"]), canonical)
            except (KeyError, TypeError, ValueError):
                same = False
            if same:
                own = 1.0 if bool(position.get("negated")) == bool(
                    plan["claim"].get("negated")) else -1.0
                evidence.append("own_turn")
                break
        revises = {}
        past = history.get("positions") or ()
        if not evidence and past:  # the same meaning said earlier than the recent window
            own = 1.0 if bool(past[0]["claim"].get("negated")) == bool(
                plan["claim"].get("negated")) else -1.0
            evidence.append("own_turn")
        if own < 0 and past and history.get("grounds"):
            # A changed position needs a new ground after it: the person's own statement.
            own = 0.0
            revises = {"revises": {
                "intention_id": past[0]["intention_id"], "receipt": past[0]["receipt_id"],
                "sent_at": past[0]["sent_at"], "negated": bool(past[0]["claim"].get("negated"))},
                "revised_because": list(history["grounds"])}
        if past:  # said before, however long ago: a verbatim repeat is counted over all of it
            said = tuple(p["text"] for p in past)
        valence = lexical_valence(candidate["text"])
        felt = dict(getattr(context, "affect_features", ()) or ())
        mood = felt.get("emotion_valence", 0.0) * 0.72 + felt.get("mood_valence", 0.0) * 0.28
        affect = copysign(1.0, valence) * mood if valence and mood else 0.0
        if affect:
            evidence.append("own_affect")
        result = dict(revises)
        sign = copysign(1.0, valence) if valence else 0.0
        moved = [(e, v) for e, v in affect_events if sign * v > 0]
        if affect > 0 and moved:  # only events that back the polarity the feeling backs
            # Strongest stimulus first; on a tie the earlier event (rows are newest first). All
            # are kept (at most the 32 read): the reader finds which are statements it can restate.
            ranked = sorted(reversed(moved), key=lambda item: -sign * item[1])
            result["affect_events"] = [e for e, _ in ranked]
            if max(sign * v for _, v in moved) - min(sign * v for _, v in moved) < 0.01:
                result["affect_basis"] = "familiarity"
        if evidence:
            result.update({"stance_consistency": 0.5 + 0.5 * max(-1.0, min(1.0, own + affect)),
                           "stance_evidence": evidence})
        asking = event.turn_form.kind in {"question", "repair"}
        if own < 0 and asking:
            result["hard_rejection"] = "own_turn_contradiction"
        elif asking and (repeats := [t for t in said if t.strip().casefold()
                                     == candidate["text"].strip().casefold()]):
            # A repeated question does not erase the position the agent already took: it
            # may be stated once more (the one realization of its plan). Said twice
            # already, it would only loop, so the repetition is rejected again.
            if len(repeats) > 1 or own <= 0:
                result["hard_rejection"] = "repeats_own_turn"
            else:
                result["restated_position"] = True
        return result

    async def record_sleep_trial(self, event, reply, receipt_id):
        decision = getattr(reply, "winner_candidate_id", None)
        if decision not in self._trial_candidates:
            return
        row = await self._database.fetchone(
            "SELECT s.details_json FROM language_response_traces t "
            "JOIN language_response_trace_steps s ON s.trace_id=t.trace_id "
            "WHERE t.source_event_id=? AND s.component='response_arbitrator' "
            "ORDER BY s.sequence DESC LIMIT 1",
            (event.event_id,),
        )
        if row is None:
            return
        # The winner metadata is recorded by the existing arbiter trace.
        identifier = _json_mapping(row[0]).get("sleep_hypothesis_id")
        if identifier:
            hypothesis = await self._database.fetchone(
                "SELECT roles_json FROM continuous_cognition_hypotheses WHERE hypothesis_id=?",
                (identifier,),
            )
            data = json.loads(hypothesis[0])
            data["receipts"] = [*data.get("receipts", ())[-15:], receipt_id]
            await self._database.execute(
                "UPDATE continuous_cognition_hypotheses SET roles_json=? WHERE hypothesis_id=?",
                (json.dumps(data, ensure_ascii=False), identifier),
            )

    @staticmethod
    def _stance_toward_claim(event: Event, hypothesis: dict) -> str | None:
        """How a person's reply relates to the position a guess put forward.

        ``premise``: the reply is a question or a retelling of that very meaning with the same
        sign, so it presupposes the position rather than judging it. ``supports`` /
        ``contradicts``: a statement of the same meaning with the same / opposite sign.
        ``None``: the reply is about something else, the general attribution stands.
        """
        from debilgi.language.dialogue_obligations import thought_from_data
        from debilgi.language.proposition_memory import meaning_key
        from debilgi.language.semantic_graph import parse_proposition, reported_proposition
        from debilgi.language.structured_utterance import analyze_structured_utterance

        metadata = (hypothesis.get("candidate") or {}).get("metadata") or {}
        claim = (metadata.get("message_plan") or {}).get("claim")
        content = event.payload.get("content")
        if not claim or not isinstance(content, str):
            return None
        try:
            canonical = metadata.get("sleep_canonical") or None
            own = thought_from_data(claim)
            structure = analyze_structured_utterance(content)
            retold = reported_proposition(structure)
            thought = retold.content if retold else parse_proposition(
                structure, allow_question=True, perspective="user")
            if thought is None or meaning_key(thought, canonical) != meaning_key(own, canonical):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        same = bool(thought.negated) == bool(own.negated)
        if retold or "?" in content or any(c.clause_kind == "question" for c in structure.clauses):
            return "premise" if same else None  # an opposite-sign question may be a challenge
        return "supports" if same else "contradicts"

    async def consolidate_chat_norms(self) -> int:
        """Night, by server (4.43, flag only): the norms of the chat (what is register and what an
        attack, what the chat laughs at, what it answers with hostility) are recounted from the
        reactions of ≥2 different people. Norms are hypotheses; they are per server and hold no
        names. Seed and corpus exchanges do not count: C starts empty."""
        from debilgi.cognition.chat_norms import consolidate
        from debilgi.config import thought_content_enabled

        if not thought_content_enabled():
            return 0
        guilds = await self._database.fetchall(
            "SELECT DISTINCT guild_id FROM language_observed_exchanges ORDER BY guild_id")
        return sum([await consolidate(self._database, row[0], self._now()) for row in guilds])

    async def review_person_hypotheses(self) -> int:
        """Night, by person (4.42, flag only): the trust of every hypothesis about a person is
        recounted from its own evidence at the night's clock — what was not confirmed for days
        fades. A hypothesis stays a hypothesis: no status here makes it a fact or a claim."""
        from debilgi.config import thought_content_enabled
        from debilgi.experience.meaning_intents import reweigh_person

        if not thought_content_enabled():
            return 0
        rows = await self._database.fetchall(
            "SELECT hypothesis_id FROM continuous_cognition_hypotheses "
            "WHERE signature LIKE 'person:%' AND status != 'rejected' ORDER BY subject_key")
        return await reweigh_person(self._database, [r[0] for r in rows], self._now())

    async def observe_sleep_feedback(self, event: Event) -> int:
        from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR

        if event.payload.get("author_is_bot") or "corpus" in event.source:
            return 0
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        if not feedback.is_learnable() or feedback.target_message_id is None:
            return 0
        rows = await self._database.fetchall(
            "SELECT * FROM continuous_cognition_hypotheses WHERE guild_id=? "
            "AND predicate_key='sleep_response'", (event.guild_id,)
        )
        changed = 0
        for row in rows:
            data = json.loads(row["roles_json"])
            if (event.actor_id != data["actor_id"] or event.channel_id != data["channel_id"]
                    or feedback.target_message_id not in data.get("receipts", ())):
                continue
            relation = "supports" if feedback.reward > 0 else "contradicts"
            stance = self._stance_toward_claim(event, data)
            if stance == "premise":
                continue  # a question or a retelling takes the position as given: no verdict
            relation = stance or relation
            cursor = await self._database.execute(
                "INSERT OR IGNORE INTO continuous_cognition_evidence "
                "(evidence_id,hypothesis_id,source_event_id,relation,confidence,observed_at) "
                "VALUES (?,?,?,?,?,?)",
                (row[0] + ':' + event.event_id, row[0], event.event_id, relation,
                 feedback.attribution_confidence, event.utc_timestamp.isoformat()),
            )
            if cursor.rowcount:
                from debilgi.language.proposition_memory import count_meaning_evidence

                evidence = await self._database.fetchall(
                    "SELECT e.actor_id,e.source,e.payload_json,x.* FROM "
                    "continuous_cognition_evidence x JOIN events e "
                    "ON e.event_id=x.source_event_id WHERE x.hypothesis_id=?", (row[0],))
                support = count_meaning_evidence([
                    {**dict(r), "polarity": 1 if r["relation"] == "supports" else -1,
                     "author_is_bot": json.loads(r["payload_json"]).get("author_is_bot")}
                    for r in evidence])
                await self._database.execute(
                    "UPDATE continuous_cognition_hypotheses SET confidence=?,"
                    "positive_evidence=?,negative_evidence=?,"
                    "status=CASE WHEN ? THEN 'rejected' ELSE status END,"
                    "last_reviewed_at=? WHERE hypothesis_id=?",
                    (max(0.05, min(0.8, 0.5 + 0.15 * (support.supports - support.contradicts))),
                     support.supports, support.contradicts, support.opposition_events > 0,
                     self._now().isoformat(), row[0]),
                )
                changed += 1
        return changed

    async def record_response_hypothesis(self, event, descriptor, confidence, receipt_id):
        """A sent question is a proposal with zero confirming observations."""
        descriptor = dict(descriptor)
        if not descriptor.get("proposition"):
            return None
        signature = (
            "response:"
            + hashlib.sha256(
                (event.event_id + json.dumps(descriptor, sort_keys=True)).encode()
            ).hexdigest()[:32]
        )
        identifier = "response-hypothesis:" + signature[9:]
        data = {
            "descriptor": descriptor,
            "receipt_id": receipt_id,
            "prompt_event_id": event.event_id,
            "speaker_id": event.actor_id,
            "channel_id": event.channel_id,
        }
        await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_hypotheses
               (hypothesis_id, guild_id, signature, subject_key, predicate_key, object_key,
                roles_json, epistemic_status, confidence, positive_evidence, negative_evidence,
                status, first_seen_at, last_reviewed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'hypothesis', ?, 0, 0, 'working', ?, ?)""",
            (
                identifier,
                event.guild_id,
                signature,
                f"discord:user:{event.actor_id}",
                descriptor["predicate"],
                descriptor.get("topic"),
                json.dumps(data, ensure_ascii=False),
                min(0.95, max(0.0, confidence)),
                event.utc_timestamp.isoformat(),
                event.utc_timestamp.isoformat(),
            ),
        )
        return identifier

    async def observe_response_hypotheses(self, event):
        """Only human confirmation or independent canonical observations supply evidence.

        Receipts, confidence thresholds and agent repetitions cannot confirm a
        proposition. The working table never becomes the Known reader.
        """
        from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
        from debilgi.language.tokenizer import tokenize
        from debilgi.social.signals import AFFIRMATION, REFUSAL

        if event.payload.get("author_is_bot") or "corpus" in event.source:
            return 0
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        words = {
            t for t in tokenize(str(event.payload.get("content", "")).casefold()) if t[:1].isalnum()
        }
        rows = await self._database.fetchall(
            "SELECT * FROM continuous_cognition_hypotheses WHERE guild_id=? "
            "AND signature LIKE 'response:%' AND first_seen_at < ? LIMIT 128",
            (event.guild_id, event.utc_timestamp.isoformat()),
        )
        changed = 0
        for row in rows:
            data = json.loads(row["roles_json"])
            descriptor = data["descriptor"]
            target = (
                feedback.target_message_id == data["receipt_id"]
                and event.actor_id == data["speaker_id"]
                and event.channel_id == data["channel_id"]
            )
            explicit = bool(
                target
                and descriptor.get("confirmation") == "polar"
                and words
                and words <= AFFIRMATION
            )
            negative = bool(target and (feedback.reward < 0 or words & REFUSAL))
            from debilgi.language.dialogue_obligations import thought_from_data
            from debilgi.language.lexicon import LexiconService
            from debilgi.language.semantic_graph import proposition_key

            proposed = json.loads(descriptor["proposition"])
            proposed_thought = thought_from_data(proposed)
            canon = await LexiconService.for_meaning(self._database).canonical_lemmas(
                event.guild_id
            )
            candidates = await self._database.fetchall(
                "SELECT c.claim_id,c.thought_json FROM meaning_claims c "
                "JOIN meaning_evidence x ON x.claim_id=c.claim_id WHERE x.source_event_id=? "
                "AND x.actor_id IS NOT NULL AND c.subject_ref=? AND c.resolved=1 "
                "AND c.source_kind NOT IN ('agent_observed','corpus_form') AND c.polarity=? "
                "AND NOT EXISTS (SELECT 1 FROM json_each(c.modality_json) "
                "WHERE json_extract(value,'$.operator')!='negation')",
                (event.event_id, row["subject_key"], -1 if proposed_thought.negated else 1),
            )
            canonical = None
            for candidate in candidates:
                actual = thought_from_data(json.loads(candidate["thought_json"]))
                if proposition_key(actual, canonical=canon) == proposition_key(
                    proposed_thought, canonical=canon
                ) and all(
                    getattr(actual.semantic_time, k) == getattr(proposed_thought.semantic_time, k)
                    for k in ("tense", "start", "end")
                ):
                    canonical = candidate
                    break
            if not (explicit or negative or canonical):
                continue
            relation = "contradicts" if negative else "supports"
            cursor = await self._database.execute(
                """INSERT OR IGNORE INTO continuous_cognition_evidence
                   (evidence_id,hypothesis_id,source_event_id,relation,confidence,details_json,observed_at)
                   VALUES (?, ?, ?, ?, 1.0, ?, ?)""",
                (
                    f"response-evidence:{row['hypothesis_id']}:{event.event_id}",
                    row["hypothesis_id"],
                    event.event_id,
                    relation,
                    json.dumps(
                        {
                            "explicit_confirmation": explicit,
                            "canonical_claim": canonical[0] if canonical else None,
                        }
                    ),
                    event.utc_timestamp.isoformat(),
                ),
            )
            if not cursor.rowcount:
                continue
            evidence = await self._database.fetchall(
                "SELECT x.*,e.actor_id,e.source FROM continuous_cognition_evidence x "
                "JOIN events e ON e.event_id=x.source_event_id WHERE x.hypothesis_id=? "
                "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0",
                (row["hypothesis_id"],),
            )
            from debilgi.language.proposition_memory import count_meaning_evidence

            support = count_meaning_evidence(
                [
                    {
                        **dict(e),
                        "polarity": 1 if e["relation"] == "supports" else -1,
                        "observed_at": e["observed_at"],
                        "source_event_id": e["source_event_id"],
                        "explicit_confirmation": json.loads(e["details_json"]).get(
                            "explicit_confirmation"
                        ),
                    }
                    for e in evidence
                ]
            )
            positive, negatives = support.supports, support.contradicts
            retained = support.hypothesis_confirmed
            confidence = min(0.95, row["confidence"] + (0.15 if not negative else -0.25))
            await self._database.execute(
                "UPDATE continuous_cognition_hypotheses SET confidence=?,positive_evidence=?, "
                "negative_evidence=?,status=?,last_reviewed_at=? WHERE hypothesis_id=?",
                (
                    max(0.0, confidence),
                    positive,
                    negatives,
                    "rejected"
                    if support.opposition_events
                    else "retained"
                    if retained
                    else "working",
                    event.utc_timestamp.isoformat(),
                    row["hypothesis_id"],
                ),
            )
            from debilgi.language.proposition_memory import PropositionMemory

            await PropositionMemory.for_meaning(self._database).review_response_hypothesis(
                row["hypothesis_id"], event, promote=retained
            )
            changed += 1
        return changed

    async def pulse_once(self, now: datetime | None = None) -> CognitivePulseStats:
        async with self._pulse_lock:
            started = self._aware(now or self._now())
            pulse_id = f"continuous-pulse:{uuid4()}"
            await self._database.execute(
                """INSERT INTO continuous_cognition_pulses (
                       pulse_id, started_at, status
                   ) VALUES (?, ?, 'running')""",
                (pulse_id, started.isoformat()),
            )
            claimed = hypotheses = gaps = signals = 0
            try:
                rows = await self._database.fetchall(
                    """SELECT work_id, source_event_id, guild_id, attempts
                       FROM continuous_cognition_work_items
                       WHERE status = 'queued' AND next_attempt_at <= ?
                       ORDER BY priority DESC, created_at
                       LIMIT ?""",
                    (started.isoformat(), self.config.work_per_pulse),
                )
                for row in rows:
                    if not await self._claim(str(row["work_id"]), started):
                        continue
                    claimed += 1
                    try:
                        result = await self._process_work_item(row, started)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await self._fail_work(row, started, exc)
                        logger.exception(
                            "work=%s event=%s status=failed",
                            row["work_id"],
                            row["source_event_id"],
                        )
                        continue
                    hypotheses += result[0]
                    gaps += result[1]
                    signals += result[2]
                stats = CognitivePulseStats(
                    pulse_id,
                    claimed,
                    hypotheses,
                    gaps,
                    signals,
                )
                await self._finish_pulse(stats, started, status="completed")
                if claimed or hypotheses or gaps or signals:
                    logger.info(
                        "pulse=%s status=completed work=%d hypotheses=%d gaps=%d signals=%d",
                        pulse_id,
                        claimed,
                        hypotheses,
                        gaps,
                        signals,
                    )
                return stats
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._database.execute(
                    """UPDATE continuous_cognition_pulses
                       SET finished_at = ?, status = 'failed', error_type = ?,
                           error_message = ? WHERE pulse_id = ?""",
                    (
                        self._now().isoformat(),
                        type(exc).__name__,
                        str(exc)[:1000],
                        pulse_id,
                    ),
                )
                await self._database.execute(
                    """UPDATE continuous_cognition_runtime
                       SET status = 'degraded', last_heartbeat_at = ?,
                           total_pulses = total_pulses + 1, last_error = ?
                       WHERE runtime_id = 1""",
                    (
                        self._now().isoformat(),
                        f"{type(exc).__name__}: {exc}"[:1000],
                    ),
                )
                logger.exception("pulse=%s status=failed", pulse_id)
                raise

    async def relevant_facts(
        self,
        guild_id: int,
        terms: Iterable[str],
        *,
        limit: int = 8,
    ) -> tuple[str, ...]:
        """Recall working conclusions without promoting hypotheses to facts."""

        normalized = {
            _normalize_term(term)
            for term in terms
            if isinstance(term, str) and _normalize_term(term)
        }
        if not normalized:
            return ()
        rows = await self._database.fetchall(
            """SELECT subject_key, predicate_key, object_key, epistemic_status,
                      confidence, status
               FROM continuous_cognition_hypotheses
               WHERE guild_id = ? AND status != 'rejected' AND signature NOT LIKE 'response:%'
                 AND signature NOT LIKE 'person:%' AND signature NOT LIKE 'chat:%'
               ORDER BY
                   CASE status WHEN 'retained' THEN 0 ELSE 1 END,
                   confidence DESC, last_reviewed_at DESC
               LIMIT 100""",
            (guild_id,),
        )
        ranked: list[tuple[int, float, str]] = []
        for row in rows:
            values = tuple(
                _normalize_term(str(row[key]))
                for key in ("subject_key", "predicate_key", "object_key")
                if row[key] is not None
            )
            overlap = len(normalized & set(values))
            if overlap == 0:
                continue
            subject = str(row["subject_key"] or "").strip()
            predicate = str(row["predicate_key"]).strip()
            object_key = str(row["object_key"] or "").strip()
            statement = " ".join(part for part in (subject, predicate, object_key) if part)
            marker = str(row["epistemic_status"])
            ranked.append(
                (
                    overlap,
                    float(row["confidence"]),
                    f"{statement} [{marker}; continuous_cognition]",
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in ranked[: max(1, min(limit, 24))])

    async def _run_loop(self) -> None:
        while True:
            self._wake.clear()
            try:
                await self.rhythm_once()
                stats = await self.pulse_once()
                if self.sleeping:
                    await self.sleep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                stats = None
            if stats is not None and stats.work_claimed >= self.config.work_per_pulse:
                continue
            delay = (
                self.config.active_interval
                if stats is not None and stats.work_claimed
                else self.config.idle_interval
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=delay.total_seconds(),
                )

    async def _claim(self, work_id: str, now: datetime) -> bool:
        cursor = await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'processing', attempts = attempts + 1, updated_at = ?
               WHERE work_id = ? AND status = 'queued'""",
            (now.isoformat(), work_id),
        )
        return cursor.rowcount == 1

    async def _process_work_item(
        self,
        row: Mapping[str, Any],
        now: datetime,
    ) -> tuple[int, int, int]:
        work_id = str(row["work_id"])
        event_id = str(row["source_event_id"])
        guild_id = int(row["guild_id"])
        attempts = int(row["attempts"]) + 1
        trace = await self._database.fetchone(
            """SELECT status, selected_route, failure_stage, failure_reason
               FROM language_response_traces
               WHERE source_event_id = ?""",
            (event_id,),
        )
        if trace is None or str(trace["status"]) == "started":
            if attempts < self.config.trace_retry_limit:
                await self._defer(work_id, now, attempts, "response_trace_not_complete")
                return 0, 0, 0
            await self._complete(
                work_id,
                now,
                "trace_unavailable_after_retry_budget",
            )
            return 0, 0, 0

        step = await self._database.fetchone(
            """SELECT details_json
               FROM language_response_trace_steps AS step
               JOIN language_response_traces AS trace
                 ON trace.trace_id = step.trace_id
               WHERE trace.source_event_id = ? AND step.stage = 'inquiry'
               ORDER BY step.sequence DESC LIMIT 1""",
            (event_id,),
        )
        hypotheses = gaps = 0
        if step is not None:
            details = _json_mapping(step["details_json"])
            hypotheses = await self._review_conclusions(
                guild_id,
                event_id,
                details.get("conclusions"),
                now,
            )
            gaps = await self._record_gaps(
                guild_id,
                event_id,
                details.get("steps"),
                now,
            )
        await self.capture_sleep_gap(event_id, now=now)
        signals = await self._record_signal(guild_id, event_id, trace, now)
        summary = (
            f"trace={trace['status']}; hypotheses={hypotheses}; gaps={gaps}; signals={signals}"
        )
        await self._complete(work_id, now, summary)
        return hypotheses, gaps, signals

    async def _review_conclusions(
        self,
        guild_id: int,
        event_id: str,
        raw_conclusions: object,
        now: datetime,
    ) -> int:
        if not isinstance(raw_conclusions, list):
            return 0
        source = await self._database.fetchone(
            "SELECT actor_id FROM events WHERE event_id=? AND guild_id=?", (event_id, guild_id)
        )
        reviewed = 0
        for raw in raw_conclusions[:12]:
            if not isinstance(raw, dict):
                continue
            predicate = _optional_text(raw.get("predicate"))
            epistemic_status = _optional_text(raw.get("epistemic_status"))
            roles = _roles(raw.get("roles"))
            if predicate is None or epistemic_status not in {"inference", "hypothesis"}:
                continue
            subject = next((value for role, value in roles if role == "subject"), None)
            object_key = next(
                (value for role, value in roles if role != "subject"),
                None,
            )
            confidence = _unit_float(raw.get("confidence"))
            from debilgi.language.lexicon import LexiconService

            canon = await LexiconService.for_meaning(self._database).canonical_lemmas(guild_id)
            from debilgi.language.dialogue_obligations import thought_from_data

            proposition = raw.get("proposition")
            thought = thought_from_data(proposition) if isinstance(proposition, dict) else None
            signature = _signature(predicate, roles, canonical=canon, thought=thought)
            from debilgi.language.proposition_memory import world_thought

            participants = (
                sorted(
                    {
                        word.participant_ref
                        for dep in thought.dependencies
                        for word in (dep.argument.head, *dep.argument.modifiers)
                        if word.participant_ref and word.participant_ref.startswith("discord:user:")
                    }
                )
                if thought
                else []
            )
            private = (
                thought is not None
                and not world_thought(thought)
                or any(value in {"я", "ты", "мы", "вы", "мой", "наш"} for _, value in roles)
            )
            if private:
                owner = ",".join(participants) or f"speaker:{source[0] if source else event_id}"
                signature = "personal:" + owner + ":" + signature
                subject = owner
            hypothesis_id = f"continuous-hypothesis:{guild_id}:{signature}"
            evidence_id = f"continuous-evidence:{event_id}:{signature}:supports"
            async with self._database.transaction() as connection:
                existing = await (
                    await connection.execute(
                        """SELECT confidence, positive_evidence, negative_evidence
                           FROM continuous_cognition_hypotheses
                           WHERE guild_id = ? AND signature = ?""",
                        (guild_id, signature),
                    )
                ).fetchone()
                if existing is None:
                    await connection.execute(
                        """INSERT INTO continuous_cognition_hypotheses (
                               hypothesis_id, guild_id, signature, subject_key,
                               predicate_key, object_key, roles_json,
                               epistemic_status, confidence, positive_evidence,
                               status, first_seen_at, last_reviewed_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'working', ?, ?)""",
                        (
                            hypothesis_id,
                            guild_id,
                            signature,
                            subject,
                            predicate,
                            object_key,
                            json.dumps(roles, ensure_ascii=False, separators=(",", ":")),
                            epistemic_status,
                            confidence,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )
                inserted = await connection.execute(
                    """INSERT OR IGNORE INTO continuous_cognition_evidence (
                           evidence_id, hypothesis_id, source_event_id, relation,
                           confidence, details_json, observed_at
                       ) VALUES (?, ?, ?, 'supports', ?, ?, ?)""",
                    (
                        evidence_id,
                        hypothesis_id,
                        event_id,
                        confidence,
                        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                        now.isoformat(),
                    ),
                )
                if inserted.rowcount == 0:
                    continue
                from debilgi.language.proposition_memory import count_meaning_evidence

                evidence = await self._database.fetchall(
                    "SELECT e.actor_id,e.source,e.payload_json,x.* "
                    "FROM continuous_cognition_evidence x JOIN events e "
                    "ON e.event_id=x.source_event_id WHERE x.hypothesis_id=? AND e.guild_id=?",
                    (hypothesis_id, guild_id),
                )
                support = count_meaning_evidence(
                    [
                        {
                            **dict(e),
                            "polarity": 1 if e["relation"] == "supports" else -1,
                            "author_is_bot": json.loads(e["payload_json"]).get("author_is_bot"),
                        }
                        for e in evidence
                    ]
                )
                combined_confidence = max(
                    confidence, float(existing["confidence"]) if existing else 0
                )
                status = (
                    "rejected"
                    if support.opposition_events
                    else "retained"
                    if support.hypothesis_confirmed and combined_confidence >= 0.66
                    else "working"
                )
                await connection.execute(
                    "UPDATE continuous_cognition_hypotheses SET confidence=?,positive_evidence=?,"
                    "negative_evidence=?,status=?,last_reviewed_at=? WHERE hypothesis_id=?",
                    (
                        combined_confidence,
                        support.supports,
                        support.contradicts,
                        status,
                        now.isoformat(),
                        hypothesis_id,
                    ),
                )
            reviewed += 1
        return reviewed

    async def _record_gaps(
        self,
        guild_id: int,
        event_id: str,
        raw_steps: object,
        now: datetime,
    ) -> int:
        if not isinstance(raw_steps, list):
            return 0
        recorded = 0
        for raw in raw_steps[:24]:
            if not isinstance(raw, dict) or raw.get("status") != "unresolved":
                continue
            question_kind = _optional_text(raw.get("question_kind"))
            answer_role = _optional_text(raw.get("answer_role"))
            if question_kind is None or answer_role is None:
                continue
            subject = _optional_text(raw.get("subject"))
            predicate = _optional_text(raw.get("predicate"))
            signature = _signature(
                question_kind,
                (
                    ("subject", subject or ""),
                    ("predicate", predicate or ""),
                    ("answer_role", answer_role),
                ),
            )
            cursor = await self._database.execute(
                """INSERT INTO continuous_cognition_gaps (
                       gap_id, guild_id, signature, question_kind, subject_key,
                       predicate_key, answer_role, source_event_id,
                       first_seen_at, last_seen_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id, signature) DO UPDATE SET
                       occurrence_count = occurrence_count + 1,
                       source_event_id = excluded.source_event_id,
                       last_seen_at = excluded.last_seen_at,
                       status = CASE
                           WHEN status = 'abandoned' THEN 'open' ELSE status END""",
                (
                    f"continuous-gap:{guild_id}:{signature}",
                    guild_id,
                    signature,
                    question_kind,
                    subject,
                    predicate,
                    answer_role,
                    event_id,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            recorded += int(cursor.rowcount > 0)
        return recorded

    async def _record_signal(
        self,
        guild_id: int,
        event_id: str,
        trace: Mapping[str, Any],
        now: datetime,
    ) -> int:
        status = str(trace["status"])
        kind = {
            "sent": "reply_sent",
            "failed": "reply_failed",
            "withheld": "reply_withheld",
            "no_reply": "initiative_not_selected",
        }.get(status, f"reply_{status}")
        details = {
            "status": status,
            "failure_stage": trace["failure_stage"],
            "failure_reason": trace["failure_reason"],
        }
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_signals (
                   signal_id, source_event_id, guild_id, signal_kind, route,
                   details_json, observed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"continuous-signal:{event_id}",
                event_id,
                guild_id,
                kind,
                trace["selected_route"],
                json.dumps(details, ensure_ascii=False, separators=(",", ":")),
                now.isoformat(),
            ),
        )
        return int(cursor.rowcount == 1)

    async def _defer(
        self,
        work_id: str,
        now: datetime,
        attempts: int,
        reason: str,
    ) -> None:
        retry_at = now + self.config.active_interval
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'queued', next_attempt_at = ?, updated_at = ?,
                   result_summary = ?
               WHERE work_id = ?""",
            (retry_at.isoformat(), now.isoformat(), reason, work_id),
        )

    async def _complete(self, work_id: str, now: datetime, summary: str) -> None:
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'completed', result_summary = ?, updated_at = ?,
                   completed_at = ?, last_error = NULL
               WHERE work_id = ?""",
            (summary, now.isoformat(), now.isoformat(), work_id),
        )

    async def _fail_work(
        self,
        row: Mapping[str, Any],
        now: datetime,
        exc: Exception,
    ) -> None:
        attempts = int(row["attempts"]) + 1
        retry = attempts < self.config.trace_retry_limit
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = ?, next_attempt_at = ?, updated_at = ?, last_error = ?
               WHERE work_id = ?""",
            (
                "queued" if retry else "failed",
                (now + self.config.active_interval).isoformat(),
                now.isoformat(),
                f"{type(exc).__name__}: {exc}"[:1000],
                str(row["work_id"]),
            ),
        )

    async def _finish_pulse(
        self,
        stats: CognitivePulseStats,
        started: datetime,
        *,
        status: str,
    ) -> None:
        await self._database.execute(
            """UPDATE continuous_cognition_pulses
               SET finished_at = ?, status = ?, work_claimed = ?,
                   hypotheses_reviewed = ?, gaps_recorded = ?,
                   signals_recorded = ?
               WHERE pulse_id = ?""",
            (
                max(started, self._now()).isoformat(),
                status,
                stats.work_claimed,
                stats.hypotheses_reviewed,
                stats.gaps_recorded,
                stats.signals_recorded,
                stats.pulse_id,
            ),
        )
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
                   SET status = 'running', last_heartbeat_at = ?,
                   total_pulses = total_pulses + 1,
                   total_work_processed = total_work_processed + ?,
                   last_error = NULL
               WHERE runtime_id = 1""",
            (
                max(started, self._now()).isoformat(),
                stats.work_claimed,
            ),
        )
        if stats.work_claimed == 0:
            await self._database.execute(
                "DELETE FROM continuous_cognition_pulses WHERE pulse_id = ?",
                (stats.pulse_id,),
            )

    def _now(self) -> datetime:
        return self._aware(self._clock())

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("continuous cognition clock must be timezone-aware")
        return value.astimezone(UTC)


def _roles(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        (str(pair[0]), str(pair[1]))
        for pair in value
        if isinstance(pair, list)
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
    )


def _signature(
    predicate: str, roles: Iterable[tuple[str, str]], *, canonical=None, thought=None
) -> str:
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )
    from debilgi.language.semantic_graph import proposition_key

    thought = thought or ClauseThought(
        ConceptLexeme(predicate, predicate),
        tuple(
            SemanticDependency(role, ConceptArgument(ConceptLexeme(value, value)))
            for role, value in roles
        ),
    )
    body = json.dumps(
        proposition_key(thought, canonical=canonical), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _merge_gap_details(base, update, occurrences=(), *, earlier=None, later=None) -> dict:
    """One hole keeps every occurrence, failure kind and tried plan, in time order."""
    parts = [earlier or {}, base, later or {}]
    result = {**dict(base), **dict(update)}
    result["occurrences"] = list(dict.fromkeys(
        [*(o for d in parts for o in d.get("occurrences", ())), *occurrences]))[-16:]
    result["kinds"] = list(dict.fromkeys(
        [*(k for d in parts for k in d.get("kinds", ())), *update.get("kinds", ())]))
    result["tried"] = list(dict.fromkeys(t for d in parts for t in d.get("tried", ())))[-64:]
    return result


def _json_mapping(value: object) -> Mapping[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize_term(value: str) -> str:
    return value.casefold().replace("ё", "е").strip(".,!?;:—–-()[]{}«»… ")


def _unit_float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.0
