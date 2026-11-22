"""Learn durable, attributed propositions from human events without a teacher."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo

from debilgi.config import thought_content_enabled
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import (
    RELATIVE_DAYS,
    SCHEMA_VERSION,
    CanonicalClaim,
    ClaimQuery,
    Known,
    Unknown,
    activity_thought,
    answer_thought,
    argument_ref,
    bound_participants,
    fact_thought,
    query_from_event,
    self_query,
    semantic_time,
    source_kind,
)
from debilgi.language.content import content_lemmas
from debilgi.language.dialogue_obligations import (
    BoundDialogueReport,
    exchange_from_events,
    thought_from_data,
)
from debilgi.language.generative_planner import proposition_sentences, statement_thought
from debilgi.language.morphology import person_cell
from debilgi.language.productive_nlg import identity_graph, realize_thought
from debilgi.language.semantic_graph import (
    MeaningProof,
    MeaningRecord,
    contextual,
    extract_meanings,
    meaning_from_data,
    proposition_key,
)
from debilgi.language.structured_utterance import SemanticTime, analyze_structured_utterance
from debilgi.language.word_relations import WordRelation, observe_word_relations
from debilgi.storage.database import Database


@dataclass(frozen=True)
class MeaningSupport:
    people: int
    days: int
    supports: int
    contradicts: int
    events: int
    explicit: bool = False
    weight: float = 0.0
    incumbent: int = 1
    established_people: int = 0
    opposition_events: int = 0

    @property
    def status(self) -> str:
        if self.supports and self.contradicts:
            return "disputed"
        return "retained" if max(self.supports, self.contradicts) >= 2 else "hypothesis"

    @property
    def hypothesis_confirmed(self) -> bool:
        return not self.opposition_events and (
            self.explicit or self.supports >= 2 and self.events >= 3
        )

    @property
    def prevailing_polarity(self) -> int:
        return self.incumbent


def count_meaning_evidence(rows) -> MeaningSupport:
    """One person per side, regardless of wording, repetitions or confidence.

    Adapters supply actor_id, observed_at, polarity and source_event_id. Corpus,
    bots and unresolved sources never vote. Events only serve the hypothesis gate.
    """
    humans = [dict(r) for r in rows]
    humans = [
        r
        for r in humans
        if r.get("actor_id") is not None
        and not r.get("author_is_bot")
        and "corpus" not in r.get("source", "")
        and not r.get("source", "").startswith("seed:")
    ]
    latest = {}
    incumbent, peak = 1, 0
    votes = Counter()
    for row in sorted(humans, key=lambda r: (str(r["observed_at"]), r["source_event_id"])):
        previous = latest.get(row["actor_id"])
        if previous is not None:
            votes[previous["polarity"]] -= 1
        latest[row["actor_id"]] = row
        votes[row["polarity"]] += 1
        if votes[incumbent] >= 2:
            peak = max(peak, votes[incumbent])
        challenger = -incumbent
        if votes[challenger] >= 2 and votes[challenger] > max(peak, votes[incumbent]):
            incumbent, peak = challenger, votes[challenger]
    positive = {actor for actor, r in latest.items() if r["polarity"] == 1}
    negative = set(latest) - positive
    return MeaningSupport(
        len(latest),
        len({str(r["observed_at"])[:10] for r in humans}),
        len(positive),
        len(negative),
        len(
            {
                r["source_event_id"]
                for r in humans
                if r["actor_id"] in positive and r["polarity"] == 1
            }
        ),
        any(r.get("explicit_confirmation") for r in latest.values() if r["polarity"] == 1),
        sum(min(1.0, max(0.25, latest[actor].get("reliability", 1.0))) for actor in latest),
        incumbent,
        peak,
        len({r["source_event_id"] for r in humans if r["polarity"] == -1}),
    )


def _asks_only(text: str) -> bool:
    """Every clause of the text is a question (the reader decides, with or without «?»)."""
    clauses = analyze_structured_utterance(text).clauses
    return bool(clauses) and all(c.clause_kind == "question" for c in clauses)


def meaning_key(thought, canonical=None) -> str:
    return json.dumps(
        proposition_key(thought, signed=False, canonical=canonical),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def world_thought(thought) -> bool:
    """Any personal participant or possessive keeps the entire report private."""
    subject = next((d.argument for d in thought.dependencies if d.role == "subject"), None)
    return bool(
        subject
        and not subject.head.participant_ref
        and not any(
            word.participant_ref
            or word.lemma in {"я", "мы", "ты", "вы", "мой", "наш", "твой", "ваш"}
            for dep in thought.dependencies
            for word in (dep.argument.head, *dep.argument.modifiers, *dep.argument.postmodifiers)
        )
    )


class PropositionMemory:
    def __init__(self, database: Database, *, timezone: tzinfo | None = None) -> None:
        self._database = database
        self._timezone = timezone
        if getattr(database, "_meaning_memory", None) is None:
            database._meaning_memory = self

    @classmethod
    def for_meaning(cls, database):
        existing = getattr(database, "_meaning_memory", None)
        return existing if existing is not None else cls(database)

    async def rebuild_synonym_claims(self, guild_id, terms):
        """Reproject shared assertions from their retained sources, also after revocation.

        Merging must never make a revoked synonym keep somebody else's vote.
        Private records keep their bindings; they do not supply public votes.
        """
        rows = await self._database.fetchall(
            "SELECT * FROM meaning_claims WHERE guild_id=? AND source_kind='other_report'",
            (guild_id,),
        )
        affected = []
        for row in rows:
            thought = thought_from_data(json.loads(row["thought_json"]))
            words = {
                thought.predicate.lemma,
                *(
                    w.lemma
                    for d in thought.dependencies
                    for w in (d.argument.head, *d.argument.modifiers, *d.argument.postmodifiers)
                ),
            }
            if world_thought(thought) and words & terms:
                affected.append(row["claim_id"])
        sources = {}
        for claim_id in affected:
            rows = await self._database.fetchall(
                "SELECT DISTINCT e.* FROM events e JOIN meaning_evidence x "
                "ON x.source_event_id=e.event_id JOIN meaning_claims c ON c.claim_id=x.claim_id "
                "WHERE c.claim_id=? AND e.guild_id=? "
                "AND (x.actor_id IS NOT NULL OR x.source_event_id=c.source_event_id)",
                (claim_id, guild_id),
            )
            sources.update((r["event_id"], r) for r in rows)
        if not affected:
            return
        async with self._database.transaction() as connection:
            for claim_id in affected:
                await connection.execute(
                    "DELETE FROM meaning_supersedes WHERE claim_id=? OR previous_claim_id=?",
                    (claim_id, claim_id),
                )
                await connection.execute(
                    "DELETE FROM meaning_arguments WHERE claim_id=?", (claim_id,)
                )
                await connection.execute(
                    "DELETE FROM meaning_evidence WHERE claim_id=?", (claim_id,)
                )
                await connection.execute("DELETE FROM meaning_claims WHERE claim_id=?", (claim_id,))
            await connection.executemany(
                "DELETE FROM meaning_claim_projections WHERE source_event_id=?",
                ((source,) for source in sources),
            )
        for row in sorted(sources.values(), key=lambda r: (r["occurred_at"], r["event_id"])):
            event = EventRepository._from_row(row)
            if not event.payload.get("content"):
                retained = await self._database.fetchall(
                    "SELECT text FROM language_proposition_reports WHERE source_event_id=? "
                    "UNION ALL SELECT json_extract(record_json,'$.text') "
                    "FROM language_meaning_records WHERE source_event_id=?",
                    (event.event_id, event.event_id),
                )
                event = replace(
                    event,
                    payload={
                        **event.payload,
                        "content": " ".join(dict.fromkeys(r[0] for r in retained if r[0])),
                    },
                )
            await self._project_claims(event)

    async def reindex_evidence(self, guild_id=None) -> None:
        """Upgrade historical provenance lazily; synonym changes rekey the same rows."""
        from debilgi.language.lexicon import LexiconService

        where = ("WHERE c.guild_id=?" if guild_id is not None else
                 "WHERE x.meaning_key IS NULL OR NOT EXISTS "
                 "(SELECT 1 FROM meaning_evidence_revision r WHERE r.guild_id=c.guild_id)")
        rows = await self._database.fetchall(
            "SELECT DISTINCT c.* FROM meaning_claims c "
            "JOIN meaning_evidence x ON x.claim_id=c.claim_id " + where,
            (guild_id,) if guild_id is not None else (),
        )
        maps = {}
        for row in rows:
            if row["guild_id"] not in maps:
                maps[row["guild_id"]] = await LexiconService.for_meaning(
                    self._database
                ).canonical_lemmas(row["guild_id"])
        updates = []
        for row in rows:
            thought = thought_from_data(json.loads(row["thought_json"]))
            updates.append(
                (
                    meaning_key(thought, maps[row["guild_id"]]),
                    row["guild_id"],
                    -1 if thought.negated else 1,
                    row["claim_id"],
                )
            )
        if rows:
            async with self._database.transaction() as connection:
                await connection.executemany(
                    "UPDATE meaning_claims SET scope=?,time_start=?,time_end=? WHERE claim_id=?",
                    [("world" if world_thought(t) else "private", t.semantic_time.start,
                      t.semantic_time.end, r["claim_id"])
                     for r in rows for t in [thought_from_data(json.loads(r["thought_json"]))]])
        if updates:
            async with self._database.transaction() as connection:
                await connection.executemany(
                    "UPDATE meaning_evidence SET meaning_key=?,guild_id=?,polarity=?,"
                    "actor_id=CASE WHEN (SELECT source_kind FROM meaning_claims c "
                    "WHERE c.claim_id=meaning_evidence.claim_id)='agent_observed' THEN NULL "
                    "WHEN source_event_id=(SELECT source_event_id "
                    "FROM meaning_claims c "
                    "WHERE c.claim_id=meaning_evidence.claim_id) "
                    "OR source_key LIKE 'confirmed_hypothesis:%' "
                    "THEN (SELECT actor_id FROM events e WHERE e.event_id=source_event_id) "
                    "ELSE actor_id END,"
                    "observed_at=(SELECT occurred_at FROM events e "
                    "WHERE e.event_id=source_event_id) "
                    "WHERE claim_id=?",
                    updates,
                )

        if rows:
            # Upgrades and learned synonyms coalesce already retained world claims.
            async with self._database.transaction() as connection:
                grouped = {}
                for row in rows:
                    claim = row
                    if not claim or claim["source_kind"] != "other_report":
                        continue
                    thought = thought_from_data(json.loads(claim["thought_json"]))
                    if not world_thought(thought):
                        continue
                    identity = (
                        row["guild_id"],
                        meaning_key(thought, maps[row["guild_id"]]),
                        claim["polarity"],
                        claim["modality_json"],
                        thought.semantic_time.start,
                        thought.semantic_time.end,
                    )
                    retained = grouped.setdefault(identity, claim["claim_id"])
                    if retained == claim["claim_id"]:
                        continue
                    await connection.execute(
                        "INSERT OR IGNORE INTO meaning_evidence "
                        "(claim_id,source_event_id,source_key,meaning_key,guild_id,actor_id,"
                        "observed_at,polarity) SELECT "
                        "?,source_event_id,source_key,"
                        "meaning_key,guild_id,actor_id,observed_at,polarity FROM meaning_evidence "
                        "WHERE claim_id=?",
                        (retained, claim["claim_id"]),
                    )
                    await connection.execute(
                        "DELETE FROM meaning_evidence WHERE claim_id=?", (claim["claim_id"],)
                    )
                    await connection.execute(
                        "DELETE FROM meaning_arguments WHERE claim_id=?", (claim["claim_id"],)
                    )
                    await connection.execute(
                        "DELETE FROM meaning_supersedes WHERE claim_id=? OR previous_claim_id=?",
                        (claim["claim_id"], claim["claim_id"]),
                    )
                    await connection.execute(
                        "DELETE FROM meaning_claims WHERE claim_id=?", (claim["claim_id"],)
                    )
        if rows:
            async with self._database.transaction() as connection:
                await connection.execute(
                    "UPDATE meaning_evidence SET voter=EXISTS(SELECT 1 FROM events e "
                    "JOIN meaning_claims c ON c.claim_id=meaning_evidence.claim_id "
                    "WHERE e.event_id=meaning_evidence.source_event_id "
                    "AND e.guild_id=meaning_evidence.guild_id "
                    "AND meaning_evidence.actor_id IS NOT NULL "
                    "AND c.resolved=1 AND c.source_kind IN ('self_report','other_report') "
                    "AND COALESCE(json_extract(e.payload_json,'$.author_is_bot'),0)=0 "
                    "AND instr(e.source,'corpus')=0 AND e.source NOT LIKE 'seed:%'),"
                    "scope=(SELECT scope FROM meaning_claims c "
                    "WHERE c.claim_id=meaning_evidence.claim_id),"
                    "time_start=(SELECT time_start FROM meaning_claims c "
                    "WHERE c.claim_id=meaning_evidence.claim_id),"
                    "time_end=(SELECT time_end FROM meaning_claims c "
                    "WHERE c.claim_id=meaning_evidence.claim_id) "
                    + ("WHERE guild_id=?" if guild_id is not None else ""),
                    (guild_id,) if guild_id is not None else ())
                cursor = await connection.execute(
                    "SELECT DISTINCT guild_id,meaning_key FROM meaning_evidence WHERE voter=1")
                for guild, key in await cursor.fetchall():
                    await self._materialize_support(connection, guild, key)

    async def _materialize_support(self, connection, guild, key):
        cursor = await connection.execute(
            "SELECT actor_id,observed_at,polarity,source_event_id,source_key,"
            "scope,time_start,time_end "
            "FROM meaning_evidence WHERE guild_id=? AND meaning_key=? AND voter=1 "
            "ORDER BY observed_at,source_event_id", (guild, key))
        rows = [dict(r) for r in await cursor.fetchall()]
        await connection.execute(
            "DELETE FROM meaning_positions WHERE guild_id=? AND meaning_key=?", (guild, key))
        await connection.execute(
            "DELETE FROM meaning_support WHERE guild_id=? AND meaning_key=?", (guild, key))
        groups = {}
        for row in rows:
            interval = json.dumps([row["time_start"], row["time_end"]])
            for scope in ("all", row["scope"]):
                for period in ("*", interval):
                    groups.setdefault((scope, period), []).append(row)
        for (scope, period), evidence in groups.items():
            latest = {}
            for row in evidence:
                latest[row["actor_id"]] = row
            data = {"support": asdict(count_meaning_evidence(evidence)),
                    "actors": list(latest), "last_seen": max(r["observed_at"] for r in evidence)}
            await connection.execute("INSERT INTO meaning_support VALUES (?,?,?,?,?)",
                (guild, key, scope, period, json.dumps(data)))
            await connection.executemany("INSERT INTO meaning_positions VALUES (?,?,?,?,?,?,?,?)",
                [(guild,key,actor,scope,period,r["polarity"],r["observed_at"],r["source_event_id"])
                 for actor,r in latest.items()])

    async def support(
        self, guild_id, key, *, snapshot=None, exclude_event=None, interval=None, public=False
    ):
        scope = "world" if public else "all"
        period = "*" if interval is None else json.dumps(list(interval))
        # One indexed read: use ready counts when the snapshot includes all witnesses;
        # historical/excluded-event queries use the same indexed provenance, without JSON parsing.
        rows = await self._database.fetchall(
            "WITH ready AS (SELECT support_json FROM meaning_support WHERE guild_id=? "
            "AND meaning_key=? AND scope=? AND interval_key=? "
            "AND (? IS NULL OR json_extract(support_json,'$.last_seen')<?) "
            "AND NOT EXISTS(SELECT 1 FROM meaning_evidence WHERE guild_id=? AND meaning_key=? "
            "AND voter=1 AND source_event_id=?)) "
            "SELECT support_json,NULL actor_id,NULL observed_at,NULL polarity,NULL source_event_id,"
            "(SELECT revision FROM meaning_evidence_revision WHERE guild_id=?) revision "
            "FROM ready UNION ALL SELECT NULL,actor_id,observed_at,polarity,source_event_id,"
            "(SELECT revision FROM meaning_evidence_revision WHERE guild_id=?) revision "
            "FROM meaning_evidence WHERE guild_id=? AND meaning_key=? AND voter=1 "
            "AND (?=0 OR scope='world') AND (?=1 OR (time_start IS ? AND time_end IS ?)) "
            "AND (? IS NULL OR observed_at<?) AND (? IS NULL OR source_event_id!=?) "
            "AND NOT EXISTS(SELECT 1 FROM ready)",
            (guild_id,key,scope,period,snapshot,snapshot,guild_id,key,exclude_event,
             guild_id,guild_id,guild_id,key,int(public),int(interval is None),
             *(interval or (None,None)),
             snapshot,snapshot,exclude_event,exclude_event))
        if rows and rows[0]["support_json"]:
            data = json.loads(rows[0]["support_json"])
            result = MeaningSupport(**data["support"])
            actors = data["actors"]
        else:
            result = count_meaning_evidence(rows)
            actors = {r["actor_id"] for r in rows}
        if not actors:
            return result
        reliability = await self._source_history(guild_id, revision=rows[0]["revision"] or 0)
        weight = sum(self._reliability(reliability.get(actor, {}), key) for actor in actors)
        return replace(result, weight=weight)

    async def _source_history(self, guild_id, *, revision=None):
        if revision is None:
            row = await self._database.fetchone(
                "SELECT revision FROM meaning_evidence_revision WHERE guild_id=?", (guild_id,))
            revision = row[0] if row else 0
        cache = getattr(self._database, "_source_reliability", {})
        if guild_id in cache and cache[guild_id][0] == revision:
            return cache[guild_id][1]
        rows = await self._database.fetchall(
            "SELECT actor_id,meaning_key,polarity,observed_at,source_event_id "
            "FROM meaning_evidence WHERE guild_id=? AND voter=1 AND scope='world' "
            "ORDER BY observed_at,source_event_id", (guild_id,))
        grouped = {}
        for row in rows:
            grouped.setdefault(row["meaning_key"], []).append(row)
        histories = getattr(self._database, "_source_history_keys", {})
        previous = histories.get(guild_id, {})
        current = {}
        history = {}
        for key, evidence in grouped.items():
            fingerprint = tuple(tuple(r) for r in evidence)
            retained = previous.get(key)
            if retained and retained[0] == fingerprint:
                contributions = retained[1]
            else:
                latest, votes, peaks = {}, Counter(), Counter()
                for row in evidence:
                    actor = row["actor_id"]
                    if actor in latest:
                        votes[latest[actor]["polarity"]] -= 1
                    latest[actor] = row
                    votes[row["polarity"]] += 1
                    peaks[1] = max(peaks[1], votes[1])
                    peaks[-1] = max(peaks[-1], votes[-1])
                contributions = {}
                for actor, row in latest.items():
                    positive = votes[1] - int(row["polarity"] == 1)
                    negative = votes[-1] - int(row["polarity"] == -1)
                    if positive and negative or max(positive, negative) < 2:
                        continue
                    prevailing = 1 if positive else -1
                    # A final quorum above every opposing historical peak must prevail,
                    # even after removing this one actor. Ambiguous history uses the exact reader.
                    if max(positive, negative) <= peaks[-prevailing]:
                        support = count_meaning_evidence(
                            [r for r in evidence if r["actor_id"] != actor])
                        prevailing = support.prevailing_polarity
                    contributions[actor] = int(row["polarity"] == prevailing)
            current[key] = (fingerprint, contributions)
            for actor, correct in contributions.items():
                history.setdefault(actor, {})[key] = correct
        histories[guild_id] = current
        self._database._source_history_keys = histories
        cache[guild_id] = (revision, history)
        self._database._source_reliability = cache
        return history

    @staticmethod
    def _reliability(contributions, exclude_key=None):
        correct = sum(contributions.values())
        total = len(contributions)
        if exclude_key in contributions:
            correct -= contributions[exclude_key]
            total -= 1
        return min(1.0,max(0.25,(correct+2.0)/(total+2.0)))

    async def source_reliability(self, guild_id, actor_id, *, exclude_key=None):
        history = await self._source_history(guild_id)
        return self._reliability(history.get(actor_id, {}), exclude_key)

    async def review_response_hypothesis(self, hypothesis_id, event, *, promote):
        """Known requires human evidence, never a proposal's weight or receipt."""
        row = await self._database.fetchone(
            "SELECT * FROM continuous_cognition_hypotheses WHERE hypothesis_id=?",
            (hypothesis_id,),
        )
        if (
            row is None
            or not row["signature"].startswith("response:")
            or event.payload.get("author_is_bot")
            or event.guild_id != row["guild_id"]
        ):
            return False
        key = "confirmed_hypothesis:" + hypothesis_id
        if not promote:
            async with self._database.transaction() as connection:
                cursor = await connection.execute(
                    "SELECT DISTINCT guild_id,meaning_key FROM meaning_evidence WHERE source_key=?",
                    (key,))
                affected = await cursor.fetchall()
                await connection.execute(
                    "UPDATE meaning_claims SET resolved=0 WHERE claim_id IN "
                    "(SELECT claim_id FROM meaning_evidence WHERE source_key=?)", (key,))
                for guild, semantic_key in affected:
                    await self._materialize_support(connection, guild, semantic_key)
            return False
        evidence = await self._database.fetchall(
            "SELECT x.*,e.actor_id,e.channel_id,e.payload_json,e.source "
            "FROM continuous_cognition_evidence x "
            "JOIN events e ON e.event_id=x.source_event_id WHERE x.hypothesis_id=?",
            (hypothesis_id,),
        )
        if any(x["relation"] == "contradicts" for x in evidence):
            return False
        data = json.loads(row["roles_json"])
        descriptor = data["descriptor"]
        proposed = json.loads(descriptor["proposition"])
        candidates = [
            x
            for x in evidence
            if x["relation"] == "supports"
            and not json.loads(x["payload_json"]).get("author_is_bot")
            and "corpus" not in x["source"]
        ]
        from debilgi.language.lexicon import LexiconService

        canon = await LexiconService.for_meaning(self._database).canonical_lemmas(event.guild_id)
        proposed_thought = thought_from_data(proposed)
        human, explicit = [], False
        for observation in candidates:
            detail = json.loads(observation["details_json"])
            payload = json.loads(observation["payload_json"])
            confirmed = bool(
                detail.get("explicit_confirmation")
                and descriptor.get("confirmation") == "polar"
                and row["subject_key"] == f"discord:user:{observation['actor_id']}"
                and observation["channel_id"] == data["channel_id"]
                and payload.get("reply_to_message_id") == data["receipt_id"]
            )
            canonical = await self._database.fetchone(
                "SELECT thought_json FROM meaning_claims c WHERE claim_id=? "
                "AND (source_event_id=? OR EXISTS (SELECT 1 FROM meaning_evidence x "
                "WHERE x.claim_id=c.claim_id AND x.source_event_id=? AND x.actor_id IS NOT NULL)) "
                "AND subject_ref=? AND resolved=1 AND polarity=? "
                "AND NOT EXISTS (SELECT 1 FROM json_each(modality_json) "
                "WHERE json_extract(value,'$.operator')!='negation') "
                "AND source_kind IN ('self_report','other_report')",
                (
                    detail.get("canonical_claim"),
                    observation["source_event_id"],
                    observation["source_event_id"],
                    row["subject_key"],
                    -1 if proposed_thought.negated else 1,
                ),
            )
            independent = bool(
                canonical
                and proposition_key(thought_from_data(json.loads(canonical[0])), canonical=canon)
                == proposition_key(proposed_thought, canonical=canon)
                and bound_participants(thought_from_data(json.loads(canonical[0])))
                == bound_participants(proposed_thought)
                and all(
                    (proposed.get("semantic_time") or {}).get(k)
                    == (json.loads(canonical[0]).get("semantic_time") or {}).get(k)
                    for k in ("tense", "start", "end")
                )
            )
            if confirmed or independent:
                human.append(
                    {**dict(observation), "polarity": 1, "explicit_confirmation": confirmed}
                )
                explicit |= confirmed
        support = count_meaning_evidence(human)
        if not support.hypothesis_confirmed:
            return False
        thought = thought_from_data(proposed)
        sources = tuple(dict.fromkeys(x["source_event_id"] for x in human))
        thought = replace(
            thought,
            clause_kind="statement",
            confidence=0.95,
            evidence=sources,
            source_event_id=event.event_id,
        )
        subject = next(d.argument for d in thought.dependencies if d.role == "subject")
        if argument_ref(subject) != row["subject_key"]:
            return False
        owner = (
            int(row["subject_key"].rsplit(":", 1)[1])
            if row["subject_key"].startswith("discord:user:")
            else event.actor_id
        )
        semantic_key = meaning_key(thought, canon)
        claim_id = hashlib.sha256((key + event.event_id).encode()).hexdigest()
        time = thought.semantic_time
        async with self._database.transaction() as connection:
            candidates = await (
                await connection.execute(
                    "SELECT DISTINCT c.claim_id,c.thought_json FROM meaning_claims c "
                    "JOIN meaning_evidence x "
                    "ON x.claim_id=c.claim_id WHERE c.guild_id=? AND c.channel_id=? "
                    "AND c.speaker_id=? AND c.subject_ref=? AND x.meaning_key=? AND c.polarity=? "
                    "AND c.resolved=1 AND c.source_kind IN ('self_report','other_report') "
                    "AND c.time_start IS ? "
                    "AND c.time_end IS ? "
                    "AND (c.valid_to IS NULL OR c.valid_to>?)",
                    (
                        event.guild_id,
                        event.channel_id,
                        owner,
                        row["subject_key"],
                        semantic_key,
                        -1 if thought.negated else 1,
                        time.start,
                        time.end,
                        event.utc_timestamp.isoformat(),
                    ),
                )
            ).fetchall()
            existing = next(
                (
                    c
                    for c in candidates
                    if bound_participants(thought_from_data(json.loads(c["thought_json"])))
                    == bound_participants(thought)
                ),
                None,
            )
            if existing:
                claim_id = existing[0]
            await connection.execute(
                "INSERT OR IGNORE INTO meaning_claims "
                "(claim_id,source_event_id,guild_id,channel_id,speaker_id,subject_ref,predicate,polarity,"
                "modality_json,source_kind,source_time,valid_from,valid_to,"
                "confidence,resolved,parser_version,thought_json,correction_target,"
                "is_correction) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    claim_id,
                    event.event_id,
                    event.guild_id,
                    event.channel_id,
                    owner,
                    row["subject_key"],
                    thought.predicate.lemma,
                    -1 if thought.negated else 1,
                    json.dumps([asdict(s) for s in thought.scopes]),
                    "self_report" if explicit else "other_report",
                    event.utc_timestamp.isoformat(),
                    time.start,
                    time.end,
                    0.95,
                    1,
                    thought.parser_version,
                    json.dumps(identity_graph(thought), ensure_ascii=False),
                    None,
                    0,
                ),
            )
            await connection.executemany(
                "INSERT OR IGNORE INTO meaning_arguments VALUES (?,?,?,?,?)",
                [
                    (
                        claim_id,
                        i,
                        d.role,
                        argument_ref(d.argument),
                        json.dumps(asdict(d.argument), ensure_ascii=False),
                    )
                    for i, d in enumerate(thought.dependencies)
                ],
            )
            await connection.executemany(
                "INSERT OR IGNORE INTO meaning_evidence "
                "(claim_id,source_event_id,source_key,meaning_key,guild_id,actor_id,observed_at,"
                "polarity) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        claim_id,
                        h["source_event_id"],
                        key,
                        semantic_key,
                        event.guild_id,
                        h["actor_id"],
                        h["observed_at"],
                        -1 if thought.negated else 1,
                    )
                    for h in human
                ],
            )
            await self._materialize_support(connection, event.guild_id, semantic_key)
        return True

    async def restore_from_events(self, limit: int = 2000) -> None:
        """Reuse bounded historical human evidence when upgrading an existing DB."""
        await self.reindex_evidence()
        self._database._meaning_evidence_ready = True
        await self.backfill_claims()
        events = await EventRepository(self._database).since(
            datetime.min.replace(tzinfo=UTC),
            limit,
        )
        for event in events:
            await self.observe(event)

    async def observe(self, event: Event) -> None:
        if (
            event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
            or event.payload.get("author_is_bot")
            or event.channel_id is None
            or event.actor_id is None
        ):
            return
        text = event.payload.get("content")
        if not isinstance(text, str):
            return
        bot_id = event.payload.get("bot_user_id")
        if isinstance(bot_id, int):
            text = text.replace(f"<@{bot_id}>", " ").replace(f"<@!{bot_id}>", " ")
        await self._observe_relations(event, text)
        await self._observe_dialogue(event)
        await self._observe_meanings(event, text)
        await self._project_claims(event)
        if source_kind(event) != "corpus_form":
            from debilgi.language.lexicon import LexiconService

            await LexiconService.for_meaning(self._database).observe_synonym_text(event, text)
        from debilgi.cognition.continuous import ContinuousCognitionService

        await ContinuousCognitionService(self._database).observe_response_hypotheses(event)
        retained = await self._database.fetchone(
            "SELECT 1 FROM language_proposition_reports WHERE source_event_id = ? LIMIT 1",
            (event.event_id,),
        )
        if retained:
            return
        reports = []
        for sentence in proposition_sentences(text):
            thought = statement_thought(
                event.meaning if sentence.casefold() == event.meaning.text else sentence,
                speaker="user",
            )
            if thought is None:
                continue
            terms = content_lemmas(sentence)
            heads = {dep.argument.head.lemma for dep in thought.dependencies}
            # Unresolved references and relative dates need their original
            # context. Retaining them as standalone evidence changes meaning.
            if heads & {"он", "она", "оно", "они", "это"} or set(terms) & {
                "вчера",
                "позавчера",
                "сегодня",
                "завтра",
                "послезавтра",
                "сейчас",
                "недавно",
                "тут",
                "здесь",
                "там",
                "туда",
                "сюда",
            }:
                continue
            report_id = hashlib.sha256(f"{event.event_id}|{sentence}".encode()).hexdigest()
            reports.append((report_id, sentence))
        if not reports:
            return
        async with self._database.transaction() as connection:
            for report_id, sentence in reports:
                await connection.execute(
                    """INSERT OR IGNORE INTO language_proposition_reports
                       (report_id, source_event_id, guild_id, channel_id, speaker_id,
                        text, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report_id,
                        event.event_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        sentence,
                        event.utc_timestamp.isoformat(),
                    ),
                )

    async def _observe_meanings(self, event: Event, text: str) -> None:
        records = extract_meanings(text, event.event_id, structure=event.meaning)
        if not records:
            return
        async with self._database.transaction() as connection:
            await connection.executemany(
                "INSERT OR IGNORE INTO language_meaning_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        r.meaning_id,
                        event.event_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        r.kind,
                        json.dumps(asdict(r), ensure_ascii=False),
                        event.utc_timestamp.isoformat(),
                    )
                    for r in records
                ),
            )

    async def backfill_claims(self, batch_size: int = 256) -> None:
        """Drain versioned source receipts in bounded pages, including redacted legacy DBs.

        Source tables stay intact. A receipt also records unparsed/empty input,
        so restarting never repeatedly reparses an unsupported source.
        """
        while True:
            rows = await self._database.fetchall(
                """SELECT e.* FROM events e
                   LEFT JOIN meaning_claim_projections p ON p.source_event_id = e.event_id
                   WHERE p.source_event_id IS NULL AND e.channel_id IS NOT NULL
                     AND e.actor_id IS NOT NULL AND e.event_type IN (?, ?)
                     AND COALESCE(json_extract(e.payload_json, '$.author_is_bot'), 0) = 0
                   ORDER BY e.occurred_at, e.event_id LIMIT ?""",
                (
                    EventType.MESSAGE_CREATED.value,
                    EventType.MESSAGE_REPLIED.value,
                    max(1, min(batch_size, 2000)),
                ),
            )
            if not rows:
                return
            for row in rows:
                event = EventRepository._from_row(row)
                if event.payload.get("content") and source_kind(event) != "corpus_form":
                    await self._observe_dialogue(event)
                if not event.payload.get("content"):
                    retained = await self._database.fetchall(
                        """SELECT text FROM language_proposition_reports WHERE source_event_id = ?
                           UNION ALL SELECT json_extract(record_json, '$.text') AS text
                           FROM language_meaning_records WHERE source_event_id = ?""",
                        (event.event_id, event.event_id),
                    )
                    if retained:
                        event = replace(
                            event,
                            payload={
                                **event.payload,
                                "content": " ".join(
                                    dict.fromkeys(row["text"] for row in retained if row["text"])
                                ),
                            },
                        )
                await self._project_claims(event)

    async def _project_claims(self, event: Event) -> None:
        if await self._database.fetchone(
            "SELECT 1 FROM meaning_claim_projections WHERE source_event_id = ?",
            (event.event_id,),
        ):
            return
        thoughts = []
        structure = event.meaning
        thought = fact_thought(structure)
        if thought:
            thoughts.append(thought)
        elif not structure.corrections:
            # Preserve the conservative sentence adapter for multi-sentence inputs.
            for sentence in proposition_sentences(structure.text):
                if sentence == structure.text:
                    continue
                part = replace(event, payload={**event.payload, "content": sentence})
                if candidate := fact_thought(part.meaning):
                    thoughts.append(candidate)
        dialogue = await self._database.fetchone(
            "SELECT * FROM language_dialogue_reports WHERE answer_event_id = ?", (event.event_id,)
        )
        if dialogue:
            thoughts.append(thought_from_data(json.loads(dialogue["thought_json"])))
        legacy = await self._database.fetchall(
            "SELECT record_json FROM language_meaning_records WHERE source_event_id = ? "
            "AND kind = 'fact'",
            (event.event_id,),
        )
        if not thoughts:
            records = [meaning_from_data(json.loads(row["record_json"])) for row in legacy]
            if thought_content_enabled():
                # A question without «?» («че делаешь») was once stored as a legacy fact; it
                # must not come back as a report about the one asked (stage 4.32).
                records = [r for r in records if not _asks_only(r.text)]
            thoughts.extend(t for record in records for t in record.conclusions)
        from debilgi.language.lexicon import LexiconService

        canon = await LexiconService.for_meaning(self._database).canonical_lemmas(event.guild_id)
        unique = {}
        for thought in thoughts:
            # Legacy thoughts already use the reader's grammatical perspective.
            # Bind only explicit participant pronouns; never map an unknown 'ты'
            # to the bot merely because bot_user_id exists.
            refs = dict(structure.participants)
            dependencies = []
            for dep in thought.dependencies:
                head = dep.argument.head
                ref = refs.get({"ты": "speaker", "я": "addressee"}.get(head.lemma, ""))
                if not head.participant_ref and ref:
                    head = replace(head, participant_ref=ref.ref)
                elif not head.participant_ref:
                    mentions = {
                        t.participant.ref
                        for t in structure.tokens
                        if t.lemma == head.lemma and t.participant
                    }
                    if len(mentions) == 1:
                        head = replace(head, participant_ref=next(iter(mentions)))
                if source_kind(event) == "corpus_form" and head.participant_ref:
                    head = replace(
                        head, participant_ref=(f"corpus:{event.event_id}:{head.participant_ref}")
                    )
                dependencies.append(replace(dep, argument=replace(dep.argument, head=head)))
            time = thought.semantic_time
            deictics = {
                d.argument.head.lemma
                for d in thought.dependencies
                if d.argument.head.lemma in {*RELATIVE_DAYS, "сейчас"}
            }
            if time.deictic is None and len(deictics) == 1:
                time = replace(time, deictic=next(iter(deictics)))
            time = semantic_time(time, event, self._timezone)
            if time.tense is None:
                time = replace(time, tense=dict(thought.predicate.features).get("tense"))
            thought = replace(
                thought,
                dependencies=tuple(dependencies),
                semantic_time=time,
                source_event_id=event.event_id,
            )
            key = meaning_key(thought, canon)
            unique[key, thought.negated] = thought
        async with self._database.transaction() as connection:
            for (key, negated), thought in sorted(unique.items()):
                # Preserve existing opaque IDs for ordinary source events;
                # opposite assertions inside one event still need separate rows.
                suffix = "|negative" if negated and (key, False) in unique else ""
                claim_id = hashlib.sha256(
                    f"claim:1|{event.event_id}|{key}{suffix}".encode()
                ).hexdigest()
                subject = next(
                    (d.argument for d in thought.dependencies if d.role == "subject"), None
                )
                if subject is None:
                    continue
                subject_ref = argument_ref(subject)
                kind = source_kind(event)
                # World claims share an identity inside a server. Personal facts
                # keep their owner/channel, even if the subject is a concept.
                world = world_thought(thought)
                candidates = [] if kind == "corpus_form" else await (
                    await connection.execute(
                        "SELECT DISTINCT c.claim_id,c.thought_json FROM meaning_claims c "
                        "JOIN meaning_evidence x "
                        "ON c.claim_id=x.claim_id WHERE c.guild_id=? AND x.meaning_key=? "
                        "AND c.polarity=? AND c.source_kind=? AND c.scope=? AND c.modality_json=? "
                        "AND NOT EXISTS (SELECT 1 FROM meaning_supersedes s "
                        "WHERE s.previous_claim_id=c.claim_id) "
                        "AND (? OR (c.speaker_id=? AND c.channel_id=? AND c.subject_ref=?)) "
                        "AND c.time_start IS ? "
                        "AND c.time_end IS ?",
                        (
                            event.guild_id,
                            key,
                            -1 if thought.negated else 1,
                            "other_report" if world and kind != "corpus_form" else kind,
                            "world" if world else "private",
                            json.dumps([asdict(s) for s in thought.scopes], ensure_ascii=False),
                            int(world and kind != "corpus_form"),
                            event.actor_id,
                            event.channel_id,
                            subject_ref,
                            thought.semantic_time.start,
                            thought.semantic_time.end,
                        ),
                    )
                ).fetchall()
                existing = next((c for c in candidates
                    if (candidate := thought_from_data(json.loads(c["thought_json"])))
                    and world_thought(candidate) == world
                    and bound_participants(candidate) == bound_participants(thought)), None)
                if existing and kind != "corpus_form":
                    claim_id = existing[0]
                if kind != "corpus_form" and subject_ref != f"discord:user:{event.actor_id}":
                    kind = "other_report"
                unresolved = {
                    t.participant.ref
                    for t in structure.tokens
                    if t.participant
                    and not t.participant.resolved
                    and ":named:" not in t.participant.ref
                }
                unresolved.update(r.ref for _, r in structure.participants if not r.resolved)
                resolved = not any(
                    d.argument.head.participant_ref in unresolved
                    or d.argument.head.lemma
                    in {
                        "он",
                        "она",
                        "оно",
                        "они",
                        "это",
                        "недавно",
                        "тут",
                        "там",
                        "здесь",
                        "туда",
                        "сюда",
                    }
                    for d in thought.dependencies
                )
                start, end = thought.semantic_time.start, thought.semantic_time.end
                if not start and thought.semantic_time.tense == "present":
                    start = event.utc_timestamp.isoformat()
                    end = (event.utc_timestamp + timedelta(hours=1)).isoformat()
                await connection.execute(
                    """INSERT OR IGNORE INTO meaning_claims 
                        (claim_id,source_event_id,guild_id,channel_id,speaker_id,subject_ref,predicate,polarity,
                        modality_json,source_kind,source_time,valid_from,valid_to,
                        confidence,resolved,parser_version,thought_json,correction_target,
                    is_correction) VALUES
                       (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        claim_id,
                        event.event_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        subject_ref,
                        thought.predicate.lemma,
                        -1 if thought.negated else 1,
                        json.dumps([asdict(s) for s in thought.scopes], ensure_ascii=False),
                        kind,
                        event.utc_timestamp.isoformat(),
                        start,
                        end,
                        thought.confidence,
                        int(resolved),
                        thought.parser_version,
                        json.dumps(identity_graph(thought), ensure_ascii=False),
                        structure.corrections[0].target_event_id if structure.corrections else None,
                        int(bool(structure.corrections)),
                    ),
                )
                await connection.executemany(
                    "INSERT OR IGNORE INTO meaning_arguments VALUES (?, ?, ?, ?, ?)",
                    (
                        (
                            claim_id,
                            i,
                            d.role,
                            argument_ref(d.argument),
                            json.dumps(asdict(d.argument), ensure_ascii=False),
                        )
                        for i, d in enumerate(thought.dependencies)
                    ),
                )
                sources = tuple(dict.fromkeys((event.event_id, *thought.evidence)))
                if dialogue:
                    sources = tuple(dict.fromkeys((*sources, dialogue["question_event_id"])))
                await connection.executemany(
                    "INSERT OR IGNORE INTO meaning_evidence "
                    "(claim_id,source_event_id,source_key,meaning_key,guild_id,actor_id,"
                    "observed_at,polarity) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        (
                            claim_id,
                            source,
                            "event",
                            key,
                            event.guild_id,
                            event.actor_id if source == event.event_id else None,
                            event.utc_timestamp.isoformat() if source == event.event_id else None,
                            -1 if thought.negated else 1,
                        )
                        for source in sources
                    ),
                )
                await self._link_updates(connection, event, subject_ref, thought.predicate.lemma)
            cursor = await connection.execute(
                "SELECT DISTINCT meaning_key FROM meaning_evidence "
                "WHERE source_event_id=? AND voter=1",
                (event.event_id,))
            for row in await cursor.fetchall():
                await self._materialize_support(connection, event.guild_id, row[0])
            await connection.execute(
                "INSERT OR IGNORE INTO meaning_claim_projections VALUES (?, ?)",
                (event.event_id, SCHEMA_VERSION),
            )

    async def _link_updates(self, connection, event: Event, subject: str, predicate: str) -> None:
        """Reconcile a topic chronologically, even when replay arrives out of order."""
        cursor = await connection.execute(
            """SELECT * FROM meaning_claims WHERE guild_id = ? AND channel_id = ?
               AND speaker_id = ? AND subject_ref = ? AND predicate = ?
               ORDER BY source_time, source_event_id, claim_id""",
            (event.guild_id, event.channel_id, event.actor_id, subject, predicate),
        )
        rows = list(await cursor.fetchall())
        for index, current in enumerate(rows):
            await connection.execute(
                "DELETE FROM meaning_supersedes WHERE claim_id = ?", (current["claim_id"],)
            )
            thought = thought_from_data(json.loads(current["thought_json"]))
            candidates = []
            for previous in rows[:index]:
                if previous["source_event_id"] == current["source_event_id"]:
                    continue
                if current["subject_ref"].startswith("concept:"):
                    cursor = await connection.execute(
                        "SELECT COUNT(DISTINCT actor_id) FROM meaning_evidence WHERE claim_id=?",
                        (previous["claim_id"],),
                    )
                    if (await cursor.fetchone())[0] >= 2:
                        continue
                older = thought_from_data(json.loads(previous["thought_json"]))
                if thought.semantic_time.tense != older.semantic_time.tense or tuple(
                    s.operator for s in thought.scopes if s.operator != "negation"
                ) != tuple(s.operator for s in older.scopes if s.operator != "negation"):
                    continue
                if (
                    current["valid_from"]
                    and previous["valid_to"]
                    and current["valid_from"] >= previous["valid_to"]
                    or previous["valid_from"]
                    and current["valid_to"]
                    and previous["valid_from"] >= current["valid_to"]
                ):
                    continue
                if current["is_correction"] or (
                    proposition_key(thought, signed=False) == proposition_key(older, signed=False)
                    and thought.negated != older.negated
                ):
                    candidates.append(previous)
            target = current["correction_target"]
            explicit = [p for p in candidates if p["source_event_id"] == target]
            if target and not explicit:
                # Replying to an actual human fact must not replace a different one.
                cursor = await connection.execute(
                    "SELECT payload_json FROM events WHERE event_id = ?", (target,)
                )
                target_event = await cursor.fetchone()
                if target_event and not json.loads(target_event["payload_json"]).get(
                    "author_is_bot"
                ):
                    candidates = []
            selected = explicit or candidates[-1:]
            await connection.executemany(
                "INSERT OR IGNORE INTO meaning_supersedes VALUES (?, ?)",
                ((current["claim_id"], p["claim_id"]) for p in selected),
            )

    async def record_agent_activity(
        self,
        event: Event,
        *,
        source_key: str,
        operation: str,
        thought,
        occurred_at: datetime,
        status: str = "completed",
        receipt_id: int | None = None,
    ) -> None:
        """Project an operational source, never a generated assertion about itself.

        Callers supply a receipt/completion, not prepared text or a desired goal.
        The raw event anchors provenance; source_key identifies the actual operation.
        """
        bot_id = event.payload.get("bot_user_id")
        if not isinstance(bot_id, int) or source_kind(event) == "corpus_form":
            return
        agent = f"discord:agent:{bot_id}"
        activity_id = hashlib.sha256(f"activity:1|{agent}|{source_key}".encode()).hexdigest()
        claim_id = f"activity:{activity_id}:{status}"
        start = occurred_at.astimezone(UTC)
        end = start + (timedelta(minutes=1) if status == "active" else timedelta(microseconds=1))
        thought = replace(
            thought,
            dependencies=tuple(
                replace(
                    d,
                    argument=replace(
                        d.argument,
                        head=replace(
                            d.argument.head,
                            participant_ref=agent,
                        ),
                    ),
                )
                if d.role == "subject"
                else d
                for d in thought.dependencies
            ),
            source_event_id=event.event_id,
            predicate=replace(
                thought.predicate,
                features=(
                    (("tense", "present"), ("person", "first"), ("number", "singular"))
                    if status == "active"
                    else (("tense", "past"), ("gender", "masculine"))
                ),
            ),
            semantic_time=SemanticTime(tense="present" if status == "active" else "past"),
            evidence=(source_key,),
        )
        # Identity is persistent configuration, not a completed autobiographical action.
        if operation == "identity":
            end = None
            thought = replace(
                thought,
                semantic_time=SemanticTime(tense="present"),
                predicate=replace(thought.predicate, features=(("tense", "present"),)),
            )
        stamp = start.isoformat()
        until = end.isoformat() if end else None
        from debilgi.language.lexicon import LexiconService

        canon = await LexiconService.for_meaning(self._database).canonical_lemmas(event.guild_id)
        semantic_key = meaning_key(thought, canon)
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                "SELECT status FROM agent_activities WHERE activity_id = ?",
                (activity_id,),
            )
            prior = await cursor.fetchone()
            if prior and prior["status"] == "completed":
                return
            # A status transition replaces the active claim, retaining its proof/history.
            await connection.execute(
                """UPDATE meaning_claims SET valid_to = ? WHERE claim_id IN
                   (SELECT claim_id FROM agent_activities WHERE activity_id = ?)
                   AND ? = 'completed' AND valid_to > ?""",
                (stamp, activity_id, status, stamp),
            )
            await connection.execute(
                """INSERT OR IGNORE INTO meaning_claims 
                    (claim_id,source_event_id,guild_id,channel_id,speaker_id,subject_ref,predicate,polarity,
                    modality_json,source_kind,source_time,valid_from,valid_to,
                    confidence,resolved,parser_version,thought_json,correction_target,
                    is_correction) VALUES
                   (?, ?, ?, ?, ?, ?, ?, 1, '[]', 'agent_observed', ?, ?, ?, 1, 1,
                    'activity:1', ?, NULL, 0)""",
                (
                    claim_id,
                    event.event_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    agent,
                    thought.predicate.lemma,
                    stamp,
                    stamp,
                    until,
                    json.dumps(identity_graph(thought), ensure_ascii=False),
                ),
            )
            if status == "active":
                await connection.execute(
                    """UPDATE meaning_claims SET source_time = ?, valid_from = ?, valid_to = ?
                       WHERE claim_id = ? AND source_time < ?""",
                    (stamp, stamp, until, claim_id, stamp),
                )
            await connection.executemany(
                "INSERT OR IGNORE INTO meaning_arguments VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        claim_id,
                        i,
                        d.role,
                        argument_ref(d.argument),
                        json.dumps(asdict(d.argument), ensure_ascii=False),
                    )
                    for i, d in enumerate(thought.dependencies)
                ),
            )
            await connection.execute(
                "INSERT OR IGNORE INTO meaning_evidence "
                "(claim_id,source_event_id,source_key,meaning_key,guild_id,actor_id,observed_at,"
                "polarity) VALUES (?,?,?,?,?,NULL,?,1)",
                (claim_id, event.event_id, source_key, semantic_key, event.guild_id, stamp),
            )
            await connection.execute(
                """INSERT INTO agent_activities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(activity_id) DO UPDATE SET status = excluded.status,
                   completed_at = excluded.completed_at, claim_id = excluded.claim_id
                   WHERE agent_activities.status != 'completed'""",
                (
                    activity_id,
                    agent,
                    event.event_id,
                    source_key,
                    operation,
                    status,
                    stamp,
                    stamp if status == "completed" else None,
                    receipt_id,
                    claim_id,
                ),
            )

    async def _agent_source_rows(self, event: Event):
        """Keyset batches cover older sources too, without traversing derived claims."""
        cursor = (event.utc_timestamp.isoformat(), "\uffff")
        while True:
            rows = await self._database.fetchall(
                """SELECT e.* FROM events e WHERE e.guild_id = ? AND e.channel_id = ?
                   AND (e.occurred_at, e.event_id) < (?, ?) AND e.source NOT LIKE '%corpus%'
                   AND e.source NOT LIKE 'seed:%' AND e.actor_id != ?
                   ORDER BY e.occurred_at DESC, e.event_id DESC LIMIT 512""",
                (event.guild_id, event.channel_id, *cursor, event.payload["bot_user_id"]),
            )
            if not rows:
                return
            for row in rows:
                yield row
            cursor = (rows[-1]["occurred_at"], rows[-1]["event_id"])

    async def project_agent_sources(self, event: Event) -> None:
        """Bounded reconciliation of existing operational records, safe after restart.

        Corpus input still teaches language but supplies no own experience. Never
        traverse claims as projection sources: derived self reports cannot recurse.
        """
        if (
            not isinstance(event.payload.get("bot_user_id"), int)
            or source_kind(event) == "corpus_form"
        ):
            return
        async for row in self._agent_source_rows(event):
            source = EventRepository._from_row(row)
            if source.payload.get("author_is_bot"):
                continue
            # Historical sources without an explicit runtime identity stay unknown.
            if source.payload.get("bot_user_id") != event.payload["bot_user_id"]:
                continue
            sources = []
            processed = await self._database.fetchone(
                "SELECT * FROM event_processing WHERE event_id = ? AND status = 'processed'",
                (source.event_id,),
            )
            if processed and processed["completed_at"]:
                sources.append(
                    (
                        f"event_processing:{source.event_id}",
                        "process",
                        "обработать",
                        "сообщение",
                        processed["completed_at"],
                        "completed",
                        None,
                    )
                )
            work = await self._database.fetchone(
                "SELECT * FROM continuous_cognition_work_items WHERE source_event_id = ?",
                (source.event_id,),
            )
            if work and work["status"] in {"queued", "failed"}:
                # Retries and failures end a previously observed running interval.
                async with self._database.transaction() as connection:
                    await connection.execute(
                        """UPDATE meaning_claims SET valid_to = MIN(valid_to, ?)
                           WHERE claim_id IN (SELECT claim_id FROM agent_activities
                           WHERE source_key = ? AND status = 'active')""",
                        (work["updated_at"], f"continuous_cognition_work_items:{work['work_id']}"),
                    )
                    await connection.execute(
                        """UPDATE agent_activities SET status = ? WHERE source_key = ?
                           AND status = 'active'""",
                        (work["status"], f"continuous_cognition_work_items:{work['work_id']}"),
                    )
            if work and work["status"] in {"processing", "completed"}:
                active = work["status"] == "processing"
                stamp = work["updated_at"] if active else work["completed_at"]
                if stamp:
                    sources.append(
                        (
                            f"continuous_cognition_work_items:{work['work_id']}",
                            "work",
                            "обрабатывать" if active else "обработать",
                            "сообщение",
                            stamp,
                            "active" if active else "completed",
                            None,
                        )
                    )
            deliveries = await self._database.fetchall(
                """SELECT response_message_id, sent_at FROM reply_delivery_jobs
                   WHERE event_id = ? AND status IN ('sent', 'completed')
                     AND response_message_id IS NOT NULL AND sent_at IS NOT NULL
                   UNION SELECT response_message_id, sent_at FROM language_meaning_intentions
                   WHERE prompt_event_id = ? AND status IN ('sent', 'observed')
                     AND response_message_id IS NOT NULL AND sent_at IS NOT NULL""",
                (source.event_id, source.event_id),
            )
            for delivery in deliveries:
                sources.append(
                    (
                        f"receipt:{source.channel_id}:{delivery['response_message_id']}",
                        "send",
                        "отправить",
                        "сообщение",
                        delivery["sent_at"],
                        "completed",
                        delivery["response_message_id"],
                    )
                )
            queries = await self._database.fetchall(
                """SELECT q.*, k.title FROM knowledge_queries q LEFT JOIN knowledge_concepts k
                   USING (concept_id) WHERE q.event_id = ? AND q.completed_at IS NOT NULL
                   AND q.status IN ('resolved', 'cached', 'no_result')""",
                (source.event_id,),
            )
            for q in queries:
                sources.append(
                    (
                        f"knowledge_queries:{q['query_id']}:search",
                        "search",
                        "искать",
                        q["query_text"],
                        q["completed_at"],
                        "completed",
                        None,
                    )
                )
                if q["title"] and q["status"] in {"resolved", "cached"}:
                    # An extract proves receiving a source, never reading a whole book.
                    sources.append(
                        (
                            f"knowledge_queries:{q['query_id']}:receive",
                            "receive",
                            "получить",
                            q["title"],
                            q["completed_at"],
                            "completed",
                            None,
                        )
                    )
            lessons = await self._database.fetchall(
                """SELECT l.lesson_id, v.input_hash, v.observed_at FROM
                   language_semantic_lesson_evidence v JOIN language_semantic_lessons l
                   USING (lesson_id) JOIN language_semantic_consultations c
                   ON c.input_hash = v.input_hash AND c.guild_id = l.guild_id
                   WHERE c.event_id = ? AND l.status = 'active' AND c.status = 'confirmed'
                   UNION SELECT l.lesson_id, v.input_hash, v.observed_at FROM
                   language_semantic_lesson_evidence v JOIN language_semantic_lessons l
                   USING (lesson_id) WHERE v.source_event_id = ? AND l.status = 'active'""",
                (source.event_id, source.event_id),
            )
            for lesson in lessons:
                sources.append(
                    (
                        f"lesson:{lesson['lesson_id']}:{lesson['input_hash']}",
                        "learn",
                        "изучить",
                        "конструкция",
                        lesson["observed_at"],
                        "completed",
                        None,
                    )
                )
            for key, operation, predicate, target, stamp, status, receipt in sources:
                when = datetime.fromisoformat(stamp)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
                if when > event.utc_timestamp:
                    continue
                await self.record_agent_activity(
                    source,
                    source_key=key,
                    operation=operation,
                    thought=activity_thought(predicate, target),
                    occurred_at=when,
                    status=status,
                    receipt_id=receipt,
                )
        from debilgi.language.entities import identity_thought

        identity = identity_thought("как тебя зовут?", event.payload)
        if identity and event.payload["bot_identity"].get("id") == event.payload["bot_user_id"]:
            record = event.payload["bot_identity"]
            name = identity.dependencies[-1].argument.head.lemma
            agent = f"discord:agent:{record['id']}"
            stamp = event.utc_timestamp.isoformat()
            prior = await self._database.fetchone(
                """SELECT thought_json FROM meaning_claims WHERE subject_ref = ?
                   AND predicate = 'называться' AND source_kind = 'agent_observed'
                   AND guild_id = ? AND channel_id = ?
                   AND source_time <= ? ORDER BY source_time DESC LIMIT 1""",
                (agent, event.guild_id, event.channel_id, stamp),
            )
            if (
                prior
                and thought_from_data(json.loads(prior["thought_json"]))
                .dependencies[-1]
                .argument.head.lemma
                == name
            ):
                return
            await self.record_agent_activity(
                event,
                source_key=f"gateway_identity:{record['id']}:{event.event_id}",
                operation="identity",
                thought=identity,
                occurred_at=event.utc_timestamp,
            )
            await self._database.execute(
                """UPDATE meaning_claims SET valid_to = ? WHERE subject_ref = ?
                   AND predicate = 'называться' AND source_kind = 'agent_observed'
                   AND guild_id = ? AND channel_id = ?
                   AND source_time < ? AND (valid_to IS NULL OR valid_to > ?)""",
                (stamp, agent, event.guild_id, event.channel_id, stamp, stamp),
            )
            await self._database.execute(
                """UPDATE meaning_claims SET valid_to = (SELECT MIN(source_time)
                   FROM meaning_claims n WHERE n.subject_ref = ? AND n.predicate = 'называться'
                   AND n.guild_id = ? AND n.channel_id = ?
                   AND n.source_kind = 'agent_observed' AND n.source_time > ?)
                   WHERE source_event_id = ? AND predicate = 'называться'
                   AND source_kind = 'agent_observed'""",
                (agent, event.guild_id, event.channel_id, stamp, event.event_id),
            )

    async def retrieve_claims(
        self,
        event: Event,
        text: str | None = None,
        *,
        hypotheses=(),
        reconcile=True,
    ) -> Known | Unknown:
        query = query_from_event(event, text, timezone=self._timezone, hypotheses=hypotheses)
        if query.requested_attribute and query.predicate is None:
            # Resolve one elliptical slot against a receipt-backed bot turn;
            # never turn that question into a request for an arbitrary self fact.
            previous = next(
                (
                    e
                    for e in reversed(await self.recent_conversation(event))
                    if e.actor_id == event.payload.get("bot_user_id")
                    and e.target_id == event.actor_id
                ),
                None,
            )
            if previous is not None:
                thought = statement_thought(previous.meaning)
                if thought is not None:
                    query = replace(
                        query,
                        predicate=thought.predicate.lemma,
                        arguments=(("subject", f"discord:agent:{previous.actor_id}"),),
                    )
        if reconcile and self_query(query):
            await self.project_agent_sources(event)
        result = await self.query_claims(query, reconcile=reconcile)
        if not self_query(query):
            checked = await self._own_action_check(event, text, query, reconcile=reconcile)
            if checked is not None:
                return checked
        return result

    async def _own_action_check(
        self, event: Event, text: str | None, query, *, reconcile=True
    ) -> Known | None:
        """«Ты же X»: an addressed statement whose subject is the addressee is checked against
        the agent's own observed actions (the same records «ты X?» reads).

        Only a match is returned (the statement is then a Known fact about itself); no record,
        a negated statement or any other shape leaves the ordinary result untouched, because
        the journal of own actions is not a proof of absence.
        """
        structure = event.meaning
        if text is not None and text != structure.text:
            structure = replace(event, payload={**event.payload, "content": text}).meaning
        bot_id = event.payload.get("bot_user_id")
        addressed = bool(event.payload.get("bot_mentioned")) or (
            isinstance(bot_id, int)
            and bot_id in {event.target_id, event.payload.get("reply_to_author_id")}
        )
        if not addressed or not isinstance(bot_id, int) or len(structure.clauses) != 1:
            return None
        clause = structure.clauses[0]
        if (
            clause.clause_kind != "statement"
            or clause.negated
            or clause.predicate_index is None
            or structure.tokens[clause.predicate_index].part_of_speech != "verb"
        ):
            return None
        subjects = [structure.tokens[i] for i in clause.subject_indices]
        addressee_subject = (
            len(subjects) == 1
            and person_cell(subjects[0].lemma) is not None
            and person_cell(subjects[0].lemma)[:2] in {("second", "singular"), ("second", "plural")}
            or not subjects
            and clause.implicit_subject == "addressee"
        )
        if not addressee_subject or not query.predicate:
            return None
        own = replace(
            query,
            arguments=(
                *((r, v) for r, v in query.arguments if r != "subject"),
                ("subject", f"discord:agent:{bot_id}"),
            ),
        )
        if reconcile:
            await self.project_agent_sources(event)
        found = await self.query_claims(own, reconcile=reconcile)
        return found if isinstance(found, Known) else None

    async def query_claims(self, query: ClaimQuery, *, reconcile=True) -> Known | Unknown:
        """A scoped structural query. Empty results are never negative assertions."""
        if reconcile and not getattr(self._database, "_meaning_evidence_ready", False):
            await self.reindex_evidence()
            self._database._meaning_evidence_ready = True
        if (
            query.channel_id is None
            or query.speaker_id is None
            or not (query.predicate or query.arguments)
            # A bare participant is not a request for everything ever done by
            # that participant. An explicit predicate-parent action query is.
            or self_query(query)
            and not (
                query.predicate
                or query.predicate_parent
                or any(role != "subject" for role, _ in query.arguments)
            )
        ):
            return Unknown(query)
        clauses = [
            "c.guild_id = ?",
            "c.source_kind IN ('self_report','other_report','agent_observed')",
            "(c.channel_id = ? OR (c.subject_ref LIKE 'concept:%' AND NOT EXISTS "
            "(SELECT 1 FROM meaning_arguments a WHERE a.claim_id=c.claim_id "
            "AND a.ref NOT LIKE 'concept:%')))",
            "(c.speaker_id = ? OR (c.source_kind = 'agent_observed' AND ?) OR "
            "(c.subject_ref LIKE 'concept:%' AND NOT EXISTS (SELECT 1 FROM meaning_arguments a "
            "WHERE a.claim_id=c.claim_id AND a.ref NOT LIKE 'concept:%')))",
            "(c.source_time < ? OR (c.predicate = 'называться' AND c.source_time = ?))",
            "(c.source_event_id != ? OR c.predicate = 'называться')",
        ]
        params = [
            query.guild_id,
            query.channel_id,
            query.speaker_id,
            int(self_query(query)),
            query.snapshot_at,
            query.snapshot_at,
            query.exclude_event_id,
        ]
        if self_query(query):
            clauses.append("c.source_kind = 'agent_observed'")
        from debilgi.language.lexicon import LexiconService

        canon = await LexiconService.for_meaning(self._database).canonical_lemmas(query.guild_id)

        def alternatives(word):
            canonical = canon.get(word, word)
            return sorted({word, canonical, *(w for w, c in canon.items() if c == canonical)})

        if query.predicate:
            words = alternatives(query.predicate)
            if self_query(query):
                # Own recorded actions are named in whichever aspect the operation used;
                # the asked verb in the other aspect is the same action (morphology decides).
                from debilgi.language.lexical_matching import aspect_pair

                own = await self._database.fetchall(
                    "SELECT DISTINCT predicate FROM meaning_claims "
                    "WHERE guild_id = ? AND source_kind = 'agent_observed'",
                    (query.guild_id,),
                )
                words = sorted(
                    {
                        *words,
                        *(
                            r["predicate"]
                            for r in own
                            if aspect_pair(query.predicate, r["predicate"])
                        ),
                    }
                )
            clauses.append("c.predicate IN (" + ",".join("?" for _ in words) + ")")
            params.extend(words)
        if not query.include_unresolved:
            clauses.append("c.resolved = 1")
        if not query.include_history:
            clauses.append("""NOT EXISTS (SELECT 1 FROM meaning_supersedes s
                JOIN meaning_claims n ON n.claim_id = s.claim_id
                WHERE s.previous_claim_id = c.claim_id AND n.source_time < ?)""")
            params.append(query.snapshot_at)
        for role, ref in query.arguments:
            refs = [ref]
            if ref.startswith("concept:"):
                refs = ["concept:" + w for w in alternatives(ref[8:])]
            clauses.append(
                "EXISTS (SELECT 1 FROM meaning_arguments a WHERE a.claim_id=c.claim_id "
                "AND (?='*' OR a.role=?) AND a.ref IN (" + ",".join("?" for _ in refs) + "))"
            )
            params.extend((role, role, *refs))
        if query.requested_role:
            clauses.append(
                "EXISTS (SELECT 1 FROM meaning_arguments a "
                "WHERE a.claim_id = c.claim_id "
                "AND a.role = ?)"
            )
            params.append(query.requested_role)
        if query.valid_from and query.valid_to:
            # An unspecified interval cannot prove a dated query.
            clauses.extend(("c.valid_from < ?", "c.valid_to > ?"))
            params.extend((query.valid_to, query.valid_from))
        rows = await self._database.fetchall(
            "SELECT c.*, (SELECT operation FROM agent_activities a WHERE a.claim_id = c.claim_id) "
            "AS agent_operation FROM meaning_claims c WHERE "
            + " AND ".join(clauses)
            + " ORDER BY c.source_time DESC, c.claim_id",
            params,
        )
        claims = []
        reports = []
        seen_meanings = set()
        children, rejected = {}, frozenset()
        if query.predicate_parent:
            from debilgi.language.interlocutor_intent import InterlocutorIntentRepository

            children, rejected = await InterlocutorIntentRepository(
                self._database
            ).predicate_children(query)
        for row in rows:
            thought = thought_from_data(json.loads(row["thought_json"]))
            if row["subject_ref"].startswith("discord:user:") and row["subject_ref"] != (
                f"discord:user:{query.speaker_id}"
            ):
                continue
            from debilgi.personality.genome import DEBILGI_GENOME

            features = {
                "predicate:" + thought.predicate_class,
                "polarity:" + str(not thought.negated),
                *("role:" + d.role for d in thought.dependencies),
            }
            if DEBILGI_GENOME.protection_weight(features):
                continue
            key = meaning_key(thought, canon)
            world = world_thought(thought)
            if not world and row["speaker_id"] != query.speaker_id and not self_query(query):
                continue
            epistemic = "fact"
            evidence_weight = 1.0
            if world:
                support = await self.support(
                    query.guild_id,
                    key,
                    snapshot=query.snapshot_at,
                    exclude_event=query.exclude_event_id,
                    interval=(thought.semantic_time.start, thought.semantic_time.end),
                    public=True,
                )
                evidence_weight = support.weight / max(1, support.people)
                if support.supports and support.contradicts:
                    epistemic = "disputed"
                elif support.people < 2:
                    epistemic = "hypothesis"
                if epistemic == "disputed" or (
                    row["speaker_id"] != query.speaker_id and epistemic != "fact"
                ):
                    # Keep it as an attributed hypothesis, never a fact premise.
                    epistemic = "disputed" if epistemic == "disputed" else "hypothesis"
            if row["source_kind"] != "agent_observed":
                scoped = (key, row["polarity"], bound_participants(thought))
                if scoped in seen_meanings:
                    continue
                seen_meanings.add(scoped)
            if query.requested_attribute and not any(
                d.role == query.requested_role
                and (
                    d.argument.modifiers
                    or d.argument.postmodifiers
                    or d.argument.head.entity_kind in {"named_entity", "proper_name"}
                )
                for d in thought.dependencies
            ):
                continue
            generalization = ()
            if query.predicate_parent:
                from debilgi.language.morphology import DEFAULT_MORPHOLOGY

                predicate = thought.predicate.lemma
                reading = DEFAULT_MORPHOLOGY.best(predicate)
                # A finite verb alone does not prove an executed action.
                # Intransitive states need a direct demonstrated query edge;
                # transitive patient frames supply a conservative language cue.
                if (
                    thought.negated
                    or thought.scopes
                    or thought.predicate_class == "modal"
                    or thought.semantic_time.tense == "future"
                    or predicate in rejected
                    or predicate in {"делать", "заниматься"}
                    or row["agent_operation"] == "identity"
                ):
                    continue
                morphological = bool(
                    reading
                    and reading.transitivity == "transitive"
                    and any(d.role == "direct_object" for d in thought.dependencies)
                )
                if not (
                    predicate in children or morphological or row["agent_operation"] is not None
                ):
                    continue
                generalization = (
                    f"predicate_parent:{predicate}:{query.predicate_parent}:depth=1",
                    *(
                        children.get(predicate)
                        or (
                            "morphology:transitive_patient"
                            if morphological
                            else f"agent_operation:{row['agent_operation']}",
                        )
                    ),
                )
            if query.tense and thought.semantic_time.tense != query.tense:
                continue
            if (
                not query.include_history
                and not query.valid_from
                and thought.semantic_time.tense == "present"
                and (
                    (row["valid_to"] and row["valid_to"] <= query.snapshot_at)
                    or (row["valid_from"] and row["valid_from"] > query.snapshot_at)
                )
            ):
                continue
            evidence = await self._database.fetchall(
                "SELECT DISTINCT source_event_id FROM meaning_evidence WHERE claim_id = ? "
                "ORDER BY source_event_id",
                (row["claim_id"],),
            )
            supersedes = await self._database.fetchall(
                "SELECT previous_claim_id FROM meaning_supersedes WHERE claim_id = ? "
                "ORDER BY previous_claim_id",
                (row["claim_id"],),
            )
            proof = MeaningProof(
                thought,
                tuple(r["source_event_id"] for r in evidence),
                rule_ids=generalization,
                claim_ids=(row["claim_id"],),
            )
            claims.append(
                CanonicalClaim(
                    row["claim_id"],
                    row["subject_ref"],
                    row["predicate"],
                    row["polarity"],
                    row["source_kind"],
                    row["source_event_id"],
                    row["speaker_id"],
                    row["source_time"],
                    row["valid_from"],
                    row["valid_to"],
                    row["confidence"] * evidence_weight,
                    thought,
                    proof,
                    tuple(r["previous_claim_id"] for r in supersedes),
                    epistemic,
                )
            )
        facts = []
        for claim in claims:
            if claim.epistemic_status == "disputed" or (
                claim.epistemic_status == "hypothesis" and claim.speaker_id != query.speaker_id
            ):
                reports.append(claim)
            else:
                facts.append(claim)
        return (
            Known(query, tuple(facts), reports=tuple(reports))
            if facts
            else Unknown(query, reports=tuple(reports))
        )

    async def learned_meanings(self, event: Event) -> tuple[MeaningRecord, ...]:
        if event.channel_id is None or event.actor_id is None:
            return ()
        rows = await self._database.fetchall(
            """SELECT m.record_json FROM language_meaning_records m
               JOIN events e ON e.event_id = m.source_event_id
               WHERE m.guild_id = ? AND m.channel_id = ? AND m.speaker_id = ?
                 AND m.source_event_id != ? AND m.observed_at < ? AND m.kind != 'fact'
                 AND e.source NOT LIKE '%corpus%' AND e.source NOT LIKE 'seed:%'
               ORDER BY m.observed_at DESC, m.meaning_id LIMIT 256""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
            ),
        )
        result = []
        for row in rows:
            record = meaning_from_data(json.loads(row["record_json"]))
            if not any(contextual(t) for t in (*record.premises, *record.conclusions)):
                result.append(record)
        predicates = await self._database.fetchall(
            """SELECT DISTINCT predicate FROM meaning_claims WHERE guild_id = ?
               AND channel_id = ? AND speaker_id = ? AND source_kind != 'corpus_form'""",
            (event.guild_id, event.channel_id, event.actor_id),
        )
        for row in predicates:
            query = ClaimQuery(
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.utc_timestamp.isoformat(),
                event.event_id,
                row["predicate"],
                calendar_zone=getattr(self._timezone, "key", None),
            )
            for claim in (await self.query_claims(query)).claims:
                result.append(
                    MeaningRecord(
                        claim.claim_id,
                        claim.source_event_id,
                        "",
                        "fact",
                        (),
                        (answer_thought(claim, query),),
                        (claim.claim_id,),
                    )
                )
        return tuple(result)

    async def relevant_reports(
        self, event: Event, query: str, *, current_only: bool = False
    ) -> tuple[tuple[str, str], ...]:
        """Legacy historical evidence, or active graph facts for the surface gate."""
        request = replace(
            query_from_event(event, query, timezone=self._timezone),
            include_unresolved=not current_only,
            include_history=not current_only,
            requested_role=None,
        )
        result = await self.query_claims(request)
        reports = []
        for claim in result.claims:
            if current_only:
                # Background for the existing surface gate is rendered from
                # active claims; no stale/raw sentence becomes a fact again.
                surfaces = realize_thought(answer_thought(claim, result.query), limit=1)
                if surfaces:
                    reports.append((claim.source_event_id, surfaces[0].text))
                continue
            row = await self._database.fetchone(
                "SELECT text FROM language_proposition_reports WHERE source_event_id = ? "
                "ORDER BY report_id LIMIT 1",
                (claim.source_event_id,),
            )
            if row:
                reports.append((claim.source_event_id, row["text"]))
        return tuple(dict.fromkeys(reports))

    async def recent_conversation(self, event: Event) -> tuple[Event, ...]:
        """Read actual chronological turns, independently of lexical retrieval."""
        if event.channel_id is None:
            return ()
        rows = await self._database.fetchall(
            """SELECT * FROM (SELECT * FROM events
               WHERE guild_id = ? AND channel_id = ? AND event_id != ?
                 AND occurred_at >= ? AND occurred_at < ? AND event_type IN (?, ?, ?)
               ORDER BY occurred_at DESC, recorded_at DESC LIMIT 100)
               ORDER BY occurred_at ASC, recorded_at ASC""",
            (
                event.guild_id,
                event.channel_id,
                event.event_id,
                (event.utc_timestamp - timedelta(hours=1)).isoformat(),
                event.utc_timestamp.isoformat(),
                EventType.MESSAGE_CREATED.value,
                EventType.MESSAGE_REPLIED.value,
                EventType.BOT_REPLIED.value,
            ),
        )
        return tuple(EventRepository._from_row(row) for row in rows)

    async def _observe_dialogue(self, event: Event) -> None:
        retained = await self._database.fetchone(
            "SELECT 1 FROM language_dialogue_reports WHERE answer_event_id = ?", (event.event_id,)
        )
        if retained:
            return
        exchange = exchange_from_events(await self.recent_conversation(event), event)
        if not exchange or exchange.status != "matched" or exchange.bound_proposition is None:
            return
        thought = replace(
            exchange.bound_proposition,
            clause_kind="statement",
            evidence=(exchange.question_event_id, event.event_id),
        )
        await self._database.execute(
            "INSERT OR IGNORE INTO language_dialogue_reports VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                exchange.question_event_id,
                event.guild_id,
                event.channel_id,
                event.actor_id,
                exchange.answer_message,
                json.dumps(identity_graph(thought), ensure_ascii=False),
                event.utc_timestamp.isoformat(),
            ),
        )

    async def relevant_dialogue_reports(
        self, event: Event, query: str
    ) -> tuple[BoundDialogueReport, ...]:
        result = await self.retrieve_claims(event, query)
        reports = []
        for claim in result.claims:
            row = await self._database.fetchone(
                "SELECT * FROM language_dialogue_reports WHERE answer_event_id = ?",
                (claim.source_event_id,),
            )
            if row:
                reports.append(
                    BoundDialogueReport(
                        row["question_event_id"],
                        row["answer_event_id"],
                        row["answer_text"],
                        claim.thought,
                    )
                )
        return tuple(reports)

    async def relevant_utterances(
        self,
        event: Event,
        query: str,
        *,
        speaker_only: bool = False,
    ) -> tuple[tuple[str, str], ...]:
        """Use the event database as a durable corpus, including unparsed chat.

        The raw corpus is language evidence, not an independently verified fact.
        Restrict it to the current conversation and exclude bot self-training.
        """
        if event.channel_id is None:
            return ()
        rows = await self._database.fetchall(
            """SELECT event_id, payload_json FROM events
               WHERE guild_id = ? AND channel_id = ? AND event_id != ?
                 AND occurred_at < ? AND event_type IN (?, ?)
                 AND COALESCE(json_extract(payload_json, '$.author_is_bot'), 0) = 0
                 AND (? = 0 OR actor_id = ?)
               ORDER BY occurred_at DESC, recorded_at DESC LIMIT 512""",
            (
                event.guild_id,
                event.channel_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
                EventType.MESSAGE_CREATED.value,
                EventType.MESSAGE_REPLIED.value,
                int(speaker_only),
                event.actor_id,
            ),
        )
        query_terms = set(content_lemmas(query))
        ranked = []
        for index, row in enumerate(rows):
            payload = json.loads(row["payload_json"])
            text = payload.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            bot_id = payload.get("bot_user_id")
            if isinstance(bot_id, int):
                text = text.replace(f"<@{bot_id}>", " ").replace(f"<@!{bot_id}>", " ")
            overlap = len(query_terms & set(content_lemmas(text)))
            ranked.append((overlap, -index, row["event_id"], text))
        ranked.sort(reverse=True)
        return tuple((event_id, text) for _, _, event_id, text in ranked[:64])

    async def _observe_relations(self, event: Event, text: str) -> None:
        retained = await self._database.fetchone(
            "SELECT 1 FROM language_generation_relation_observations WHERE source_event_id = ?",
            (event.event_id,),
        )
        if retained:
            return
        relations = Counter(observe_word_relations(event.meaning, human_perspective=True))
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                "INSERT OR IGNORE INTO language_generation_relation_observations VALUES (?)",
                (event.event_id,),
            )
            if cursor.rowcount == 0:
                return
            await connection.executemany(
                """INSERT INTO language_generation_word_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id, channel_id, governor, dependent, relation,
                               dependent_pos, grammatical_case)
                   DO UPDATE SET count = count + excluded.count""",
                (
                    (
                        event.guild_id,
                        event.channel_id,
                        link.governor,
                        link.dependent,
                        link.relation,
                        link.dependent_pos,
                        link.case,
                        count,
                    )
                    for link, count in relations.items()
                ),
            )

    async def learned_word_relations(self, event: Event) -> tuple[WordRelation, ...]:
        rows = await self._database.fetchall(
            """SELECT governor, dependent, relation, dependent_pos, grammatical_case, count
               FROM language_generation_word_relations WHERE guild_id = ? AND channel_id = ?
               ORDER BY count DESC, governor, dependent, relation LIMIT 2000""",
            (event.guild_id, event.channel_id),
        )
        return tuple(WordRelation(*row) for row in rows)
