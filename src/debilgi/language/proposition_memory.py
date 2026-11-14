"""Learn durable, attributed propositions from human events without a teacher."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.content import content_lemmas
from debilgi.language.dialogue_obligations import (
    BoundDialogueReport,
    exchange_from_events,
    thought_from_data,
)
from debilgi.language.generative_planner import proposition_sentences, statement_thought
from debilgi.language.semantic_graph import (
    MeaningRecord,
    contextual,
    extract_meanings,
    meaning_from_data,
    proposition_key,
)
from debilgi.language.tokenizer import tokenize
from debilgi.language.word_relations import WordRelation, grammatical_tokens, observe_word_relations
from debilgi.storage.database import Database


class PropositionMemory:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def restore_from_events(self, limit: int = 2000) -> None:
        """Reuse bounded historical human evidence when upgrading an existing DB."""
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
        await self._observe_language(event, text)
        await self._observe_relations(event, text)
        await self._observe_dialogue(event)
        await self._observe_meanings(event, text)
        retained = await self._database.fetchone(
            "SELECT 1 FROM language_proposition_reports WHERE source_event_id = ? LIMIT 1",
            (event.event_id,),
        )
        if retained:
            return
        reports = []
        for sentence in proposition_sentences(text):
            thought = statement_thought(sentence, speaker="user")
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
            reports.append((report_id, sentence, terms))
        if not reports:
            return
        async with self._database.transaction() as connection:
            for report_id, sentence, terms in reports:
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
                await connection.executemany(
                    "INSERT OR IGNORE INTO language_proposition_terms (report_id, term) "
                    "VALUES (?, ?)",
                    ((report_id, term) for term in terms),
                )

    async def _observe_meanings(self, event: Event, text: str) -> None:
        records = extract_meanings(text, event.event_id)
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

    async def learned_meanings(self, event: Event) -> tuple[MeaningRecord, ...]:
        if event.channel_id is None or event.actor_id is None:
            return ()
        rows = await self._database.fetchall(
            """SELECT record_json FROM language_meaning_records
               WHERE guild_id = ? AND channel_id = ? AND speaker_id = ?
                 AND source_event_id != ? AND observed_at < ?
                 AND (kind != 'fact' OR observed_at >= ?)
               ORDER BY observed_at DESC, meaning_id LIMIT 256""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
                (event.utc_timestamp - timedelta(hours=1)).isoformat(),
            ),
        )
        result, latest = [], set()
        for row in rows:
            record = meaning_from_data(json.loads(row["record_json"]))
            if record.kind == "fact":
                conclusions = []
                for thought in record.conclusions:
                    key = proposition_key(thought, signed=False)
                    if key in latest or contextual(thought):
                        continue
                    latest.add(key)
                    conclusions.append(thought)
                if not conclusions:
                    continue
                record = replace(record, conclusions=tuple(conclusions))
            elif any(contextual(t) for t in (*record.premises, *record.conclusions)):
                continue
            result.append(record)
        return tuple(result)

    async def relevant_reports(self, event: Event, query: str) -> tuple[tuple[str, str], ...]:
        terms = content_lemmas(query)
        if not terms or event.channel_id is None or event.actor_id is None:
            return ()
        placeholders = ",".join("?" for _ in terms)
        rows = await self._database.fetchall(
            f"""SELECT r.source_event_id, r.text, COUNT(*) AS matches
                FROM language_proposition_reports r
                JOIN language_proposition_terms t ON t.report_id = r.report_id
                WHERE r.guild_id = ? AND r.channel_id = ? AND r.speaker_id = ?
                  AND r.source_event_id != ? AND r.observed_at < ?
                  AND t.term IN ({placeholders})
                GROUP BY r.report_id
                ORDER BY matches DESC, r.observed_at DESC, r.report_id
                LIMIT 24""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
                *terms,
            ),
        )
        return tuple((row["source_event_id"], row["text"]) for row in rows)

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
                json.dumps(asdict(thought), ensure_ascii=False),
                event.utc_timestamp.isoformat(),
            ),
        )

    async def relevant_dialogue_reports(
        self, event: Event, query: str
    ) -> tuple[BoundDialogueReport, ...]:
        terms = set(content_lemmas(query))
        if not terms or event.channel_id is None or event.actor_id is None:
            return ()
        rows = await self._database.fetchall(
            """SELECT * FROM language_dialogue_reports
               WHERE guild_id = ? AND channel_id = ? AND speaker_id = ?
                 AND answer_event_id != ? AND observed_at < ?
               ORDER BY observed_at DESC LIMIT 100""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
            ),
        )
        ranked = []
        for index, row in enumerate(rows):
            thought = thought_from_data(json.loads(row["thought_json"]))
            heads = {
                thought.predicate.lemma,
                *(dep.argument.head.lemma for dep in thought.dependencies),
            }
            # Relative dates/references stay retained, but aren't timeless facts.
            if heads & {
                "сейчас",
                "сегодня",
                "вчера",
                "завтра",
                "недавно",
                "тут",
                "там",
                "здесь",
                "он",
                "она",
                "оно",
                "они",
                "это",
            }:
                continue
            overlap = len(terms & heads)
            if overlap:
                ranked.append(
                    (
                        overlap,
                        -index,
                        BoundDialogueReport(
                            row["question_event_id"],
                            row["answer_event_id"],
                            row["answer_text"],
                            thought,
                        ),
                    )
                )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(report for _, _, report in ranked[:24])

    async def relevant_utterances(self, event: Event, query: str) -> tuple[tuple[str, str], ...]:
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
               ORDER BY occurred_at DESC, recorded_at DESC LIMIT 512""",
            (
                event.guild_id,
                event.channel_id,
                event.event_id,
                event.utc_timestamp.isoformat(),
                EventType.MESSAGE_CREATED.value,
                EventType.MESSAGE_REPLIED.value,
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

    async def _observe_language(self, event: Event, text: str) -> None:
        """Persist transitions independently of the stricter proposition parser."""
        tokens = tuple(
            token for token in tokenize(text, max_tokens=256) if not token.startswith("<")
        )
        if not tokens:
            return
        observed = await self._database.fetchone(
            "SELECT 1 FROM language_generation_observations WHERE source_event_id = ?",
            (event.event_id,),
        )
        if observed:
            return
        sequence = ("\0start", *tokens, "\0end")
        counts = Counter(
            json.dumps(sequence[index : index + size], ensure_ascii=False)
            for size in (2, 3)
            for index in range(len(sequence) - size + 1)
        )
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                "INSERT OR IGNORE INTO language_generation_observations VALUES (?)",
                (event.event_id,),
            )
            if cursor.rowcount == 0:
                return
            await connection.executemany(
                """INSERT INTO language_generation_ngrams VALUES (?, ?, ?, ?)
                   ON CONFLICT(guild_id, channel_id, gram_json)
                   DO UPDATE SET count = count + excluded.count""",
                ((event.guild_id, event.channel_id, gram, count) for gram, count in counts.items()),
            )

    async def learned_ngrams(self, event: Event) -> tuple[tuple[tuple[str, ...], int], ...]:
        rows = await self._database.fetchall(
            """SELECT gram_json, count FROM language_generation_reader_ngrams
               WHERE guild_id = ? AND channel_id = ? ORDER BY count DESC, gram_json LIMIT 2000""",
            (event.guild_id, event.channel_id),
        )
        return tuple((tuple(json.loads(row["gram_json"])), row["count"]) for row in rows)

    async def _observe_relations(self, event: Event, text: str) -> None:
        retained = await self._database.fetchone(
            "SELECT 1 FROM language_generation_relation_observations WHERE source_event_id = ?",
            (event.event_id,),
        )
        if retained:
            return
        tokens = grammatical_tokens(text, human_perspective=True)
        relations = Counter(observe_word_relations(" ".join(tokens)))
        sentences = []
        sentence = []
        for token in tokens:
            if token.startswith("<"):
                continue
            sentence.append(token)
            if token in {".", "!", "?", "…"}:
                sentences.append(tuple(sentence))
                sentence = []
        if sentence:
            sentences.append(tuple(sentence))
        grams = Counter()
        for sentence_tokens in sentences:
            sequence = ("\0start", *sentence_tokens, "\0end")
            grams.update(
                json.dumps(sequence[index : index + size], ensure_ascii=False)
                for size in (2, 3)
                for index in range(len(sequence) - size + 1)
            )
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
            await connection.executemany(
                """INSERT INTO language_generation_reader_ngrams VALUES (?, ?, ?, ?)
                   ON CONFLICT(guild_id, channel_id, gram_json)
                   DO UPDATE SET count = count + excluded.count""",
                ((event.guild_id, event.channel_id, gram, count) for gram, count in grams.items()),
            )

    async def learned_word_relations(self, event: Event) -> tuple[WordRelation, ...]:
        rows = await self._database.fetchall(
            """SELECT governor, dependent, relation, dependent_pos, grammatical_case, count
               FROM language_generation_word_relations WHERE guild_id = ? AND channel_id = ?
               ORDER BY count DESC, governor, dependent, relation LIMIT 2000""",
            (event.guild_id, event.channel_id),
        )
        return tuple(WordRelation(*row) for row in rows)
