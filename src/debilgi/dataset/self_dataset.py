"""Build a feedback-aware Instruction/Q&A corpus and a growing concept graph."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from debilgi.events.models import Event, EventType
from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
from debilgi.experience.models import PreparedReply
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.content import content_lemmas
from debilgi.language.function_words import is_graph_concept
from debilgi.language.lexical_matching import same_lexical_family
from debilgi.language.semantics import extract_semantic_frames
from debilgi.language.surface_policy import internal_surface_reason
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

_relation_filter_logger = logging.getLogger("debilgi.relation_filter")

_MENTION_RE = re.compile(r"<@!?\d+>")
_NO_SPACE_BEFORE = frozenset(",.!?:;)]}»…")


@dataclass(frozen=True, slots=True)
class DatasetStats:
    candidates: int
    accepted: int
    rejected: int
    graph_nodes: int
    graph_edges: int


class SelfDatasetRepository:
    def __init__(self, database: Database, *, max_graph_terms: int = 12) -> None:
        self._database = database
        self.max_graph_terms = max(2, min(max_graph_terms, 40))

    async def capture_exchange(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
    ) -> str | None:
        content = prompt.payload.get("content")
        if not isinstance(content, str) or not content.strip() or not reply.text.strip():
            return None
        instruction = self._clean_instruction(content)
        if not instruction:
            return None
        answer = " ".join(reply.text.split())[:4000]
        prompt_hash = _digest(instruction.casefold())
        answer_hash = _digest(answer.casefold())
        example_id = f"instruction:{_digest(f'{prompt.guild_id}:{prompt.event_id}')[:24]}"
        topics = content_lemmas(instruction)[:32]
        reasoning, plan_quality = await self._reasoning_summary(prompt.event_id)
        quality = self._initial_quality(instruction, answer, plan_quality)
        status = "rejected" if quality <= 0.25 else "candidate"
        now = created_at.isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO learning_instruction_examples (
                    example_id, guild_id, channel_id, user_id, source_event_id,
                    response_message_id, instruction_text, input_text, answer_text,
                    reasoning_json, topic_terms_json, prompt_hash, answer_hash,
                    quality_score, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (guild_id, source_event_id) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    quality_score = MAX(quality_score, excluded.quality_score),
                    updated_at = excluded.updated_at
                ON CONFLICT (guild_id, prompt_hash, answer_hash) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    quality_score = MAX(quality_score, excluded.quality_score),
                    updated_at = excluded.updated_at""",
                (
                    example_id,
                    prompt.guild_id,
                    prompt.channel_id,
                    prompt.actor_id,
                    prompt.event_id,
                    response_message_id,
                    instruction,
                    answer,
                    _json(reasoning),
                    _json(topics),
                    prompt_hash,
                    answer_hash,
                    quality,
                    status,
                    now,
                    now,
                ),
            )
            row = await (
                await connection.execute(
                    """SELECT example_id FROM learning_instruction_examples
                       WHERE guild_id = ? AND prompt_hash = ? AND answer_hash = ?""",
                    (prompt.guild_id, prompt_hash, answer_hash),
                )
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to store instruction example")
            stored_id = str(row["example_id"])
            await connection.execute(
                """INSERT OR IGNORE INTO learning_example_sources (
                    guild_id, source_event_id, example_id, response_message_id, observed_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    prompt.guild_id,
                    prompt.event_id,
                    stored_id,
                    response_message_id,
                    now,
                ),
            )
            await self._grow_graph(
                connection,
                guild_id=prompt.guild_id,
                user_id=prompt.actor_id,
                example_id=stored_id,
                instruction=instruction,
                answer=answer,
                quality=quality,
                observed_at=now,
            )
        return stored_id

    async def observe(self, event: Event) -> None:
        if event.payload.get("author_is_bot") is True:
            return
        if event.payload.get("suppress_full_parse") is True:
            return
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        response_message_id = feedback.target_message_id
        if response_message_id is None:
            return
        row = await self._database.fetchone(
            """SELECT example_id FROM learning_example_sources
               WHERE guild_id = ? AND response_message_id = ?""",
            (event.guild_id, response_message_id),
        )
        if row is None:
            return
        if not feedback.is_learnable():
            return
        await self.record_feedback(
            example_id=str(row["example_id"]),
            feedback_event_id=feedback.feedback_event_id,
            kind=feedback.kind,
            score=feedback.reward,
            observed_at=event.utc_timestamp,
        )

    async def record_short_act_feedback(
        self,
        event: Event,
        *,
        response_message_id: int,
        accepted: bool,
    ) -> bool:
        row = await self._database.fetchone(
            """SELECT example_id FROM learning_example_sources
               WHERE guild_id = ? AND response_message_id = ?""",
            (event.guild_id, response_message_id),
        )
        if row is None:
            return False
        return await self.record_feedback(
            example_id=str(row["example_id"]),
            feedback_event_id=event.event_id,
            kind="short_act_confirmation" if accepted else "short_act_rejection",
            score=0.80 if accepted else -0.90,
            observed_at=event.utc_timestamp,
        )

    async def record_feedback(
        self,
        *,
        example_id: str,
        feedback_event_id: str,
        kind: str,
        score: float,
        observed_at: datetime,
    ) -> bool:
        bounded = max(-1.0, min(1.0, score))
        now = observed_at.isoformat()
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """INSERT OR IGNORE INTO learning_dataset_feedback (
                    feedback_event_id, example_id, feedback_kind, score, observed_at
                ) VALUES (?, ?, ?, ?, ?)""",
                (feedback_event_id, example_id, kind, bounded, now),
            )
            if cursor.rowcount != 1:
                return False
            row = await (
                await connection.execute(
                    """SELECT instruction_text, answer_text, quality_score, feedback_score
                       FROM learning_instruction_examples
                       WHERE example_id = ?""",
                    (example_id,),
                )
            ).fetchone()
            if row is None:
                return False
            old_quality = float(row["quality_score"])
            old_feedback = float(row["feedback_score"])
            feedback = max(-1.0, min(1.0, old_feedback * 0.55 + bounded * 0.45))
            surface_issue = _surface_issue(str(row["instruction_text"]), str(row["answer_text"]))
            if surface_issue is not None or bounded <= -0.8:
                quality = min(old_quality, 0.20 if surface_issue else 0.25)
                status = "rejected"
            else:
                quality = max(0.0, min(1.0, old_quality + bounded * 0.25))
                status = (
                    "accepted"
                    if quality >= 0.72 and feedback >= 0.25
                    else "rejected"
                    if quality <= 0.30
                    else "candidate"
                )
            await connection.execute(
                """UPDATE learning_instruction_examples
                   SET quality_score = ?, feedback_score = ?, status = ?, updated_at = ?
                   WHERE example_id = ?""",
                (quality, feedback, status, now, example_id),
            )
        return True

    async def related_concepts(
        self,
        guild_id: int,
        concept: str,
        *,
        limit: int = 32,
    ) -> tuple[str, ...]:
        keys = tuple(
            dict.fromkeys(
                (
                    concept.casefold().replace("ё", "е"),
                    *content_lemmas(concept),
                )
            )
        )
        if not keys:
            return ()
        placeholders = ",".join("?" for _ in keys)
        rows = await self._database.fetchall(
            f"""WITH origin(node_id) AS (
                    SELECT node_id FROM learning_graph_nodes
                    WHERE guild_id = ? AND node_type = 'concept'
                      AND node_key IN ({placeholders})
                ), related(node_id) AS (
                    SELECT target_node_id FROM learning_graph_edges
                    WHERE guild_id = ? AND source_node_id IN (SELECT node_id FROM origin)
                    UNION
                    SELECT source_node_id FROM learning_graph_edges
                    WHERE guild_id = ? AND target_node_id IN (SELECT node_id FROM origin)
                )
                SELECT node_key FROM learning_graph_nodes
                WHERE guild_id = ? AND node_type = 'concept'
                  AND node_id IN (SELECT node_id FROM related)
                ORDER BY occurrence_count DESC LIMIT ?""",
            (guild_id, *keys, guild_id, guild_id, guild_id, max(1, min(limit, 100))),
        )
        return tuple(str(row["node_key"]) for row in rows)

    async def link_explanation(
        self,
        *,
        guild_id: int,
        concept: str,
        explanation: str,
        source_event_id: str,
        observed_at: datetime,
        confidence: float = 0.72,
    ) -> None:
        async with self._database.transaction() as connection:
            await _grow_explanation_graph(
                connection,
                guild_id=guild_id,
                concept=concept,
                explanation=explanation,
                source_event_id=source_event_id,
                confidence=confidence,
                observed_at=observed_at.isoformat(),
            )

    async def link_reference_concepts(
        self,
        *,
        guild_id: int,
        word: str,
        concepts: tuple[str, ...],
        relations: tuple[tuple[str, str, str, float], ...] = (),
        source: str,
        observed_at: datetime,
        confidence: float = 0.76,
    ) -> None:
        """Persist parsed reference knowledge without retaining the definition quote."""

        normalized = word.casefold().replace("ё", "е").strip()
        selected = tuple(dict.fromkeys(item for item in concepts if item and item != normalized))
        now = observed_at.isoformat()
        async with self._database.transaction() as connection:
            for concept in selected:
                await connection.execute(
                    """INSERT INTO language_reference_concepts (
                           guild_id, normalized_form, concept_key, source, confidence,
                           first_seen_at, last_seen_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (guild_id, normalized_form, concept_key, source) DO UPDATE SET
                           evidence_count = evidence_count + 1,
                           confidence = MAX(confidence, excluded.confidence),
                           last_seen_at = excluded.last_seen_at""",
                    (guild_id, normalized, concept, source, confidence, now, now),
                )
            await _grow_reference_learning_graph(
                connection,
                guild_id=guild_id,
                word=normalized,
                concepts=selected,
                relations=relations,
                source=source,
                confidence=confidence,
                observed_at=now,
            )

    async def link_usage_pattern(
        self,
        *,
        guild_id: int,
        word: str,
        lemma: str,
        preferred_case: str | None,
        preferred_number: str | None,
        preferred_role: str | None,
        neighbor_lemmas: tuple[str, ...],
        pattern_status: str,
        confidence: float,
        example_count: int,
        source: str,
        observed_at: datetime,
    ) -> None:
        """Store only a generalized usage rule; source examples remain transient."""

        normalized = word.casefold().replace("ё", "е").strip()
        neighbors = tuple(dict.fromkeys(neighbor_lemmas))[:8]
        now = observed_at.isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO language_usage_patterns (
                       guild_id, normalized_form, lemma, preferred_case, preferred_number,
                       preferred_role, neighbor_lemmas_json, pattern_status, confidence,
                       example_count, source, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (guild_id, normalized_form, source) DO UPDATE SET
                       lemma = excluded.lemma,
                       preferred_case = excluded.preferred_case,
                       preferred_number = excluded.preferred_number,
                       preferred_role = excluded.preferred_role,
                       neighbor_lemmas_json = excluded.neighbor_lemmas_json,
                       pattern_status = excluded.pattern_status,
                       confidence = excluded.confidence,
                       example_count = excluded.example_count,
                       updated_at = excluded.updated_at""",
                (
                    guild_id,
                    normalized,
                    lemma,
                    preferred_case,
                    preferred_number,
                    preferred_role,
                    _json(neighbors),
                    pattern_status,
                    confidence,
                    max(0, example_count),
                    source,
                    now,
                ),
            )
            await _grow_usage_pattern_graph(
                connection,
                guild_id=guild_id,
                word=normalized,
                lemma=lemma,
                preferred_case=preferred_case,
                preferred_number=preferred_number,
                preferred_role=preferred_role,
                neighbor_lemmas=neighbors,
                pattern_status=pattern_status,
                confidence=confidence,
                example_count=example_count,
                source=source,
                observed_at=now,
            )

    async def stats(self, guild_id: int) -> DatasetStats:
        rows = await self._database.fetchall(
            """SELECT status, COUNT(*) AS count FROM learning_instruction_examples
               WHERE guild_id = ? GROUP BY status""",
            (guild_id,),
        )
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        nodes = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM learning_graph_nodes WHERE guild_id = ?", (guild_id,)
        )
        edges = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM learning_graph_edges WHERE guild_id = ?", (guild_id,)
        )
        return DatasetStats(
            counts.get("candidate", 0),
            counts.get("accepted", 0),
            counts.get("rejected", 0),
            int(nodes["count"]) if nodes else 0,
            int(edges["count"]) if edges else 0,
        )

    async def reassess_examples(self, guild_id: int) -> int:
        """Quarantine historical examples that fail the current surface-quality gate."""

        rows = await self._database.fetchall(
            """SELECT example_id, instruction_text, answer_text
               FROM learning_instruction_examples WHERE guild_id = ? AND status != 'rejected'""",
            (guild_id,),
        )
        rejected = 0
        for row in rows:
            if _surface_issue(str(row["instruction_text"]), str(row["answer_text"])) is None:
                continue
            cursor = await self._database.execute(
                """UPDATE learning_instruction_examples
                   SET status = 'rejected', quality_score = MIN(quality_score, 0.20),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE example_id = ? AND status != 'rejected'""",
                (str(row["example_id"]),),
            )
            rejected += cursor.rowcount
        return rejected

    async def rebuild_graph(self, guild_id: int) -> tuple[int, int]:
        """Rebuild derived graph data from non-rejected examples after quality changes."""

        rows = await self._database.fetchall(
            """SELECT * FROM learning_instruction_examples
               WHERE guild_id = ? AND status != 'rejected' AND quality_score >= 0.45
               ORDER BY created_at ASC""",
            (guild_id,),
        )
        explanations = await self._database.fetchall(
            """SELECT concept_key, response_text, response_event_id, resolved_at
               FROM language_open_questions
               WHERE guild_id = ? AND status = 'learned' AND response_text IS NOT NULL
               ORDER BY resolved_at ASC""",
            (guild_id,),
        )
        reference_concepts = await self._database.fetchall(
            """SELECT normalized_form, concept_key, source, confidence, last_seen_at
               FROM language_reference_concepts WHERE guild_id = ?
               ORDER BY last_seen_at ASC""",
            (guild_id,),
        )
        usage_patterns = await self._database.fetchall(
            """SELECT * FROM language_usage_patterns WHERE guild_id = ?
               ORDER BY updated_at ASC""",
            (guild_id,),
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                "DELETE FROM learning_graph_edges WHERE guild_id = ?", (guild_id,)
            )
            await connection.execute(
                "DELETE FROM learning_graph_nodes WHERE guild_id = ?", (guild_id,)
            )
            for row in rows:
                await self._grow_graph(
                    connection,
                    guild_id=guild_id,
                    user_id=int(row["user_id"]) if row["user_id"] is not None else None,
                    example_id=str(row["example_id"]),
                    instruction=str(row["instruction_text"]),
                    answer=str(row["answer_text"]),
                    quality=float(row["quality_score"]),
                    observed_at=str(row["updated_at"]),
                )
            for row in explanations:
                await _grow_explanation_graph(
                    connection,
                    guild_id=guild_id,
                    concept=str(row["concept_key"]),
                    explanation=str(row["response_text"]),
                    source_event_id=str(row["response_event_id"]),
                    confidence=0.72,
                    observed_at=str(row["resolved_at"]),
                )
            grouped_reference: dict[tuple[str, str], list[Any]] = {}
            for row in reference_concepts:
                grouped_reference.setdefault(
                    (str(row["normalized_form"]), str(row["source"])), []
                ).append(row)
            for (word, source), concept_rows in grouped_reference.items():
                await _grow_reference_learning_graph(
                    connection,
                    guild_id=guild_id,
                    word=word,
                    concepts=tuple(str(row["concept_key"]) for row in concept_rows),
                    relations=(),
                    source=source,
                    confidence=max(float(row["confidence"]) for row in concept_rows),
                    observed_at=max(str(row["last_seen_at"]) for row in concept_rows),
                )
            for row in usage_patterns:
                await _grow_usage_pattern_graph(
                    connection,
                    guild_id=guild_id,
                    word=str(row["normalized_form"]),
                    lemma=str(row["lemma"]),
                    preferred_case=(
                        str(row["preferred_case"]) if row["preferred_case"] else None
                    ),
                    preferred_number=(
                        str(row["preferred_number"]) if row["preferred_number"] else None
                    ),
                    preferred_role=(
                        str(row["preferred_role"]) if row["preferred_role"] else None
                    ),
                    neighbor_lemmas=tuple(json.loads(str(row["neighbor_lemmas_json"]))),
                    pattern_status=str(row["pattern_status"]),
                    confidence=float(row["confidence"]),
                    example_count=int(row["example_count"]),
                    source=str(row["source"]),
                    observed_at=str(row["updated_at"]),
                )
        stats = await self.stats(guild_id)
        return stats.graph_nodes, stats.graph_edges

    async def export(
        self,
        guild_id: int,
        output_directory: Path,
        *,
        include_candidates: bool = False,
        minimum_quality: float = 0.55,
    ) -> dict[str, int]:
        output_directory.mkdir(parents=True, exist_ok=True)
        statuses = ("accepted", "candidate") if include_candidates else ("accepted",)
        placeholders = ",".join("?" for _ in statuses)
        examples = await self._database.fetchall(
            f"""SELECT * FROM learning_instruction_examples
                WHERE guild_id = ? AND status IN ({placeholders}) AND quality_score >= ?
                ORDER BY quality_score DESC, created_at ASC""",
            (guild_id, *statuses, max(0.0, min(1.0, minimum_quality))),
        )
        nodes = await self._database.fetchall(
            "SELECT * FROM learning_graph_nodes WHERE guild_id = ? ORDER BY node_id", (guild_id,)
        )
        edges = await self._database.fetchall(
            "SELECT * FROM learning_graph_edges WHERE guild_id = ? ORDER BY edge_id", (guild_id,)
        )
        _write_jsonl(
            output_directory / "instruction_qa.jsonl",
            (
                {
                    "instruction": str(row["instruction_text"]),
                    "input": str(row["input_text"]),
                    "output": str(row["answer_text"]),
                    "reasoning": json.loads(str(row["reasoning_json"])),
                    "metadata": {
                        "example_id": str(row["example_id"]),
                        "topics": json.loads(str(row["topic_terms_json"])),
                        "quality": float(row["quality_score"]),
                        "feedback": float(row["feedback_score"]),
                        "status": str(row["status"]),
                    },
                }
                for row in examples
            ),
        )
        _write_jsonl(output_directory / "graph_nodes.jsonl", (_node_export(row) for row in nodes))
        _write_jsonl(output_directory / "graph_edges.jsonl", (_edge_export(row) for row in edges))
        return {
            "instruction_qa": len(examples),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
        }

    async def backfill_experiences(self, guild_id: int, *, limit: int = 10_000) -> int:
        rows = await self._database.fetchall(
            """SELECT experience.*, event.payload_json, event.occurred_at
               FROM language_utterance_experiences AS experience
               JOIN events AS event ON event.event_id = experience.prompt_event_id
               LEFT JOIN learning_example_sources AS source
                 ON source.guild_id = experience.guild_id
                AND source.source_event_id = experience.prompt_event_id
               WHERE experience.guild_id = ? AND source.example_id IS NULL
               ORDER BY experience.created_at ASC LIMIT ?""",
            (guild_id, max(1, min(limit, 100_000))),
        )
        added = 0
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            prompt = Event(
                event_id=str(row["prompt_event_id"]),
                guild_id=int(row["guild_id"]),
                channel_id=int(row["channel_id"]),
                actor_id=int(row["user_id"]) if row["user_id"] is not None else None,
                target_id=None,
                event_type=EventType.MESSAGE_CREATED,
                timestamp=datetime.fromisoformat(str(row["occurred_at"])),
                payload=payload,
                source="dataset_backfill",
            )
            tokens = tuple(str(item) for item in json.loads(str(row["response_tokens_json"])))
            reply = PreparedReply(
                text=_detokenize(tokens),
                intent_id=str(row["intent_id"]),
                tokens=tokens,
                genome_signature=str(row["genome_signature"]),
                candidate_count=int(row["candidate_count"]),
            )
            captured = await self.capture_exchange(
                prompt,
                response_message_id=int(row["response_message_id"]),
                reply=reply,
                created_at=datetime.fromisoformat(str(row["created_at"])),
            )
            added += int(captured is not None)
            if captured is not None:
                for feedback_event in await self._feedback_events_for_response(
                    guild_id,
                    int(row["response_message_id"]),
                ):
                    await self.observe(feedback_event)
        return added

    async def _feedback_events_for_response(
        self,
        guild_id: int,
        response_message_id: int,
    ) -> tuple[Event, ...]:
        rows = await self._database.fetchall(
            """SELECT event_id, guild_id, channel_id, actor_id, target_id,
                      event_type, occurred_at, payload_json, source
               FROM events
               WHERE guild_id = ? AND json_extract(payload_json, '$.author_is_bot') IS NOT 1
                 AND (
                     json_extract(payload_json, '$.reply_to_message_id') = ?
                     OR (
                         event_type IN ('REACTION_ADDED', 'REACTION_REMOVED')
                         AND json_extract(payload_json, '$.message_id') = ?
                     )
                 )
               ORDER BY occurred_at ASC""",
            (guild_id, response_message_id, response_message_id),
        )
        return tuple(
            Event(
                event_id=str(row["event_id"]),
                guild_id=int(row["guild_id"]),
                channel_id=(int(row["channel_id"]) if row["channel_id"] is not None else None),
                actor_id=int(row["actor_id"]) if row["actor_id"] is not None else None,
                target_id=int(row["target_id"]) if row["target_id"] is not None else None,
                event_type=EventType(str(row["event_type"])),
                timestamp=datetime.fromisoformat(str(row["occurred_at"])),
                payload=json.loads(str(row["payload_json"])),
                source=str(row["source"]),
            )
            for row in rows
        )

    async def _reasoning_summary(self, source_event_id: str) -> tuple[dict[str, Any], float]:
        row = await self._database.fetchone(
            """SELECT message_intent, response_goal, key_points_json, tone, unknowns_json,
                      confidence, status, validation_json
               FROM language_response_plans WHERE source_event_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (source_event_id,),
        )
        if row is None:
            return {"kind": "observed_exchange", "chain_of_thought": False}, 0.0
        validation = json.loads(str(row["validation_json"])) if row["validation_json"] else {}
        summary = {
            "kind": "response_plan_summary",
            "chain_of_thought": False,
            "intent": str(row["message_intent"]),
            "goal": str(row["response_goal"]),
            "key_points": json.loads(str(row["key_points_json"])),
            "tone": str(row["tone"]),
            "unknowns": json.loads(str(row["unknowns_json"])),
            "confidence": float(row["confidence"]),
            "validation": validation,
        }
        plan_quality = (
            0.15
            if str(row["status"]) == "used"
            else -0.20
            if str(row["status"]) == "failed"
            else 0.0
        )
        return summary, plan_quality

    @staticmethod
    def _initial_quality(instruction: str, answer: str, plan_quality: float) -> float:
        if _surface_issue(instruction, answer) is not None:
            return 0.10
        score = 0.45 + plan_quality
        score += 0.05 if 3 <= len(tokenize(answer)) <= 160 else -0.10
        score += 0.04 if set(content_lemmas(instruction)) & set(content_lemmas(answer)) else 0.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _clean_instruction(content: str) -> str:
        without_mentions = _MENTION_RE.sub("", content)
        return DEFAULT_ADDRESSING.analyze(" ".join(without_mentions.split())).content_text[:4000]

    async def _grow_graph(
        self,
        connection: aiosqlite.Connection,
        *,
        guild_id: int,
        user_id: int | None,
        example_id: str,
        instruction: str,
        answer: str,
        quality: float,
        observed_at: str,
    ) -> None:
        instruction_node = await _upsert_node(
            connection,
            guild_id,
            "instruction",
            _digest(instruction.casefold()),
            instruction,
            quality,
            {},
            observed_at,
        )
        answer_node = await _upsert_node(
            connection,
            guild_id,
            "answer",
            _digest(answer.casefold()),
            answer,
            quality,
            {},
            observed_at,
        )
        agent_node = await _upsert_node(
            connection, guild_id, "agent", "debilgi", "DEBiLGI", 0.95, {}, observed_at
        )
        await _upsert_edge(
            connection,
            guild_id,
            instruction_node,
            "answered_by",
            answer_node,
            quality,
            {"example_id": example_id},
            observed_at,
        )
        await _upsert_edge(
            connection,
            guild_id,
            agent_node,
            "produced",
            answer_node,
            quality,
            {"example_id": example_id},
            observed_at,
        )
        if user_id is not None:
            actor_node = await _upsert_node(
                connection,
                guild_id,
                "actor",
                str(user_id),
                f"user:{user_id}",
                0.95,
                {},
                observed_at,
            )
            await _upsert_edge(
                connection,
                guild_id,
                actor_node,
                "asked",
                instruction_node,
                0.9,
                {"example_id": example_id},
                observed_at,
            )
        prompt_terms = content_lemmas(instruction)[: self.max_graph_terms]
        answer_terms = content_lemmas(answer)[: self.max_graph_terms]
        concept_nodes: dict[str, str] = {}
        for term in dict.fromkeys((*prompt_terms, *answer_terms)):
            concept_nodes[term] = await _upsert_node(
                connection, guild_id, "concept", term, term, quality, {}, observed_at
            )
        for term in prompt_terms:
            await _upsert_edge(
                connection,
                guild_id,
                instruction_node,
                "mentions",
                concept_nodes[term],
                quality,
                {"example_id": example_id},
                observed_at,
            )
        for term in answer_terms:
            await _upsert_edge(
                connection,
                guild_id,
                answer_node,
                "uses",
                concept_nodes[term],
                quality,
                {"example_id": example_id},
                observed_at,
            )
        for frame in extract_semantic_frames(answer)[:8]:
            if not frame.target:
                continue
            source_key = next(iter(content_lemmas(frame.actor)), frame.actor.casefold())
            target_key = next(iter(content_lemmas(frame.target)), frame.target.casefold())
            relation = next(iter(content_lemmas(frame.action)), frame.action.casefold())
            source = concept_nodes.get(source_key) or await _upsert_node(
                connection,
                guild_id,
                "concept",
                source_key,
                source_key,
                frame.confidence,
                {},
                observed_at,
            )
            target = concept_nodes.get(target_key) or await _upsert_node(
                connection,
                guild_id,
                "concept",
                target_key,
                target_key,
                frame.confidence,
                {},
                observed_at,
            )
            await _upsert_edge(
                connection,
                guild_id,
                source,
                relation,
                target,
                min(quality, frame.confidence),
                {"example_id": example_id, "negated": frame.negated},
                observed_at,
            )


async def _grow_explanation_graph(
    connection: aiosqlite.Connection,
    *,
    guild_id: int,
    concept: str,
    explanation: str,
    source_event_id: str,
    confidence: float,
    observed_at: str,
) -> None:
    concept_key = concept.casefold().replace("ё", "е").strip()
    if not _allow_graph_endpoint(concept_key, source="human_explanation"):
        return
    concept_node = await _upsert_node(
        connection,
        guild_id,
        "concept",
        concept_key,
        concept_key,
        confidence,
        {"source": "human_explanation"},
        observed_at,
    )
    explanation_key = f"explanation:{_digest(source_event_id)[:28]}"
    explanation_node = await _upsert_node(
        connection,
        guild_id,
        "explanation",
        explanation_key,
        explanation,
        confidence,
        {"source_event_id": source_event_id},
        observed_at,
    )
    await _upsert_edge(
        connection,
        guild_id,
        concept_node,
        "explained_by",
        explanation_node,
        confidence,
        {"source_event_id": source_event_id},
        observed_at,
    )
    for term in content_lemmas(explanation)[:24]:
        if term == concept_key:
            continue
        related_node = await _upsert_node(
            connection,
            guild_id,
            "concept",
            term,
            term,
            confidence,
            {"source": "human_explanation"},
            observed_at,
        )
        await _upsert_edge(
            connection,
            guild_id,
            concept_node,
            "associated_with",
            related_node,
            confidence,
            {"source_event_id": source_event_id},
            observed_at,
        )


async def _grow_reference_learning_graph(
    connection: aiosqlite.Connection,
    *,
    guild_id: int,
    word: str,
    concepts: tuple[str, ...],
    relations: tuple[tuple[str, str, str, float], ...],
    source: str,
    confidence: float,
    observed_at: str,
) -> None:
    properties = {"knowledge_source": "reference", "source": source}
    if not _allow_graph_endpoint(word, source=source):
        return
    word_node = await _upsert_node(
        connection, guild_id, "concept", word, word, confidence, properties, observed_at
    )
    concept_nodes: dict[str, str] = {word: word_node}
    for concept in concepts:
        if not _allow_graph_endpoint(concept, source=source):
            continue
        concept_nodes[concept] = await _upsert_node(
            connection,
            guild_id,
            "concept",
            concept,
            concept,
            confidence,
            properties,
            observed_at,
        )
        await _upsert_edge(
            connection,
            guild_id,
            word_node,
            "reference_association",
            concept_nodes[concept],
            confidence,
            properties,
            observed_at,
        )
    for actor, relation, target, relation_confidence in relations:
        if not actor or not relation or not target:
            continue
        if not _allow_graph_endpoint(actor, source=source) or not _allow_graph_endpoint(
            target, source=source
        ):
            continue
        actor_node = concept_nodes.get(actor) or await _upsert_node(
            connection,
            guild_id,
            "concept",
            actor,
            actor,
            relation_confidence,
            properties,
            observed_at,
        )
        target_node = concept_nodes.get(target) or await _upsert_node(
            connection,
            guild_id,
            "concept",
            target,
            target,
            relation_confidence,
            properties,
            observed_at,
        )
        await _upsert_edge(
            connection,
            guild_id,
            actor_node,
            relation,
            target_node,
            min(confidence, relation_confidence),
            properties,
            observed_at,
        )


async def _grow_usage_pattern_graph(
    connection: aiosqlite.Connection,
    *,
    guild_id: int,
    word: str,
    lemma: str,
    preferred_case: str | None,
    preferred_number: str | None,
    preferred_role: str | None,
    neighbor_lemmas: tuple[str, ...],
    pattern_status: str,
    confidence: float,
    example_count: int,
    source: str,
    observed_at: str,
) -> None:
    if not _allow_graph_endpoint(lemma or word, source=source):
        return
    properties = {
        "source": source,
        "status": pattern_status,
        "preferred_case": preferred_case,
        "preferred_number": preferred_number,
        "preferred_role": preferred_role,
        "neighbors": neighbor_lemmas,
        "example_count": max(0, example_count),
    }
    word_node = await _upsert_node(
        connection,
        guild_id,
        "concept",
        lemma or word,
        lemma or word,
        confidence,
        {"knowledge_source": "reference"},
        observed_at,
    )
    pattern_key = f"usage:{word}:{source}"
    labels = tuple(
        value
        for value in (preferred_case, preferred_number, preferred_role, *neighbor_lemmas)
        if value
    )
    pattern_node = await _upsert_node(
        connection,
        guild_id,
        "usage_pattern",
        pattern_key,
        f"{word}: {', '.join(labels) if labels else 'сочетаемость неопределена'}",
        confidence,
        properties,
        observed_at,
    )
    await connection.execute(
        """UPDATE learning_graph_nodes SET properties_json = ?, confidence = ?, updated_at = ?
           WHERE node_id = ?""",
        (_json(properties), confidence, observed_at, pattern_node),
    )
    await _upsert_edge(
        connection,
        guild_id,
        word_node,
        "has_usage_pattern",
        pattern_node,
        confidence,
        {"soft_constraint": True, "source": source},
        observed_at,
    )
    for neighbor in neighbor_lemmas:
        if not _allow_graph_endpoint(neighbor, source=source):
            continue
        neighbor_node = await _upsert_node(
            connection,
            guild_id,
            "concept",
            neighbor,
            neighbor,
            confidence,
            {"knowledge_source": "usage_examples"},
            observed_at,
        )
        await _upsert_edge(
            connection,
            guild_id,
            pattern_node,
            "frequent_neighbor",
            neighbor_node,
            confidence,
            {"soft_constraint": True, "source": source},
            observed_at,
        )


async def _upsert_node(
    connection: aiosqlite.Connection,
    guild_id: int,
    node_type: str,
    key: str,
    label: str,
    confidence: float,
    properties: dict[str, Any],
    now: str,
) -> str:
    node_id = f"node:{_digest(f'{guild_id}:{node_type}:{key}')[:28]}"
    await connection.execute(
        """INSERT INTO learning_graph_nodes (
            node_id, guild_id, node_type, node_key, label, properties_json,
            confidence, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (guild_id, node_type, node_key) DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            confidence = MAX(confidence, excluded.confidence),
            label = CASE
                WHEN length(excluded.label) > length(label) THEN excluded.label
                ELSE label
            END,
            updated_at = excluded.updated_at""",
        (
            node_id,
            guild_id,
            node_type,
            key[:500],
            label[:1000],
            _json(properties),
            max(0.0, min(1.0, confidence)),
            now,
            now,
        ),
    )
    row = await (
        await connection.execute(
            """SELECT node_id FROM learning_graph_nodes
               WHERE guild_id = ? AND node_type = ? AND node_key = ?""",
            (guild_id, node_type, key[:500]),
        )
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to upsert graph node")
    return str(row["node_id"])


def _allow_graph_endpoint(value: str, *, source: str) -> bool:
    if is_graph_concept(value):
        return True
    _relation_filter_logger.info(
        "filtered graph endpoint value=%r source=%s reason=service_word",
        value,
        source,
    )
    return False


async def _upsert_edge(
    connection: aiosqlite.Connection,
    guild_id: int,
    source: str,
    relation: str,
    target: str,
    confidence: float,
    properties: dict[str, Any],
    now: str,
) -> None:
    edge_id = f"edge:{_digest(f'{guild_id}:{source}:{relation}:{target}')[:28]}"
    await connection.execute(
        """INSERT INTO learning_graph_edges (
            edge_id, guild_id, source_node_id, relation, target_node_id,
            properties_json, confidence, first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (guild_id, source_node_id, relation, target_node_id) DO UPDATE SET
            evidence_count = evidence_count + 1,
            confidence = MAX(confidence, excluded.confidence),
            last_seen_at = excluded.last_seen_at""",
        (
            edge_id,
            guild_id,
            source,
            relation[:120],
            target,
            _json(properties),
            max(0.0, min(1.0, confidence)),
            now,
            now,
        ),
    )


def _surface_issue(instruction: str, answer: str) -> str | None:
    normalized_instruction = " ".join(instruction.casefold().split())
    normalized_answer = " ".join(answer.casefold().split())
    if not normalized_answer:
        return "empty"
    if internal_surface_reason(normalized_answer) is not None:
        return "known_broken_marker"
    if re.search(r"<(?:user|role|channel|url|emoji:[^>]+)>", normalized_answer):
        return "technical_token"
    if normalized_answer.strip(" .,!?…") == normalized_instruction.strip(" .,!?…"):
        return "exact_echo"
    prompt_words = tuple(token for token in tokenize(instruction) if token[:1].isalnum())
    answer_words = tuple(token for token in tokenize(answer) if token[:1].isalnum())
    prompt_topics = set(content_lemmas(instruction))
    answer_topics = set(content_lemmas(answer))
    if (
        len(prompt_words) >= 2
        and answer_topics
        and len(answer_words) <= len(prompt_words) + 1
        and all(
            any(_same_lexical_family(answer_topic, prompt_topic) for prompt_topic in prompt_topics)
            for answer_topic in answer_topics
        )
    ):
        return "shallow_echo"
    if re.search(r"\b(?:это|так|только)\s+[а-яё]+(?:ть|ться)\b", normalized_answer):
        return "broken_fragment"
    return None


def _same_lexical_family(left: str, right: str) -> bool:
    return same_lexical_family(left, right)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _detokenize(tokens: tuple[str, ...]) -> str:
    text = ""
    for token in tokens:
        if not text or token in _NO_SPACE_BEFORE:
            text += token
        else:
            text += f" {token}"
    return text.strip()


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _node_export(row: aiosqlite.Row) -> dict[str, Any]:
    return {
        "id": str(row["node_id"]),
        "type": str(row["node_type"]),
        "key": str(row["node_key"]),
        "label": str(row["label"]),
        "properties": json.loads(str(row["properties_json"])),
        "confidence": float(row["confidence"]),
        "occurrences": int(row["occurrence_count"]),
    }


def _edge_export(row: aiosqlite.Row) -> dict[str, Any]:
    return {
        "id": str(row["edge_id"]),
        "source": str(row["source_node_id"]),
        "relation": str(row["relation"]),
        "target": str(row["target_node_id"]),
        "properties": json.loads(str(row["properties_json"])),
        "confidence": float(row["confidence"]),
        "evidence_count": int(row["evidence_count"]),
    }
