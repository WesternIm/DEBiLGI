"""Persistence boundary for archive jobs and aggregate language statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from debilgi.language.discourse import store_discourse_turn
from debilgi.language.models import ArchivedMessage, ArchiveImportJob, ImportStatus
from debilgi.language.ngrams import ngram_frequencies, token_frequencies
from debilgi.language.reply_patterns import build_reply_pattern
from debilgi.language.semantics import store_frames
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class LanguageStats:
    raw_messages: int
    distinct_tokens: int
    distinct_bigrams: int
    distinct_trigrams: int
    reply_patterns: int
    semantic_frames: int
    discourse_turns: int
    open_questions: int


class ArchiveRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def create_job(
        self,
        *,
        guild_id: int,
        channel_id: int,
        requested_by: int,
        after_at: datetime,
        before_at: datetime,
        max_messages: int,
        delete_raw_after: bool,
    ) -> ArchiveImportJob:
        job = ArchiveImportJob(
            job_id=f"archive:{uuid4()}",
            guild_id=guild_id,
            channel_id=channel_id,
            requested_by=requested_by,
            after_at=after_at,
            before_at=before_at,
            max_messages=max_messages,
            status=ImportStatus.PENDING,
            delete_raw_after=delete_raw_after,
        )
        await self._database.execute(
            """
            INSERT INTO archive_import_jobs (
                job_id, guild_id, channel_id, requested_by, after_at, before_at,
                max_messages, status, delete_raw_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.guild_id,
                job.channel_id,
                job.requested_by,
                job.after_at.astimezone(UTC).isoformat(),
                job.before_at.astimezone(UTC).isoformat(),
                job.max_messages,
                job.status.value,
                int(job.delete_raw_after),
            ),
        )
        return job

    async def get_job(self, job_id: str) -> ArchiveImportJob | None:
        row = await self._database.fetchone(
            "SELECT * FROM archive_import_jobs WHERE job_id = ?", (job_id,)
        )
        return self._job_from_row(row) if row else None

    async def recent_jobs(self, guild_id: int, limit: int = 10) -> list[ArchiveImportJob]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM archive_import_jobs
            WHERE guild_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (guild_id, max(1, min(limit, 20))),
        )
        return [self._job_from_row(row) for row in rows]

    async def resumable_jobs(self) -> list[ArchiveImportJob]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM archive_import_jobs
            WHERE status IN ('pending', 'running')
            ORDER BY created_at ASC
            """
        )
        return [self._job_from_row(row) for row in rows]

    async def mark_running(self, job_id: str) -> None:
        await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET status = 'running', attempts = attempts + 1,
                last_error = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status IN ('pending', 'running')
            """,
            (job_id,),
        )

    async def resume_job(self, job_id: str) -> bool:
        cursor = await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET status = 'pending', last_error = NULL, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status IN ('paused', 'failed')
            """,
            (job_id,),
        )
        return cursor.rowcount == 1

    async def pause_job(self, job_id: str) -> bool:
        cursor = await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET status = 'paused', updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status IN ('pending', 'running')
            """,
            (job_id,),
        )
        return cursor.rowcount == 1

    async def cancel_job(self, job_id: str) -> bool:
        cursor = await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status NOT IN ('completed', 'cancelled')
            """,
            (job_id,),
        )
        return cursor.rowcount == 1

    async def advance_job(
        self, job_id: str, *, last_message_id: int, seen: int, processed: int
    ) -> None:
        await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET last_message_id = ?, messages_seen = messages_seen + ?,
                messages_processed = messages_processed + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'running'
            """,
            (last_message_id, seen, processed, job_id),
        )

    async def fail_job(self, job_id: str, error: str) -> None:
        await self._database.execute(
            """
            UPDATE archive_import_jobs
            SET status = 'failed', last_error = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND status = 'running'
            """,
            (error[:1000], job_id),
        )

    async def complete_job(self, job_id: str) -> None:
        async with self._database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT delete_raw_after FROM archive_import_jobs WHERE job_id = ?",
                    (job_id,),
                )
            ).fetchone()
            if row is None:
                raise LookupError(f"Archive job does not exist: {job_id}")
            if bool(row[0]):
                await connection.execute(
                    """
                    UPDATE archive_messages
                    SET content = NULL, raw_deleted = 1
                    WHERE message_id IN (
                        SELECT message_id FROM archive_job_messages WHERE job_id = ?
                    )
                    """,
                    (job_id,),
                )
            await connection.execute(
                """
                UPDATE archive_import_jobs
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP,
                    completed_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND status = 'running'
                """,
                (job_id,),
            )

    async def ingest_message(self, job_id: str, message: ArchivedMessage) -> bool:
        tokens = tokenize(message.content)
        token_counts = token_frequencies(tokens)
        ngram_counts = {
            2: ngram_frequencies(tokens, 2),
            3: ngram_frequencies(tokens, 3),
        }

        async with self._database.transaction() as connection:
            job_row = await (
                await connection.execute(
                    """
                    SELECT guild_id, channel_id, status
                    FROM archive_import_jobs WHERE job_id = ?
                    """,
                    (job_id,),
                )
            ).fetchone()
            if job_row is None:
                raise LookupError(f"Archive job does not exist: {job_id}")
            if str(job_row[2]) != ImportStatus.RUNNING.value:
                raise RuntimeError("archive job must be running before message ingestion")
            if int(job_row[0]) != message.guild_id or int(job_row[1]) != message.channel_id:
                raise ValueError("message does not belong to the archive job scope")

            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO archive_messages (
                    message_id, guild_id, channel_id, author_id,
                    created_at, content, reply_to_message_id, author_is_bot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.guild_id,
                    message.channel_id,
                    message.author_id,
                    message.created_at.astimezone(UTC).isoformat(),
                    message.content,
                    message.reply_to_message_id,
                    int(message.author_is_bot),
                ),
            )
            await connection.execute(
                """
                INSERT OR IGNORE INTO archive_job_messages (job_id, message_id)
                VALUES (?, ?)
                """,
                (job_id, message.message_id),
            )
            if cursor.rowcount != 1:
                return False

            await store_frames(
                connection,
                source_id=f"archive:{message.message_id}",
                source_kind="archive",
                guild_id=message.guild_id,
                channel_id=message.channel_id,
                actor_id=message.author_id,
                text=message.content,
                observed_at=message.created_at.astimezone(UTC).isoformat(),
            )
            await store_discourse_turn(
                connection,
                source_id=f"archive:{message.message_id}",
                guild_id=message.guild_id,
                channel_id=message.channel_id,
                message_id=message.message_id,
                parent_message_id=message.reply_to_message_id,
                actor_id=message.author_id,
                addressed_to_id=None,
                content=message.content,
                observed_at=message.created_at.astimezone(UTC).isoformat(),
            )

            if token_counts:
                await connection.executemany(
                    """
                    INSERT INTO language_token_counts (guild_id, token, count)
                    VALUES (?, ?, ?)
                    ON CONFLICT (guild_id, token)
                    DO UPDATE SET count = count + excluded.count
                    """,
                    ((message.guild_id, token, count) for token, count in token_counts.items()),
                )

            for n, counts in ngram_counts.items():
                if counts:
                    await connection.executemany(
                        """
                        INSERT INTO language_ngram_counts (guild_id, n, gram_json, count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (guild_id, n, gram_json)
                        DO UPDATE SET count = count + excluded.count
                        """,
                        (
                            (
                                message.guild_id,
                                n,
                                json.dumps(gram, ensure_ascii=False, separators=(",", ":")),
                                count,
                            )
                            for gram, count in counts.items()
                        ),
                    )

            if message.reply_to_message_id is not None:
                parent = await (
                    await connection.execute(
                        """
                        SELECT content FROM archive_messages
                        WHERE message_id = ? AND guild_id = ? AND content IS NOT NULL
                        """,
                        (message.reply_to_message_id, message.guild_id),
                    )
                ).fetchone()
                if parent is not None:
                    pattern = build_reply_pattern(tokenize(str(parent[0])), tokens)
                    if pattern is not None:
                        prompt_tail, response_head = pattern
                        await connection.execute(
                            """
                            INSERT INTO language_reply_patterns (
                                guild_id, prompt_tail_json, response_head_json, count
                            ) VALUES (?, ?, ?, 1)
                            ON CONFLICT (guild_id, prompt_tail_json, response_head_json)
                            DO UPDATE SET count = count + 1
                            """,
                            (
                                message.guild_id,
                                json.dumps(prompt_tail, ensure_ascii=False, separators=(",", ":")),
                                json.dumps(
                                    response_head, ensure_ascii=False, separators=(",", ":")
                                ),
                            ),
                        )
        return True

    async def language_stats(self, guild_id: int) -> LanguageStats:
        row = await self._database.fetchone(
            """
            SELECT
                (SELECT COUNT(*) FROM archive_messages
                 WHERE guild_id = ? AND content IS NOT NULL) AS raw_messages,
                (SELECT COUNT(*) FROM language_token_counts WHERE guild_id = ?) AS tokens,
                (SELECT COUNT(*) FROM language_ngram_counts WHERE guild_id = ? AND n = 2)
                    AS bigrams,
                (SELECT COUNT(*) FROM language_ngram_counts WHERE guild_id = ? AND n = 3)
                    AS trigrams,
                (SELECT COUNT(*) FROM language_reply_patterns WHERE guild_id = ?)
                    AS replies,
                (SELECT COUNT(*) FROM language_semantic_frames WHERE guild_id = ?)
                    AS semantic_frames,
                (SELECT COUNT(*) FROM language_discourse_turns WHERE guild_id = ?)
                    AS discourse_turns,
                (SELECT COUNT(*) FROM language_discourse_turns
                 WHERE guild_id = ? AND question_status = 'open') AS open_questions
            """,
            (
                guild_id,
                guild_id,
                guild_id,
                guild_id,
                guild_id,
                guild_id,
                guild_id,
                guild_id,
            ),
        )
        assert row is not None
        return LanguageStats(
            raw_messages=int(row["raw_messages"]),
            distinct_tokens=int(row["tokens"]),
            distinct_bigrams=int(row["bigrams"]),
            distinct_trigrams=int(row["trigrams"]),
            reply_patterns=int(row["replies"]),
            semantic_frames=int(row["semantic_frames"]),
            discourse_turns=int(row["discourse_turns"]),
            open_questions=int(row["open_questions"]),
        )

    async def token_count(self, guild_id: int, token: str) -> int:
        row = await self._database.fetchone(
            "SELECT count FROM language_token_counts WHERE guild_id = ? AND token = ?",
            (guild_id, token),
        )
        return int(row["count"]) if row else 0

    async def recent_texts(
        self, guild_id: int, channel_id: int, limit: int = 50
    ) -> tuple[str, ...]:
        rows = await self._database.fetchall(
            """
            SELECT content FROM archive_messages
            WHERE guild_id = ? AND channel_id = ?
              AND content IS NOT NULL AND author_is_bot = 0
            ORDER BY created_at DESC, message_id DESC LIMIT ?
            """,
            (guild_id, channel_id, max(1, min(limit, 200))),
        )
        return tuple(str(row["content"]) for row in reversed(rows))

    async def matching_reply_heads(
        self, guild_id: int, prompt_tokens: tuple[str, ...], limit: int = 8
    ) -> tuple[tuple[str, ...], ...]:
        rows = await self._database.fetchall(
            """
            SELECT prompt_tail_json, response_head_json, count
            FROM language_reply_patterns
            WHERE guild_id = ?
            ORDER BY count DESC LIMIT 500
            """,
            (guild_id,),
        )
        ranked: list[tuple[int, int, tuple[str, ...]]] = []
        for row in rows:
            prompt_tail = tuple(json.loads(str(row["prompt_tail_json"])))
            response_head = tuple(json.loads(str(row["response_head_json"])))
            overlap = self._suffix_overlap(prompt_tokens, prompt_tail)
            ranked.append((overlap, int(row["count"]), response_head))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        matching = [item for item in ranked if item[0] > 0]
        selected = matching or ranked
        unique: list[tuple[str, ...]] = []
        for _, _, response_head in selected:
            if response_head and response_head not in unique:
                unique.append(response_head)
            if len(unique) >= max(1, min(limit, 50)):
                break
        return tuple(unique)

    async def common_bigrams(
        self, guild_id: int, limit: int = 500
    ) -> tuple[tuple[tuple[str, str], int], ...]:
        ngrams = await self.common_ngrams(guild_id, limit)
        return tuple(((gram[0], gram[1]), count) for gram, count in ngrams if len(gram) == 2)

    async def common_ngrams(
        self, guild_id: int, limit: int = 1000
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        rows = await self._database.fetchall(
            """
            SELECT n, gram_json, count FROM language_ngram_counts
            WHERE guild_id = ? AND n IN (2, 3)
            ORDER BY count DESC, n DESC, gram_json ASC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 2000))),
        )
        result: list[tuple[tuple[str, ...], int]] = []
        for row in rows:
            gram = tuple(str(token) for token in json.loads(str(row["gram_json"])))
            if len(gram) == int(row["n"]):
                result.append((gram, int(row["count"])))
        return tuple(result)

    @staticmethod
    def _suffix_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> int:
        for size in range(min(len(left), len(right)), 0, -1):
            if left[-size:] == right[-size:]:
                return size
        return 0

    @staticmethod
    def _job_from_row(row: Any) -> ArchiveImportJob:
        return ArchiveImportJob(
            job_id=str(row["job_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            requested_by=int(row["requested_by"]),
            after_at=datetime.fromisoformat(str(row["after_at"])),
            before_at=datetime.fromisoformat(str(row["before_at"])),
            max_messages=int(row["max_messages"]),
            status=ImportStatus(str(row["status"])),
            last_message_id=(
                int(row["last_message_id"]) if row["last_message_id"] is not None else None
            ),
            messages_seen=int(row["messages_seen"]),
            messages_processed=int(row["messages_processed"]),
            delete_raw_after=bool(row["delete_raw_after"]),
            attempts=int(row["attempts"]),
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        )
