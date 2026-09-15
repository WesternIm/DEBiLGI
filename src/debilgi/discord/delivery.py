"""Single-consumer durable replies with conservative handling of ambiguous sends."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import suppress
from datetime import datetime
from types import SimpleNamespace

import discord

from debilgi.discord.delivery_repository import DeliveryRepository, decode_reply
from debilgi.discord.reply_workflow import ReplyWorkflow
from debilgi.events.repository import EventRepository
from debilgi.language.research_wait import LexicalResearchPending

logger = logging.getLogger(__name__)


def delivery_nonce(event_id: str) -> str:
    return hashlib.sha256(event_id.encode()).hexdigest()[:24]


class DiscordReplyTransport:
    def __init__(self, bot) -> None:
        self.bot = bot

    async def channel(self, event):
        channel = self.bot.get_channel(event.channel_id)
        if channel is None:
            channel = await self.bot.fetch_channel(event.channel_id)
        if getattr(getattr(channel, 'guild', None), 'id', None) != event.guild_id:
            raise ValueError('delivery channel does not belong to the event guild')
        return channel

    async def send(self, channel, event, reply):
        reference = None if reply.initiative else discord.MessageReference(
            message_id=int(event.payload['message_id']), channel_id=event.channel_id,
            guild_id=event.guild_id, fail_if_not_exists=True,
        )
        return await channel.send(
            reply.text, reference=reference, mention_author=False,
            nonce=delivery_nonce(event.event_id), allowed_mentions=discord.AllowedMentions.none(),
        )

    async def find_sent(self, event, reply):
        channel = await self.channel(event)
        async for message in channel.history(
            limit=100, after=event.utc_timestamp, oldest_first=False
        ):
            if (self.bot.user is not None and message.author.id == self.bot.user.id
                    and str(message.nonce) == delivery_nonce(event.event_id)
                    and message.content == reply.text):
                return message
        return None


class ReplyDeliveryService:
    def __init__(self, repository: DeliveryRepository, events: EventRepository,
                 workflow: ReplyWorkflow, transport: DiscordReplyTransport) -> None:
        self.repository = repository
        self.events = events
        self.workflow = workflow
        self.transport = transport
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        async with self._lock:
            if self.running:
                return
            await self.repository.recover()
            self._task = asyncio.create_task(self._run(), name='debilgi-reply-delivery')

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            try:
                if await self.run_once():
                    continue
            except Exception:
                logger.exception('Reply delivery worker failed; retrying persisted work')
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=1)

    async def run_once(self) -> bool:
        async with self._lock:
            job = await self.repository.next_job()
            if job is None:
                return False
            event_id = str(job['event_id'])
            try:
                await self._process(job)
            except LexicalResearchPending as pending:
                await self.repository.state(
                    event_id, 'pending', str(pending), pending.retry_after,
                )
            except Exception as exc:
                logger.exception('Reply delivery failed for %s', event_id)
                current = await self.repository.get(event_id)
                status = current['status']
                if status == 'sending':
                    # A timeout, 5xx, or missing local receipt cannot prove non-delivery.
                    status = 'uncertain'
                elif status == 'prepared' and isinstance(
                    exc, (discord.Forbidden, discord.NotFound, ValueError)
                ):
                    status = 'failed'
                elif status == 'preparing':
                    status = 'failed' if current['attempts'] >= 3 else 'pending'
                await self.repository.state(event_id, status, f'{type(exc).__name__}: {exc}', 30)
            return True

    async def _process(self, job) -> None:
        event_id = str(job['event_id'])
        event = await self.events.get(event_id)
        if event is None:
            raise RuntimeError('delivery event is missing')
        status = job['status']
        if status == 'pending':
            await self.repository.preparing(event_id)
            reply = await self.workflow.prepare(event)
            if reply is None:
                await self.repository.state(event_id, 'no_reply')
                return
            await self.repository.save_reply(event_id, reply)
            status = 'prepared'
        else:
            reply = decode_reply(job['prepared_json'])
        if status == 'uncertain':
            sent = await self.transport.find_sent(event, reply)
            if sent is None:
                await self.repository.state(event_id, 'uncertain', 'receipt_not_found', 300)
                logger.warning('Delivery requires confirmation: event=%s guild=%s channel=%s',
                               event_id, event.guild_id, event.channel_id)
                return
            await self.repository.sent(event_id, sent.id, sent.created_at)
        elif status == 'prepared':
            reply = await self.workflow.validate(event, reply)
            if reply is None:
                await self.repository.state(event_id, 'no_reply', 'context_gate_withheld')
                return
            await self.repository.save_reply(event_id, reply)
            # Resolve the channel before the irreversible boundary.
            channel = await self.transport.channel(event)
            await self.repository.state(event_id, 'sending')
            try:
                sent = await self.transport.send(channel, event, reply)
            except (discord.Forbidden, discord.NotFound) as exc:
                await self.repository.state(event_id, 'failed', str(exc))
                return
            await self.repository.sent(event_id, sent.id, sent.created_at)
        else:
            sent = SimpleNamespace(id=int(job['response_message_id']),
                                   created_at=datetime.fromisoformat(job['sent_at']))
        await self.workflow.record_sent(event, reply, sent)
        await self.repository.state(event_id, 'completed')
