"""Discord gateway listeners that only normalize and persist observations."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from debilgi.discord.replies import SafeMentionReplyService
from debilgi.events.normalizer import (
    normalize_command_used,
    normalize_message_created,
    normalize_message_deleted,
    normalize_reaction,
)
from debilgi.events.pipeline import EventPipeline

logger = logging.getLogger(__name__)


class EventListener(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        pipeline: EventPipeline,
        reply_service: SafeMentionReplyService,
    ) -> None:
        self._bot = bot
        self._pipeline = pipeline
        self._reply_service = reply_service

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return

        attachments = [
            {
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "size": attachment.size,
            }
            for attachment in message.attachments
        ]
        reply_to = message.reference.message_id if message.reference else None
        resolved = message.reference.resolved if message.reference else None
        reply_to_author_id = resolved.author.id if isinstance(resolved, discord.Message) else None
        bot_mentioned = self._bot.user is not None and self._bot.user in message.mentions
        event = normalize_message_created(
            message_id=message.id,
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            author_id=message.author.id,
            created_at=message.created_at,
            content=message.content,
            author_is_bot=message.author.bot,
            reply_to_message_id=reply_to,
            reply_to_author_id=reply_to_author_id,
            bot_mentioned=bot_mentioned,
            mentioned_user_ids=(member.id for member in message.mentions),
            attachments=attachments,
        )
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist message event %s", event.event_id)
            return

        if message.author.bot:
            return
        if not await self._pipeline.wait_until_processed(event.event_id, timeout=5.0):
            logger.warning("Message event was not processed in time: %s", event.event_id)
            return
        prepared = (
            await self._reply_service.prepare_reply(event)
            if bot_mentioned
            else await self._reply_service.prepare_initiative(event)
        )
        if prepared:
            try:
                if prepared.initiative:
                    sent = await message.channel.send(
                        prepared.text,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    sent = await message.reply(
                        prepared.text,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
            except discord.HTTPException:
                logger.exception("Failed to send reply for event %s", event.event_id)
                return
            try:
                await self._reply_service.record_sent(
                    event,
                    response_message_id=sent.id,
                    reply=prepared,
                    created_at=sent.created_at,
                )
            except Exception:
                logger.exception("Failed to register experience for event %s", event.event_id)

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        await self._store_reaction(payload, added=True)

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        await self._store_reaction(payload, added=False)

    @commands.Cog.listener()
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if payload.guild_id is None:
            return
        cached = payload.cached_message
        event = normalize_message_deleted(
            message_id=payload.message_id,
            guild_id=payload.guild_id,
            channel_id=payload.channel_id,
            author_id=cached.author.id if cached else None,
        )
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist deletion event %s", event.event_id)

    @commands.Cog.listener()
    async def on_interaction(self, interaction: discord.Interaction) -> None:
        if (
            interaction.guild_id is None
            or interaction.type is not discord.InteractionType.application_command
        ):
            return
        command = interaction.command
        command_name = command.qualified_name if command else "unknown"
        event = normalize_command_used(
            interaction_id=interaction.id,
            guild_id=interaction.guild_id,
            channel_id=interaction.channel_id,
            actor_id=interaction.user.id,
            command_name=command_name,
            observed_at=interaction.created_at,
        )
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist command event %s", event.event_id)

    async def _store_reaction(
        self, payload: discord.RawReactionActionEvent, *, added: bool
    ) -> None:
        if payload.guild_id is None:
            return
        event = normalize_reaction(
            added=added,
            guild_id=payload.guild_id,
            channel_id=payload.channel_id,
            message_id=payload.message_id,
            actor_id=payload.user_id,
            target_id=payload.message_author_id,
            emoji=str(payload.emoji),
            emoji_animated=bool(payload.emoji.animated),
        )
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist reaction event %s", event.event_id)
