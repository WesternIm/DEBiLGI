"""Discord gateway listeners that only normalize and persist observations."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from debilgi.discord.delivery import ReplyDeliveryService
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
        delivery: ReplyDeliveryService,
    ) -> None:
        self._delivery = delivery
        self._bot = bot
        self._pipeline = pipeline

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
        bot_addressed = bot_mentioned or (
            self._bot.user is not None and reply_to_author_id == self._bot.user.id
        )
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
            bot_mentioned=bot_addressed,
            bot_user_id=self._bot.user.id if self._bot.user is not None else None,
            mentioned_user_ids=(member.id for member in message.mentions),
            author_identity=_user_identity(message.author),
            bot_identity=_user_identity(self._bot.user) if self._bot.user is not None else None,
            reply_to_author_identity=(
                _user_identity(resolved.author) if isinstance(resolved, discord.Message) else None
            ),
            mentioned_users=(_user_identity(member) for member in message.mentions),
            mentioned_roles=(
                {"id": role.id, "name": role.name, "kind": "role"}
                for role in message.role_mentions
            ),
            mentioned_channels=(
                {"id": channel.id, "name": channel.name, "kind": "channel"}
                for channel in message.channel_mentions
            ),
            attachments=attachments,
        )
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist message event %s", event.event_id)
        finally:
            self._delivery.wake()

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


def _user_identity(user: discord.abc.User) -> dict[str, object]:
    """Copy Discord's public naming layers without retaining the gateway object."""

    return {
        "id": user.id,
        "username": user.name,
        "display_name": user.display_name,
        "global_name": getattr(user, "global_name", None),
        "kind": "bot" if user.bot else "person",
    }
