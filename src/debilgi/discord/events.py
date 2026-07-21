"""Discord gateway listeners that only normalize and persist observations."""

from __future__ import annotations

import logging
from dataclasses import replace

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
from debilgi.language.context_gate import WorkingMemoryContextGate
from debilgi.language.tokenizer import tokenize

logger = logging.getLogger(__name__)


class EventListener(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        pipeline: EventPipeline,
        reply_service: SafeMentionReplyService,
        context_gate: WorkingMemoryContextGate,
    ) -> None:
        self._bot = bot
        self._pipeline = pipeline
        self._reply_service = reply_service
        self._context_gate = context_gate

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
        upstream_failure: str | None = None
        try:
            await self._pipeline.publish(event)
        except Exception:
            logger.exception("Failed to persist message event %s", event.event_id)
            upstream_failure = "event_persistence_failed"

        if message.author.bot:
            return
        if upstream_failure is None and not await self._pipeline.wait_until_processed(
            event.event_id, timeout=5.0
        ):
            logger.warning("Message event was not processed in time: %s", event.event_id)
            upstream_failure = "event_processing_timeout"
        bound_explanation = (
            await self._reply_service.prepare_bound_explanation(event)
            if upstream_failure is None
            else None
        )
        bound_short_act = (
            await self._reply_service.prepare_bound_short_act(event)
            if upstream_failure is None and bound_explanation is None
            else None
        )
        if bound_explanation is not None:
            prepared = bound_explanation
        elif bound_short_act is not None:
            prepared = bound_short_act
        elif bot_addressed:
            prepared = await self._reply_service.prepare_guaranteed_reply(
                event,
                upstream_failure=upstream_failure,
            )
        elif upstream_failure is None:
            try:
                prepared = await self._reply_service.prepare_initiative(event)
            except Exception:
                logger.exception("Failed to prepare initiative for event %s", event.event_id)
                prepared = None
        else:
            prepared = None
        if prepared:
            decision = await self._context_gate.evaluate(
                guild_id=event.guild_id,
                channel_id=message.channel.id,
                candidate_text=prepared.text,
                source=prepared.response_source,
                now=message.created_at,
            )
            if not decision.allowed or decision.text is None:
                logger.info(
                    "Context gate withheld event=%s source=%s reason=%s",
                    event.event_id,
                    prepared.response_source,
                    decision.reason,
                )
                return
            if decision.text != prepared.text:
                prepared = replace(
                    prepared,
                    text=decision.text,
                    tokens=tokenize(decision.text),
                    fallback=True,
                    response_source="fallback",
                )
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
            await self._context_gate.commit_outgoing(
                guild_id=event.guild_id,
                channel_id=message.channel.id,
                message_id=sent.id,
                text=prepared.text,
                bot_user_id=self._bot.user.id if self._bot.user is not None else None,
                target_user_id=event.actor_id,
                observed_at=sent.created_at,
            )
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


def _user_identity(user: discord.abc.User) -> dict[str, object]:
    """Copy Discord's public naming layers without retaining the gateway object."""

    return {
        "id": user.id,
        "username": user.name,
        "display_name": user.display_name,
        "global_name": getattr(user, "global_name", None),
        "kind": "bot" if user.bot else "person",
    }
