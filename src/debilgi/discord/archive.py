"""Discord history adapter for the language archive importer."""

from __future__ import annotations

from datetime import timedelta

import discord
from discord.ext import commands

from debilgi.language.models import ArchivedMessage, ArchiveImportJob


class DiscordHistorySource:
    def __init__(self, bot: commands.Bot) -> None:
        self._bot = bot

    async def fetch_batch(self, job: ArchiveImportJob, limit: int) -> list[ArchivedMessage]:
        channel = self._bot.get_channel(job.channel_id)
        if channel is None:
            channel = await self._bot.fetch_channel(job.channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise TypeError(f"Channel {job.channel_id} does not expose message history")
        if channel.guild.id != job.guild_id:
            raise ValueError("Fetched archive channel belongs to another guild")

        after = (
            discord.Object(id=job.last_message_id)
            if job.last_message_id
            else job.after_at - timedelta(microseconds=1)
        )
        messages: list[ArchivedMessage] = []
        async for message in channel.history(
            limit=limit,
            after=after,
            before=job.before_at,
            oldest_first=True,
        ):
            messages.append(
                ArchivedMessage(
                    message_id=message.id,
                    guild_id=channel.guild.id,
                    channel_id=channel.id,
                    author_id=message.author.id,
                    created_at=message.created_at,
                    content=message.content,
                    reply_to_message_id=(
                        message.reference.message_id if message.reference else None
                    ),
                    author_is_bot=message.author.bot,
                )
            )
        return messages
