"""Minimal Discord gateway intents and bot permissions."""

from __future__ import annotations

import discord


def build_intents() -> discord.Intents:
    intents = discord.Intents.none()
    intents.guilds = True
    intents.messages = True
    intents.reactions = True
    intents.message_content = True
    return intents


def recommended_bot_permissions() -> discord.Permissions:
    permissions = discord.Permissions.none()
    permissions.view_channel = True
    permissions.read_message_history = True
    permissions.send_messages = True
    permissions.add_reactions = True
    return permissions
