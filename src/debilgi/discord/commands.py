"""Read-only diagnostics exposed as administrator-only slash commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from debilgi.cognition.core import CognitiveCore
from debilgi.cognition.features import BehaviorFeatureExtractor
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.pipeline import EventPipeline
from debilgi.events.repository import EventRepository
from debilgi.language.archive_importer import ArchiveImportManager
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.models import ArchiveImportJob
from debilgi.memory.short_term import ShortTermMemory
from debilgi.storage.database import Database

if TYPE_CHECKING:
    from debilgi.discord.bot import DebilgiBot


class Diagnostics(
    commands.GroupCog,
    group_name="debilgi",
    group_description="Read-only DEBiLGI diagnostics",
):
    def __init__(
        self,
        bot: DebilgiBot,
        database: Database,
        repository: EventRepository,
        pipeline: EventPipeline,
        archive_repository: ArchiveRepository,
        archive_manager: ArchiveImportManager,
        memory: ShortTermMemory,
        feature_extractor: BehaviorFeatureExtractor,
        world_model: WorldModelRepository,
        cognitive_core: CognitiveCore,
    ) -> None:
        self.bot = bot
        self._database = database
        self._repository = repository
        self._pipeline = pipeline
        self._archive_repository = archive_repository
        self._archive_manager = archive_manager
        self._memory = memory
        self._feature_extractor = feature_extractor
        self._world_model = world_model
        self._cognitive_core = cognitive_core

    @app_commands.command(name="dna", description="Show immutable personality genome")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def dna(self, interaction: discord.Interaction) -> None:
        genome = self.bot.genome
        traits = "\n".join(f"{name}: {value:.2f}" for name, value in genome.traits().items())
        await interaction.response.send_message(
            f"genome: {genome.genome_id} v{genome.version}\n"
            f"signature: {genome.signature}\n{traits}",
            ephemeral=True,
        )

    @app_commands.command(name="experience", description="Show consolidated reply experience")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def experience(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        stats = await self.bot.experience_repository.stats(interaction.guild_id)
        self_model = await self.bot.experience_repository.self_model(interaction.guild_id)
        average = "-" if stats.average_reward is None else f"{stats.average_reward:.3f}"
        await interaction.response.send_message(
            f"pending: {stats.pending}\n"
            f"resolved: {stats.resolved}\n"
            f"expired: {stats.expired}\n"
            f"average reward: {average}\n"
            f"learned transitions: {stats.learned_transitions}\n"
            f"self efficacy: {self_model.efficacy:.3f}\n"
            f"initiative success: {self_model.initiative_successes}/"
            f"{self_model.initiative_attempts}",
            ephemeral=True,
        )

    @app_commands.command(name="reflection", description="Show private outcome reflection")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def reflection(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        repository = self.bot.reflection_repository
        stats = await repository.stats(interaction.guild_id)
        guidance = await repository.guidance(interaction.guild_id)
        beliefs = await repository.beliefs(interaction.guild_id, 5)
        lines = [
            f"reflections: {stats.reflections}",
            f"strategy revisions: {stats.revisions}",
            f"average surprise: {stats.average_surprise:.3f}",
            f"strategy beliefs: {stats.strategy_beliefs}",
            f"guidance length scale: {guidance.length_scale:.3f}",
            f"guidance exploration: {guidance.exploration:.3f}",
            f"guidance caution: {guidance.caution:.3f}",
        ]
        lines.extend(
            f"{belief.strategy}: reward={belief.mean_reward:.3f} "
            f"confidence={belief.confidence:.3f} n={belief.observations}"
            for belief in beliefs
        )
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="literacy", description="Show symbolic literacy statistics")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def literacy(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        stats = await self.bot.literacy_repository.stats(interaction.guild_id)
        await interaction.response.send_message(
            f"messages observed: {stats.messages_observed}\n"
            f"letters observed: {stats.letters_observed}\n"
            f"digits observed: {stats.digits_observed}\n"
            f"punctuation observed: {stats.punctuation_observed}\n"
            f"symbols observed: {stats.symbols_observed}\n"
            f"arithmetic queries: {stats.arithmetic_queries}\n"
            f"alphabet queries: {stats.alphabet_queries}\n"
            f"counting queries: {stats.counting_queries}\n"
            f"comparison queries: {stats.comparison_queries}\n"
            f"symbol queries: {stats.symbol_queries}",
            ephemeral=True,
        )

    @app_commands.command(name="reactions", description="Show composable social reactions")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def reactions(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        repository = self.bot.social_reaction_repository
        stats = await repository.stats(interaction.guild_id)
        recent = await repository.recent(interaction.guild_id, 5)
        lines = [
            f"reactions: {stats.reactions}",
            f"composites: {stats.composite_reactions}",
            f"relation models: {stats.relation_models}",
        ]
        lines.extend(
            f"{item.blend_id} confidence={item.confidence:.3f} "
            f"primary={item.vector.ranked()[0][1]:.3f}"
            for item in recent
        )
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="autonomy", description="Show goals and initiative state")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def autonomy(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel context.", ephemeral=True)
            return
        state = await self.bot.autonomy_repository.state(
            interaction.guild_id, interaction.channel_id
        )
        goal = await self.bot.autonomy_repository.open_goal(
            interaction.guild_id, interaction.channel_id
        )
        counts = await self.bot.autonomy_repository.goal_counts(interaction.guild_id)
        await interaction.response.send_message(
            f"initiative enabled: {self.bot.initiative_policy.enabled}\n"
            f"evidence balance: {state.evidence_balance:.3f}\n"
            f"human events: {state.observed_human_events}\n"
            f"initiatives started: {state.initiatives_started}\n"
            f"open goal: {goal.goal_id if goal else '-'}\n"
            f"goal states: {counts or {}}",
            ephemeral=True,
        )

    @app_commands.command(name="knowledge", description="Show Wikipedia concept memory")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def knowledge(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        if interaction.channel_id is None:
            await interaction.response.send_message("No channel context.", ephemeral=True)
            return
        stats = await self.bot.knowledge_repository.stats(interaction.guild_id)
        state = await self.bot.knowledge_repository.state(
            interaction.guild_id, interaction.channel_id
        )
        await interaction.response.send_message(
            f"wikipedia enabled: {self.bot.knowledge_service.enabled}\n"
            f"concepts: {stats.concepts}\n"
            f"aliases: {stats.aliases}\n"
            f"event links: {stats.event_links}\n"
            f"context associations: {stats.context_associations}\n"
            f"pending: {stats.pending}\n"
            f"resolved: {stats.resolved}\n"
            f"cached: {stats.cached}\n"
            f"no result: {stats.no_result}\n"
            f"failed: {stats.failed}\n"
            f"evidence balance: {state.evidence_balance:.3f}\n"
            f"queries started: {state.queries_started}\n"
            f"network queue: {self.bot.knowledge_service.queue_depth}",
            ephemeral=True,
        )

    @app_commands.command(name="health", description="Show connection and database health")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def health(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        event_count = await self._repository.count(interaction.guild_id)
        schema_version = await self._database.schema_version()
        latency_ms = round(self.bot.latency * 1000)
        await interaction.response.send_message(
            f"gateway: connected\nlatency: {latency_ms} ms\n"
            f"schema: {schema_version}\nguild events: {event_count}\n"
            f"in-memory queue: {self._pipeline.queue_depth}",
            ephemeral=True,
        )

    @app_commands.command(name="queue", description="Show durable processing queue state")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def queue(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        stats = await self._repository.processing_stats(interaction.guild_id)
        await interaction.response.send_message(
            f"pending: {stats.pending}\nprocessing: {stats.processing}\n"
            f"processed: {stats.processed}\nfailed: {stats.failed}",
            ephemeral=True,
        )

    @app_commands.command(name="stats", description="Count observed event types on this server")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def stats(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        counts = await self._repository.counts_by_type(interaction.guild_id)
        body = "\n".join(f"{name}: {count}" for name, count in counts.items())
        await interaction.response.send_message(body or "No events recorded yet.", ephemeral=True)

    @app_commands.command(name="recent", description="Show recent event metadata (no message text)")
    @app_commands.describe(limit="Number of events, from 1 to 20")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def recent(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 20] = 10,
    ) -> None:
        assert interaction.guild_id is not None
        events = await self._repository.recent(interaction.guild_id, limit)
        lines = [
            f"{event.utc_timestamp:%Y-%m-%d %H:%M:%S} {event.event_type.value} "
            f"actor={event.actor_id or '-'} channel={event.channel_id or '-'}"
            for event in events
        ]
        await interaction.response.send_message(
            "\n".join(lines) or "No events recorded yet.", ephemeral=True
        )

    @app_commands.command(
        name="replay", description="Explicitly reprocess recent events without duplicating them"
    )
    @app_commands.describe(limit="Number of recent events, from 1 to 100")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def replay(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 100] = 10,
    ) -> None:
        assert interaction.guild_id is not None
        await interaction.response.defer(ephemeral=True)
        queued = await self._pipeline.replay(interaction.guild_id, limit)
        await interaction.followup.send(
            f"Queued {queued} existing events for explicit replay.", ephemeral=True
        )

    @app_commands.command(
        name="archive-start", description="Import a channel period into language statistics"
    )
    @app_commands.describe(
        channel="Text channel to import",
        from_date="First date, inclusive, in YYYY-MM-DD",
        through_date="Last date, inclusive, in YYYY-MM-DD",
        limit="Maximum messages to scan",
        delete_raw_after="Delete raw imported text after aggregation",
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def archive_start(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
        from_date: str,
        through_date: str,
        limit: app_commands.Range[int, 1, 100_000] = 1000,
        delete_raw_after: bool = True,
    ) -> None:
        assert interaction.guild_id is not None
        if channel.guild.id != interaction.guild_id:
            await interaction.response.send_message(
                "Channel belongs to another server.", ephemeral=True
            )
            return
        bot_member = channel.guild.me
        permissions = channel.permissions_for(bot_member) if bot_member else None
        if not permissions or not permissions.view_channel or not permissions.read_message_history:
            await interaction.response.send_message(
                "Bot needs View Channel and Read Message History in that channel.",
                ephemeral=True,
            )
            return
        try:
            after_at = _parse_date(from_date)
            before_at = _parse_date(through_date) + timedelta(days=1)
            if after_at >= before_at:
                raise ValueError("from_date must not be after through_date")
        except ValueError as exc:
            await interaction.response.send_message(f"Invalid period: {exc}", ephemeral=True)
            return

        job = await self._archive_repository.create_job(
            guild_id=interaction.guild_id,
            channel_id=channel.id,
            requested_by=interaction.user.id,
            after_at=after_at,
            before_at=before_at,
            max_messages=limit,
            delete_raw_after=delete_raw_after,
        )
        self._archive_manager.submit(job)
        await interaction.response.send_message(
            f"Archive import started: `{job.job_id}`\n"
            f"channel: {channel.mention}\nlimit: {limit}\n"
            f"delete raw after: {delete_raw_after}",
            ephemeral=True,
        )

    @app_commands.command(name="archive-status", description="Show archive import progress")
    @app_commands.describe(job_id="Optional full archive job ID")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def archive_status(
        self, interaction: discord.Interaction, job_id: str | None = None
    ) -> None:
        assert interaction.guild_id is not None
        if job_id:
            job = await self._archive_repository.get_job(job_id)
            jobs = [job] if job and job.guild_id == interaction.guild_id else []
        else:
            jobs = await self._archive_repository.recent_jobs(interaction.guild_id, 5)
        if not jobs:
            await interaction.response.send_message("No matching archive jobs.", ephemeral=True)
            return
        await interaction.response.send_message(
            "\n".join(_format_job(job) for job in jobs), ephemeral=True
        )

    @app_commands.command(name="archive-pause", description="Pause an archive import")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def archive_pause(self, interaction: discord.Interaction, job_id: str) -> None:
        if not await self._job_belongs_to_interaction(interaction, job_id):
            return
        changed = await self._archive_manager.pause(job_id)
        await interaction.response.send_message(
            "Archive import paused." if changed else "Job cannot be paused.", ephemeral=True
        )

    @app_commands.command(name="archive-resume", description="Resume a paused or failed import")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def archive_resume(self, interaction: discord.Interaction, job_id: str) -> None:
        if not await self._job_belongs_to_interaction(interaction, job_id):
            return
        changed = await self._archive_manager.resume(job_id)
        await interaction.response.send_message(
            "Archive import resumed." if changed else "Job cannot be resumed.", ephemeral=True
        )

    @app_commands.command(name="archive-cancel", description="Cancel an archive import")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def archive_cancel(self, interaction: discord.Interaction, job_id: str) -> None:
        if not await self._job_belongs_to_interaction(interaction, job_id):
            return
        changed = await self._archive_manager.cancel(job_id)
        await interaction.response.send_message(
            "Archive import cancelled." if changed else "Job cannot be cancelled.", ephemeral=True
        )

    @app_commands.command(name="language-stats", description="Show imported language statistics")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def language_stats(self, interaction: discord.Interaction) -> None:
        assert interaction.guild_id is not None
        stats = await self._archive_repository.language_stats(interaction.guild_id)
        await interaction.response.send_message(
            f"raw messages: {stats.raw_messages}\n"
            f"distinct tokens: {stats.distinct_tokens}\n"
            f"distinct bigrams: {stats.distinct_bigrams}\n"
            f"distinct trigrams: {stats.distinct_trigrams}\n"
            f"reply patterns: {stats.reply_patterns}\n"
            f"semantic frames: {stats.semantic_frames}\n"
            f"discourse turns: {stats.discourse_turns}\n"
            f"open questions: {stats.open_questions}",
            ephemeral=True,
        )

    @app_commands.command(name="memory", description="Show bounded short-term memory state")
    @app_commands.describe(user="Optional participant whose recent context to inspect")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def memory(
        self, interaction: discord.Interaction, user: discord.Member | None = None
    ) -> None:
        assert interaction.guild_id is not None
        stats = await self._memory.stats(interaction.guild_id)
        body = (
            f"events: {stats.events}\nchannels: {stats.channels}\n"
            f"users: {stats.users}\nreply chains: {stats.reply_chains}\n"
            f"charged users: {stats.charged_users}"
        )
        if user is not None:
            context = await self._memory.user_context(interaction.guild_id, user.id)
            charge = await self._memory.context_charge(interaction.guild_id, user.id)
            body += f"\n\n{user.mention}: events={len(context)} charge={charge:.3f}"
        await interaction.response.send_message(body, ephemeral=True)

    @app_commands.command(
        name="features", description="Measure neutral observable behavior features"
    )
    @app_commands.describe(user="Participant to measure in the current memory window")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def features(self, interaction: discord.Interaction, user: discord.Member) -> None:
        assert interaction.guild_id is not None
        context = await self._memory.guild_context(interaction.guild_id)
        bot_user_id = self.bot.user.id if self.bot.user else None
        measured = self._feature_extractor.extract(
            context,
            guild_id=interaction.guild_id,
            subject_id=user.id,
            now=datetime.now(UTC),
            bot_user_id=bot_user_id,
        )
        await interaction.response.send_message(
            f"{user.mention} — neutral observations\n"
            f"messages: {measured.message_count}\n"
            f"messages/min: {measured.messages_per_minute:.3f}\n"
            f"average length: {measured.average_message_length:.2f}\n"
            f"uppercase ratio: {measured.uppercase_ratio:.3f}\n"
            f"repeated words: {measured.repeated_word_count}\n"
            f"duplicate messages: {measured.duplicate_message_count}\n"
            f"replies received: {measured.replies_received_count}\n"
            f"reactions received: {measured.reactions_received_count}\n"
            f"longest reply thread: {measured.longest_reply_thread_seconds:.1f}s\n"
            f"bot mentions: {measured.bot_mention_count}\n"
            f"repeats after bot action: "
            f"{measured.repeated_message_after_bot_action_count}\n"
            f"commands: {measured.command_use_count}\n"
            f"explicit reward requests: {measured.reward_request_attempt_count}\n"
            f"confirmed moderator deletions: {measured.moderator_deletion_count}\n"
            f"average response time: "
            f"{_optional_seconds(measured.average_response_time_seconds)}\n"
            f"activity delta: {measured.activity_delta_after_message:.2f}\n"
            f"conversation returns: {measured.conversation_return_count}\n"
            f"repeated scenarios: {measured.repeated_scenario_count}",
            ephemeral=True,
        )

    @app_commands.command(
        name="expectations", description="Show shadow predictions and measured differences"
    )
    @app_commands.describe(limit="Number of recent expectations, from 1 to 10")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def expectations(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 10] = 5,
    ) -> None:
        assert interaction.guild_id is not None
        stats = await self._world_model.stats(interaction.guild_id)
        recent = await self._world_model.recent_expectations(interaction.guild_id, limit)
        lines = [
            f"pending: {stats.pending_expectations} "
            f"resolved: {stats.resolved_expectations} "
            f"patterns: {stats.learned_patterns} "
            f"average d: {_optional_number(stats.average_difference)}"
        ]
        lines.extend(
            f"{item.status.value} predicted={item.prediction.outcome.value} "
            f"p={item.prediction.probability:.3f} "
            f"actual={item.actual_outcome.value if item.actual_outcome else '-'} "
            f"d={_optional_number(item.difference)}"
            for item in recent
        )
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @app_commands.command(name="core", description="Show private Cognitive Core v0.2 shadow state")
    @app_commands.describe(user="Optional participant model to inspect")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def core(
        self, interaction: discord.Interaction, user: discord.Member | None = None
    ) -> None:
        assert interaction.guild_id is not None
        snapshot = await self._cognitive_core.snapshot(
            interaction.guild_id, user_id=user.id if user else None, limit=5
        )
        model = snapshot.entity_model
        state = snapshot.internal_state
        last_intent = snapshot.intents[0] if snapshot.intents else None
        causal_chain = (
            " -> ".join(str(step.get("stage", "?")) for step in last_intent.reasoning_trace)
            if last_intent
            else "-"
        )
        body = (
            f"episodes: {len(snapshot.episodes)}\n"
            f"semantic memories: {len(snapshot.semantic_memories)}\n"
            f"recent difference vectors: {len(snapshot.differences)}\n"
            f"average processing: {_optional_number(snapshot.average_processing_ms)} ms\n"
            f"model: observations={model.observations if model else 0} "
            f"predictability={_optional_number(model.predictability if model else None)}\n"
            f"state: tension={_optional_number(state.tension if state else None)} "
            f"uncertainty={_optional_number(state.uncertainty if state else None)}\n"
            f"last shadow intent: {last_intent.action_type.value if last_intent else '-'}\n"
            f"causal chain: {causal_chain}\n"
            f"intent record: shadow-only\n"
            f"mention reply bridge: {'enabled' if self.bot.reply_service.enabled else 'disabled'}"
        )
        await interaction.response.send_message(body, ephemeral=True)

    async def _job_belongs_to_interaction(
        self, interaction: discord.Interaction, job_id: str
    ) -> bool:
        job = await self._archive_repository.get_job(job_id)
        if job is not None and job.guild_id == interaction.guild_id:
            return True
        await interaction.response.send_message("No matching archive job.", ephemeral=True)
        return False


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"expected YYYY-MM-DD, got {value!r}") from exc


def _format_job(job: ArchiveImportJob) -> str:
    error = f" error={job.last_error[:100]!r}" if job.last_error else ""
    return (
        f"`{job.job_id}` {job.status.value} channel={job.channel_id} "
        f"seen={job.messages_seen}/{job.max_messages} "
        f"processed={job.messages_processed}{error}"
    )


def _optional_seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.1f}s"


def _optional_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
