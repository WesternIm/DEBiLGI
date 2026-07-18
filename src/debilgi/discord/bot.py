"""Discord bot composition root."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import discord
from discord.ext import commands

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.autonomy.repository import AutonomyRepository
from debilgi.cognition.core import CognitiveCore, CognitiveCoreConfig
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.expectations import ExpectationPredictor
from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.cognition.learning import LearningConfig, LearningEngine
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.config import Settings
from debilgi.discord.archive import DiscordHistorySource
from debilgi.discord.commands import Diagnostics
from debilgi.discord.events import EventListener
from debilgi.discord.permissions import build_intents
from debilgi.discord.replies import SafeMentionReplyService
from debilgi.events.pipeline import EventPipeline
from debilgi.events.repository import EventRepository
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.repository import ExperienceRepository
from debilgi.knowledge.policy import KnowledgeQueryPolicy
from debilgi.knowledge.repository import KnowledgeRepository
from debilgi.knowledge.service import KnowledgeAcquisitionService
from debilgi.knowledge.wikipedia import WikipediaClient
from debilgi.language.archive_importer import ArchiveImporter, ArchiveImportManager
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.literacy.engine import LiteracyEngine
from debilgi.literacy.repository import LiteracyRepository
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.reflection.engine import ReflectionEngine
from debilgi.reflection.repository import ReflectionRepository
from debilgi.social.engine import SocialReactionEngine
from debilgi.social.repository import SocialReactionRepository
from debilgi.storage.database import Database

logger = logging.getLogger(__name__)


class DebilgiBot(commands.Bot):
    def __init__(self, settings: Settings, database: Database, repository: EventRepository) -> None:
        super().__init__(command_prefix=commands.when_mentioned, intents=build_intents())
        self.settings = settings
        self.genome = DEBILGI_GENOME
        self.personality_expression = PersonalityExpressionEngine(self.genome)
        self.database = database
        self.repository = repository
        self.memory = ShortTermMemory(
            MemoryLimits(
                retention=timedelta(minutes=settings.memory_retention_minutes),
                charge_half_life=timedelta(minutes=settings.memory_charge_half_life_minutes),
                max_total_events=settings.memory_max_total_events,
                max_channel_events=settings.memory_max_channel_events,
                max_user_events=settings.memory_max_user_events,
                max_reply_chains=settings.memory_max_reply_chains,
                max_chain_events=settings.memory_max_chain_events,
            )
        )
        self.feature_extractor = BehaviorFeatureExtractor(
            FeatureConfig(lookback=self.memory.limits.retention)
        )
        self.world_model = WorldModelRepository(database)
        self.learning_engine = LearningEngine(
            self.world_model,
            self.memory,
            self.feature_extractor,
            ExpectationPredictor(),
            LearningConfig(
                horizon=timedelta(seconds=settings.expectation_horizon_seconds),
                poll_interval=timedelta(seconds=settings.expectation_poll_seconds),
            ),
        )
        self.cognitive_repository = CognitiveRepository(database)
        self.cognitive_core = CognitiveCore(
            self.cognitive_repository,
            self.world_model,
            config=CognitiveCoreConfig(
                episode_gap=timedelta(seconds=settings.cognitive_episode_gap_seconds),
                maintenance_interval=timedelta(seconds=settings.cognitive_maintenance_seconds),
                raw_text_retention=timedelta(days=settings.raw_text_retention_days),
                consolidation_batch_size=settings.cognitive_consolidation_batch_size,
            ),
        )
        self.archive_repository = ArchiveRepository(database)
        self.discourse_memory = DiscourseMemoryRepository(
            database,
            max_turns_per_guild=settings.discourse_max_turns,
            context_turn_limit=settings.discourse_context_turns,
            content_character_limit=settings.language_context_chars,
        )
        self.semantic_memory = SemanticMemoryRepository(database)
        self.literacy_repository = LiteracyRepository(database)
        self.literacy_engine = LiteracyEngine(self.literacy_repository)
        self.social_reaction_repository = SocialReactionRepository(database)
        self.social_reaction_engine = SocialReactionEngine(
            self.social_reaction_repository,
            self.cognitive_repository,
            self.genome,
        )
        self.experience_repository = ExperienceRepository(database)
        self.reflection_repository = ReflectionRepository(database)
        self.reflection_engine = ReflectionEngine(
            self.reflection_repository,
            self.experience_repository,
        )
        self.experience_consolidator = ExperienceConsolidator(
            self.experience_repository,
            reflection=self.reflection_engine,
        )
        self.autonomy_repository = AutonomyRepository(database)
        self.initiative_policy = InitiativePolicy(
            self.genome,
            self.autonomy_repository,
            self.experience_repository,
            enabled=settings.initiative_enabled,
        )
        self.knowledge_repository = KnowledgeRepository(database)
        self.knowledge_policy = KnowledgeQueryPolicy(enabled=settings.wikipedia_enabled)
        self.wikipedia_client = WikipediaClient(
            language=settings.wikipedia_language,
            user_agent=settings.wikipedia_user_agent,
            timeout_seconds=settings.wikipedia_timeout_seconds,
        )
        self.knowledge_service = KnowledgeAcquisitionService(
            self.knowledge_repository,
            self.cognitive_repository,
            self.knowledge_policy,
            self.wikipedia_client,
            reply_wait_seconds=settings.wikipedia_reply_wait_milliseconds / 1000,
        )
        self.reply_service = SafeMentionReplyService(
            self.cognitive_repository,
            self.memory,
            self.archive_repository,
            self.personality_expression,
            self.experience_consolidator,
            self.initiative_policy,
            MemoryPhraseGenerator(
                max_output_tokens=settings.language_max_output_tokens,
                max_output_chars=settings.language_max_output_chars,
            ),
            social_reactions=self.social_reaction_repository,
            reflection=self.reflection_engine,
            literacy=self.literacy_engine,
            discourse_memory=self.discourse_memory,
            semantic_memory=self.semantic_memory,
            knowledge_service=self.knowledge_service,
            context_character_limit=settings.language_context_chars,
            enabled=settings.test_replies_enabled,
        )
        self.pipeline = EventPipeline(
            repository,
            [
                self.memory.observe,
                self.discourse_memory.observe,
                self.semantic_memory.observe,
                self.literacy_engine.observe,
                self.learning_engine.observe,
                self.cognitive_core.observe,
                self.social_reaction_engine.observe,
                self.experience_consolidator.observe,
                self.autonomy_repository.observe,
                self.knowledge_service.observe,
            ],
        )
        self.archive_importer = ArchiveImporter(self.archive_repository)
        self.archive_manager = ArchiveImportManager(
            self.archive_repository,
            self.archive_importer,
            lambda _job: DiscordHistorySource(self),
        )

    async def setup_hook(self) -> None:
        await self.database.initialize()
        reflected = await self.reflection_engine.backfill()
        if reflected:
            logger.info("Backfilled %d internal reflections", reflected)
        cutoff = datetime.now(UTC) - self.memory.limits.retention
        recent_events = await self.repository.since(cutoff, self.memory.limits.max_total_events)
        await self.memory.rebuild(recent_events)
        for event in recent_events:
            await self.discourse_memory.observe(event)
            await self.social_reaction_engine.observe(event)
            await self.literacy_engine.observe(event)
        await self.learning_engine.start()
        await self.cognitive_core.start()
        await self.knowledge_service.start()
        await self.pipeline.start()
        await self.add_cog(EventListener(self, self.pipeline, self.reply_service))
        await self.add_cog(
            Diagnostics(
                self,
                self.database,
                self.repository,
                self.pipeline,
                self.archive_repository,
                self.archive_manager,
                self.memory,
                self.feature_extractor,
                self.world_model,
                self.cognitive_core,
            )
        )

        if self.settings.sync_commands:
            if self.settings.dev_guild_id:
                guild = discord.Object(id=self.settings.dev_guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info("Synced %d commands to development guild", len(synced))
            else:
                synced = await self.tree.sync()
                logger.info("Synced %d global commands", len(synced))

    async def on_ready(self) -> None:
        user_id = self.user.id if self.user else "?"
        logger.info("Connected to Discord as %s (%s)", self.user, user_id)
        try:
            await self.archive_manager.start()
        except Exception:
            logger.exception("Failed to resume archive imports")

    async def on_command_error(
        self, context: commands.Context[commands.Bot], error: commands.CommandError
    ) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        await super().on_command_error(context, error)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            try:
                await self.archive_manager.stop()
            finally:
                try:
                    await self.pipeline.stop()
                finally:
                    try:
                        await self.knowledge_service.stop()
                    finally:
                        try:
                            await self.learning_engine.stop()
                        finally:
                            try:
                                await self.cognitive_core.stop()
                            finally:
                                await self.database.close()


def create_bot(settings: Settings, database: Database, repository: EventRepository) -> DebilgiBot:
    return DebilgiBot(settings, database, repository)
