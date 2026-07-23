"""Discord bot composition root."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import discord
from discord.ext import commands

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.autonomy.repository import AutonomyRepository
from debilgi.background import BackgroundLearningService
from debilgi.cognition.core import CognitiveCore, CognitiveCoreConfig
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.expectations import ExpectationPredictor
from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.cognition.learning import LearningConfig, LearningEngine
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.config import Settings
from debilgi.dataset.self_dataset import SelfDatasetRepository
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
from debilgi.language.abstract_reasoning import AbstractReasoningService
from debilgi.language.arbitration_repository import ArbitrationRepository
from debilgi.language.archive_importer import ArchiveImporter, ArchiveImportManager
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.causal_reasoning import CausalReasoningRepository
from debilgi.language.character_markov import CharacterMarkovConfig, CharacterMarkovRepository
from debilgi.language.context_gate import ContextGateDeferred, WorkingMemoryContextGate
from debilgi.language.conversation import ConversationRepository, ResponsePlanRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.grounded_generator import GroundedConstructionGenerator
from debilgi.language.lexicon import LexiconService, NeologismRepository, NeologismService
from debilgi.language.morphology import MorphologyRepository
from debilgi.language.open_questions import OpenLanguageQuestionService
from debilgi.language.position_reasoning import PositionReasoningService, PositionRepository
from debilgi.language.russian_reference import RussianReferenceService, WiktionaryClient
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.language.semantic_consultant import (
    CloudflareWorkersAIConsultant,
    PuterSidecarConsultant,
    SemanticConsultationService,
)
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.sentence_understanding import UtteranceUnderstandingRepository
from debilgi.language.short_acts import ShortSpeechActService
from debilgi.language.usage_learning import LexicalUsageLearningService
from debilgi.literacy.engine import LiteracyEngine
from debilgi.literacy.repository import LiteracyRepository
from debilgi.memory.long_term import LongTermMemoryRepository
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
                max_context_events=settings.memory_max_context_events,
                relevance_half_life=timedelta(
                    minutes=settings.memory_relevance_half_life_minutes
                ),
                min_context_relevance=settings.memory_min_context_relevance,
            )
        )
        self.long_term_memory = LongTermMemoryRepository(
            database,
            self.memory,
            min_importance=settings.memory_consolidation_min_importance,
            fragment_batch_size=settings.memory_consolidation_batch_size,
            capsule_limit=settings.memory_long_term_capsule_limit,
            recall_limit=settings.memory_long_term_recall_limit,
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
        character_markov_config = CharacterMarkovConfig(
            min_order=settings.character_markov_min_order,
            max_order=settings.character_markov_max_order,
            max_word_length=settings.character_markov_max_word_length,
            generation_attempts=settings.character_markov_generation_attempts,
        )
        self.archive_repository = ArchiveRepository(
            database,
            character_markov_config=character_markov_config,
        )
        self.discourse_memory = DiscourseMemoryRepository(
            database,
            max_turns_per_guild=settings.discourse_max_turns,
            context_turn_limit=settings.discourse_context_turns,
            content_character_limit=settings.language_context_chars,
        )
        self.semantic_memory = SemanticMemoryRepository(database)
        self.causal_reasoning = CausalReasoningRepository(database)
        self.character_markov = CharacterMarkovRepository(
            database,
            character_markov_config,
        )
        self.lexicon = LexiconService(
            database,
            unknown_word_threshold=settings.unknown_word_threshold,
            curiosity_threshold=settings.curiosity_threshold,
            curiosity_question_cooldown=timedelta(
                minutes=settings.curiosity_question_cooldown_minutes
            ),
            curiosity_max_consecutive_questions=settings.curiosity_max_consecutive_questions,
        )
        self.russian_reference_client = WiktionaryClient(
            user_agent=settings.russian_reference_user_agent,
            timeout_seconds=settings.russian_reference_timeout_seconds,
        )
        self.russian_reference = RussianReferenceService(
            database,
            self.lexicon,
            self.russian_reference_client,
            enabled=settings.russian_reference_enabled,
            worker_count=settings.russian_reference_workers,
        )
        self.morphology = MorphologyRepository(database)
        self.neologisms = NeologismRepository(
            database,
            expiration_days=settings.neologism_expiration_days,
            min_response_interval=settings.neologism_min_response_interval,
        )
        self.neologism_service = NeologismService(
            self.lexicon,
            self.character_markov,
            self.neologisms,
        )
        self.conversation_memory = ConversationRepository(
            database,
            idle_timeout=timedelta(minutes=settings.conversation_idle_timeout_minutes),
            max_idle_timeout=timedelta(minutes=settings.conversation_max_idle_timeout_minutes),
        )
        self.response_plans = ResponsePlanRepository(database)
        self.literacy_repository = LiteracyRepository(database)
        self.literacy_engine = LiteracyEngine(self.literacy_repository)
        self.social_reaction_repository = SocialReactionRepository(database)
        self.social_reaction_engine = SocialReactionEngine(
            self.social_reaction_repository,
            self.cognitive_repository,
            self.genome,
        )
        self.experience_repository = ExperienceRepository(database)
        self.self_dataset = SelfDatasetRepository(database)
        self.arbitration_repository = ArbitrationRepository(
            database,
            learning_enabled=settings.arbitration_learning_enabled,
            learning_rate=settings.arbitration_learning_rate,
            minimum_feedback_observations=(
                settings.arbitration_minimum_feedback_observations
            ),
            maximum_weight_delta=settings.arbitration_maximum_weight_delta,
        )
        self.position_repository = PositionRepository(database)
        self.position_reasoning = PositionReasoningService(self.position_repository)
        self.abstract_reasoning = AbstractReasoningService(database)
        self.lexical_usage_learning = LexicalUsageLearningService(database, self.self_dataset)
        self.utterance_understandings = UtteranceUnderstandingRepository(database)
        self.context_gate = WorkingMemoryContextGate(
            self.self_dataset,
            window_size=settings.context_gate_window_turns,
            minimum_matches=settings.context_gate_minimum_matches,
            initiative_idle=timedelta(seconds=settings.context_gate_initiative_idle_seconds),
        )
        self.open_language_questions = OpenLanguageQuestionService(
            database,
            self.lexicon,
            self.self_dataset,
            ttl_messages=settings.open_question_ttl_messages,
            working_context=self.context_gate,
        )
        self.reflection_repository = ReflectionRepository(database)
        self.reflection_engine = ReflectionEngine(
            self.reflection_repository,
            self.experience_repository,
        )
        self.experience_consolidator = ExperienceConsolidator(
            self.experience_repository,
            reflection=self.reflection_engine,
        )
        self.short_speech_acts = ShortSpeechActService(
            database,
            self.experience_consolidator,
            self.self_dataset,
            working_context=self.context_gate,
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
            dictionary=self.russian_reference,
            reply_wait_seconds=settings.wikipedia_reply_wait_milliseconds / 1000,
        )
        self.semantic_assimilation = SemanticAssimilationService(
            database,
            self.self_dataset,
        )
        if settings.semantic_consultant_provider == "cloudflare":
            semantic_provider = CloudflareWorkersAIConsultant(
                settings.semantic_consultant_account_id or "",
                settings.semantic_consultant_api_token or "",
                model=settings.semantic_consultant_model,
                instruction=settings.semantic_consultant_instruction_path.read_text(
                    encoding="utf-8"
                ),
                timeout_seconds=settings.semantic_consultant_timeout_seconds,
            )
        else:
            semantic_provider = PuterSidecarConsultant(
                settings.semantic_consultant_endpoint,
                model=settings.semantic_consultant_model,
                timeout_seconds=settings.semantic_consultant_timeout_seconds,
                shared_secret=settings.semantic_consultant_secret,
            )
        self.semantic_consultant = SemanticConsultationService(
            database,
            semantic_provider,
            enabled=settings.semantic_consultant_enabled,
            trigger_threshold=settings.semantic_consultant_trigger_threshold,
            minimum_result_confidence=(
                settings.semantic_consultant_minimum_confidence
            ),
            assimilator=self.semantic_assimilation,
        )
        self.background_learning = BackgroundLearningService(
            database,
            self.lexicon,
            self.russian_reference,
            self.wikipedia_client,
            self.self_dataset,
            self.long_term_memory,
            self.autonomy_repository,
            sender=self._send_background_message,
            usage_learning=self.lexical_usage_learning,
            enabled=settings.background_enabled,
            interval=timedelta(seconds=settings.background_interval_seconds),
            words_per_cycle=settings.background_words_per_cycle,
            study_worker_count=settings.background_study_workers,
            maximum_study_attempts=settings.background_max_study_attempts,
            wikipedia_enabled=settings.wikipedia_enabled,
            # The external teacher may expand abstract constructions only
            # while idle. The live reply path remains entirely local and
            # never forwards a generated answer from Puter.
            semantic_consultant=self.semantic_consultant,
            causal_reasoning=self.causal_reasoning,
            abstract_reasoning=self.abstract_reasoning,
            graph_decay_daily_rate=settings.background_graph_decay_daily_rate,
            self_check_every_cycles=settings.background_self_check_cycles,
            initiative_enabled=(
                settings.initiative_enabled and settings.background_initiative_enabled
            ),
            initiative_probability=settings.background_initiative_probability,
            initiative_cooldown=timedelta(
                minutes=settings.background_initiative_cooldown_minutes
            ),
        )
        self.lexicon.set_study_notifier(
            self.background_learning.wake_for_study
        )
        self.reply_service = SafeMentionReplyService(
            self.cognitive_repository,
            self.memory,
            self.archive_repository,
            self.personality_expression,
            self.experience_consolidator,
            self.initiative_policy,
            GroundedConstructionGenerator(
                max_output_chars=settings.language_max_output_chars,
            ),
            social_reactions=self.social_reaction_repository,
            reflection=self.reflection_engine,
            literacy=self.literacy_engine,
            discourse_memory=self.discourse_memory,
            semantic_memory=self.semantic_memory,
            causal_reasoning=self.causal_reasoning,
            knowledge_service=self.knowledge_service,
            lexicon=self.lexicon,
            conversation_memory=self.conversation_memory,
            response_plans=self.response_plans,
            long_term_memory=self.long_term_memory,
            self_dataset=self.self_dataset,
            russian_reference=self.russian_reference,
            open_questions=self.open_language_questions,
            short_speech_acts=self.short_speech_acts,
            usage_learning=self.lexical_usage_learning,
            understanding_repository=self.utterance_understandings,
            arbitration_repository=self.arbitration_repository,
            context_gate=self.context_gate,
            # The consultant teaches structured facts when local sources are
            # exhausted. Its ready-made wording is still never surfaced.
            semantic_consultant=self.semantic_consultant,
            semantic_assimilation=self.semantic_assimilation,
            position_reasoning=self.position_reasoning,
            abstract_reasoning=self.abstract_reasoning,
            arbitration_mode=settings.arbitration_mode,
            arbitration_minimum_score=settings.arbitration_minimum_score,
            arbitration_max_total_candidates=(
                settings.arbitration_max_total_candidates
            ),
            response_confidence_threshold=settings.response_confidence_threshold,
            response_plan_retry_limit=settings.response_plan_retry_limit,
            context_character_limit=settings.language_context_chars,
            enabled=settings.test_replies_enabled,
        )
        self.pipeline = EventPipeline(
            repository,
            [
                self.short_speech_acts.observe,
                self.arbitration_repository.observe_feedback,
                self.position_repository.observe_feedback,
                self.memory.observe,
                self.context_gate.observe,
                self.long_term_memory.observe,
                self.discourse_memory.observe,
                self.semantic_memory.observe,
                self.causal_reasoning.observe,
                self.character_markov.observe,
                self.lexicon.observe,
                self.open_language_questions.observe,
                self.russian_reference.observe,
                self.morphology.observe,
                self.conversation_memory.observe,
                self.literacy_engine.observe,
                self.learning_engine.observe,
                self.cognitive_core.observe,
                self.social_reaction_engine.observe,
                self.experience_consolidator.observe,
                self.self_dataset.observe,
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
        semantic_lessons = await self.semantic_assimilation.backfill()
        if semantic_lessons:
            logger.info(
                "Backfilled durable lessons from %d semantic consultations",
                semantic_lessons,
            )
        productive_constructions = (
            await self.semantic_assimilation.backfill_productive_constructions()
        )
        if productive_constructions:
            logger.info(
                "Backfilled %d reusable productive constructions",
                productive_constructions,
            )
        if self.settings.dev_guild_id is not None:
            social_curriculum = await self.semantic_assimilation.install_social_curriculum(
                guild_id=self.settings.dev_guild_id,
            )
            productive_curriculum = (
                await self.semantic_assimilation.install_productive_curriculum(
                    guild_id=self.settings.dev_guild_id,
                )
            )
            if social_curriculum or productive_curriculum:
                logger.info(
                    "Installed abstract curricula: social=%d productive=%d",
                    social_curriculum,
                    productive_curriculum,
                )
        cutoff = datetime.now(UTC) - self.memory.limits.retention
        recent_events = await self.repository.since(cutoff, self.memory.limits.max_total_events)
        await self.memory.rebuild(recent_events)
        for event in recent_events:
            await self.context_gate.observe(event)
            await self.discourse_memory.observe(event)
            await self.social_reaction_engine.observe(event)
            await self.literacy_engine.observe(event)
        await self.learning_engine.start()
        await self.cognitive_core.start()
        await self.knowledge_service.start()
        await self.russian_reference.start()
        await self.pipeline.start()
        await self.open_language_questions.restore_working_context()
        await self.add_cog(
            EventListener(self, self.pipeline, self.reply_service, self.context_gate)
        )
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
        try:
            await self.background_learning.start()
        except Exception:
            logger.exception("Failed to start background learning")

    async def _send_background_message(
        self, guild_id: int, channel_id: int, text: str
    ) -> int | None:
        decision = await self.context_gate.evaluate(
            guild_id=guild_id,
            channel_id=channel_id,
            candidate_text=text,
            source="background",
        )
        if decision.action == "deferred":
            raise ContextGateDeferred(decision.reason)
        if not decision.allowed or decision.text is None:
            return None
        text = decision.text
        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                return None
        channel_guild = getattr(channel, "guild", None)
        if channel_guild is not None and channel_guild.id != guild_id:
            return None
        send = getattr(channel, "send", None)
        if send is None:
            return None
        sent = await send(text, allowed_mentions=discord.AllowedMentions.none())
        await self.context_gate.commit_outgoing(
            guild_id=guild_id,
            channel_id=channel_id,
            message_id=int(sent.id),
            text=text,
            bot_user_id=self.user.id if self.user is not None else None,
            target_user_id=None,
            observed_at=sent.created_at,
        )
        return int(sent.id)

    async def on_command_error(
        self, context: commands.Context[commands.Bot], error: commands.CommandError
    ) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        await super().on_command_error(context, error)

    async def close(self) -> None:
        await self.background_learning.stop()
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
                        await self.russian_reference.stop()
                    finally:
                        try:
                            await self.knowledge_service.stop()
                        finally:
                            try:
                                await self.semantic_consultant.close()
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
