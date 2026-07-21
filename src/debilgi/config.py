"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing or invalid."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str, root: Path) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else root / value


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {value!r}")


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _unit_float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    discord_token: str | None
    database_path: Path
    migrations_path: Path
    log_level: str = "INFO"
    sync_commands: bool = False
    dev_guild_id: int | None = None
    test_replies_enabled: bool = False
    initiative_enabled: bool = False
    wikipedia_enabled: bool = False
    wikipedia_language: str = "ru"
    wikipedia_timeout_seconds: int = 5
    wikipedia_reply_wait_milliseconds: int = 1800
    wikipedia_user_agent: str = "DEBiLGI/0.4 (mailto:local-operator@example.invalid)"
    russian_reference_enabled: bool = False
    russian_reference_timeout_seconds: int = 5
    russian_reference_user_agent: str = "DEBiLGI/0.6 (mailto:local-operator@example.invalid)"
    language_max_output_tokens: int = 72
    language_max_output_chars: int = 1200
    language_context_chars: int = 600
    discourse_context_turns: int = 80
    discourse_max_turns: int = 20_000
    character_markov_min_order: int = 2
    character_markov_max_order: int = 8
    character_markov_max_word_length: int = 32
    character_markov_generation_attempts: int = 24
    conversation_idle_timeout_minutes: int = 45
    conversation_max_idle_timeout_minutes: int = 60
    neologism_min_response_interval: int = 75
    neologism_max_per_message: int = 1
    neologism_expiration_days: int = 30
    curiosity_question_cooldown_minutes: int = 30
    curiosity_max_consecutive_questions: int = 1
    curiosity_threshold: float = 0.68
    unknown_word_threshold: float = 0.45
    response_confidence_threshold: float = 0.45
    response_plan_retry_limit: int = 2
    semantic_consultant_enabled: bool = False
    semantic_consultant_provider: str = "puter"
    semantic_consultant_endpoint: str = "http://127.0.0.1:43127/v1/consult"
    semantic_consultant_model: str = "google/gemini-3.5-flash"
    semantic_consultant_account_id: str | None = None
    semantic_consultant_api_token: str | None = field(default=None, repr=False)
    semantic_consultant_instruction_path: Path = Path(
        "integrations/puter-sidecar/semantic-instruction.txt"
    )
    semantic_consultant_timeout_seconds: int = 45
    semantic_consultant_trigger_threshold: float = 0.72
    semantic_consultant_minimum_confidence: float = 0.58
    semantic_consultant_secret: str | None = None
    arbitration_mode: str = "active"
    arbitration_minimum_score: float = 0.45
    arbitration_max_total_candidates: int = 48
    arbitration_learning_enabled: bool = False
    arbitration_learning_rate: float = 0.03
    arbitration_minimum_feedback_observations: int = 20
    arbitration_maximum_weight_delta: float = 0.05
    open_question_ttl_messages: int = 2
    context_gate_window_turns: int = 5
    context_gate_minimum_matches: int = 1
    context_gate_initiative_idle_seconds: int = 300
    background_enabled: bool = False
    background_interval_seconds: int = 600
    background_words_per_cycle: int = 5
    background_max_study_attempts: int = 4
    background_graph_decay_daily_rate: float = 0.015
    background_self_check_cycles: int = 6
    background_initiative_enabled: bool = False
    background_initiative_probability: float = 0.25
    background_initiative_cooldown_minutes: int = 60
    memory_retention_minutes: int = 60
    memory_charge_half_life_minutes: int = 15
    memory_max_total_events: int = 5000
    memory_max_channel_events: int = 100
    memory_max_user_events: int = 50
    memory_max_reply_chains: int = 100
    memory_max_chain_events: int = 50
    memory_max_context_events: int = 24
    memory_relevance_half_life_minutes: int = 20
    memory_min_context_relevance: float = 0.10
    memory_consolidation_min_importance: float = 0.34
    memory_consolidation_batch_size: int = 12
    memory_long_term_capsule_limit: int = 5_000
    memory_long_term_recall_limit: int = 6
    expectation_horizon_seconds: int = 120
    expectation_poll_seconds: int = 5
    cognitive_episode_gap_seconds: int = 600
    cognitive_maintenance_seconds: int = 60
    cognitive_consolidation_batch_size: int = 100
    raw_text_retention_days: int = 30

    def __post_init__(self) -> None:
        if self.character_markov_min_order > self.character_markov_max_order:
            raise ConfigurationError("character Markov minimum order cannot exceed maximum")
        if self.character_markov_max_order > 16:
            raise ConfigurationError("character Markov maximum order cannot exceed 16")
        if self.conversation_idle_timeout_minutes > self.conversation_max_idle_timeout_minutes:
            raise ConfigurationError("conversation idle timeout cannot exceed maximum")

    @classmethod
    def from_env(cls) -> Settings:
        root = _project_root()
        load_dotenv(root / ".env")
        load_dotenv(root / ".env.local")

        raw_guild_id = os.getenv("DEBILGI_DEV_GUILD_ID")
        try:
            dev_guild_id = int(raw_guild_id) if raw_guild_id else None
        except ValueError as exc:
            raise ConfigurationError("DEBILGI_DEV_GUILD_ID must be an integer") from exc

        log_level = os.getenv("DEBILGI_LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(f"Unsupported DEBILGI_LOG_LEVEL: {log_level}")

        wikipedia_language = os.getenv("DEBILGI_WIKIPEDIA_LANGUAGE", "ru").strip().lower()
        if not wikipedia_language or not wikipedia_language.replace("-", "").isalpha():
            raise ConfigurationError("DEBILGI_WIKIPEDIA_LANGUAGE must be a language code")
        arbitration_mode = os.getenv("DEBILGI_ARBITRATION_MODE", "active").strip().lower()
        if arbitration_mode not in {"off", "shadow", "active"}:
            raise ConfigurationError(
                "DEBILGI_ARBITRATION_MODE must be off, shadow or active"
            )
        consultant_provider = os.getenv(
            "DEBILGI_SEMANTIC_CONSULTANT_PROVIDER", "puter"
        ).strip().lower()
        if consultant_provider not in {"puter", "cloudflare"}:
            raise ConfigurationError(
                "DEBILGI_SEMANTIC_CONSULTANT_PROVIDER must be puter or cloudflare"
            )
        consultant_endpoint = os.getenv(
            "DEBILGI_SEMANTIC_CONSULTANT_ENDPOINT",
            "http://127.0.0.1:43127/v1/consult",
        ).strip()
        if consultant_provider == "puter" and not consultant_endpoint.startswith(
            ("http://", "https://")
        ):
            raise ConfigurationError(
                "DEBILGI_SEMANTIC_CONSULTANT_ENDPOINT must be an HTTP URL"
            )
        consultant_model = (
            os.getenv(
                "DEBILGI_CLOUDFLARE_MODEL",
                "@cf/meta/llama-3.1-8b-instruct-fast",
            )
            if consultant_provider == "cloudflare"
            else os.getenv(
                "DEBILGI_SEMANTIC_CONSULTANT_MODEL",
                "google/gemini-3.5-flash",
            )
        ).strip()
        if not consultant_model:
            raise ConfigurationError("DEBILGI_SEMANTIC_CONSULTANT_MODEL cannot be empty")
        consultant_account_id = (
            os.getenv("DEBILGI_CLOUDFLARE_ACCOUNT_ID")
            or os.getenv("CLOUDFLARE_ACCOUNT_ID")
            or ""
        ).strip() or None
        consultant_api_token = (
            os.getenv("DEBILGI_CLOUDFLARE_API_TOKEN")
            or os.getenv("CLOUDFLARE_API_TOKEN")
            or ""
        ).strip() or None
        consultant_instruction_path = _path_from_env(
            "DEBILGI_SEMANTIC_CONSULTANT_INSTRUCTION_PATH",
            "integrations/puter-sidecar/semantic-instruction.txt",
            root,
        )
        consultant_enabled = _bool_from_env("DEBILGI_SEMANTIC_CONSULTANT_ENABLED")
        if consultant_enabled and consultant_provider == "cloudflare":
            if consultant_account_id is None or consultant_api_token is None:
                raise ConfigurationError(
                    "Cloudflare semantic consultant requires Account ID and API token"
                )
            if not consultant_instruction_path.is_file():
                raise ConfigurationError(
                    "DEBILGI_SEMANTIC_CONSULTANT_INSTRUCTION_PATH must be a file"
                )

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN") or None,
            database_path=_path_from_env("DEBILGI_DATABASE_PATH", "data/debilgi.sqlite3", root),
            migrations_path=_path_from_env("DEBILGI_MIGRATIONS_PATH", "migrations", root),
            log_level=log_level,
            sync_commands=_bool_from_env("DEBILGI_SYNC_COMMANDS"),
            dev_guild_id=dev_guild_id,
            test_replies_enabled=_bool_from_env("DEBILGI_TEST_REPLIES"),
            initiative_enabled=_bool_from_env("DEBILGI_INITIATIVE"),
            wikipedia_enabled=_bool_from_env("DEBILGI_WIKIPEDIA_ENABLED"),
            wikipedia_language=wikipedia_language,
            wikipedia_timeout_seconds=_positive_int_from_env(
                "DEBILGI_WIKIPEDIA_TIMEOUT_SECONDS", 5
            ),
            wikipedia_reply_wait_milliseconds=_positive_int_from_env(
                "DEBILGI_WIKIPEDIA_REPLY_WAIT_MILLISECONDS", 1800
            ),
            wikipedia_user_agent=os.getenv(
                "DEBILGI_WIKIPEDIA_USER_AGENT",
                "DEBiLGI/0.4 (mailto:local-operator@example.invalid)",
            ).strip(),
            russian_reference_enabled=_bool_from_env("DEBILGI_RUSSIAN_REFERENCE_ENABLED"),
            russian_reference_timeout_seconds=_positive_int_from_env(
                "DEBILGI_RUSSIAN_REFERENCE_TIMEOUT_SECONDS", 5
            ),
            russian_reference_user_agent=os.getenv(
                "DEBILGI_RUSSIAN_REFERENCE_USER_AGENT",
                "DEBiLGI/0.6 (mailto:local-operator@example.invalid)",
            ).strip(),
            language_max_output_tokens=_positive_int_from_env(
                "DEBILGI_LANGUAGE_MAX_OUTPUT_TOKENS", 72
            ),
            language_max_output_chars=_positive_int_from_env(
                "DEBILGI_LANGUAGE_MAX_OUTPUT_CHARS", 1200
            ),
            language_context_chars=_positive_int_from_env("DEBILGI_LANGUAGE_CONTEXT_CHARS", 600),
            discourse_context_turns=_positive_int_from_env("DEBILGI_DISCOURSE_CONTEXT_TURNS", 80),
            discourse_max_turns=_positive_int_from_env("DEBILGI_DISCOURSE_MAX_TURNS", 20_000),
            character_markov_min_order=_positive_int_from_env(
                "DEBILGI_CHARACTER_MARKOV_MIN_ORDER", 2
            ),
            character_markov_max_order=_positive_int_from_env(
                "DEBILGI_CHARACTER_MARKOV_MAX_ORDER", 8
            ),
            character_markov_max_word_length=_positive_int_from_env(
                "DEBILGI_CHARACTER_MARKOV_MAX_WORD_LENGTH", 32
            ),
            character_markov_generation_attempts=_positive_int_from_env(
                "DEBILGI_CHARACTER_MARKOV_GENERATION_ATTEMPTS", 24
            ),
            conversation_idle_timeout_minutes=_positive_int_from_env(
                "DEBILGI_CONVERSATION_IDLE_TIMEOUT_MINUTES", 45
            ),
            conversation_max_idle_timeout_minutes=_positive_int_from_env(
                "DEBILGI_CONVERSATION_MAX_IDLE_TIMEOUT_MINUTES", 60
            ),
            neologism_min_response_interval=_positive_int_from_env(
                "DEBILGI_NEOLOGISM_MIN_RESPONSE_INTERVAL", 75
            ),
            neologism_max_per_message=_positive_int_from_env(
                "DEBILGI_NEOLOGISM_MAX_PER_MESSAGE", 1
            ),
            neologism_expiration_days=_positive_int_from_env(
                "DEBILGI_NEOLOGISM_EXPIRATION_DAYS", 30
            ),
            curiosity_question_cooldown_minutes=_positive_int_from_env(
                "DEBILGI_CURIOSITY_QUESTION_COOLDOWN_MINUTES", 30
            ),
            curiosity_max_consecutive_questions=_positive_int_from_env(
                "DEBILGI_CURIOSITY_MAX_CONSECUTIVE_QUESTIONS", 1
            ),
            curiosity_threshold=_unit_float_from_env("DEBILGI_CURIOSITY_THRESHOLD", 0.68),
            unknown_word_threshold=_unit_float_from_env("DEBILGI_UNKNOWN_WORD_THRESHOLD", 0.45),
            response_confidence_threshold=_unit_float_from_env(
                "DEBILGI_RESPONSE_CONFIDENCE_THRESHOLD", 0.45
            ),
            response_plan_retry_limit=_positive_int_from_env(
                "DEBILGI_RESPONSE_PLAN_RETRY_LIMIT", 2
            ),
            semantic_consultant_enabled=consultant_enabled,
            semantic_consultant_provider=consultant_provider,
            semantic_consultant_endpoint=consultant_endpoint,
            semantic_consultant_model=consultant_model,
            semantic_consultant_account_id=consultant_account_id,
            semantic_consultant_api_token=consultant_api_token,
            semantic_consultant_instruction_path=consultant_instruction_path,
            semantic_consultant_timeout_seconds=_positive_int_from_env(
                "DEBILGI_SEMANTIC_CONSULTANT_TIMEOUT_SECONDS", 45
            ),
            semantic_consultant_trigger_threshold=_unit_float_from_env(
                "DEBILGI_SEMANTIC_CONSULTANT_TRIGGER_THRESHOLD", 0.72
            ),
            semantic_consultant_minimum_confidence=_unit_float_from_env(
                "DEBILGI_SEMANTIC_CONSULTANT_MINIMUM_CONFIDENCE", 0.58
            ),
            semantic_consultant_secret=(
                os.getenv("DEBILGI_SEMANTIC_CONSULTANT_SECRET", "").strip() or None
            ),
            arbitration_mode=arbitration_mode,
            arbitration_minimum_score=_unit_float_from_env(
                "DEBILGI_ARBITRATION_MIN_SCORE", 0.45
            ),
            arbitration_max_total_candidates=_positive_int_from_env(
                "DEBILGI_ARBITRATION_MAX_TOTAL_CANDIDATES", 48
            ),
            arbitration_learning_enabled=_bool_from_env(
                "DEBILGI_ARBITRATION_LEARNING_ENABLED", True
            ),
            arbitration_learning_rate=_unit_float_from_env(
                "DEBILGI_ARBITRATION_LEARNING_RATE", 0.03
            ),
            arbitration_minimum_feedback_observations=_positive_int_from_env(
                "DEBILGI_ARBITRATION_MIN_FEEDBACK_OBSERVATIONS", 20
            ),
            arbitration_maximum_weight_delta=_unit_float_from_env(
                "DEBILGI_ARBITRATION_MAX_WEIGHT_DELTA", 0.05
            ),
            open_question_ttl_messages=_positive_int_from_env(
                "DEBILGI_OPEN_QUESTION_TTL_MESSAGES", 2
            ),
            context_gate_window_turns=_positive_int_from_env(
                "DEBILGI_CONTEXT_GATE_WINDOW_TURNS", 5
            ),
            context_gate_minimum_matches=_positive_int_from_env(
                "DEBILGI_CONTEXT_GATE_MINIMUM_MATCHES", 1
            ),
            context_gate_initiative_idle_seconds=_positive_int_from_env(
                "DEBILGI_CONTEXT_GATE_INITIATIVE_IDLE_SECONDS", 300
            ),
            background_enabled=_bool_from_env("DEBILGI_BACKGROUND_ENABLED"),
            background_interval_seconds=_positive_int_from_env(
                "DEBILGI_BACKGROUND_INTERVAL_SECONDS", 600
            ),
            background_words_per_cycle=_positive_int_from_env(
                "DEBILGI_BACKGROUND_WORDS_PER_CYCLE", 5
            ),
            background_max_study_attempts=_positive_int_from_env(
                "DEBILGI_BACKGROUND_MAX_STUDY_ATTEMPTS", 4
            ),
            background_graph_decay_daily_rate=_unit_float_from_env(
                "DEBILGI_BACKGROUND_GRAPH_DECAY_DAILY_RATE", 0.015
            ),
            background_self_check_cycles=_positive_int_from_env(
                "DEBILGI_BACKGROUND_SELF_CHECK_CYCLES", 6
            ),
            background_initiative_enabled=_bool_from_env(
                "DEBILGI_BACKGROUND_INITIATIVE_ENABLED"
            ),
            background_initiative_probability=_unit_float_from_env(
                "DEBILGI_BACKGROUND_INITIATIVE_PROBABILITY", 0.25
            ),
            background_initiative_cooldown_minutes=_positive_int_from_env(
                "DEBILGI_BACKGROUND_INITIATIVE_COOLDOWN_MINUTES", 60
            ),
            memory_retention_minutes=_positive_int_from_env("DEBILGI_MEMORY_RETENTION_MINUTES", 60),
            memory_charge_half_life_minutes=_positive_int_from_env(
                "DEBILGI_MEMORY_CHARGE_HALF_LIFE_MINUTES", 15
            ),
            memory_max_total_events=_positive_int_from_env("DEBILGI_MEMORY_MAX_TOTAL_EVENTS", 5000),
            memory_max_channel_events=_positive_int_from_env(
                "DEBILGI_MEMORY_MAX_CHANNEL_EVENTS", 100
            ),
            memory_max_user_events=_positive_int_from_env("DEBILGI_MEMORY_MAX_USER_EVENTS", 50),
            memory_max_reply_chains=_positive_int_from_env("DEBILGI_MEMORY_MAX_REPLY_CHAINS", 100),
            memory_max_chain_events=_positive_int_from_env("DEBILGI_MEMORY_MAX_CHAIN_EVENTS", 50),
            memory_max_context_events=_positive_int_from_env(
                "DEBILGI_MEMORY_MAX_CONTEXT_EVENTS", 24
            ),
            memory_relevance_half_life_minutes=_positive_int_from_env(
                "DEBILGI_MEMORY_RELEVANCE_HALF_LIFE_MINUTES", 20
            ),
            memory_min_context_relevance=_unit_float_from_env(
                "DEBILGI_MEMORY_MIN_CONTEXT_RELEVANCE", 0.10
            ),
            memory_consolidation_min_importance=_unit_float_from_env(
                "DEBILGI_MEMORY_CONSOLIDATION_MIN_IMPORTANCE", 0.34
            ),
            memory_consolidation_batch_size=_positive_int_from_env(
                "DEBILGI_MEMORY_CONSOLIDATION_BATCH_SIZE", 12
            ),
            memory_long_term_capsule_limit=_positive_int_from_env(
                "DEBILGI_MEMORY_LONG_TERM_CAPSULE_LIMIT", 5_000
            ),
            memory_long_term_recall_limit=_positive_int_from_env(
                "DEBILGI_MEMORY_LONG_TERM_RECALL_LIMIT", 6
            ),
            expectation_horizon_seconds=_positive_int_from_env(
                "DEBILGI_EXPECTATION_HORIZON_SECONDS", 120
            ),
            expectation_poll_seconds=_positive_int_from_env("DEBILGI_EXPECTATION_POLL_SECONDS", 5),
            cognitive_episode_gap_seconds=_positive_int_from_env(
                "DEBILGI_COGNITIVE_EPISODE_GAP_SECONDS", 600
            ),
            cognitive_maintenance_seconds=_positive_int_from_env(
                "DEBILGI_COGNITIVE_MAINTENANCE_SECONDS", 60
            ),
            cognitive_consolidation_batch_size=_positive_int_from_env(
                "DEBILGI_COGNITIVE_BATCH_SIZE", 100
            ),
            raw_text_retention_days=_positive_int_from_env("DEBILGI_RAW_TEXT_RETENTION_DAYS", 30),
        )

    def require_discord_token(self) -> str:
        if not self.discord_token:
            raise ConfigurationError(
                "DISCORD_TOKEN is missing. Copy .env.example to .env and set the token."
            )
        return self.discord_token
