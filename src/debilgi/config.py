"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    language_max_output_tokens: int = 72
    language_max_output_chars: int = 1200
    language_context_chars: int = 600
    discourse_context_turns: int = 80
    discourse_max_turns: int = 20_000
    memory_retention_minutes: int = 60
    memory_charge_half_life_minutes: int = 15
    memory_max_total_events: int = 5000
    memory_max_channel_events: int = 100
    memory_max_user_events: int = 50
    memory_max_reply_chains: int = 100
    memory_max_chain_events: int = 50
    expectation_horizon_seconds: int = 120
    expectation_poll_seconds: int = 5
    cognitive_episode_gap_seconds: int = 600
    cognitive_maintenance_seconds: int = 60
    cognitive_consolidation_batch_size: int = 100
    raw_text_retention_days: int = 30

    @classmethod
    def from_env(cls) -> Settings:
        root = _project_root()
        load_dotenv(root / ".env")

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
            language_max_output_tokens=_positive_int_from_env(
                "DEBILGI_LANGUAGE_MAX_OUTPUT_TOKENS", 72
            ),
            language_max_output_chars=_positive_int_from_env(
                "DEBILGI_LANGUAGE_MAX_OUTPUT_CHARS", 1200
            ),
            language_context_chars=_positive_int_from_env("DEBILGI_LANGUAGE_CONTEXT_CHARS", 600),
            discourse_context_turns=_positive_int_from_env("DEBILGI_DISCOURSE_CONTEXT_TURNS", 80),
            discourse_max_turns=_positive_int_from_env("DEBILGI_DISCOURSE_MAX_TURNS", 20_000),
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
