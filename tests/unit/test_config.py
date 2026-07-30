from pathlib import Path

import pytest

from debilgi.config import ConfigurationError, Settings


def test_missing_token_is_allowed_for_tooling_but_rejected_for_startup(tmp_path: Path) -> None:
    settings = Settings(
        discord_token=None,
        database_path=tmp_path / "db.sqlite3",
        migrations_path=tmp_path,
    )

    with pytest.raises(ConfigurationError, match="DISCORD_TOKEN"):
        settings.require_discord_token()


def test_token_is_returned_without_being_transformed(tmp_path: Path) -> None:
    settings = Settings(
        discord_token="secret",
        database_path=tmp_path / "db.sqlite3",
        migrations_path=tmp_path,
    )

    assert settings.require_discord_token() == "secret"


def test_machine_local_config_overrides_shared_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "DISCORD_TOKEN=shared-placeholder\nDEBILGI_LOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "DISCORD_TOKEN=machine-secret\nDEBILGI_LOG_LEVEL=DEBUG\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("debilgi.config._project_root", lambda: tmp_path)
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("DEBILGI_LOG_LEVEL", raising=False)

    settings = Settings.from_env()

    assert settings.discord_token == "machine-secret"
    assert settings.log_level == "DEBUG"


def test_process_environment_overrides_machine_local_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env.local").write_text(
        "DISCORD_TOKEN=machine-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("debilgi.config._project_root", lambda: tmp_path)
    monkeypatch.setenv("DISCORD_TOKEN", "process-secret")

    assert Settings.from_env().discord_token == "process-secret"


def test_memory_environment_limits_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_MEMORY_MAX_TOTAL_EVENTS", "0")

    with pytest.raises(ConfigurationError, match="must be positive"):
        Settings.from_env()


def test_memory_lifecycle_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_MEMORY_MAX_CONTEXT_EVENTS", "18")
    monkeypatch.setenv("DEBILGI_MEMORY_RELEVANCE_HALF_LIFE_MINUTES", "12")
    monkeypatch.setenv("DEBILGI_MEMORY_MIN_CONTEXT_RELEVANCE", "0.21")
    monkeypatch.setenv("DEBILGI_MEMORY_CONSOLIDATION_MIN_IMPORTANCE", "0.44")
    monkeypatch.setenv("DEBILGI_MEMORY_CONSOLIDATION_BATCH_SIZE", "8")
    monkeypatch.setenv("DEBILGI_MEMORY_LONG_TERM_CAPSULE_LIMIT", "2400")
    monkeypatch.setenv("DEBILGI_MEMORY_LONG_TERM_RECALL_LIMIT", "4")

    settings = Settings.from_env()

    assert settings.memory_max_context_events == 18
    assert settings.memory_relevance_half_life_minutes == 12
    assert settings.memory_min_context_relevance == pytest.approx(0.21)
    assert settings.memory_consolidation_min_importance == pytest.approx(0.44)
    assert settings.memory_consolidation_batch_size == 8
    assert settings.memory_long_term_capsule_limit == 2400
    assert settings.memory_long_term_recall_limit == 4


def test_local_reply_mode_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_TEST_REPLIES", "true")

    assert Settings.from_env().test_replies_enabled


def test_productive_only_reply_experiment_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_PRODUCTIVE_ONLY_REPLIES", "true")

    assert Settings.from_env().productive_only_replies


def test_background_learning_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_BACKGROUND_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_BACKGROUND_INTERVAL_SECONDS", "420")
    monkeypatch.setenv("DEBILGI_BACKGROUND_WORDS_PER_CYCLE", "7")
    monkeypatch.setenv("DEBILGI_BACKGROUND_STUDY_WORKERS", "6")
    monkeypatch.setenv("DEBILGI_BACKGROUND_MAX_STUDY_ATTEMPTS", "3")
    monkeypatch.setenv("DEBILGI_BACKGROUND_GRAPH_DECAY_DAILY_RATE", "0.02")
    monkeypatch.setenv("DEBILGI_BACKGROUND_SELF_CHECK_CYCLES", "4")
    monkeypatch.setenv("DEBILGI_BACKGROUND_INITIATIVE_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_BACKGROUND_INITIATIVE_PROBABILITY", "0.4")
    monkeypatch.setenv("DEBILGI_BACKGROUND_INITIATIVE_COOLDOWN_MINUTES", "90")

    settings = Settings.from_env()

    assert settings.background_enabled is True
    assert settings.background_interval_seconds == 420
    assert settings.background_words_per_cycle == 7
    assert settings.background_study_workers == 6
    assert settings.background_max_study_attempts == 3
    assert settings.background_graph_decay_daily_rate == pytest.approx(0.02)
    assert settings.background_self_check_cycles == 4
    assert settings.background_initiative_enabled is True
    assert settings.background_initiative_probability == pytest.approx(0.4)
    assert settings.background_initiative_cooldown_minutes == 90


def test_continuous_cognition_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_CONTINUOUS_COGNITION_ENABLED", "false")
    monkeypatch.setenv("DEBILGI_CONTINUOUS_COGNITION_ACTIVE_SECONDS", "3")
    monkeypatch.setenv("DEBILGI_CONTINUOUS_COGNITION_IDLE_SECONDS", "20")
    monkeypatch.setenv("DEBILGI_CONTINUOUS_COGNITION_WORK_PER_PULSE", "11")

    settings = Settings.from_env()

    assert settings.continuous_cognition_enabled is False
    assert settings.continuous_cognition_active_seconds == 3
    assert settings.continuous_cognition_idle_seconds == 20
    assert settings.continuous_cognition_work_per_pulse == 11


def test_reference_worker_pool_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_RUSSIAN_REFERENCE_WORKERS", "5")

    assert Settings.from_env().russian_reference_workers == 5


def test_language_limits_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_LANGUAGE_MAX_OUTPUT_TOKENS", "90")
    monkeypatch.setenv("DEBILGI_LANGUAGE_MAX_OUTPUT_CHARS", "1500")
    monkeypatch.setenv("DEBILGI_LANGUAGE_CONTEXT_CHARS", "800")
    monkeypatch.setenv("DEBILGI_DISCOURSE_CONTEXT_TURNS", "120")
    monkeypatch.setenv("DEBILGI_DISCOURSE_MAX_TURNS", "30000")

    settings = Settings.from_env()
    assert settings.language_max_output_tokens == 90
    assert settings.language_max_output_chars == 1500
    assert settings.language_context_chars == 800
    assert settings.discourse_context_turns == 120
    assert settings.discourse_max_turns == 30000


def test_language_limits_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_LANGUAGE_CONTEXT_CHARS", "0")

    with pytest.raises(ConfigurationError, match="must be positive"):
        Settings.from_env()


def test_language_development_limits_are_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_CHARACTER_MARKOV_MIN_ORDER", "3")
    monkeypatch.setenv("DEBILGI_CHARACTER_MARKOV_MAX_ORDER", "7")
    monkeypatch.setenv("DEBILGI_CONVERSATION_IDLE_TIMEOUT_MINUTES", "35")
    monkeypatch.setenv("DEBILGI_CONVERSATION_MAX_IDLE_TIMEOUT_MINUTES", "55")
    monkeypatch.setenv("DEBILGI_CURIOSITY_THRESHOLD", "0.72")
    monkeypatch.setenv("DEBILGI_RESPONSE_CONFIDENCE_THRESHOLD", "0.51")
    monkeypatch.setenv("DEBILGI_RESPONSE_PLAN_RETRY_LIMIT", "3")
    monkeypatch.setenv("DEBILGI_OPEN_QUESTION_TTL_MESSAGES", "4")

    settings = Settings.from_env()

    assert settings.character_markov_min_order == 3
    assert settings.character_markov_max_order == 7
    assert settings.conversation_idle_timeout_minutes == 35
    assert settings.conversation_max_idle_timeout_minutes == 55
    assert settings.curiosity_threshold == pytest.approx(0.72)
    assert settings.response_confidence_threshold == pytest.approx(0.51)
    assert settings.response_plan_retry_limit == 3
    assert settings.open_question_ttl_messages == 4


def test_semantic_consultant_is_opt_in_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_SEMANTIC_CONSULTANT_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("DEBILGI_CLOUDFLARE_API_TOKEN", "secret-token")
    monkeypatch.setenv("DEBILGI_CLOUDFLARE_MODEL", "@cf/test/model")
    monkeypatch.setenv("DEBILGI_SEMANTIC_CONSULTANT_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("DEBILGI_SEMANTIC_CONSULTANT_TRIGGER_THRESHOLD", "0.66")
    monkeypatch.setenv("DEBILGI_SEMANTIC_CONSULTANT_MINIMUM_CONFIDENCE", "0.61")

    settings = Settings.from_env()

    assert settings.semantic_consultant_enabled
    assert settings.semantic_consultant_model == "@cf/test/model"
    assert settings.semantic_consultant_timeout_seconds == 9
    assert settings.semantic_consultant_trigger_threshold == pytest.approx(0.66)
    assert settings.semantic_consultant_minimum_confidence == pytest.approx(0.61)


def test_cloudflare_semantic_consultant_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_SEMANTIC_CONSULTANT_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_CLOUDFLARE_ACCOUNT_ID", "account-id")
    monkeypatch.setenv("DEBILGI_CLOUDFLARE_API_TOKEN", "secret-token")
    monkeypatch.setenv(
        "DEBILGI_CLOUDFLARE_MODEL",
        "@cf/meta/llama-3.1-8b-instruct-fast",
    )

    settings = Settings.from_env()

    assert settings.semantic_consultant_account_id == "account-id"
    assert settings.semantic_consultant_api_token == "secret-token"
    assert "secret-token" not in repr(settings)
    assert settings.semantic_consultant_instruction_path.is_file()


def test_context_gate_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_CONTEXT_GATE_WINDOW_TURNS", "4")
    monkeypatch.setenv("DEBILGI_CONTEXT_GATE_MINIMUM_MATCHES", "2")
    monkeypatch.setenv("DEBILGI_CONTEXT_GATE_INITIATIVE_IDLE_SECONDS", "240")

    settings = Settings.from_env()

    assert settings.context_gate_window_turns == 4
    assert settings.context_gate_minimum_matches == 2
    assert settings.context_gate_initiative_idle_seconds == 240


def test_response_arbitration_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_ARBITRATION_MODE", "shadow")
    monkeypatch.setenv("DEBILGI_ARBITRATION_MIN_SCORE", "0.52")
    monkeypatch.setenv("DEBILGI_ARBITRATION_MAX_TOTAL_CANDIDATES", "36")
    monkeypatch.setenv("DEBILGI_ARBITRATION_LEARNING_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_ARBITRATION_LEARNING_RATE", "0.02")
    monkeypatch.setenv("DEBILGI_ARBITRATION_MIN_FEEDBACK_OBSERVATIONS", "30")
    monkeypatch.setenv("DEBILGI_ARBITRATION_MAX_WEIGHT_DELTA", "0.04")

    settings = Settings.from_env()

    assert settings.arbitration_mode == "shadow"
    assert settings.arbitration_minimum_score == pytest.approx(0.52)
    assert settings.arbitration_max_total_candidates == 36
    assert settings.arbitration_learning_enabled is True
    assert settings.arbitration_learning_rate == pytest.approx(0.02)
    assert settings.arbitration_minimum_feedback_observations == 30
    assert settings.arbitration_maximum_weight_delta == pytest.approx(0.04)


def test_invalid_response_arbitration_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_ARBITRATION_MODE", "random")

    with pytest.raises(ConfigurationError, match="off, shadow or active"):
        Settings.from_env()


def test_language_development_ranges_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_CHARACTER_MARKOV_MIN_ORDER", "9")
    monkeypatch.setenv("DEBILGI_CHARACTER_MARKOV_MAX_ORDER", "8")

    with pytest.raises(ConfigurationError, match="minimum order"):
        Settings.from_env()


def test_wikipedia_mode_and_language_are_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBILGI_WIKIPEDIA_ENABLED", "true")
    monkeypatch.setenv("DEBILGI_WIKIPEDIA_LANGUAGE", "ru")

    settings = Settings.from_env()
    assert settings.wikipedia_enabled
    assert settings.wikipedia_language == "ru"


def test_invalid_wikipedia_language_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_WIKIPEDIA_LANGUAGE", "ru_123")

    with pytest.raises(ConfigurationError, match="language code"):
        Settings.from_env()
