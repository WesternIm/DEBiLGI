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


def test_memory_environment_limits_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_MEMORY_MAX_TOTAL_EVENTS", "0")

    with pytest.raises(ConfigurationError, match="must be positive"):
        Settings.from_env()


def test_local_reply_mode_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBILGI_TEST_REPLIES", "true")

    assert Settings.from_env().test_replies_enabled


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
