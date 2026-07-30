import json
import os
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from debilgi.secret_vault import (
    VAULT_NAME,
    VAULT_PASSPHRASE_ENV,
    VaultConfigurationError,
    VaultUnlockError,
    create_vault,
    decrypt_vault,
    prepare_runtime_secrets,
    reset_vault,
)

PASSPHRASE = "correct horse battery staple"


def test_vault_round_trip_contains_no_plaintext_secret(tmp_path: Path) -> None:
    path = tmp_path / ".debilgi-secrets.vault"
    secrets = {
        "DISCORD_TOKEN": "discord-private-token",
        "DEBILGI_CLOUDFLARE_API_TOKEN": "cloud-private-token",
    }

    create_vault(path, secrets, PASSPHRASE)

    raw = path.read_text(encoding="utf-8")
    envelope = json.loads(raw)
    assert "discord-private-token" not in raw
    assert "cloud-private-token" not in raw
    assert envelope["cipher"]["name"] == "AES-256-GCM"
    assert envelope["kdf"]["name"] == "scrypt"
    assert decrypt_vault(path, PASSPHRASE) == secrets


def test_wrong_password_and_tampering_do_not_return_secrets(tmp_path: Path) -> None:
    path = tmp_path / ".debilgi-secrets.vault"
    create_vault(path, {"DISCORD_TOKEN": "private"}, PASSPHRASE)

    with pytest.raises(VaultUnlockError):
        decrypt_vault(path, "wrong password with enough length")

    envelope = json.loads(path.read_text(encoding="utf-8"))
    ciphertext = envelope["ciphertext"]
    envelope["ciphertext"] = f"{ciphertext[:-2]}AA"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises((VaultUnlockError, VaultConfigurationError, InvalidTag)):
        decrypt_vault(path, PASSPHRASE)


def test_vault_never_overwrites_and_reset_requires_exact_confirmation(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".debilgi-secrets.vault"
    create_vault(path, {"DISCORD_TOKEN": "first"}, PASSPHRASE)

    with pytest.raises(VaultConfigurationError, match="already exists"):
        create_vault(path, {"DISCORD_TOKEN": "second"}, PASSPHRASE)
    with pytest.raises(VaultConfigurationError, match="confirmation"):
        reset_vault(path, confirmation="yes")

    assert decrypt_vault(path, PASSPHRASE)["DISCORD_TOKEN"] == "first"
    reset_vault(path, confirmation="RESET")
    assert not path.exists()


def test_vault_rejects_short_password_and_invalid_secret_names(tmp_path: Path) -> None:
    path = tmp_path / ".debilgi-secrets.vault"
    with pytest.raises(VaultConfigurationError, match="at least"):
        create_vault(path, {"DISCORD_TOKEN": "secret"}, "short")
    with pytest.raises(VaultConfigurationError, match="Invalid secret"):
        create_vault(
            path,
            {"DISCORD_TOKEN": "secret", "bad-name": "value"},
            PASSPHRASE,
        )


def test_runtime_unlock_injects_secrets_without_printing_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / VAULT_NAME
    create_vault(
        path,
        {
            "DISCORD_TOKEN": "runtime-private-token",
            "ANOTHER_API_TOKEN": "second-private-token",
        },
        PASSPHRASE,
    )
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("ANOTHER_API_TOKEN", raising=False)
    monkeypatch.setenv(VAULT_PASSPHRASE_ENV, PASSPHRASE)

    prepare_runtime_secrets(tmp_path)

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert os.environ["DISCORD_TOKEN"] == "runtime-private-token"
    assert os.environ["ANOTHER_API_TOKEN"] == "second-private-token"


def test_noninteractive_fresh_install_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setattr(
        "debilgi.secret_vault._interactive_terminal_available",
        lambda: False,
    )

    with pytest.raises(VaultConfigurationError, match="interactive console"):
        prepare_runtime_secrets(tmp_path)
