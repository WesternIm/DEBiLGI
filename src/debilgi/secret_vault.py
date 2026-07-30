"""Password-encrypted, Git-ignored storage for runtime credentials."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import re
import stat
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from dotenv import dotenv_values

VAULT_NAME = ".debilgi-secrets.vault"
VAULT_FORMAT = "debilgi-secret-vault"
VAULT_VERSION = 1
VAULT_PASSPHRASE_ENV = "DEBILGI_VAULT_PASSPHRASE"
_AAD = b"DEBiLGI secret vault v1"
_SALT_BYTES = 16
_NONCE_BYTES = 12
_KEY_BYTES = 32
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_MAX_VAULT_BYTES = 128 * 1024
_MIN_PASSPHRASE_LENGTH = 12
_SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_DEFAULT_SECRET_PROMPTS = (
    ("DISCORD_TOKEN", "Discord token", True),
    ("DEBILGI_CLOUDFLARE_ACCOUNT_ID", "Cloudflare account id", False),
    ("DEBILGI_CLOUDFLARE_API_TOKEN", "Cloudflare API token", False),
)


class SecretVaultError(RuntimeError):
    """Base error safe to show without leaking credentials."""


class VaultConfigurationError(SecretVaultError):
    """The vault cannot be created or opened in the current environment."""


class VaultUnlockError(SecretVaultError):
    """The passphrase is wrong or the authenticated vault was modified."""


def create_vault(
    path: Path,
    secrets: dict[str, str],
    passphrase: str,
) -> None:
    """Encrypt secrets and create a new vault without overwriting anything."""

    _validate_passphrase(passphrase)
    clean = _validated_secrets(secrets)
    if not clean.get("DISCORD_TOKEN"):
        raise VaultConfigurationError("DISCORD_TOKEN cannot be empty")
    salt = os.urandom(_SALT_BYTES)
    nonce = os.urandom(_NONCE_BYTES)
    key = _derive_key(passphrase, salt)
    plaintext = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _AAD)
    envelope = {
        "format": VAULT_FORMAT,
        "version": VAULT_VERSION,
        "kdf": {
            "name": "scrypt",
            "salt": _encode(salt),
            "length": _KEY_BYTES,
            "n": _SCRYPT_N,
            "r": _SCRYPT_R,
            "p": _SCRYPT_P,
        },
        "cipher": {
            "name": "AES-256-GCM",
            "nonce": _encode(nonce),
        },
        "ciphertext": _encode(ciphertext),
    }
    encoded = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise VaultConfigurationError(
            f"Secret vault already exists: {path}. Reset it before creating another."
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise
    _restrict_owner_access(path)


def decrypt_vault(path: Path, passphrase: str) -> dict[str, str]:
    """Authenticate and decrypt one vault."""

    _validate_passphrase(passphrase)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise VaultConfigurationError(f"Secret vault does not exist: {path}") from exc
    if len(raw) > _MAX_VAULT_BYTES:
        raise VaultConfigurationError("Secret vault is unexpectedly large")
    try:
        envelope = json.loads(raw)
        _validate_envelope(envelope)
        kdf = envelope["kdf"]
        salt = _decode(kdf["salt"])
        nonce = _decode(envelope["cipher"]["nonce"])
        ciphertext = _decode(envelope["ciphertext"])
        key = _derive_key(
            passphrase,
            salt,
            n=int(kdf["n"]),
            r=int(kdf["r"]),
            p=int(kdf["p"]),
        )
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
        secrets = json.loads(plaintext)
    except InvalidTag as exc:
        raise VaultUnlockError(
            "Wrong master passphrase or the secret vault was modified"
        ) from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VaultConfigurationError("Secret vault has an invalid format") from exc
    if not isinstance(secrets, dict):
        raise VaultConfigurationError("Decrypted secret payload is invalid")
    return _validated_secrets(
        {str(name): str(value) for name, value in secrets.items()}
    )


def prepare_runtime_secrets(project_root: Path) -> None:
    """Unlock the vault, or create it interactively on a fresh installation."""

    vault = project_root.resolve() / VAULT_NAME
    if vault.is_file():
        passphrase = _runtime_passphrase()
        for name, value in decrypt_vault(vault, passphrase).items():
            os.environ.setdefault(name, value)
        return
    if _plaintext_discord_token_available(project_root):
        return
    if not _interactive_terminal_available():
        raise VaultConfigurationError(
            "No DISCORD_TOKEN or encrypted vault is available. Run "
            "`debilgi-secrets setup` in an interactive console."
        )
    print("Первый запуск: создаю зашифрованное хранилище секретов.")
    secrets = _prompt_secrets(project_root)
    passphrase = _prompt_new_passphrase()
    create_vault(vault, secrets, passphrase)
    for name, value in secrets.items():
        os.environ.setdefault(name, value)
    print(f"Зашифрованный vault создан: {vault}")


def reset_vault(path: Path, *, confirmation: str) -> None:
    """Delete a vault only after an exact explicit confirmation."""

    if confirmation != "RESET":
        raise VaultConfigurationError("Reset cancelled: confirmation did not match RESET")
    try:
        path.unlink()
    except FileNotFoundError as exc:
        raise VaultConfigurationError(f"Secret vault does not exist: {path}") from exc


def _derive_key(
    passphrase: str,
    salt: bytes,
    *,
    n: int = _SCRYPT_N,
    r: int = _SCRYPT_R,
    p: int = _SCRYPT_P,
) -> bytes:
    if (
        len(salt) != _SALT_BYTES
        or n < 2**14
        or n > 2**20
        or n & (n - 1)
        or not 1 <= r <= 32
        or not 1 <= p <= 16
    ):
        raise VaultConfigurationError("Secret vault has unsafe KDF parameters")
    return Scrypt(
        salt=salt,
        length=_KEY_BYTES,
        n=n,
        r=r,
        p=p,
    ).derive(passphrase.encode("utf-8"))


def _validate_passphrase(passphrase: str) -> None:
    if len(passphrase) < _MIN_PASSPHRASE_LENGTH:
        raise VaultConfigurationError(
            f"Master passphrase must contain at least {_MIN_PASSPHRASE_LENGTH} characters"
        )


def _validated_secrets(secrets: dict[str, str]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for name, value in secrets.items():
        if not _SECRET_NAME_RE.fullmatch(name):
            raise VaultConfigurationError(f"Invalid secret variable name: {name!r}")
        if "\x00" in value:
            raise VaultConfigurationError(f"Secret {name} contains a null byte")
        if value:
            clean[name] = value
    return clean


def _validate_envelope(value: Any) -> None:
    if not isinstance(value, dict):
        raise VaultConfigurationError("Secret vault envelope must be an object")
    if value.get("format") != VAULT_FORMAT or value.get("version") != VAULT_VERSION:
        raise VaultConfigurationError("Unsupported secret vault format or version")
    kdf = value.get("kdf")
    cipher = value.get("cipher")
    if not isinstance(kdf, dict) or kdf.get("name") != "scrypt":
        raise VaultConfigurationError("Unsupported secret vault KDF")
    if kdf.get("length") != _KEY_BYTES:
        raise VaultConfigurationError("Unsupported secret vault key length")
    if not isinstance(cipher, dict) or cipher.get("name") != "AES-256-GCM":
        raise VaultConfigurationError("Unsupported secret vault cipher")


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode(value: Any) -> bytes:
    if not isinstance(value, str):
        raise VaultConfigurationError("Secret vault contains non-text binary data")
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise VaultConfigurationError("Secret vault contains invalid base64") from exc


def _prompt_secrets(project_root: Path) -> dict[str, str]:
    secrets: dict[str, str] = {}
    prompts = list(_DEFAULT_SECRET_PROMPTS)
    extra_names = _extra_secret_names(project_root)
    prompts.extend((name, name, False) for name in extra_names)
    for name, label, required in prompts:
        suffix = "" if required else " (необязательно)"
        while True:
            value = getpass.getpass(f"{label}{suffix}: ").strip()
            if value or not required:
                break
            print(f"{label} не может быть пустым.")
        if value:
            secrets[name] = value
    return secrets


def _prompt_new_passphrase() -> str:
    while True:
        first = getpass.getpass("Новый мастер-пароль vault: ")
        second = getpass.getpass("Повтори мастер-пароль: ")
        if first != second:
            print("Пароли не совпали.")
            continue
        try:
            _validate_passphrase(first)
        except VaultConfigurationError as exc:
            print(exc)
            continue
        return first


def _runtime_passphrase() -> str:
    configured = os.getenv(VAULT_PASSPHRASE_ENV)
    if configured:
        return configured
    if not _interactive_terminal_available():
        raise VaultConfigurationError(
            "Encrypted vault requires an interactive master passphrase. "
            f"For unattended startup, inject {VAULT_PASSPHRASE_ENV} from an "
            "external secret manager."
        )
    return getpass.getpass("Мастер-пароль vault: ")


def _extra_secret_names(project_root: Path) -> tuple[str, ...]:
    raw = (
        os.getenv("DEBILGI_VAULT_EXTRA_KEYS")
        or _config_value(project_root / ".env.local", "DEBILGI_VAULT_EXTRA_KEYS")
        or _config_value(project_root / ".env", "DEBILGI_VAULT_EXTRA_KEYS")
        or ""
    )
    result = tuple(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    for name in result:
        if not _SECRET_NAME_RE.fullmatch(name):
            raise VaultConfigurationError(f"Invalid extra secret variable name: {name!r}")
    return result


def _plaintext_discord_token_available(project_root: Path) -> bool:
    return bool(
        os.getenv("DISCORD_TOKEN")
        or _config_value(project_root / ".env.local", "DISCORD_TOKEN")
        or _config_value(project_root / ".env", "DISCORD_TOKEN")
    )


def _config_value(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    value = dotenv_values(path).get(name)
    return value if value else None


def _interactive_terminal_available() -> bool:
    return bool(sys.stdin.isatty() and sys.stderr.isatty())


def _restrict_owner_access(path: Path) -> None:
    with suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, inspect or reset DEBiLGI's encrypted secret vault.",
    )
    parser.add_argument(
        "command",
        choices=("setup", "status", "reset"),
        nargs="?",
        default="setup",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    vault = args.project_root.resolve() / VAULT_NAME
    try:
        if args.command == "status":
            print(f"vault={'present' if vault.is_file() else 'missing'} path={vault}")
            return 0
        if args.command == "reset":
            confirmation = input(
                "Это удалит зашифрованные токены. Введи RESET для подтверждения: "
            )
            reset_vault(vault, confirmation=confirmation)
            print("Vault удалён. Следующий запуск снова запросит токены.")
            return 0
        if vault.exists():
            raise VaultConfigurationError(
                "Vault already exists. Run `debilgi-secrets reset` before setup."
            )
        if not _interactive_terminal_available():
            raise VaultConfigurationError("Vault setup requires an interactive console")
        create_vault(vault, _prompt_secrets(args.project_root), _prompt_new_passphrase())
        print(f"Vault создан: {vault}")
        return 0
    except SecretVaultError as exc:
        print(f"Ошибка vault: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
