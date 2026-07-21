"""Portable environment setup and launcher for DEBiLGI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
MARKER_FILE = VENV_DIR / ".debilgi-bootstrap.json"
MINIMUM_PYTHON = (3, 11)


def environment_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list[str]) -> int:
    printable = " ".join(command)
    print(f"\n> {printable}", flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def require_supported_python() -> None:
    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise SystemExit(
            f"Нужен Python {required} или новее, а сейчас найден Python {current}."
        )


def environment_is_portable() -> bool:
    python = environment_python()
    if not python.is_file():
        return False

    check = (
        "import json, pathlib, sys; "
        "print(json.dumps({'prefix': str(pathlib.Path(sys.prefix).resolve()), "
        "'version': list(sys.version_info[:2])}))"
    )
    completed = subprocess.run(
        [str(python), "-c", check],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False

    try:
        details = json.loads(completed.stdout.strip())
        prefix = Path(details["prefix"])
        version = tuple(details["version"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

    return prefix == VENV_DIR.resolve() and version >= MINIMUM_PYTHON


def project_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update((PROJECT_ROOT / "pyproject.toml").read_bytes())
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode())
    return digest.hexdigest()


def read_marker() -> dict[str, object]:
    try:
        return json.loads(MARKER_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_marker(*, dev: bool) -> None:
    marker = {
        "fingerprint": project_fingerprint(),
        "dev": dev,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    MARKER_FILE.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_environment(*, clear: bool) -> None:
    action = "Пересоздаю" if clear else "Создаю"
    print(f"{action} переносимое окружение {VENV_DIR} ...", flush=True)
    venv.EnvBuilder(with_pip=True, clear=clear).create(VENV_DIR)


def setup_environment(*, dev: bool, recreate: bool) -> None:
    require_supported_python()

    usable = environment_is_portable()
    if recreate or not usable:
        create_environment(clear=VENV_DIR.exists())

    marker = read_marker()
    needs_install = (
        recreate
        or marker.get("fingerprint") != project_fingerprint()
        or (dev and not marker.get("dev"))
    )
    if not needs_install:
        print("Окружение .venv уже готово.", flush=True)
        return

    python = str(environment_python())
    if run([python, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        raise SystemExit("Не удалось обновить pip.")

    package = ".[dev]" if dev else "."
    if run([python, "-m", "pip", "install", "--editable", package]) != 0:
        raise SystemExit("Не удалось установить зависимости проекта.")

    write_marker(dev=dev)
    print("Готово: проект установлен в локальную .venv.", flush=True)


def ensure_env_file() -> bool:
    target = PROJECT_ROOT / ".env"
    if target.exists():
        return True

    example = PROJECT_ROOT / ".env.example"
    if example.exists():
        shutil.copyfile(example, target)
        print(
            "Создан файл .env. Впиши в него DISCORD_TOKEN и снова запусти проект.",
            flush=True,
        )
    else:
        print("Нет файла .env с настройками запуска.", flush=True)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Создание .venv, запуск DEBiLGI и тестов без ручной активации окружения."
    )
    parser.add_argument(
        "command",
        choices=("setup", "run", "test"),
        nargs="?",
        default="run",
        help="Что выполнить (по умолчанию: run).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Установить инструменты разработки (pytest и ruff).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Полностью пересоздать .venv для текущего компьютера.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dev = args.dev or args.command == "test"
    setup_environment(dev=dev, recreate=args.recreate)

    if args.command == "setup":
        return 0
    if args.command == "test":
        return run([str(environment_python()), "-m", "pytest"])
    if not ensure_env_file():
        return 2
    return run([str(environment_python()), "-m", "debilgi.main"])


if __name__ == "__main__":
    raise SystemExit(main())
