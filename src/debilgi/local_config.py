"""Create a Git-ignored, machine-local configuration file safely."""

from __future__ import annotations

import argparse
import os
import stat
from contextlib import suppress
from pathlib import Path

LOCAL_CONFIG_NAME = ".env.local"
LOCAL_CONFIG_EXAMPLE_NAME = ".env.local.example"
_BUILTIN_TEMPLATE = """\
# Private DEBiLGI settings for this machine.
# This file is ignored by Git. Never paste these values into .env.example.
DISCORD_TOKEN=
DEBILGI_CLOUDFLARE_ACCOUNT_ID=
DEBILGI_CLOUDFLARE_API_TOKEN=

# Any DEBILGI_* option may be overridden in this file.
"""


def initialize_local_config(project_root: Path) -> tuple[Path, bool]:
    """Create ``.env.local`` once and return ``(path, created)``.

    Exclusive creation makes the no-overwrite guarantee safe even when two
    setup processes start at nearly the same time.
    """

    root = project_root.resolve()
    target = root / LOCAL_CONFIG_NAME
    if target.exists():
        _restrict_owner_access(target)
        return target, False

    example = root / LOCAL_CONFIG_EXAMPLE_NAME
    content = (
        example.read_text(encoding="utf-8")
        if example.is_file()
        else _BUILTIN_TEMPLATE
    )
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        _restrict_owner_access(target)
        return target, False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    except BaseException:
        with suppress(OSError):
            target.unlink()
        raise
    _restrict_owner_access(target)
    return target, True


def _restrict_owner_access(path: Path) -> None:
    """Apply the strongest portable owner-only mode available.

    POSIX receives real ``0600`` permissions. On Windows, Python maps chmod to
    the supported read/write flags; Git exclusion remains the primary
    protection against accidental publication.
    """

    with suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create DEBiLGI's private, Git-ignored .env.local file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository directory (defaults to the current directory).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    path, created = initialize_local_config(args.project_root)
    action = "Создан" if created else "Уже существует"
    print(f"{action} локальный конфиг: {path}")
    if created:
        print("Заполни DISCORD_TOKEN и нужные секреты; Git этот файл не отслеживает.")
    else:
        print("Существующий файл не изменён.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
