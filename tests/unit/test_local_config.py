import os
import stat
from pathlib import Path

from debilgi.local_config import initialize_local_config


def test_local_config_is_created_from_example_without_overwrite(tmp_path: Path) -> None:
    example = tmp_path / ".env.local.example"
    example.write_text("DISCORD_TOKEN=\nCUSTOM_SETTING=true\n", encoding="utf-8")

    path, created = initialize_local_config(tmp_path)
    path.write_text("DISCORD_TOKEN=private-value\n", encoding="utf-8")
    repeated_path, repeated_created = initialize_local_config(tmp_path)

    assert created is True
    assert repeated_created is False
    assert repeated_path == path
    assert path.read_text(encoding="utf-8") == "DISCORD_TOKEN=private-value\n"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_local_config_has_a_safe_builtin_template(tmp_path: Path) -> None:
    path, created = initialize_local_config(tmp_path)

    content = path.read_text(encoding="utf-8")
    assert created is True
    assert "DISCORD_TOKEN=" in content
    assert "replace-me" not in content
    assert "API_TOKEN=" in content
